"""
Self-supervised contrastive pretraining for MRI slice encoders.

Implements SimCLR-style pretraining: pairs of augmented views of the
same image are pushed together while views of different images are
pushed apart (NT-Xent loss). The resulting encoder weights can then
be used to initialize supervised classification training via
``ClassificationTrainingConfig.pretrained_checkpoint``.

Usage (from the project root):

    python -m scripts.train_self_supervised --config configs/self_supervised.yaml
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.factory import BACKBONE_CLASSIFIERS
from training.classification_trainer import collect_samples
from utils.seed import seed_worker


LOGGER = logging.getLogger(__name__)

SUPPORTED_BACKBONES = tuple(BACKBONE_CLASSIFIERS)


@dataclass
class SelfSupervisedConfig:
    backbone: str = "densenet121"

    epochs: int = 50
    batch_size: int = 32

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    temperature: float = 0.07

    projector_hidden: int = 512
    projection_dim: int = 128

    image_size: tuple[int, int] = (224, 224)

    data_directory: str | Path = "data/raw/classification"

    model_directory: str | Path | None = None
    result_directory: str | Path | None = None

    device: str | None = None

    use_amp: bool = True

    seed: int = 42

    num_workers: int = 0

    resume_from: str | Path | None = None

    def __post_init__(self) -> None:
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {self.backbone}. "
                f"Supported: {SUPPORTED_BACKBONES}"
            )

        if self.model_directory is None:
            self.model_directory = (
                f"checkpoints/self_supervised/{self.backbone}"
            )

        if self.result_directory is None:
            self.result_directory = f"outputs/{self.backbone}_ssl"

        if self.data_directory is None:
            raise ValueError("data_directory must be set.")


@dataclass
class SSLTrainingHistory:
    epoch: int
    loss: float
    learning_rate: float
    epoch_time: float


@dataclass
class SSLTrainingResult:
    best_epoch: int
    best_loss: float
    checkpoint_path: Path
    history: list[dict[str, Any]] = field(default_factory=list)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class ContrastiveAugmentation:
    """Strong augmentation pipeline for contrastive view pairs."""

    def __init__(
        self,
        image_size: tuple[int, int],
    ) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.4, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __call__(self, image: Any) -> torch.Tensor:
        return self.transform(image)


class SSLImageDataset(Dataset):
    """Dataset returning two augmented views of each image."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        augmentation: ContrastiveAugmentation,
    ) -> None:
        self.samples = samples
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, _ = self.samples[index]

        from PIL import Image

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")

        view_one = self.augmentation(image)
        view_two = self.augmentation(image)

        return view_one, view_two


def create_ssl_dataset(
    data_directory: str | Path,
) -> tuple[SSLImageDataset, SSLImageDataset]:
    """Split the training images into pretrain/validation subsets.

    The official Test split is never touched during pretraining.
    """

    data_directory = Path(data_directory)

    train_root = data_directory / "train"

    samples = collect_samples(train_root)

    labels = [label for _, label in samples]

    from sklearn.model_selection import train_test_split

    train_samples, validation_samples = train_test_split(
        samples,
        test_size=0.1,
        random_state=42,
        stratify=labels,
    )

    LOGGER.info("SSL pretrain samples: %d", len(train_samples))
    LOGGER.info("SSL validation samples: %d", len(validation_samples))

    augmentation = ContrastiveAugmentation((224, 224))

    train_dataset = SSLImageDataset(train_samples, augmentation)
    validation_dataset = SSLImageDataset(validation_samples, augmentation)

    return train_dataset, validation_dataset


class ProjectionHead(nn.Module):
    """MLP projection head mapping encoder features to the
    contrastive embedding space."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        projection_dim: int,
    ) -> None:
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projector(features)


class SimCLRModel(nn.Module):
    """Encoder + projection head for contrastive pretraining.

    The backbone classification head is replaced with an identity so
    the encoder returns the pooled feature embedding; the projection
    head maps that embedding into the contrastive space.
    """

    def __init__(
        self,
        backbone: str,
        projector_hidden: int = 512,
        projection_dim: int = 128,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = backbone

        specification = BACKBONE_CLASSIFIERS[backbone]

        encoder = specification["builder"](pretrained)

        setattr(
            encoder,
            specification["head_attr"],
            nn.Identity(),
        )

        self.encoder = nn.Sequential(
            encoder,
            nn.Flatten(),
        )

        self.input_dim = specification["head_out_features"]

        self.projector = ProjectionHead(
            self.input_dim,
            projector_hidden,
            projection_dim,
        )

    def forward(
        self,
        view_one: torch.Tensor,
        view_two: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features_one = self.encoder(view_one)
        features_two = self.encoder(view_two)

        return (
            self.projector(features_one),
            self.projector(features_two),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)


def nt_xent_loss(
    projections_one: torch.Tensor,
    projections_two: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """NT-Xent contrastive loss for SimCLR.

    Args:
        projections_one: (B, D) normalized projections of view one.
        projections_two: (B, D) normalized projections of view two.

    Returns:
        Scalar loss averaged over the batch.
    """

    projections_one = F.normalize(projections_one, dim=1)
    projections_two = F.normalize(projections_two, dim=1)

    batch_size = projections_one.shape[0]

    representations = torch.cat(
        [projections_one, projections_two],
        dim=0,
    )

    similarity = torch.matmul(
        representations,
        representations.T,
    ) / temperature

    similarity = similarity - torch.eye(
        2 * batch_size,
        device=similarity.device,
    ) * 1e9

    labels = torch.cat(
        [
            torch.arange(batch_size, device=similarity.device) + batch_size,
            torch.arange(batch_size, device=similarity.device),
        ]
    )

    loss = F.cross_entropy(
        similarity,
        labels,
    )

    return loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    temperature: float,
    use_amp: bool,
) -> float:
    model.train()

    total_loss = 0.0
    steps = 0

    for view_one, view_two in loader:
        view_one = view_one.to(device, non_blocking=True)
        view_two = view_two.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
        ):
            projection_one, projection_two = model(view_one, view_two)

            loss = criterion(
                projection_one,
                projection_two,
                temperature,
            )

        optimizer.zero_grad()

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    temperature: float,
    use_amp: bool,
) -> float:
    model.eval()

    total_loss = 0.0
    steps = 0

    for view_one, view_two in loader:
        view_one = view_one.to(device, non_blocking=True)
        view_two = view_two.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
        ):
            projection_one, projection_two = model(view_one, view_two)

            loss = criterion(
                projection_one,
                projection_two,
                temperature,
            )

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


def save_ssl_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: SelfSupervisedConfig,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
        },
        path,
    )

    return path


def save_ssl_history(
    history: list[SSLTrainingHistory],
    result_directory: Path,
) -> Path:
    import json

    result_directory.mkdir(parents=True, exist_ok=True)

    path = result_directory / "ssl_training_history.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            [asdict(entry) for entry in history],
            file,
            indent=4,
        )

    return path


def train_self_supervised(
    config: SelfSupervisedConfig | None = None,
) -> SSLTrainingResult:
    if config is None:
        config = SelfSupervisedConfig()

    set_seed(config.seed)

    device = resolve_device(config.device)

    LOGGER.info("Device: %s", device)

    if device.type == "cuda":
        LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))

    train_dataset, validation_dataset = create_ssl_dataset(
        config.data_directory
    )

    pin_memory = device.type == "cuda"

    worker_init_fn = seed_worker if config.num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )

    model = SimCLRModel(
        backbone=config.backbone,
        projector_hidden=config.projector_hidden,
        projection_dim=config.projection_dim,
        pretrained=False,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=config.use_amp and device.type == "cuda",
    )

    model_directory = Path(config.model_directory)
    result_directory = Path(config.result_directory)

    model_directory.mkdir(parents=True, exist_ok=True)
    result_directory.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_epoch = 0

    best_checkpoint = model_directory / "best.pt"
    final_checkpoint = model_directory / "last.pt"

    start_epoch = 1

    if config.resume_from is not None:
        resume_path = Path(config.resume_from)
        if resume_path.exists():
            LOGGER.info("Resuming from checkpoint: %s", resume_path)
            resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(resume_ckpt["model_state_dict"])
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
            saved_loss = resume_ckpt.get("loss")
            if saved_loss is not None:
                best_loss = float(saved_loss)
                best_epoch = int(resume_ckpt.get("epoch", 0))
            LOGGER.info("Resumed from epoch %d, best_loss=%.4f", start_epoch - 1, best_loss)
        else:
            LOGGER.warning("Resume checkpoint not found: %s", resume_path)

    history: list[SSLTrainingHistory] = []

    LOGGER.info("=" * 78)
    LOGGER.info("BrainTumorDiagnosis Self-Supervised Pretraining (SimCLR)")
    LOGGER.info("=" * 78)
    LOGGER.info("Backbone: %s", config.backbone)
    LOGGER.info("Epochs: %d", config.epochs)
    LOGGER.info("Batch size: %d", config.batch_size)
    LOGGER.info("Learning rate: %.6f", config.learning_rate)
    LOGGER.info("Temperature: %.2f", config.temperature)
    LOGGER.info("Projection dim: %d", config.projection_dim)
    LOGGER.info("-" * 78)

    for epoch in range(start_epoch, config.epochs + 1):
        start_time = time.perf_counter()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=nt_xent_loss,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            temperature=config.temperature,
            use_amp=config.use_amp,
        )

        validation_loss = evaluate_loss(
            model=model,
            loader=validation_loader,
            criterion=nt_xent_loss,
            device=device,
            temperature=config.temperature,
            use_amp=config.use_amp,
        )

        scheduler.step()

        epoch_time = time.perf_counter() - start_time

        current_lr = optimizer.param_groups[0]["lr"]

        LOGGER.info(
            "Epoch %d/%d | SSL Loss: %.4f | Validation Loss: %.4f | "
            "LR: %.6f | Time: %.2fs",
            epoch,
            config.epochs,
            train_loss,
            validation_loss,
            current_lr,
            epoch_time,
        )

        history.append(
            SSLTrainingHistory(
                epoch=epoch,
                loss=validation_loss,
                learning_rate=current_lr,
                epoch_time=epoch_time,
            )
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch

            save_ssl_checkpoint(
                best_checkpoint,
                model,
                optimizer,
                epoch,
                validation_loss,
                config,
            )

            LOGGER.info("Best SSL checkpoint saved: %s", best_checkpoint)

        save_ssl_checkpoint(
            final_checkpoint,
            model,
            optimizer,
            epoch,
            validation_loss,
            config,
        )

    LOGGER.info("=" * 78)
    LOGGER.info("Self-supervised pretraining completed.")
    LOGGER.info("Best epoch: %d | Best validation loss: %.4f", best_epoch, best_loss)
    LOGGER.info("Best checkpoint: %s", best_checkpoint)
    LOGGER.info("=" * 78)

    return SSLTrainingResult(
        best_epoch=best_epoch,
        best_loss=best_loss,
        checkpoint_path=best_checkpoint,
        history=[asdict(entry) for entry in history],
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    train_self_supervised()


if __name__ == "__main__":
    main()