"""
BrainTumorDiagnosis unified classification trainer.

One config-driven training engine for every 2D backbone:

    - cnn2d              (project baseline CNN)
    - densenet121        (ImageNet transfer)
    - resnet50           (ImageNet transfer)
    - efficientnet_v2_s  (ImageNet transfer)
    - convnext_tiny      (ImageNet transfer)
    - swin_t             (ImageNet transfer)

Protocol:
    - validation split is held out from the official training data
      (stratified, deterministic seed)
    - the official Test split is NEVER used for validation
    - deterministic seeds, class weights, label smoothing, AdamW,
      ReduceLROnPlateau, mixed precision, gradient accumulation,
      gradient clipping, early stopping
    - checkpoints contain a full environment fingerprint
"""

from __future__ import annotations

import json
import logging
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from losses.classification import ClassificationLoss
from models.factory import (
    ModelConfig,
    create_backbone_classifier,
    create_classification_model,
    model_summary,
)
from utils.seed import seed_worker


LOGGER = logging.getLogger(__name__)


CLASS_NAMES = (
    "Glioma",
    "Meningioma",
    "Pituitary",
    "No Tumor",
)


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


SUPPORTED_BACKBONES = (
    "cnn2d",
    "densenet121",
    "resnet50",
    "efficientnet_v2_s",
    "convnext_tiny",
    "swin_t",
    "vit_b_16",
)


@dataclass
class ClassificationTrainingConfig:
    backbone: str = "cnn2d"

    epochs: int = 30
    batch_size: int = 16

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    num_classes: int = 4
    input_channels: int = 3

    image_size: tuple[int, int] = (224, 224)

    data_directory: str | Path = "data/raw/classification"

    model_directory: str | Path | None = None

    result_directory: str | Path | None = None

    device: str | None = None

    use_amp: bool = True

    seed: int = 42

    validation_split: float = 0.2

    validation_seed: int = 42

    label_smoothing: float = 0.0

    use_class_weights: bool = True

    use_augmentation: bool = True

    num_workers: int = 0

    patience: int = 7

    min_delta: float = 1e-3

    pretrained: bool = True

    pretrained_checkpoint: str | Path | None = None

    resume_from: str | Path | None = None

    stop_flag_path: str | Path | None = None

    progress_file: str | Path | None = None

    grad_accumulation_steps: int = 1

    grad_clip: float | None = 1.0

    base_channels: int = 32

    def __post_init__(self) -> None:
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {self.backbone}. "
                f"Supported: {SUPPORTED_BACKBONES}"
            )

        if self.backbone == "cnn2d":
            self.pretrained = False

        if self.model_directory is None:
            self.model_directory = f"checkpoints/classification/{self.backbone}"

        if self.result_directory is None:
            self.result_directory = f"outputs/{self.backbone}_training"


@dataclass
class EpochResult:
    loss: float
    accuracy: float


@dataclass
class TrainingHistory:
    epoch: int
    train_loss: float
    validation_loss: float
    train_accuracy: float
    validation_accuracy: float
    learning_rate: float
    train_samples: int
    validation_samples: int
    epoch_time: float


@dataclass
class ClassificationTrainingResult:
    best_epoch: int
    best_validation_accuracy: float

    final_train_loss: float
    final_train_accuracy: float

    final_validation_loss: float
    final_validation_accuracy: float

    checkpoint_path: Path


class BrainTumorClassificationDataset(Dataset):
    """Image classification dataset for the unified trainer."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: transforms.Compose,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Unable to read image: {image_path}") from exc

        image = self.transform(image)

        return image, label


def collect_samples(
    root_dir: Path,
) -> list[tuple[Path, int]]:
    """Collect image paths and class labels (case-insensitive class dirs)."""

    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root_dir}")

    class_dirs = {
        directory.name.lower(): directory
        for directory in root_dir.iterdir()
        if directory.is_dir()
    }

    samples: list[tuple[Path, int]] = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = class_dirs.get(class_name.lower())

        if class_dir is None:
            LOGGER.warning(
                "Class directory missing for %s: %s",
                class_name,
                class_dir,
            )
            continue

        image_files = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )

        LOGGER.info(
            "%-10s : %d images",
            class_name,
            len(image_files),
        )

        for image_path in image_files:
            samples.append((image_path, class_index))

    if not samples:
        raise RuntimeError(f"No images found in {root_dir}")

    return samples


def create_train_transform(
    image_size: tuple[int, int],
    augment: bool,
) -> transforms.Compose:
    """Create the training transform (with optional augmentation)."""

    operations: list[Any] = [transforms.Resize(image_size)]

    if augment:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                ),
            ]
        )

    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    return transforms.Compose(operations)


def create_validation_transform(
    image_size: tuple[int, int],
) -> transforms.Compose:
    """Create the deterministic validation/test transform."""

    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def create_dataloaders(
    config: ClassificationTrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders (test set untouched)."""

    data_directory = Path(config.data_directory)

    train_root = data_directory / "train"
    test_root = data_directory / "test"

    all_train_samples = collect_samples(train_root)

    if test_root.exists():
        test_samples = collect_samples(test_root)

        LOGGER.info(
            "Test samples (kept untouched): %d",
            len(test_samples),
        )

    labels = [label for _, label in all_train_samples]

    train_samples, validation_samples = train_test_split(
        all_train_samples,
        test_size=config.validation_split,
        random_state=config.validation_seed,
        stratify=labels,
    )

    LOGGER.info("Total training samples: %d", len(all_train_samples))
    LOGGER.info("Training samples: %d", len(train_samples))
    LOGGER.info("Validation samples: %d", len(validation_samples))

    train_dataset = BrainTumorClassificationDataset(
        train_samples,
        create_train_transform(
            config.image_size,
            config.use_augmentation,
        ),
    )

    validation_dataset = BrainTumorClassificationDataset(
        validation_samples,
        create_validation_transform(config.image_size),
    )

    pin_memory = torch.cuda.is_available()

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

    return train_loader, validation_loader


def create_class_weights(
    samples: list[tuple[Path, int]],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Create inverse-frequency class weights normalized to mean 1."""

    counts = torch.zeros(num_classes, dtype=torch.float32)

    for _, label in samples:
        counts[label] += 1.0

    counts = counts.clamp(min=1.0)

    weights = counts.sum() / (num_classes * counts)

    weights = weights / weights.mean()

    LOGGER.info("Class weights: %s", weights.tolist())

    return weights.to(device)


def create_model(
    config: ClassificationTrainingConfig,
    device: torch.device,
) -> nn.Module:
    """Create the classification model for the configured backbone."""

    if config.backbone == "cnn2d":
        model = create_classification_model(
            ModelConfig(
                model_type="classification",
                in_channels=config.input_channels,
                num_classes=config.num_classes,
                classification_input_size=config.image_size,
                base_channels=config.base_channels,
            )
        )
    else:
        model = create_backbone_classifier(
            backbone=config.backbone,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
        )

    if config.pretrained_checkpoint is not None:
        checkpoint_path = Path(config.pretrained_checkpoint)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)

        if isinstance(state_dict, dict) and "encoder" in state_dict:
            state_dict = state_dict["encoder"]

        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=False,
        )

        LOGGER.info(
            "Initialized from pretrained checkpoint: %s "
            "(missing %d, unexpected %d)",
            checkpoint_path,
            len(missing),
            len(unexpected),
        )

    model = model.to(device)

    summary = model_summary(model)

    LOGGER.info(
        "Model: %s | parameters: %d (trainable %d)",
        config.backbone,
        summary["total_parameters"],
        summary["trainable_parameters"],
    )

    return model


def set_seed(seed: int) -> None:
    """Set all random seeds deterministically."""

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


def environment_fingerprint() -> dict[str, Any]:
    """Capture the training environment for checkpoint provenance."""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _torchvision_version(),
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _torchvision_version() -> str:
    try:
        import torchvision

        return torchvision.__version__
    except ImportError:
        return "unknown"


def save_training_history(
    history: list[TrainingHistory],
    result_directory: Path,
) -> Path:
    result_directory.mkdir(parents=True, exist_ok=True)

    history_path = result_directory / "training_history.json"

    payload = [asdict(item) for item in history]

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)

    return history_path


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    epoch: int,
    validation_accuracy: float,
    config: ClassificationTrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "validation_accuracy": validation_accuracy,
        "class_names": CLASS_NAMES,
        "num_classes": config.num_classes,
        "input_channels": config.input_channels,
        "image_size": config.image_size,
        "model_name": config.backbone,
        "backbone": config.backbone,
        "dataset_root": str(config.data_directory),
        "config": asdict(config),
        "environment": environment_fingerprint(),
    }

    torch.save(checkpoint, path)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    grad_accumulation_steps: int,
    grad_clip: float | None,
) -> EpochResult:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    amp_enabled = use_amp and device.type == "cuda"

    optimizer.zero_grad(set_to_none=True)

    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaled_loss = loss / grad_accumulation_steps

        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if batch_index % grad_accumulation_steps == 0:
            if grad_clip is not None:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                )

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

        if batch_index % 5 == 0:
            running_loss = total_loss / total_samples
            running_acc = total_correct / total_samples
            print(
                f"  [Batch {batch_index}/{len(loader)}] loss={running_loss:.4f} acc={running_acc*100:.1f}%",
                flush=True,
            )

    if total_samples == 0:
        return EpochResult(loss=0.0, accuracy=0.0)

    return EpochResult(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> EpochResult:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    amp_enabled = use_amp and device.type == "cuda"

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        return EpochResult(loss=0.0, accuracy=0.0)

    return EpochResult(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
    )


def create_training_components(
    config: ClassificationTrainingConfig,
):
    device = resolve_device(config.device)

    LOGGER.info("Device: %s", device)

    if device.type == "cuda":
        LOGGER.info("GPU: %s", torch.cuda.get_device_name(0))
        LOGGER.info("GPU memory: %.2f GB",
                    torch.cuda.get_device_properties(0).total_memory / (1024**3))

    LOGGER.info(
        "Validation split: %.2f (held out from training data; "
        "official Test split is never used for validation).",
        config.validation_split,
    )

    train_loader, validation_loader = create_dataloaders(config)

    model = create_model(config, device)

    class_weights = None

    if config.use_class_weights:
        class_weights = create_class_weights(
            train_loader.dataset.samples,
            config.num_classes,
            device,
        )

    criterion = ClassificationLoss(
        label_smoothing=config.label_smoothing,
        class_weights=class_weights,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(config.use_amp and device.type == "cuda"),
    )

    return (
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        scheduler,
        scaler,
        device,
    )


def _write_progress_file(
    config: ClassificationTrainingConfig,
    epoch: int,
    best_accuracy: float,
    best_epoch: int,
    validation_accuracy: float,
    device: torch.device,
) -> None:
    """Write a small JSON progress file after each epoch.

    The training manager / watch command polls this file for
    live progress reporting without coupling to the trainer.
    """

    import datetime as _datetime

    path = Path(config.progress_file)

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "backbone": config.backbone,
        "epoch": epoch,
        "total_epochs": config.epochs,
        "best_accuracy": round(best_accuracy, 6),
        "best_epoch": best_epoch,
        "validation_accuracy": round(validation_accuracy, 6),
        "device": str(device),
        "updated_at": _datetime.datetime.now().isoformat(timespec="seconds"),
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _load_resume_state(
    config: ClassificationTrainingConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, int, int, list[TrainingHistory]]:
    """Restore training state from a checkpoint for resumption.

    Returns:
        (start_epoch, best_accuracy, best_epoch, epochs_without_improvement,
         history)
    """

    checkpoint_path = Path(config.resume_from)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Resume checkpoint has no model_state_dict: {checkpoint_path}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1

    best_accuracy = float(checkpoint.get("validation_accuracy", -1.0))

    best_epoch = int(checkpoint.get("epoch", 0)) if best_accuracy > 0 else 0

    epochs_without_improvement = 0

    history: list[TrainingHistory] = []

    history_file = Path(config.result_directory) / "training_history.json"

    if history_file.exists():
        with history_file.open("r", encoding="utf-8") as file:
            history = [
                TrainingHistory(**entry)
                for entry in json.load(file)
            ]

    LOGGER.info(
        "Resuming %s from epoch %d (checkpoint: %s)",
        config.backbone,
        start_epoch - 1,
        checkpoint_path,
    )

    return start_epoch, best_accuracy, best_epoch, epochs_without_improvement, history


def _stop_requested(config: ClassificationTrainingConfig) -> bool:
    if config.stop_flag_path is None:
        return False

    return Path(config.stop_flag_path).exists()


def train_classification_model(
    config: ClassificationTrainingConfig | None = None,
) -> ClassificationTrainingResult:

    if config is None:
        config = ClassificationTrainingConfig()

    set_seed(config.seed)

    result_directory = Path(config.result_directory)
    model_directory = Path(config.model_directory)

    result_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    (
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        scheduler,
        scaler,
        device,
    ) = create_training_components(config)

    best_accuracy = -1.0
    best_epoch = 0

    best_checkpoint = model_directory / "best.pt"
    final_checkpoint = model_directory / "last.pt"

    epochs_without_improvement = 0

    history: list[TrainingHistory] = []

    start_epoch = 1

    if config.resume_from is not None:
        start_epoch, best_accuracy, best_epoch, epochs_without_improvement, history = (
            _load_resume_state(
                config,
                model,
                optimizer,
                scheduler,
                scaler,
                device,
            )
        )

    final_train = EpochResult(loss=0.0, accuracy=0.0)
    final_validation = EpochResult(loss=0.0, accuracy=0.0)

    LOGGER.info("=" * 78)
    LOGGER.info("BrainTumorDiagnosis Classification Training")
    LOGGER.info("=" * 78)
    LOGGER.info("Backbone: %s", config.backbone)
    LOGGER.info("Classes: %s", CLASS_NAMES)
    LOGGER.info("Input channels: %d", config.input_channels)
    LOGGER.info("Image size: %s", config.image_size)
    LOGGER.info("Epochs: %d", config.epochs)
    LOGGER.info("Batch size: %d", config.batch_size)
    LOGGER.info("Learning rate: %.6f", config.learning_rate)
    LOGGER.info("Pretrained: %s", config.pretrained)
    LOGGER.info("Label smoothing: %.2f", config.label_smoothing)
    LOGGER.info("Class weights: %s", config.use_class_weights)
    LOGGER.info("Augmentation: %s", config.use_augmentation)
    LOGGER.info("Gradient accumulation: %d", config.grad_accumulation_steps)
    LOGGER.info("Gradient clipping: %s", config.grad_clip)
    LOGGER.info("AMP: %s", config.use_amp and device.type == "cuda")
    LOGGER.info("Resume from: %s", config.resume_from)
    LOGGER.info("Stop flag: %s", config.stop_flag_path)

    for epoch in range(start_epoch, config.epochs + 1):
        if _stop_requested(config):
            LOGGER.info("Stop flag detected. Saving state and exiting.")
            break

        LOGGER.info("-" * 78)
        LOGGER.info("Epoch %d/%d", epoch, config.epochs)

        start_time = time.perf_counter()

        final_train = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=config.use_amp,
            grad_accumulation_steps=config.grad_accumulation_steps,
            grad_clip=config.grad_clip,
        )

        final_validation = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            use_amp=config.use_amp,
        )

        scheduler.step(final_validation.accuracy)

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.perf_counter() - start_time

        print(
            f"  Epoch {epoch}/{config.epochs} | "
            f"train_loss={final_train.loss:.4f} train_acc={final_train.accuracy*100:.1f}% | "
            f"val_loss={final_validation.loss:.4f} val_acc={final_validation.accuracy*100:.1f}% | "
            f"lr={current_lr:.6f} | time={epoch_time:.1f}s",
            flush=True,
        )

        epoch_history = TrainingHistory(
            epoch=epoch,
            train_loss=final_train.loss,
            validation_loss=final_validation.loss,
            train_accuracy=final_train.accuracy,
            validation_accuracy=final_validation.accuracy,
            learning_rate=current_lr,
            train_samples=len(train_loader.dataset),
            validation_samples=len(validation_loader.dataset),
            epoch_time=epoch_time,
        )

        history.append(epoch_history)

        save_training_history(history, result_directory)

        if config.progress_file is not None:
            _write_progress_file(
                config,
                epoch,
                best_accuracy,
                best_epoch,
                final_validation.accuracy,
                device,
            )

        save_checkpoint(
            path=final_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            validation_accuracy=final_validation.accuracy,
            config=config,
        )

        if final_validation.accuracy > best_accuracy + config.min_delta:
            best_accuracy = final_validation.accuracy
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=best_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                validation_accuracy=best_accuracy,
                config=config,
            )

            LOGGER.info("Best checkpoint saved: %s", best_checkpoint)
        else:
            epochs_without_improvement += 1

            LOGGER.info(
                "No improvement. Patience: %d/%d",
                epochs_without_improvement,
                config.patience,
            )

        if epochs_without_improvement >= config.patience:
            LOGGER.info("Early stopping.")
            break

    save_training_history(history, result_directory)

    LOGGER.info("=" * 78)
    LOGGER.info("Classification training completed.")
    LOGGER.info("Best epoch: %d", best_epoch)
    LOGGER.info("Best validation accuracy: %.2f%%", best_accuracy * 100)
    LOGGER.info("Best checkpoint: %s", best_checkpoint)
    LOGGER.info("=" * 78)

    return ClassificationTrainingResult(
        best_epoch=best_epoch,
        best_validation_accuracy=best_accuracy,
        final_train_loss=final_train.loss,
        final_train_accuracy=final_train.accuracy,
        final_validation_loss=final_validation.loss,
        final_validation_accuracy=final_validation.accuracy,
        checkpoint_path=best_checkpoint,
    )