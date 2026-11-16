"""
Segmentation trainer for BrainTumorDiagnosis.

Trains a 3D U-Net on BraTS 2024 post-treatment patches with:

    - patient-level split manifest
    - foreground-biased patch dataset
    - combined Dice + cross-entropy loss
    - AdamW + cosine annealing
    - AMP (mixed precision) when CUDA is available
    - patch-based validation each epoch
    - early stopping on mean foreground Dice
    - CheckpointManager best/last checkpoints
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from datasets.segmentation import (
    SegmentationDataset,
    SegmentationDatasetConfig,
    load_split_cases,
)
from engine.checkpoint import CheckpointManager
from losses.segmentation import DiceLoss3D, SegmentationLoss
from models.segmentation import create_segmentation_model
from utils.device import get_device
from utils.metrics import dice_score
from utils.seed import set_seed


LOGGER = logging.getLogger(__name__)

FOREGROUND_LABELS = (1, 2, 3, 4)


@dataclass
class SegmentationTrainerConfig:
    """Configuration for the segmentation trainer."""

    in_channels: int = 4
    out_channels: int = 5
    base_channels: int = 16
    dropout: float = 0.1

    patch_size: tuple[int, int, int] = (96, 96, 96)
    samples_per_case: int = 8
    foreground_ratio: float = 0.6
    augment: bool = True

    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = False

    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    use_amp: bool = True
    gradient_clip_norm: float | None = 1.0

    dice_weight: float = 0.5
    ce_weight: float = 0.5

    early_stopping_patience: int = 15

    manifest_path: str = "data/splits/brats/split_manifest.json"
    checkpoint_dir: str = "checkpoints/segmentation/unet3d"
    history_path: str = "outputs/segmentation_unet3d/training_history.json"

    seed: int = 42


class SegmentationTrainer:
    """
    Train and validate a 3D segmentation model.
    """

    def __init__(
        self,
        config: SegmentationTrainerConfig,
    ) -> None:
        self.config = config

        self.device = get_device()

        self._run_epochs: int | None = None

        self.manager = CheckpointManager(
            config.checkpoint_dir,
            experiment_name="unet3d_segmentation",
        )

        self.history: dict[str, Any] = {
            "config": asdict(config),
            "epochs": [],
        }

        set_seed(config.seed)

    def _build_loaders(
        self,
        train_cases,
        validation_cases,
        smoke: bool = False,
    ) -> tuple[DataLoader, DataLoader, SegmentationDataset, SegmentationDataset]:
        dataset_config = SegmentationDatasetConfig(
            patch_size=self.config.patch_size,
            samples_per_case=(
                2 if smoke else self.config.samples_per_case
            ),
            foreground_ratio=self.config.foreground_ratio,
            augment=self.config.augment,
        )

        train_dataset = SegmentationDataset(
            train_cases,
            config=dataset_config,
            seed=self.config.seed,
        )

        validation_dataset = SegmentationDataset(
            validation_cases,
            config=dataset_config,
            seed=self.config.seed + 1,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=(
                2 if smoke else self.config.batch_size
            ),
            shuffle=True,
            num_workers=(
                0 if smoke else self.config.num_workers
            ),
            pin_memory=self.config.pin_memory,
            drop_last=False,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=(
                2 if smoke else self.config.batch_size
            ),
            shuffle=False,
            num_workers=(
                0 if smoke else self.config.num_workers
            ),
            pin_memory=self.config.pin_memory,
        )

        return (
            train_loader,
            validation_loader,
            train_dataset,
            validation_dataset,
        )

    def _build_model(
        self,
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer, Any]:
        model = create_segmentation_model(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            base_channels=self.config.base_channels,
            dropout=self.config.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs,
        )

        return model, optimizer, scheduler

    def _forward_loss(
        self,
        batch: dict[str, torch.Tensor],
        model: torch.nn.Module,
        criterion: torch.nn.Module,
    ) -> torch.Tensor:
        image = batch["image"].to(
            self.device,
            non_blocking=True,
        )

        label = batch["label"].to(
            self.device,
            non_blocking=True,
        )

        logits = model(image)

        return criterion(
            logits,
            label,
        )

    def _train_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scheduler: Any,
        loader: DataLoader,
        scaler: Any,
        epoch: int,
        batch_log_interval: int = 25,
    ) -> float:
        model.train()

        running_loss = 0.0
        num_batches = 0
        total_batches = len(loader)

        use_amp = (
            self.config.use_amp
            and self.device.type == "cuda"
            and scaler is not None
        )

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    loss = self._forward_loss(
                        batch,
                        model,
                        criterion,
                    )

                scaler.scale(loss).backward()

                if self.config.gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clip_norm,
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss = self._forward_loss(
                    batch,
                    model,
                    criterion,
                )

                loss.backward()

                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clip_norm,
                    )

                optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

            if (
                batch_log_interval > 0
                and total_batches > 0
                and (
                    num_batches % batch_log_interval == 0
                    or num_batches == total_batches
                )
            ):
                LOGGER.info(
                    "Train epoch %d/%d | batch %d/%d | %.1f%% | loss %.4f",
                    epoch + 1,
                    self._run_epochs or self.config.epochs,
                    num_batches,
                    total_batches,
                    num_batches / total_batches * 100.0,
                    running_loss / num_batches,
                )

        scheduler.step()

        epoch_loss = running_loss / max(num_batches, 1)

        LOGGER.info(
            "Train epoch %d/%d | loss %.4f",
            epoch + 1,
            self._run_epochs or self.config.epochs,
            epoch_loss,
        )

        return epoch_loss

    @torch.no_grad()
    def _validate_epoch(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        model.eval()

        running_loss = 0.0
        num_batches = 0

        class_intersections = {
            label: 0.0 for label in FOREGROUND_LABELS
        }
        class_denominators = {
            label: 0.0 for label in FOREGROUND_LABELS
        }

        for batch in loader:
            image = batch["image"].to(self.device)
            label = batch["label"].to(self.device)

            logits = model(image)

            loss = criterion(
                logits,
                label,
            )

            running_loss += float(loss.item())
            num_batches += 1

            prediction = torch.argmax(
                logits,
                dim=1,
            )

            for class_label in FOREGROUND_LABELS:
                prediction_mask = (prediction == class_label).float()
                target_mask = (label == class_label).float()

                intersection = torch.sum(
                    prediction_mask * target_mask
                ).item()

                denominator = (
                    torch.sum(prediction_mask).item()
                    + torch.sum(target_mask).item()
                )

                class_intersections[class_label] += intersection
                class_denominators[class_label] += denominator

        class_dice = {}

        for class_label in FOREGROUND_LABELS:
            intersection = class_intersections[class_label]
            denominator = class_denominators[class_label]

            class_dice[f"dice_label_{class_label}"] = (
                (2.0 * intersection + 1e-5) / (denominator + 1e-5)
            )

        mean_dice = sum(class_dice.values()) / len(class_dice)

        mean_loss = running_loss / max(num_batches, 1)

        metrics = {
            "val_loss": mean_loss,
            "val_dice_mean": mean_dice,
            **class_dice,
        }

        LOGGER.info(
            "Val epoch %d/%d | loss %.4f | mean foreground dice %.4f",
            epoch + 1,
            self._run_epochs or self.config.epochs,
            mean_loss,
            mean_dice,
        )

        return metrics

    def run(
        self,
        smoke: bool = False,
        max_epochs: int | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        epochs = (
            max_epochs
            if max_epochs is not None and max_epochs > 0
            else self.config.epochs
        )

        self._run_epochs = epochs

        train_cases = load_split_cases(
            self.config.manifest_path,
            "train",
        )

        validation_cases = load_split_cases(
            self.config.manifest_path,
            "validation",
        )

        if smoke:
            train_cases = train_cases[:2]
            validation_cases = validation_cases[:2]

        (
            train_loader,
            validation_loader,
            train_dataset,
            validation_dataset,
        ) = self._build_loaders(
            train_cases,
            validation_cases,
            smoke=smoke,
        )

        model, optimizer, scheduler = self._build_model()

        criterion = SegmentationLoss(
            dice_weight=self.config.dice_weight,
            ce_weight=self.config.ce_weight,
        )

        scaler = None

        if self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")

        LOGGER.info(
            "Training pipeline ready | train cases=%d | val cases=%d | "
            "patch=%s | epochs=%d | params=%d",
            len(train_cases),
            len(validation_cases),
            self.config.patch_size,
            epochs,
            sum(
                parameter.numel() for parameter in model.parameters()
            ),
        )

        best_dice = -1.0
        patience_counter = 0
        stopped_early = False

        start_epoch = 0

        if resume and (
            self.manager.exists_last() or self.manager.exists_best()
        ):
            if self.manager.exists_last():
                resume_path = self.manager.last_path
            else:
                resume_path = self.manager.best_path

            if (
                self.manager.exists_best()
                and self.manager.exists_last()
                and self.manager.best_path.stat().st_mtime
                > self.manager.last_path.stat().st_mtime
            ):
                resume_path = self.manager.best_path

            checkpoint = self.manager.load(
                resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                map_location=self.device,
            )

            start_epoch = int(checkpoint.get("epoch", 0))

            if self.manager.exists_best():
                best_path = self.manager.best_path

                best_checkpoint = torch.load(
                    best_path,
                    map_location="cpu",
                    weights_only=False,
                )

                best_dice = float(
                    best_checkpoint.get("monitor_value", -1.0)
                )

            history_path = Path(self.config.history_path)

            if history_path.exists():
                try:
                    with history_path.open("r", encoding="utf-8") as file:
                        previous_history = json.load(file)

                    self.history["epochs"] = previous_history.get(
                        "epochs",
                        [],
                    )
                except (json.JSONDecodeError, OSError):
                    LOGGER.warning(
                        "Could not read previous history: %s",
                        history_path,
                    )

            LOGGER.info(
                "Resuming training from checkpoint %s (epoch %d/%d, "
                "best val dice so far: %.4f).",
                resume_path,
                start_epoch + 1,
                epochs,
                best_dice,
            )

            if start_epoch >= epochs:
                LOGGER.info(
                    "Checkpoint epoch %d already reached the target "
                    "of %d epochs. Nothing to run.",
                    start_epoch,
                    epochs,
                )

                return {
                    "best_val_dice_mean": best_dice,
                    "epochs_completed": start_epoch,
                    "early_stopped": False,
                    "history_path": self.config.history_path,
                }

        completed_epochs = start_epoch

        for epoch in range(start_epoch, epochs):
            train_dataset.reseed(epoch)
            validation_dataset.reseed(epoch)

            LOGGER.info(
                "Start epoch %d/%d",
                epoch + 1,
                epochs,
            )

            train_loss = self._train_epoch(
                model,
                optimizer,
                criterion,
                scheduler,
                train_loader,
                scaler,
                epoch,
            )

            validation_metrics = self._validate_epoch(
                model,
                criterion,
                validation_loader,
                epoch,
            )

            current_dice = validation_metrics["val_dice_mean"]

            entry = {
                "epoch": epoch + 1,
                "learning_rate": scheduler.get_last_lr()[0],
                "train_loss": train_loss,
                **validation_metrics,
            }

            self.history["epochs"].append(entry)

            is_best = current_dice > best_dice

            if is_best:
                best_dice = current_dice
                patience_counter = 0
            else:
                patience_counter += 1

            completed_epochs = epoch + 1

            self.manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                monitor_name="val_dice_mean",
                monitor_value=current_dice,
                config=self.config,
            )

            if is_best:
                self.manager.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    monitor_name="val_dice_mean",
                    monitor_value=current_dice,
                    config=self.config,
                    is_best=True,
                )

            if (
                self.config.early_stopping_patience > 0
                and patience_counter >= self.config.early_stopping_patience
            ):
                LOGGER.info(
                    "Early stopping after %d epochs without improvement.",
                    patience_counter,
                )
                stopped_early = True
                break

        self._save_history(best_dice, stopped_early)

        return {
            "best_val_dice_mean": best_dice,
            "epochs_completed": completed_epochs,
            "early_stopped": stopped_early,
            "history_path": self.config.history_path,
        }

    def _save_history(
        self,
        best_dice: float,
        stopped_early: bool,
    ) -> None:
        history_path = Path(self.config.history_path)

        history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history["best_val_dice_mean"] = best_dice
        self.history["early_stopped"] = stopped_early

        with history_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.history,
                file,
                indent=4,
            )

        LOGGER.info(
            "Training history saved: %s",
            history_path,
        )


def _test() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    import tempfile

    from datasets.segmentation import create_synthetic_brats_cases

    temp_dir = tempfile.TemporaryDirectory()

    cases = create_synthetic_brats_cases(
        Path(temp_dir.name),
        num_cases=6,
        volume_shape=(32, 36, 40),
    )

    config = SegmentationTrainerConfig(
        patch_size=(16, 16, 16),
        samples_per_case=2,
        batch_size=2,
        base_channels=2,
        epochs=3,
        learning_rate=1e-3,
        use_amp=False,
        checkpoint_dir=str(Path(temp_dir.name) / "checkpoints"),
        history_path=str(Path(temp_dir.name) / "history.json"),
        seed=11,
    )

    trainer = SegmentationTrainer(config)

    train_cases = cases[:4]
    validation_cases = cases[4:]

    (
        train_loader,
        validation_loader,
        train_dataset,
        validation_dataset,
    ) = trainer._build_loaders(
        train_cases,
        validation_cases,
        smoke=True,
    )

    model, optimizer, scheduler = trainer._build_model()

    criterion = SegmentationLoss()

    scaler = None

    if trainer.device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    for epoch in range(config.epochs):
        train_dataset.reseed(epoch)
        validation_dataset.reseed(epoch)

        train_loss = trainer._train_epoch(
            model,
            optimizer,
            criterion,
            scheduler,
            train_loader,
            scaler,
            epoch,
        )

        assert torch.isfinite(torch.tensor(train_loss))

        metrics = trainer._validate_epoch(
            model,
            criterion,
            validation_loader,
            epoch,
        )

        assert 0.0 <= metrics["val_dice_mean"] <= 1.0

    LOGGER.info(
        "Final validation dice: %.4f",
        metrics["val_dice_mean"],
    )

    temp_dir.cleanup()

    LOGGER.info("Segmentation trainer test passed.")


if __name__ == "__main__":
    _test()