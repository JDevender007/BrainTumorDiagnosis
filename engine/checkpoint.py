"""
Checkpoint management for BrainTumorDiagnosis.

Handles:
- Saving model checkpoints
- Loading checkpoints
- Best-model tracking
- Last-model tracking
- Optimizer and scheduler state
- Training metadata
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


LOGGER = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside a checkpoint."""

    epoch: int
    monitor_name: str
    monitor_value: float
    is_best: bool = False


class CheckpointManager:
    """
    Manage training checkpoints.

    Directory structure:

        checkpoints/
        └── experiment/
            ├── best/
            │   └── experiment.pt
            ├── last/
            │   └── experiment.pt
            └── metadata.json
    """

    def __init__(
        self,
        directory: str | Path,
        experiment_name: str = "model",
    ) -> None:
        self.directory = Path(directory)
        self.experiment_name = experiment_name

        self.best_directory = self.directory / "best"
        self.last_directory = self.directory / "last"

        self.best_directory.mkdir(parents=True, exist_ok=True)
        self.last_directory.mkdir(parents=True, exist_ok=True)

        self.best_path = self.best_directory / f"{self.experiment_name}.pt"

        self.last_path = self.last_directory / f"{self.experiment_name}.pt"

        self.metadata_path = self.directory / "metadata.json"

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        epoch: int,
        monitor_name: str,
        monitor_value: float,
        config: Any | None = None,
        history: Any | None = None,
        extra: dict[str, Any] | None = None,
        is_best: bool = False,
        path: str | Path | None = None,
    ) -> Path:
        """
        Save a complete checkpoint.
        """

        checkpoint: dict[str, Any] = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "monitor_name": monitor_name,
            "monitor_value": float(monitor_value),
            "is_best": bool(is_best),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        if config is not None:
            if hasattr(config, "__dataclass_fields__"):
                checkpoint["config"] = asdict(config)
            else:
                checkpoint["config"] = config

        if history is not None:
            if hasattr(history, "__dataclass_fields__"):
                checkpoint["history"] = asdict(history)
            elif hasattr(history, "to_dict"):
                checkpoint["history"] = history.to_dict()
            else:
                checkpoint["history"] = history

        if extra:
            checkpoint["extra"] = extra

        target_path = (
            Path(path)
            if path is not None
            else self.best_path if is_best else self.last_path
        )

        target_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            checkpoint,
            target_path,
        )

        metadata = CheckpointMetadata(
            epoch=epoch,
            monitor_name=monitor_name,
            monitor_value=monitor_value,
            is_best=is_best,
        )

        self._write_metadata(metadata)

        LOGGER.info(
            "Checkpoint saved: %s",
            target_path,
        )

        return target_path

    def load(
        self,
        path: str | Path,
        *,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> dict[str, Any]:
        """
        Load a checkpoint and optionally restore training state.
        """

        checkpoint_path = Path(path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )

        if model is not None:
            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=strict,
            )

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        LOGGER.info(
            "Checkpoint loaded: %s",
            checkpoint_path,
        )

        return checkpoint

    def load_best(
        self,
        *,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load the best checkpoint."""

        return self.load(
            self.best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=map_location,
            strict=strict,
        )

    def load_last(
        self,
        *,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load the last checkpoint."""

        return self.load(
            self.last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=map_location,
            strict=strict,
        )

    def exists_best(self) -> bool:
        """Return whether a best checkpoint exists."""

        return self.best_path.exists()

    def exists_last(self) -> bool:
        """Return whether a last checkpoint exists."""

        return self.last_path.exists()

    def _write_metadata(
        self,
        metadata: CheckpointMetadata,
    ) -> None:
        """Write checkpoint metadata to JSON."""

        self.metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                asdict(metadata),
                file,
                indent=4,
            )

    def get_paths(self) -> dict[str, str]:
        """Return checkpoint paths."""

        return {
            "best": str(self.best_path),
            "last": str(self.last_path),
            "metadata": str(self.metadata_path),
        }


def _test_checkpoint_manager() -> None:
    """Run a small checkpoint manager test."""

    import tempfile

    model = torch.nn.Linear(4, 2)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager(
            temp_dir,
            experiment_name="checkpoint_test",
        )

        best_path = manager.save(
            model=model,
            optimizer=optimizer,
            epoch=1,
            monitor_name="val_loss",
            monitor_value=0.5,
            is_best=True,
        )

        last_path = manager.save(
            model=model,
            optimizer=optimizer,
            epoch=1,
            monitor_name="val_loss",
            monitor_value=0.5,
            is_best=False,
        )

        assert best_path.exists()
        assert last_path.exists()

        restored_model = torch.nn.Linear(4, 2)

        manager.load_best(
            model=restored_model,
        )

        manager.load_last(
            model=restored_model,
        )

        LOGGER.info("Checkpoint manager test passed.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | %(levelname)s | " "%(name)s | %(message)s"),
    )

    _test_checkpoint_manager()
