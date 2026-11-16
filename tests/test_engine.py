"""Tests for checkpoint management and inference utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from engine.checkpoint import CheckpointManager
from engine.inference import InferenceResult, _resolve_classification_checkpoint
from utils.device import get_cuda_memory, get_device, get_device_name
from utils.metrics import dice_score, iou_score


def make_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
    )


def test_checkpoint_save_and_load_roundtrip(tmp_path) -> None:
    manager = CheckpointManager(
        directory=tmp_path,
        experiment_name="roundtrip",
    )

    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    best_path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        monitor_name="val_accuracy",
        monitor_value=0.87,
        is_best=True,
    )

    assert best_path.exists()
    assert manager.exists_best()
    assert not manager.exists_last()

    restored = make_model()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)

    checkpoint = manager.load_best(
        model=restored,
        optimizer=restored_optimizer,
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["monitor_value"] == pytest.approx(0.87)

    for p1, p2 in zip(model.parameters(), restored.parameters()):
        assert torch.equal(p1.detach(), p2.detach())


def test_checkpoint_metadata_written(tmp_path) -> None:
    manager = CheckpointManager(
        directory=tmp_path,
        experiment_name="meta",
    )

    manager.save(
        model=make_model(),
        epoch=1,
        monitor_name="val_loss",
        monitor_value=0.5,
    )

    metadata = json.loads(manager.metadata_path.read_text(encoding="utf-8"))

    assert metadata["epoch"] == 1
    assert metadata["monitor_name"] == "val_loss"


def test_checkpoint_missing_raises(tmp_path) -> None:
    manager = CheckpointManager(
        directory=tmp_path,
        experiment_name="missing",
    )

    with pytest.raises(FileNotFoundError):
        manager.load_best()


def test_resolve_checkpoint_precedence(tmp_path, monkeypatch) -> None:
    explicit = tmp_path / "explicit.pt"
    fallback = Path("checkpoints/classification/densenet121/best.pt")

    existing = {str(explicit), str(fallback)}

    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: str(self) in existing,
    )

    resolved = _resolve_classification_checkpoint(explicit, "densenet121")

    assert resolved == explicit

    resolved = _resolve_classification_checkpoint(tmp_path / "missing.pt", "densenet121")

    assert resolved == fallback

    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: False,
    )

    with pytest.raises(FileNotFoundError):
        _resolve_classification_checkpoint(tmp_path / "missing.pt", "densenet121")


def test_inference_result_to_dict_excludes_mask() -> None:
    result = InferenceResult(
        tumor_detected=True,
        tumor_class="Glioma",
        tumor_class_index=0,
        classification_confidence=0.95,
        tumor_voxel_count=100,
        tumor_percentage=0.5,
        segmentation_shape=(182, 218, 182),
        inference_time_seconds=1.5,
        segmentation_mask=torch.ones((182, 218, 182)),
    )

    data = result.to_dict()

    assert data["tumor_class"] == "Glioma"
    assert data["tumor_detected"] is True
    assert "segmentation_mask" not in data

    assert result.segmentation is not None
    assert result.segmentation.shape == (182, 218, 182)


def test_dice_and_iou_perfect_match() -> None:
    prediction = torch.tensor([1.0, 1.0, 0.0, 0.0])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])

    assert dice_score(prediction, target).item() == pytest.approx(1.0)
    assert iou_score(prediction, target).item() == pytest.approx(1.0)


def test_device_helpers() -> None:
    device = get_device()

    assert isinstance(device, torch.device)

    name = get_device_name(device)

    assert isinstance(name, str)
    assert name

    memory = get_cuda_memory()

    assert set(memory) == {"allocated_mb", "reserved_mb"}