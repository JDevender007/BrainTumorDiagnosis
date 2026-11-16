"""Tests for the self-supervised contrastive pretraining module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from models.factory import BACKBONE_CLASSIFIERS
from training.self_supervised_trainer import (
    ContrastiveAugmentation,
    ProjectionHead,
    SSLImageDataset,
    SelfSupervisedConfig,
    SimCLRModel,
    nt_xent_loss,
    resolve_device,
    train_self_supervised,
)


@pytest.fixture()
def tiny_dataset(tmp_path) -> Path:
    data_directory = tmp_path / "data"

    for class_name in ("Glioma", "Meningioma"):
        directory = data_directory / "train" / class_name
        directory.mkdir(parents=True, exist_ok=True)

        for index in range(6):
            Image.new("RGB", (32, 32), "white").save(
                directory / f"{index:03d}.jpg"
            )

    return data_directory


def test_nt_xent_loss_identical_views() -> None:
    projections = torch.randn(4, 64)

    loss_identical = nt_xent_loss(
        projections,
        projections.clone(),
        temperature=0.5,
    )

    shuffled = projections[torch.randperm(4)]

    loss_shuffled = nt_xent_loss(
        projections,
        shuffled,
        temperature=0.5,
    )

    assert loss_identical.item() <= loss_shuffled.item() + 1e-6
    assert torch.isfinite(loss_identical)


def test_nt_xent_loss_finite() -> None:
    view_one = torch.randn(8, 128)
    view_two = torch.randn(8, 128)

    loss = nt_xent_loss(
        view_one,
        view_two,
        temperature=0.07,
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_projection_head_shape() -> None:
    head = ProjectionHead(
        input_dim=128,
        hidden_dim=64,
        projection_dim=32,
    )

    features = torch.randn(5, 128)

    output = head(features)

    assert output.shape == (5, 32)


def test_simclr_model_forward_shapes() -> None:
    model = SimCLRModel(
        backbone="densenet121",
        projector_hidden=64,
        projection_dim=32,
        pretrained=False,
    )
    model.eval()

    view_one = torch.randn(2, 3, 64, 64)
    view_two = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        projection_one, projection_two = model(view_one, view_two)

    assert projection_one.shape == (2, 32)
    assert projection_two.shape == (2, 32)

    assert model.backbone == "densenet121"


def test_backbone_zoo_exposed() -> None:
    assert "densenet121" in BACKBONE_CLASSIFIERS
    assert "vit_b_16" in BACKBONE_CLASSIFIERS


def test_augmentation_output_shape() -> None:
    augmentation = ContrastiveAugmentation((64, 64))

    image = Image.new("RGB", (100, 80), "gray")

    view = augmentation(image)

    assert view.shape == (3, 64, 64)


def test_ssl_dataset_returns_views(tiny_dataset: Path) -> None:
    samples = []

    for class_dir in (tiny_dataset / "train").iterdir():
        for image_path in class_dir.glob("*.jpg"):
            samples.append((image_path, 0))

    augmentation = ContrastiveAugmentation((32, 32))

    dataset = SSLImageDataset(samples, augmentation)

    assert len(dataset) == 12

    view_one, view_two = dataset[0]

    assert view_one.shape == (3, 32, 32)
    assert view_two.shape == (3, 32, 32)


def test_unsupported_backbone_raises() -> None:
    with pytest.raises(ValueError):
        SelfSupervisedConfig(backbone="not_a_backbone")


def test_resolve_device_defaults() -> None:
    device = resolve_device(None)

    assert device.type in ("cuda", "cpu")


def test_end_to_end_tiny_ssl(tiny_dataset: Path, tmp_path: Path) -> None:
    config = SelfSupervisedConfig(
        backbone="densenet121",
        epochs=1,
        batch_size=4,
        image_size=(64, 64),
        data_directory=str(tiny_dataset),
        model_directory=str(tmp_path / "checkpoints" / "ssl"),
        result_directory=str(tmp_path / "results" / "ssl"),
        device="cpu",
        use_amp=False,
        num_workers=0,
        seed=42,
    )

    result = train_self_supervised(config)

    assert result.best_epoch == 1
    assert result.checkpoint_path.exists()

    checkpoint = torch.load(
        result.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert "encoder_state_dict" in checkpoint
    assert checkpoint["config"]["backbone"] == "densenet121"