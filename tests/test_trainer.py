"""Tests for the unified classification trainer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from training.classification_trainer import (
    ClassificationTrainingConfig,
    collect_samples,
    create_class_weights,
    create_train_transform,
    create_validation_transform,
    environment_fingerprint,
    train_classification_model,
)

from models.factory import ModelConfig, create_classification_model


CLASS_DIRS = (
    "Glioma",
    "Meningioma",
    "Pituitary",
    "No Tumor",
)


def _make_images(directory: Path, count: int, size: int = 16) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        Image.new("RGB", (size, size), "white").save(
            directory / f"{index:03d}.jpg"
        )


@pytest.fixture()
def tiny_dataset(tmp_path) -> Path:
    data_directory = tmp_path / "data"

    for class_name in CLASS_DIRS:
        _make_images(data_directory / "train" / class_name, 6)
        _make_images(data_directory / "test" / class_name, 2)

    return data_directory


def test_collect_samples_case_insensitive(tiny_dataset: Path) -> None:
    samples = collect_samples(tiny_dataset / "train")

    assert len(samples) == 24

    labels = {label for _, label in samples}

    assert labels == set(range(4))


def test_collect_samples_missing_directory_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_samples(tmp_path / "does_not_exist")


def test_create_class_weights_inverse_frequency() -> None:
    samples = [
        (Path("a.jpg"), 0),
        (Path("b.jpg"), 0),
        (Path("c.jpg"), 1),
    ]

    weights = create_class_weights(
        samples,
        num_classes=2,
        device=torch.device("cpu"),
    )

    assert weights.shape == (2,)

    assert weights[0] < weights[1]

    assert weights.mean().item() == pytest.approx(1.0)


def test_transforms_output_shape() -> None:
    image = Image.new("RGB", (64, 64), "gray")

    train_transform = create_train_transform(
        (32, 32),
        augment=True,
    )

    validation_transform = create_validation_transform((32, 32))

    augmented = train_transform(image)

    assert augmented.shape == (3, 32, 32)

    plain = validation_transform(image)

    assert plain.shape == (3, 32, 32)


def test_environment_fingerprint() -> None:
    fingerprint = environment_fingerprint()

    assert "python" in fingerprint
    assert "torch" in fingerprint
    assert "torchvision" in fingerprint
    assert "cuda" in fingerprint


def test_unsupported_backbone_raises() -> None:
    with pytest.raises(ValueError):
        ClassificationTrainingConfig(backbone="not_a_backbone")


def test_cnn2d_config_forces_pretrained_false() -> None:
    config = ClassificationTrainingConfig(
        backbone="cnn2d",
        pretrained=True,
    )

    assert config.pretrained is False


def test_resume_from_last_checkpoint(tiny_dataset: Path, tmp_path: Path) -> None:
    base_config = ClassificationTrainingConfig(
        backbone="cnn2d",
        epochs=1,
        batch_size=4,
        image_size=(32, 32),
        data_directory=str(tiny_dataset),
        model_directory=str(tmp_path / "checkpoints" / "cnn2d"),
        result_directory=str(tmp_path / "results" / "cnn2d"),
        device="cpu",
        use_amp=False,
        use_augmentation=False,
        num_workers=0,
        seed=42,
    )

    first = train_classification_model(base_config)

    last_checkpoint = tmp_path / "checkpoints" / "cnn2d" / "last.pt"

    assert last_checkpoint.exists()

    resume_config = ClassificationTrainingConfig(
        backbone="cnn2d",
        epochs=2,
        batch_size=4,
        image_size=(32, 32),
        data_directory=str(tiny_dataset),
        model_directory=str(tmp_path / "checkpoints" / "cnn2d"),
        result_directory=str(tmp_path / "results" / "cnn2d"),
        device="cpu",
        use_amp=False,
        use_augmentation=False,
        num_workers=0,
        seed=42,
        resume_from=str(last_checkpoint),
    )

    resumed = train_classification_model(resume_config)

    assert resumed.best_epoch >= 1

    history_file = tmp_path / "results" / "cnn2d" / "training_history.json"

    import json

    with history_file.open("r", encoding="utf-8") as file:
        history = json.load(file)

    assert len(history) == 2

    assert history[0]["epoch"] == 1
    assert history[1]["epoch"] == 2


def test_stop_flag_halts_training(tiny_dataset: Path, tmp_path: Path) -> None:
    stop_flag = tmp_path / "stop.flag"

    stop_flag.touch()

    config = ClassificationTrainingConfig(
        backbone="cnn2d",
        epochs=3,
        batch_size=4,
        image_size=(32, 32),
        data_directory=str(tiny_dataset),
        model_directory=str(tmp_path / "checkpoints" / "cnn2d"),
        result_directory=str(tmp_path / "results" / "cnn2d"),
        device="cpu",
        use_amp=False,
        use_augmentation=False,
        num_workers=0,
        seed=42,
        stop_flag_path=str(stop_flag),
        progress_file=str(tmp_path / "results" / "cnn2d" / "progress.json"),
    )

    result = train_classification_model(config)

    assert result.best_epoch == 0

    progress_file = tmp_path / "results" / "cnn2d" / "progress.json"

    assert not progress_file.exists()


def test_progress_file_written(tiny_dataset: Path, tmp_path: Path) -> None:
    config = ClassificationTrainingConfig(
        backbone="cnn2d",
        epochs=1,
        batch_size=4,
        image_size=(32, 32),
        data_directory=str(tiny_dataset),
        model_directory=str(tmp_path / "checkpoints" / "cnn2d"),
        result_directory=str(tmp_path / "results" / "cnn2d"),
        device="cpu",
        use_amp=False,
        use_augmentation=False,
        num_workers=0,
        seed=42,
        progress_file=str(tmp_path / "results" / "cnn2d" / "progress.json"),
    )

    train_classification_model(config)

    progress_file = tmp_path / "results" / "cnn2d" / "progress.json"

    assert progress_file.exists()

    import json

    with progress_file.open("r", encoding="utf-8") as file:
        progress = json.load(file)

    assert progress["backbone"] == "cnn2d"
    assert progress["epoch"] == 1
    assert progress["total_epochs"] == 1
    assert "best_accuracy" in progress
    assert "updated_at" in progress


def test_end_to_end_tiny_training(tiny_dataset: Path, tmp_path: Path) -> None:
    config = ClassificationTrainingConfig(
        backbone="cnn2d",
        epochs=1,
        batch_size=4,
        image_size=(32, 32),
        data_directory=str(tiny_dataset),
        model_directory=str(tmp_path / "checkpoints" / "cnn2d"),
        result_directory=str(tmp_path / "results" / "cnn2d"),
        device="cpu",
        use_amp=False,
        use_augmentation=False,
        num_workers=0,
        seed=42,
    )

    result = train_classification_model(config)

    assert result.best_epoch == 1
    assert result.checkpoint_path.exists()

    checkpoint = torch.load(
        result.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint["config"]["backbone"] == "cnn2d"

    model = create_classification_model(
        ModelConfig(
            model_type="classification",
            in_channels=3,
            num_classes=4,
            base_channels=32,
        )
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    history_file = tmp_path / "results" / "cnn2d" / "training_history.json"

    assert history_file.exists()