"""Tests for the Grad-CAM explainability module."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from models.factory import ModelConfig, create_classification_model
from evaluation.explainability import (
    GradCAM,
    create_test_transform,
    explain_image,
    load_model,
    save_overlay,
)


@pytest.fixture()
def cnn_model():
    model = create_classification_model(
        ModelConfig(
            model_type="classification",
            in_channels=3,
            num_classes=4,
            base_channels=8,
        )
    )

    model.eval()

    return model


@pytest.fixture()
def sample_image() -> torch.Tensor:
    image = Image.new("RGB", (64, 64), "white")

    return create_test_transform()(image).unsqueeze(0)


def test_gradcam_heatmap_shape_and_range(cnn_model, sample_image) -> None:
    target_layer = cnn_model.layer4

    cam = GradCAM(cnn_model, target_layer)

    heatmap, probabilities, target_class = cam.generate_heatmap(
        sample_image,
        target_class=None,
    )

    assert heatmap.shape == (224, 224)

    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0

    assert probabilities.shape == (4,)

    assert 0 <= target_class < 4

    assert probabilities.sum() == pytest.approx(1.0)


def test_gradcam_target_class(cnn_model, sample_image) -> None:
    target_layer = cnn_model.layer4

    cam = GradCAM(cnn_model, target_layer)

    heatmap, _, target_class = cam.generate_heatmap(
        sample_image,
        target_class=2,
    )

    assert target_class == 2

    assert heatmap.shape == (224, 224)


def test_gradcam_remove_hooks(cnn_model, sample_image) -> None:
    target_layer = cnn_model.layer4

    cam = GradCAM(cnn_model, target_layer)

    cam.generate_heatmap(sample_image)

    cam.remove_hooks()

    assert cam.activations is not None


def test_explain_image_helper(cnn_model, sample_image) -> None:
    device = torch.device("cpu")

    heatmap, probabilities, target_class = explain_image(
        cnn_model,
        "cnn2d",
        sample_image,
        device,
    )

    assert heatmap.shape == (224, 224)
    assert probabilities.shape == (4,)
    assert 0 <= target_class < 4


def test_load_model_cnn2d(tmp_path, sample_image) -> None:
    model = create_classification_model(
        ModelConfig(
            model_type="classification",
            in_channels=3,
            num_classes=4,
            base_channels=16,
        )
    )

    checkpoint_path = tmp_path / "best.pt"

    torch.save(
        {"model_state_dict": model.state_dict()},
        checkpoint_path,
    )

    loaded = load_model(
        "cnn2d",
        torch.device("cpu"),
        checkpoint_path,
    )

    assert isinstance(loaded, torch.nn.Module)

    loaded.eval()

    with torch.no_grad():
        output = loaded(sample_image)

    assert output.shape == (1, 4)


def test_explain_image_densenet_patched_forward() -> None:
    """DenseNet's in-place ReLU must be patched for Grad-CAM backward."""

    from models.factory import create_backbone_classifier

    model = create_backbone_classifier(
        backbone="densenet121",
        num_classes=4,
        pretrained=False,
    )
    model.eval()

    image = create_test_transform()(
        Image.new("RGB", (64, 64), "white")
    ).unsqueeze(0)

    heatmap, probabilities, target_class = explain_image(
        model,
        "densenet121",
        image,
        torch.device("cpu"),
    )

    assert heatmap.shape == (224, 224)
    assert probabilities.shape == (4,)
    assert 0 <= target_class < 4


def test_save_overlay_writes_png(tmp_path) -> None:
    image_path = tmp_path / "input.jpg"

    Image.new("RGB", (64, 64), "gray").save(image_path)

    heatmap = np.zeros((224, 224), dtype=np.float32)
    heatmap[50:170, 70:150] = 1.0

    output_path = save_overlay(
        image_path,
        heatmap,
        tmp_path / "overlay.png",
        class_name="Glioma",
        probability=0.9,
    )

    assert output_path.exists()

    with Image.open(output_path) as image:
        assert image.format == "PNG"