"""Tests for the model factory and backbone classifiers."""

from __future__ import annotations

import pytest
import torch

from models.factory import (
    BACKBONE_CLASSIFIERS,
    ModelConfig,
    _replace_final_linear,
    count_parameters,
    create_backbone_classifier,
    create_classification_model,
    create_hybrid_model,
    create_segmentation_model,
    model_summary,
)


def test_classification_model_forward_shape() -> None:
    model = create_classification_model(
        ModelConfig(
            model_type="classification",
            in_channels=3,
            num_classes=4,
            base_channels=8,
        )
    )
    model.eval()

    image = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        output = model(image)

    assert output.shape == (2, 4)


def test_segmentation_model_forward_shape() -> None:
    model = create_segmentation_model(
        ModelConfig(
            model_type="segmentation",
            in_channels=4,
            segmentation_classes=5,
            base_channels=4,
        )
    )
    model.eval()

    volume = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        output = model(volume)

    assert output.shape == (1, 5, 32, 32, 32)


def test_hybrid_model_forward_shape() -> None:
    model = create_hybrid_model(
        ModelConfig(
            model_type="hybrid",
            in_channels=4,
            num_classes=4,
            segmentation_classes=5,
            base_channels=4,
        )
    )
    model.eval()

    volume = torch.randn(1, 4, 32, 32, 32)
    image = torch.randn(1, 4, 32, 32)

    with torch.no_grad():
        output = model(volume, image)

    assert output["segmentation"].shape == (1, 5, 32, 32, 32)
    assert output["classification"].shape == (1, 4)


@pytest.mark.parametrize("backbone", ["densenet121", "resnet50", "efficientnet_v2_s"])
def test_backbone_classifier_forward_shape(backbone: str) -> None:
    model = create_backbone_classifier(
        backbone=backbone,
        num_classes=4,
        pretrained=False,
    )
    model.eval()

    image = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(image)

    assert output.shape == (1, 4)


@pytest.mark.parametrize("backbone", ["convnext_tiny", "swin_t", "vit_b_16"])
def test_backbone_classifier_forward_shape_fast(backbone: str) -> None:
    model = create_backbone_classifier(
        backbone=backbone,
        num_classes=4,
        pretrained=False,
    )
    model.eval()

    image = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(image)

    assert output.shape == (1, 4)


def test_backbone_classifier_head_replacement() -> None:
    for backbone in BACKBONE_CLASSIFIERS:
        model = create_backbone_classifier(
            backbone=backbone,
            num_classes=4,
            pretrained=False,
        )

        head = getattr(model, BACKBONE_CLASSIFIERS[backbone]["head_attr"])

        assert isinstance(head, torch.nn.Sequential)

        final_layer = head[-1]

        if isinstance(final_layer, torch.nn.Sequential):
            final_layer = final_layer[-1]

        assert isinstance(final_layer, torch.nn.Linear)
        assert final_layer.out_features == 4


def test_unsupported_backbone_raises() -> None:
    with pytest.raises(ValueError):
        create_backbone_classifier(
            backbone="not_a_real_backbone",
            num_classes=4,
        )


def test_replace_final_linear_single_linear() -> None:
    head = torch.nn.Linear(10, 2)

    replaced = _replace_final_linear(head, 4, dropout=0.3)

    assert isinstance(replaced, torch.nn.Sequential)
    assert isinstance(replaced[-1], torch.nn.Linear)
    assert replaced[-1].out_features == 4


def test_model_summary_counts() -> None:
    model = create_classification_model(
        ModelConfig(
            model_type="classification",
            in_channels=3,
            num_classes=4,
            base_channels=8,
        )
    )

    summary = model_summary(model)

    assert summary["total_parameters"] > 0
    assert summary["trainable_parameters"] == summary["total_parameters"]
    assert summary["non_trainable_parameters"] == 0
    assert summary["total_parameters"] == count_parameters(model)