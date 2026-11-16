"""Tests for segmentation metric utilities."""

from __future__ import annotations

import pytest
import torch

from utils.metrics import (
    batch_segmentation_metrics,
    binary_accuracy,
    dice_score,
    iou_score,
    mean_metrics,
    sensitivity,
    specificity,
)


def test_dice_perfect_and_empty() -> None:
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])

    perfect = dice_score(target.clone(), target)

    assert perfect.item() == pytest.approx(1.0)

    empty = dice_score(
        torch.zeros(4),
        torch.zeros(4),
    )

    assert empty.item() == pytest.approx(1.0)

    zero_overlap = dice_score(
        torch.zeros(4),
        target,
    )

    assert zero_overlap.item() == pytest.approx(0.0, abs=1e-5)


def test_iou_perfect() -> None:
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])

    perfect = iou_score(target.clone(), target)

    assert perfect.item() == pytest.approx(1.0)


def test_accuracy_sensitivity_specificity() -> None:
    prediction = torch.tensor([1.0, 1.0, 1.0, 0.0])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])

    assert binary_accuracy(prediction, target).item() == pytest.approx(0.75)

    assert sensitivity(prediction, target).item() == pytest.approx(1.0)

    assert specificity(prediction, target).item() == pytest.approx(0.5)


def test_batch_metrics_keys() -> None:
    prediction = torch.tensor([1.0, 0.0])
    target = torch.tensor([1.0, 0.0])

    metrics = batch_segmentation_metrics(prediction, target)

    assert set(metrics) == {
        "dice",
        "iou",
        "accuracy",
        "sensitivity",
        "specificity",
    }


def test_mean_metrics() -> None:
    first = {"dice": 1.0, "iou": 1.0}
    second = {"dice": 0.5, "iou": 0.5}

    averaged = mean_metrics([first, second])

    assert averaged["dice"] == pytest.approx(0.75)
    assert averaged["iou"] == pytest.approx(0.75)

    assert mean_metrics([]) == {}