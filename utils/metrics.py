from __future__ import annotations

from typing import Iterable

import torch


def _flatten_binary(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten prediction and target tensors.
    """
    prediction = prediction.detach().reshape(-1).float()
    target = target.detach().reshape(-1).float()

    return prediction, target


def dice_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Calculate binary Dice score.
    """
    prediction, target = _flatten_binary(
        prediction,
        target,
    )

    intersection = torch.sum(prediction * target)

    denominator = torch.sum(prediction) + torch.sum(target)

    return (2.0 * intersection + smooth) / (denominator + smooth)


def iou_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Calculate binary Intersection over Union.
    """
    prediction, target = _flatten_binary(
        prediction,
        target,
    )

    intersection = torch.sum(prediction * target)

    union = torch.sum(prediction) + torch.sum(target) - intersection

    return (intersection + smooth) / (union + smooth)


def binary_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate binary pixel/voxel accuracy.
    """
    prediction, target = _flatten_binary(
        prediction,
        target,
    )

    return torch.mean((prediction == target).float())


def sensitivity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Calculate sensitivity / recall.
    """
    prediction, target = _flatten_binary(
        prediction,
        target,
    )

    true_positive = torch.sum(prediction * target)

    false_negative = torch.sum((1.0 - prediction) * target)

    return (true_positive + smooth) / (true_positive + false_negative + smooth)


def specificity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Calculate specificity.
    """
    prediction, target = _flatten_binary(
        prediction,
        target,
    )

    true_negative = torch.sum((1.0 - prediction) * (1.0 - target))

    false_positive = torch.sum(prediction * (1.0 - target))

    return (true_negative + smooth) / (true_negative + false_positive + smooth)


def batch_segmentation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """
    Calculate common segmentation metrics.
    """
    return {
        "dice": float(dice_score(prediction, target)),
        "iou": float(iou_score(prediction, target)),
        "accuracy": float(binary_accuracy(prediction, target)),
        "sensitivity": float(sensitivity(prediction, target)),
        "specificity": float(specificity(prediction, target)),
    }


def mean_metrics(
    metrics: Iterable[dict[str, float]],
) -> dict[str, float]:
    """
    Calculate mean values across metric dictionaries.
    """
    metric_list = list(metrics)

    if not metric_list:
        return {}

    keys = metric_list[0].keys()

    return {
        key: sum(item[key] for item in metric_list) / len(metric_list) for key in keys
    }
