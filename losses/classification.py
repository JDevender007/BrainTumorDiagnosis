"""
BrainTumorDiagnosis
Classification losses.

Baseline-first: the default loss is plain cross-entropy.
Label smoothing is available as an option.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """
    Cross-entropy loss for classification.

    Parameters:
        label_smoothing:
            Optional smoothing value in [0, 1). Zero disables
            smoothing (plain cross-entropy).

        class_weights:
            Optional per-class weight tensor. When provided it
            must have one entry per class.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1): {label_smoothing}"
            )

        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )


def create_classification_loss(
    name: str = "cross_entropy",
    label_smoothing: float = 0.0,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """
    Factory for classification losses.

    Supported names:
        - "cross_entropy"
        - "ce"
    """

    normalized_name = name.strip().lower().replace("-", "_")

    if normalized_name in {"cross_entropy", "ce"}:
        return ClassificationLoss(
            label_smoothing=label_smoothing,
            class_weights=class_weights,
        )

    raise ValueError(f"Unsupported classification loss: {name}")


def _test() -> None:
    print("=" * 78)
    print("BrainTumorDiagnosis Classification Loss Test")
    print("=" * 78)

    logits = torch.randn(
        8,
        4,
        requires_grad=True,
    )

    targets = torch.randint(
        0,
        4,
        (8,),
    )

    plain = ClassificationLoss(label_smoothing=0.0)

    loss = plain(logits, targets)

    loss.backward()

    assert loss.ndim == 0

    assert logits.grad is not None

    print(f"Plain cross-entropy loss : {loss.item():.4f}")

    smoothed = ClassificationLoss(label_smoothing=0.1)

    smoothed_loss = smoothed(logits, targets)

    print(f"Smoothed loss            : {smoothed_loss.item():.4f}")

    factory_loss = create_classification_loss(
        name="cross_entropy",
        label_smoothing=0.05,
    )

    assert isinstance(factory_loss, ClassificationLoss)

    try:
        create_classification_loss(name="unknown")
        raise AssertionError("Expected ValueError for unknown loss name.")
    except ValueError:
        pass

    print("=" * 78)
    print("Classification loss test passed.")
    print("=" * 78)


if __name__ == "__main__":
    _test()
