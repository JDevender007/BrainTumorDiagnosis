"""
Segmentation losses for BrainTumorDiagnosis.

BraTS 2024 post-treatment label schema:

    Label 0: background
    Label 1: non-enhancing tumor core (NETC)
    Label 2: surrounding non-enhancing FLAIR hyperintensity (SNFH / edema)
    Label 3: enhancing tumor (ET)
    Label 4: resection cavity (RC)

Expected shapes:

    Prediction: [B, C, D, H, W]  (logits)
    Target:     [B, D, H, W]     (integer labels)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


LOGGER = logging.getLogger(__name__)


class DiceLoss3D(nn.Module):
    """
    Multi-class soft Dice loss for 3D segmentation.
    """

    def __init__(
        self,
        smooth: float = 1e-5,
        include_background: bool = True,
        softmax: bool = True,
    ) -> None:
        super().__init__()

        self.smooth = smooth
        self.include_background = include_background
        self.softmax = softmax

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        if prediction.ndim != 5:
            raise ValueError(
                "Prediction must have shape (B, C, D, H, W). "
                f"Received {tuple(prediction.shape)}"
            )

        if target.ndim != 4:
            raise ValueError(
                "Target must have shape (B, D, H, W). "
                f"Received {tuple(target.shape)}"
            )

        if prediction.shape[0] != target.shape[0]:
            raise ValueError(
                "Batch dimensions do not match. "
                f"Prediction={prediction.shape[0]}, "
                f"Target={target.shape[0]}"
            )

        if prediction.shape[2:] != target.shape[1:]:
            raise ValueError(
                "Spatial dimensions do not match. "
                f"Prediction={tuple(prediction.shape[2:])}, "
                f"Target={tuple(target.shape[1:])}"
            )

        prediction = prediction.float()
        target = target.long()

        num_classes = prediction.shape[1]

        target = F.one_hot(
            target,
            num_classes=num_classes,
        )

        target = target.permute(
            0,
            4,
            1,
            2,
            3,
        ).float()

        if self.softmax:
            prediction = torch.softmax(
                prediction,
                dim=1,
            )

        dims = (2, 3, 4)

        intersection = torch.sum(
            prediction * target,
            dim=dims,
        )

        prediction_sum = torch.sum(
            prediction,
            dim=dims,
        )

        target_sum = torch.sum(
            target,
            dim=dims,
        )

        dice_score = (2.0 * intersection + self.smooth) / (
            prediction_sum + target_sum + self.smooth
        )

        if not self.include_background and num_classes > 1:
            dice_score = dice_score[:, 1:]

        return (1.0 - dice_score).mean()


class SegmentationLoss(nn.Module):
    """
    Weighted combination of soft Dice loss and cross-entropy.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1e-5,
        include_background: bool = True,
    ) -> None:
        super().__init__()

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        self.dice_loss = DiceLoss3D(
            smooth=smooth,
            include_background=include_background,
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        dice_term = self.dice_loss(
            prediction,
            target,
        )

        ce_term = F.cross_entropy(
            prediction,
            target.long(),
        )

        return (
            self.dice_weight * dice_term + self.ce_weight * ce_term
        )


def segmentation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 0.5,
    ce_weight: float = 0.5,
) -> torch.Tensor:
    """
    Compute the combined segmentation loss in one call.
    """
    return SegmentationLoss(
        dice_weight=dice_weight,
        ce_weight=ce_weight,
    )(
        prediction,
        target,
    )


def _test() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("Loss self-test on device: %s", device)

    torch.manual_seed(42)

    prediction = torch.randn(
        2,
        5,
        16,
        16,
        16,
        device=device,
    )

    target = torch.randint(
        0,
        5,
        (2, 16, 16, 16),
        device=device,
    )

    criterion = SegmentationLoss().to(device)

    loss = criterion(
        prediction,
        target,
    )

    dice_loss = DiceLoss3D(
        include_background=True,
    ).to(device)

    dice_value = dice_loss(
        prediction,
        target,
    )

    foreground_dice = DiceLoss3D(
        include_background=False,
    ).to(device)

    foreground_value = foreground_dice(
        prediction,
        target,
    )

    LOGGER.info("Combined loss : %.6f", loss.item())
    LOGGER.info("Dice loss     : %.6f", dice_value.item())
    LOGGER.info("Foreground-only dice loss: %.6f", foreground_value.item())

    assert torch.isfinite(loss)
    assert torch.isfinite(dice_value)
    assert torch.isfinite(foreground_value)
    assert loss.ndim == 0

    LOGGER.info("Segmentation loss test passed.")


if __name__ == "__main__":
    _test()