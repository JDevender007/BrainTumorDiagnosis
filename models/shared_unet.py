"""Shared-encoder + U-Net-style multi-task baseline (Phase 3B, docs/TODO.md).

Fair-comparison baseline for Table I: the *same* pretrained DenseNet-121
encoder as the densenet121 cls-only row, a U-Net decoder with skip
connections for the auxiliary segmentation task, and a GAP + FC
classification head. This isolates "shared-encoder multi-task with a proper
U-Net decoder" as the standard baseline against TAMNet's attention/fusion
architecture.

The forward contract matches ``TumorAwareMultiTaskNet``::

    {"logits": (B, num_classes),
     "probabilities": (B, num_classes),
     "seg_logits": (B, seg_classes, H, W)}          # when return_seg=True

so ``training/multitask_trainer.py`` (CE + lambda_seg * (Dice2D + BCE)) and
the evaluation scripts consume it unchanged — identical training loop,
identical protocol (``--protocol brisc``, the only dataset with aligned
masks).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121


class ConvBNReLU(nn.Module):
    """3x3 conv -> BN -> ReLU (U-Net style block)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Nearest-upsample to the skip's size, concat, then ConvBNReLU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.conv(torch.cat([x, skip], dim=1))


class SharedUNetMultiTask(nn.Module):
    """Shared DenseNet-121 encoder + U-Net decoder + GAP/FC classifier.

    Encoder feature taps (torchvision ``densenet121.features`` indices):

    ==========  =========  ==========
    index       module     out
    ==========  ==========
    2           relu0      64 @ 112
    3           pool0      64 @ 56
    5           trans1     128 @ 28
    7           trans2     256 @ 14
    9           trans3     512 @ 7
    10          dense4     1024 @ 7  (bottleneck, cls head input)
    ==========  ==========
    """

    def __init__(
        self,
        num_classes: int = 4,
        num_seg_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        dense = densenet121(weights=weights)
        self.encoder = dense.features

        # U-Net decoder: bottleneck(1024) + trans3(512) -> ... -> 224x224
        self.up1 = UpBlock(1024, 512, 256)  # @7  + trans3
        self.up2 = UpBlock(256, 256, 192)  # @14 + trans2
        self.up3 = UpBlock(192, 128, 128)  # @28 + trans1
        self.up4 = UpBlock(128, 64, 64)  # @56 + pool0
        self.up5 = UpBlock(64, 64, 64)  # @112 + relu0
        self.seg_head = nn.Conv2d(64, num_seg_classes, kernel_size=3, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_seg: bool = True,
        seg_mask: torch.Tensor | None = None,  # noqa: ARG002 - contract compat
    ) -> dict[str, torch.Tensor]:
        skips: dict[int, torch.Tensor] = {}
        feat = x
        for i, layer in enumerate(self.encoder):
            feat = layer(feat)
            if i in (2, 3, 5, 7, 9):
                skips[i] = feat
        bottleneck = feat  # denseblock4: (B, 1024, 7, 7)

        out = self.avgpool(bottleneck).flatten(1)
        logits = self.fc(self.dropout(out))
        result: dict[str, torch.Tensor] = {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=1),
        }

        if return_seg:
            d = self.up1(bottleneck, skips[9])  # 256 @ 7
            d = self.up2(d, skips[7])  # 192 @ 14
            d = self.up3(d, skips[5])  # 128 @ 28
            d = self.up4(d, skips[3])  # 64 @ 56
            d = self.up5(d, skips[2])  # 64 @ 112
            d = F.interpolate(d, scale_factor=2, mode="nearest")  # -> 224
            result["seg_logits"] = self.seg_head(d)
        return result


def count_parameters(model: nn.Module) -> int:
    """Total (trainable + frozen) parameter count, matching repo tooling."""
    return sum(p.numel() for p in model.parameters())
