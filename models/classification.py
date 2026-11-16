from __future__ import annotations

import logging

import torch
import torch.nn as nn


LOGGER = logging.getLogger(__name__)


class ClassificationBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BrainTumorClassifier3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_channels: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ClassificationBlock3D(
                in_channels,
                base_channels,
            ),
            ClassificationBlock3D(
                base_channels,
                base_channels * 2,
            ),
            ClassificationBlock3D(
                base_channels * 2,
                base_channels * 4,
            ),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                base_channels * 4,
                base_channels * 4,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(
                base_channels * 4,
                num_classes,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def create_classification_model(
    in_channels: int = 4,
    num_classes: int = 4,
    base_channels: int = 16,
) -> BrainTumorClassifier3D:
    return BrainTumorClassifier3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = create_classification_model()

    x = torch.randn(
        1,
        4,
        32,
        32,
        32,
    )

    with torch.no_grad():
        y = model(x)

    LOGGER.info("Input shape: %s", tuple(x.shape))
    LOGGER.info("Output shape: %s", tuple(y.shape))
    LOGGER.info("Parameters: %d", count_parameters(model))

    assert y.shape == (1, 4)

    LOGGER.info("Classification model test passed.")
