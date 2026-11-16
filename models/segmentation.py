from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SegmentationConfig:
    in_channels: int = 4
    out_channels: int = 4
    base_channels: int = 8
    dropout: float = 0.0


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(
                negative_slope=0.01,
                inplace=True,
            ),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(
                negative_slope=0.01,
                inplace=True,
            ),
        ]

        if dropout > 0.0:
            layers.append(nn.Dropout3d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class EncoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.pool = nn.MaxPool3d(
            kernel_size=2,
            stride=2,
        )

        self.conv = ConvBlock3D(
            in_channels,
            out_channels,
            dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class DecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = ConvBlock3D(
            out_channels + skip_channels,
            out_channels,
            dropout,
        )

    @staticmethod
    def _match_size(
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        diff_depth = target.size(2) - x.size(2)
        diff_height = target.size(3) - x.size(3)
        diff_width = target.size(4) - x.size(4)

        if diff_depth != 0 or diff_height != 0 or diff_width != 0:
            x = F.pad(
                x,
                [
                    diff_width // 2,
                    diff_width - diff_width // 2,
                    diff_height // 2,
                    diff_height - diff_height // 2,
                    diff_depth // 2,
                    diff_depth - diff_depth // 2,
                ],
            )

        return x

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)
        x = self._match_size(x, skip)

        x = torch.cat(
            [skip, x],
            dim=1,
        )

        return self.conv(x)


class UNet3D(nn.Module):
    """
    Lightweight 3D U-Net for BraTS MRI segmentation.

    Input:
        [B, 4, D, H, W]

    Output:
        [B, 4, D, H, W]

    The four input channels represent:
        t1n
        t1c
        t2w
        t2f

    The output channels represent segmentation logits.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = ConvBlock3D(
            in_channels,
            c1,
            dropout,
        )

        self.enc2 = EncoderBlock3D(
            c1,
            c2,
            dropout,
        )

        self.enc3 = EncoderBlock3D(
            c2,
            c3,
            dropout,
        )

        self.bottleneck = EncoderBlock3D(
            c3,
            c4,
            dropout,
        )

        self.dec3 = DecoderBlock3D(
            c4,
            c3,
            c3,
            dropout,
        )

        self.dec2 = DecoderBlock3D(
            c3,
            c2,
            c2,
            dropout,
        )

        self.dec1 = DecoderBlock3D(
            c2,
            c1,
            c1,
            dropout,
        )

        self.output = nn.Conv3d(
            c1,
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        bottleneck = self.bottleneck(e3)

        d3 = self.dec3(
            bottleneck,
            e3,
        )

        d2 = self.dec2(
            d3,
            e2,
        )

        d1 = self.dec1(
            d2,
            e1,
        )

        return self.output(d1)


def create_segmentation_model(
    in_channels: int = 4,
    out_channels: int = 4,
    base_channels: int = 8,
    dropout: float = 0.0,
    config: SegmentationConfig | None = None,
) -> UNet3D:
    """
    Create a 3D U-Net segmentation model.

    Direct arguments are supported for compatibility with
    the training and testing pipeline.

    If config is supplied, config values take precedence.
    """

    if config is not None:
        in_channels = config.in_channels
        out_channels = config.out_channels
        base_channels = config.base_channels
        dropout = config.dropout

    model = UNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        dropout=dropout,
    )

    LOGGER.info(
        "3D U-Net created. "
        "Input channels=%d, output channels=%d, "
        "base channels=%d",
        in_channels,
        out_channels,
        base_channels,
    )

    return model


def count_parameters(
    model: nn.Module,
) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def _test() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | " "%(levelname)s | " "%(name)s | " "%(message)s"),
    )

    LOGGER.info("=" * 70)
    LOGGER.info("BrainTumorDiagnosis 3D Segmentation Model Test")
    LOGGER.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info(
        "Device: %s",
        device,
    )

    model = create_segmentation_model(
        in_channels=4,
        out_channels=4,
        base_channels=2,
    ).to(device)

    model.eval()

    test_input = torch.randn(
        1,
        4,
        32,
        32,
        32,
        device=device,
    )

    with torch.no_grad():
        output = model(test_input)

    LOGGER.info(
        "Input shape: %s",
        tuple(test_input.shape),
    )

    LOGGER.info(
        "Output shape: %s",
        tuple(output.shape),
    )

    LOGGER.info(
        "Model parameters: %d",
        count_parameters(model),
    )

    LOGGER.info(
        "Trainable parameters: %d",
        count_trainable_parameters(model),
    )

    assert output.shape == test_input.shape

    LOGGER.info("Segmentation model test passed.")

    LOGGER.info("=" * 70)


if __name__ == "__main__":
    _test()
