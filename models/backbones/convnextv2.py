"""
==============================================================================
ConvNeXt V2 Backbone

Author  : Devender J.
Project : Hybrid Brain Tumor Diagnosis

Purpose
-------
3D ConvNeXt V2 style backbone for extracting volumetric MRI features.

Input
-----
(B, C, H, W, D)

Output
------
Feature maps at multiple resolutions.
==============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================


@dataclass(slots=True)
class ConvNeXtV2Config:

    in_channels: int = 4

    base_channels: int = 32

    depths: tuple[int, int, int, int] = (
        2,
        2,
        2,
        2,
    )

    drop_path: float = 0.0


# ==============================================================================
# Layer Normalization
# ==============================================================================


class LayerNorm3D(nn.Module):
    """
    Layer normalization for 3D feature maps.

    Input shape:
        (B, C, H, W, D)
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
    ) -> None:

        super().__init__()

        self.norm = nn.LayerNorm(
            channels,
            eps=eps,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = x.permute(
            0,
            2,
            3,
            4,
            1,
        )

        x = self.norm(x)

        x = x.permute(
            0,
            4,
            1,
            2,
            3,
        )

        return x


# ==============================================================================
# GRN
# ==============================================================================


class GlobalResponseNorm3D(nn.Module):
    """
    Global Response Normalization.

    Adapted for volumetric 3D feature maps.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:

        super().__init__()

        self.gamma = nn.Parameter(
            torch.zeros(
                1,
                channels,
                1,
                1,
                1,
            )
        )

        self.beta = nn.Parameter(
            torch.zeros(
                1,
                channels,
                1,
                1,
                1,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        gx = torch.norm(
            x,
            p=2,
            dim=(2, 3, 4),
            keepdim=True,
        )

        nx = gx / (
            gx.mean(
                dim=1,
                keepdim=True,
            )
            + 1e-6
        )

        return self.gamma * (x * nx) + self.beta + x


# ==============================================================================
# ConvNeXt V2 Block
# ==============================================================================


class ConvNeXtV2Block3D(nn.Module):
    """
    3D ConvNeXt V2 style block.

    Structure:

        Depthwise Conv
            ↓
        LayerNorm
            ↓
        Pointwise Expansion
            ↓
        GELU
            ↓
        GRN
            ↓
        Pointwise Projection
            ↓
        Residual
    """

    def __init__(
        self,
        channels: int,
    ) -> None:

        super().__init__()

        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=True,
        )

        self.norm = LayerNorm3D(
            channels,
        )

        self.pw1 = nn.Conv3d(
            channels,
            channels * 4,
            kernel_size=1,
        )

        self.activation = nn.GELU()

        self.grn = GlobalResponseNorm3D(
            channels * 4,
        )

        self.pw2 = nn.Conv3d(
            channels * 4,
            channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        x = self.depthwise(x)

        x = self.norm(x)

        x = self.pw1(x)

        x = self.activation(x)

        x = self.grn(x)

        x = self.pw2(x)

        return residual + x


# ==============================================================================
# Downsampling Block
# ==============================================================================


class Downsample3D(nn.Module):
    """
    Downsamples a volumetric feature map by a factor of 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.norm = LayerNorm3D(
            in_channels,
        )

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.norm(x)

        x = self.conv(x)

        return x


# ==============================================================================
# ConvNeXt V2 Stage
# ==============================================================================


class ConvNeXtV2Stage3D(nn.Module):

    def __init__(
        self,
        channels: int,
        depth: int,
    ) -> None:

        super().__init__()

        self.blocks = nn.Sequential(
            *[
                ConvNeXtV2Block3D(
                    channels,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.blocks(x)


# ==============================================================================
# ConvNeXt V2 Backbone
# ==============================================================================


class ConvNeXtV2Backbone3D(nn.Module):
    """
    3D ConvNeXt V2 feature extractor.

    Returns features from four stages.
    """

    def __init__(
        self,
        config: ConvNeXtV2Config | None = None,
    ) -> None:

        super().__init__()

        if config is None:

            config = ConvNeXtV2Config()

        self.config = config

        base = config.base_channels

        channels = (
            base,
            base * 2,
            base * 4,
            base * 8,
        )

        self.stem = nn.Conv3d(
            config.in_channels,
            channels[0],
            kernel_size=4,
            stride=4,
        )

        self.stage1 = ConvNeXtV2Stage3D(
            channels[0],
            config.depths[0],
        )

        self.down1 = Downsample3D(
            channels[0],
            channels[1],
        )

        self.stage2 = ConvNeXtV2Stage3D(
            channels[1],
            config.depths[1],
        )

        self.down2 = Downsample3D(
            channels[1],
            channels[2],
        )

        self.stage3 = ConvNeXtV2Stage3D(
            channels[2],
            config.depths[2],
        )

        self.down3 = Downsample3D(
            channels[2],
            channels[3],
        )

        self.stage4 = ConvNeXtV2Stage3D(
            channels[3],
            config.depths[3],
        )

        self.feature_channels = channels

        LOGGER.info("ConvNeXt V2 3D backbone initialized.")

        LOGGER.info(
            "Input channels : %d",
            config.in_channels,
        )

        LOGGER.info(
            "Feature channels : %s",
            channels,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> List[torch.Tensor]:

        features = []

        x = self.stem(x)

        x = self.stage1(x)

        features.append(x)

        x = self.down1(x)

        x = self.stage2(x)

        features.append(x)

        x = self.down2(x)

        x = self.stage3(x)

        features.append(x)

        x = self.down3(x)

        x = self.stage4(x)

        features.append(x)

        return features


# ==============================================================================
# Model Factory
# ==============================================================================


def create_convnextv2(
    in_channels: int = 4,
    base_channels: int = 32,
) -> ConvNeXtV2Backbone3D:

    config = ConvNeXtV2Config(
        in_channels=in_channels,
        base_channels=base_channels,
    )

    return ConvNeXtV2Backbone3D(
        config,
    )


# ==============================================================================
# Model Test
# ==============================================================================


def _test() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s : %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":

        LOGGER.info(
            "Using CUDA device: %s",
            torch.cuda.get_device_name(0),
        )

    else:

        LOGGER.info("Using CPU device.")

    model = create_convnextv2(
        in_channels=4,
        base_channels=16,
    )

    model = model.to(device)

    model.eval()

    input_tensor = torch.randn(
        1,
        4,
        64,
        64,
        64,
        device=device,
    )

    with torch.no_grad():

        features = model(
            input_tensor,
        )

    print()
    print("=" * 70)
    print("ConvNeXt V2 3D Test")
    print("=" * 70)

    print(
        "Input Shape:",
        tuple(input_tensor.shape),
    )

    for index, feature in enumerate(features):

        print(
            f"Stage {index + 1} Shape:",
            tuple(feature.shape),
        )

    parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print(
        "Trainable Parameters:",
        f"{parameters:,}",
    )

    print(
        "Device:",
        device,
    )

    print("=" * 70)
    print("ConvNeXt V2 Test Passed")
    print("=" * 70)


# ==============================================================================
# Main
# ==============================================================================


if __name__ == "__main__":

    _test()
