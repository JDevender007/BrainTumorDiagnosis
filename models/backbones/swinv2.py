"""
==============================================================================
Swin Transformer V2 3D Backbone

Author  : Devender J.
Project : Hybrid Brain Tumor Diagnosis

Purpose
-------
3D Swin Transformer V2 backbone for BraTS 2024 MRI volumes.

Input
-----
(B, 4, D, H, W)

Output
------
Multi-scale feature maps:

features[0] -> (B, 16, D/2, H/2, W/2)
features[1] -> (B, 32, D/4, H/4, W/4)
features[2] -> (B, 64, D/8, H/8, W/8)
features[3] -> (B, 128, D/16, H/16, W/16)

The implementation supports 64x64x64 test volumes.
==============================================================================

"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Logging
# ==============================================================================

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s",
)


# ==============================================================================
# Configuration
# ==============================================================================


@dataclass(slots=True)
class SwinV2Config:

    in_channels: int = 4

    feature_channels: tuple[int, int, int, int] = (
        16,
        32,
        64,
        128,
    )

    window_size: int = 4

    input_size: tuple[int, int, int] = (
        64,
        64,
        64,
    )

    depths: tuple[int, int, int, int] = (
        1,
        1,
        2,
        1,
    )

    num_heads: tuple[int, int, int, int] = (
        1,
        2,
        4,
        8,
    )

    dropout: float = 0.0

    attention_dropout: float = 0.0

    drop_path: float = 0.0

    use_checkpoint: bool = False


# ==============================================================================
# Utility
# ==============================================================================


def get_effective_window_size(
    spatial_size: Sequence[int],
    requested_window: int,
) -> tuple[int, int, int]:

    effective = []

    for size in spatial_size:

        if size <= 0:
            raise ValueError(f"Invalid spatial dimension: {size}")

        window = min(
            requested_window,
            size,
        )

        while window > 1 and size % window != 0:
            window -= 1

        effective.append(window)

    return tuple(effective)


# ==============================================================================
# Window Partition
# ==============================================================================


def window_partition(
    x: torch.Tensor,
    window_size: int | tuple[int, int, int],
) -> tuple[torch.Tensor, tuple[int, int, int]]:

    if x.ndim != 5:
        raise ValueError(
            "Expected tensor with shape "
            "(B, D, H, W, C). "
            f"Received {tuple(x.shape)}"
        )

    batch_size, depth, height, width, channels = x.shape

    spatial_size = (
        depth,
        height,
        width,
    )

    if isinstance(window_size, int):

        window = get_effective_window_size(
            spatial_size,
            window_size,
        )

    else:

        window = tuple(
            min(
                int(window_size[index]),
                spatial_size[index],
            )
            for index in range(3)
        )

    wd, wh, ww = window

    if depth % wd != 0 or height % wh != 0 or width % ww != 0:

        raise ValueError(
            "Input dimensions must be divisible by "
            "window size. "
            f"Received spatial size={spatial_size}, "
            f"window={window}."
        )

    x = x.view(
        batch_size,
        depth // wd,
        wd,
        height // wh,
        wh,
        width // ww,
        ww,
        channels,
    )

    windows = (
        x.permute(
            0,
            1,
            3,
            5,
            2,
            4,
            6,
            7,
        )
        .contiguous()
        .view(
            -1,
            wd * wh * ww,
            channels,
        )
    )

    return windows, window


# ==============================================================================
# Window Reverse
# ==============================================================================


def window_reverse(
    windows: torch.Tensor,
    window_size: int | tuple[int, int, int],
    spatial_size: tuple[int, int, int],
    batch_size: int,
) -> torch.Tensor:

    depth, height, width = spatial_size

    if isinstance(window_size, int):

        window = get_effective_window_size(
            spatial_size,
            window_size,
        )

    else:

        window = tuple(window_size)

    wd, wh, ww = window

    channels = windows.shape[-1]

    windows = windows.view(
        batch_size,
        depth // wd,
        height // wh,
        width // ww,
        wd,
        wh,
        ww,
        channels,
    )

    x = (
        windows.permute(
            0,
            1,
            4,
            2,
            5,
            3,
            6,
            7,
        )
        .contiguous()
        .view(
            batch_size,
            depth,
            height,
            width,
            channels,
        )
    )

    return x


# ==============================================================================
# MLP
# ==============================================================================


class MLP(nn.Module):

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        expansion: int = 4,
    ):

        super().__init__()

        hidden_channels = channels * expansion

        self.fc1 = nn.Linear(
            channels,
            hidden_channels,
        )

        self.activation = nn.GELU()

        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(
            hidden_channels,
            channels,
        )

        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.fc1(x)

        x = self.activation(x)

        x = self.dropout1(x)

        x = self.fc2(x)

        x = self.dropout2(x)

        return x


# ==============================================================================
# Window Attention
# ==============================================================================


class WindowAttention3D(nn.Module):

    def __init__(
        self,
        channels: int,
        num_heads: int,
        dropout: float = 0.0,
    ):

        super().__init__()

        if channels % num_heads != 0:

            raise ValueError(
                f"channels={channels} must be divisible " f"by num_heads={num_heads}"
            )

        self.channels = channels

        self.num_heads = num_heads

        self.head_dim = channels // num_heads

        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(
            channels,
            channels * 3,
            bias=True,
        )

        self.attention_dropout = nn.Dropout(dropout)

        self.projection = nn.Linear(
            channels,
            channels,
        )

        self.projection_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, tokens, channels = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            batch_size,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4,
        )

        query = qkv[0]

        key = qkv[1]

        value = qkv[2]

        attention = query @ key.transpose(-2, -1)

        attention = attention * self.scale

        attention = F.softmax(
            attention,
            dim=-1,
        )

        attention = self.attention_dropout(attention)

        output = attention @ value

        output = output.transpose(
            1,
            2,
        )

        output = output.reshape(
            batch_size,
            tokens,
            channels,
        )

        output = self.projection(output)

        output = self.projection_dropout(output)

        return output


# ==============================================================================
# Swin V2 Block
# ==============================================================================


class SwinV2Block3D(nn.Module):

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):

        super().__init__()

        self.channels = channels

        self.window_size = window_size

        self.norm1 = nn.LayerNorm(channels)

        self.attention = WindowAttention3D(
            channels=channels,
            num_heads=num_heads,
            dropout=attention_dropout,
        )

        self.norm2 = nn.LayerNorm(channels)

        self.mlp = MLP(
            channels=channels,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 5:

            raise ValueError(
                "Expected input shape " "(B, D, H, W, C). " f"Received {tuple(x.shape)}"
            )

        batch_size = x.shape[0]

        spatial_size = (
            x.shape[1],
            x.shape[2],
            x.shape[3],
        )

        shortcut = x

        x = self.norm1(x)

        windows, effective_window = window_partition(
            x,
            self.window_size,
        )

        windows = self.attention(windows)

        x = window_reverse(
            windows=windows,
            window_size=effective_window,
            spatial_size=spatial_size,
            batch_size=batch_size,
        )

        x = shortcut + x

        x = x + self.mlp(self.norm2(x))

        return x


# ==============================================================================
# Patch Embedding
# ==============================================================================


class PatchEmbedding3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):

        super().__init__()

        self.projection = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.projection(x)

        x = x.permute(
            0,
            2,
            3,
            4,
            1,
        )

        x = self.norm(x)

        return x


# ==============================================================================
# Patch Merging
# ==============================================================================


class PatchMerging3D(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ):

        super().__init__()

        self.reduction = nn.Linear(
            input_channels * 8,
            output_channels,
            bias=False,
        )

        self.norm = nn.LayerNorm(input_channels * 8)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, depth, height, width, channels = x.shape

        if depth < 2 or height < 2 or width < 2:

            return x

        depth_even = depth - depth % 2

        height_even = height - height % 2

        width_even = width - width % 2

        x = x[
            :,
            :depth_even,
            :height_even,
            :width_even,
            :,
        ]

        x000 = x[
            :,
            0::2,
            0::2,
            0::2,
            :,
        ]

        x001 = x[
            :,
            0::2,
            0::2,
            1::2,
            :,
        ]

        x010 = x[
            :,
            0::2,
            1::2,
            0::2,
            :,
        ]

        x011 = x[
            :,
            0::2,
            1::2,
            1::2,
            :,
        ]

        x100 = x[
            :,
            1::2,
            0::2,
            0::2,
            :,
        ]

        x101 = x[
            :,
            1::2,
            0::2,
            1::2,
            :,
        ]

        x110 = x[
            :,
            1::2,
            1::2,
            0::2,
            :,
        ]

        x111 = x[
            :,
            1::2,
            1::2,
            1::2,
            :,
        ]

        x = torch.cat(
            [
                x000,
                x001,
                x010,
                x011,
                x100,
                x101,
                x110,
                x111,
            ],
            dim=-1,
        )

        x = self.norm(x)

        x = self.reduction(x)

        return x


# ==============================================================================
# Swin Stage
# ==============================================================================


class SwinStage3D(nn.Module):

    def __init__(
        self,
        channels: int,
        depth: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):

        super().__init__()

        self.blocks = nn.ModuleList(
            [
                SwinV2Block3D(
                    channels=channels,
                    num_heads=num_heads,
                    window_size=window_size,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        for block in self.blocks:

            x = block(x)

        return x


# ==============================================================================
# Swin Transformer V2 3D Backbone
# ==============================================================================


class SwinTransformerV2_3D(nn.Module):

    def __init__(
        self,
        config: SwinV2Config | None = None,
    ):

        super().__init__()

        if config is None:

            config = SwinV2Config()

        self.config = config

        channels = config.feature_channels

        self.patch_embedding = PatchEmbedding3D(
            in_channels=config.in_channels,
            out_channels=channels[0],
        )

        self.stage1 = SwinStage3D(
            channels=channels[0],
            depth=config.depths[0],
            num_heads=config.num_heads[0],
            window_size=config.window_size,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
        )

        self.downsample1 = PatchMerging3D(
            input_channels=channels[0],
            output_channels=channels[1],
        )

        self.stage2 = SwinStage3D(
            channels=channels[1],
            depth=config.depths[1],
            num_heads=config.num_heads[1],
            window_size=config.window_size,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
        )

        self.downsample2 = PatchMerging3D(
            input_channels=channels[1],
            output_channels=channels[2],
        )

        self.stage3 = SwinStage3D(
            channels=channels[2],
            depth=config.depths[2],
            num_heads=config.num_heads[2],
            window_size=config.window_size,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
        )

        self.downsample3 = PatchMerging3D(
            input_channels=channels[2],
            output_channels=channels[3],
        )

        self.stage4 = SwinStage3D(
            channels=channels[3],
            depth=config.depths[3],
            num_heads=config.num_heads[3],
            window_size=config.window_size,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
        )

        LOGGER.info("Swin Transformer V2 3D backbone initialized.")

        LOGGER.info(
            "Input channels : %d",
            config.in_channels,
        )

        LOGGER.info(
            "Feature channels : %s",
            config.feature_channels,
        )

    # --------------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:

        if x.ndim != 5:

            raise ValueError(
                "Expected input shape " "(B, C, D, H, W). " f"Received {tuple(x.shape)}"
            )

        x = self.patch_embedding(x)

        x = self.stage1(x)

        feature1 = x.permute(
            0,
            4,
            1,
            2,
            3,
        ).contiguous()

        x = self.downsample1(x)

        x = self.stage2(x)

        feature2 = x.permute(
            0,
            4,
            1,
            2,
            3,
        ).contiguous()

        x = self.downsample2(x)

        x = self.stage3(x)

        feature3 = x.permute(
            0,
            4,
            1,
            2,
            3,
        ).contiguous()

        x = self.downsample3(x)

        x = self.stage4(x)

        feature4 = x.permute(
            0,
            4,
            1,
            2,
            3,
        ).contiguous()

        return (
            feature1,
            feature2,
            feature3,
            feature4,
        )

    # --------------------------------------------------------------------------

    def summary(self) -> None:

        parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        LOGGER.info("=" * 70)

        LOGGER.info("Swin Transformer V2 3D Backbone Summary")

        LOGGER.info("=" * 70)

        LOGGER.info(
            "Input Channels : %d",
            self.config.in_channels,
        )

        LOGGER.info(
            "Feature Channels : %s",
            self.config.feature_channels,
        )

        LOGGER.info(
            "Window Size : %d",
            self.config.window_size,
        )

        LOGGER.info(
            "Trainable Parameters : %s",
            f"{parameters:,}",
        )

        LOGGER.info("=" * 70)


# ==============================================================================
# Factory
# ==============================================================================


def create_swinv2_3d(
    in_channels: int = 4,
    feature_channels: tuple[int, int, int, int] = (
        16,
        32,
        64,
        128,
    ),
    window_size: int = 4,
    input_size: tuple[int, int, int] = (
        64,
        64,
        64,
    ),
) -> SwinTransformerV2_3D:

    config = SwinV2Config(
        in_channels=in_channels,
        feature_channels=feature_channels,
        window_size=window_size,
        input_size=input_size,
    )

    return SwinTransformerV2_3D(config)


# ==============================================================================
# Parameter Count
# ==============================================================================


def count_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


# ==============================================================================
# Test
# ==============================================================================


def test_model() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":

        LOGGER.info(
            "Using CUDA device: %s",
            torch.cuda.get_device_name(0),
        )

    else:

        LOGGER.info("Using CPU device.")

    model = create_swinv2_3d(
        in_channels=4,
        feature_channels=(
            16,
            32,
            64,
            128,
        ),
        window_size=4,
        input_size=(
            64,
            64,
            64,
        ),
    )

    model = model.to(device)

    model.eval()

    model.summary()

    input_volume = torch.randn(
        1,
        4,
        64,
        64,
        64,
        device=device,
    )

    LOGGER.info(
        "Input Shape : %s",
        tuple(input_volume.shape),
    )

    with torch.no_grad():

        features = model(input_volume)

    LOGGER.info(
        "Feature 1 Shape : %s",
        tuple(features[0].shape),
    )

    LOGGER.info(
        "Feature 2 Shape : %s",
        tuple(features[1].shape),
    )

    LOGGER.info(
        "Feature 3 Shape : %s",
        tuple(features[2].shape),
    )

    LOGGER.info(
        "Feature 4 Shape : %s",
        tuple(features[3].shape),
    )

    parameters = count_parameters(model)

    expected_shapes = [
        (
            1,
            16,
            32,
            32,
            32,
        ),
        (
            1,
            32,
            16,
            16,
            16,
        ),
        (
            1,
            64,
            8,
            8,
            8,
        ),
        (
            1,
            128,
            4,
            4,
            4,
        ),
    ]

    actual_shapes = [tuple(feature.shape) for feature in features]

    if actual_shapes != expected_shapes:

        raise RuntimeError(
            "Unexpected feature shapes.\n"
            f"Expected: {expected_shapes}\n"
            f"Received: {actual_shapes}"
        )

    if not torch.isfinite(features[-1]).all():

        raise RuntimeError("Model output contains NaN or Inf values.")

    print()

    print("=" * 70)

    print("Swin Transformer V2 3D Test Passed")

    print("=" * 70)

    print(
        "Input Shape :",
        tuple(input_volume.shape),
    )

    for index, feature in enumerate(
        features,
        start=1,
    ):

        print(
            f"Feature {index} Shape :",
            tuple(feature.shape),
        )

    print(
        "Parameters :",
        f"{parameters:,}",
    )

    print(
        "Device :",
        device,
    )

    print("=" * 70)


# ==============================================================================
# Main
# ==============================================================================


if __name__ == "__main__":

    test_model()
