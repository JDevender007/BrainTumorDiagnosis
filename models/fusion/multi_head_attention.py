"""
==============================================================================
3D Multi-Head Attention Fusion

Author  : Devender J.
Project : Hybrid Brain Tumor Diagnosis

Memory-efficient multi-head self-attention for 3D feature maps.

Designed for:
    NVIDIA RTX 3050 4 GB VRAM

Input format:
    (B, C, D, H, W)

Feature levels:
    Level 1: (B, 16, 32, 32, 32)
    Level 2: (B, 32, 16, 16, 16)
    Level 3: (B, 64, 8, 8, 8)
    Level 4: (B, 128, 4, 4, 4)

The implementation uses local 3D windows instead of global attention.
==============================================================================

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn

# ==============================================================================
# Logging
# ==============================================================================

LOGGER = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================


@dataclass(slots=True)
class MultiHeadAttentionConfig:
    """
    Configuration for 3D multi-head attention fusion.
    """

    channels: Tuple[int, ...] = (16, 32, 64, 128)

    num_heads: int = 4

    window_size: int = 4

    dropout: float = 0.0

    mlp_ratio: float = 2.0

    use_bias: bool = True


# ==============================================================================
# Device
# ==============================================================================


def get_device() -> torch.device:
    """
    Select CUDA when available.
    """

    if torch.cuda.is_available():

        device = torch.device("cuda")

        LOGGER.info(
            "Using CUDA device: %s",
            torch.cuda.get_device_name(0),
        )

        return device

    device = torch.device("cpu")

    LOGGER.info("CUDA unavailable. Using CPU.")

    return device


# ==============================================================================
# Utility Functions
# ==============================================================================


def partition_windows(
    x: torch.Tensor,
    window_size: int,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Partition a 3D feature map into local cubic windows.

    Input:
        x:
            (B, C, D, H, W)

    Output:
        windows:
            (B * num_windows, window_volume, C)

        spatial_shape:
            Original spatial dimensions.
    """

    if x.ndim != 5:

        raise ValueError(
            "Expected input shape (B, C, D, H, W). " f"Received {tuple(x.shape)}"
        )

    batch_size, channels, depth, height, width = x.shape

    if depth % window_size != 0:
        raise ValueError(
            f"Depth {depth} must be divisible by " f"window_size {window_size}."
        )

    if height % window_size != 0:
        raise ValueError(
            f"Height {height} must be divisible by " f"window_size {window_size}."
        )

    if width % window_size != 0:
        raise ValueError(
            f"Width {width} must be divisible by " f"window_size {window_size}."
        )

    windows = x.view(
        batch_size,
        channels,
        depth // window_size,
        window_size,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
    )

    windows = windows.permute(
        0,
        2,
        4,
        6,
        3,
        5,
        7,
        1,
    ).contiguous()

    windows = windows.view(
        -1,
        window_size * window_size * window_size,
        channels,
    )

    return windows, (depth, height, width)


# ==============================================================================
# Reverse Windows
# ==============================================================================


def reverse_windows(
    windows: torch.Tensor,
    window_size: int,
    spatial_shape: Tuple[int, int, int],
    batch_size: int,
    channels: int,
) -> torch.Tensor:
    """
    Reconstruct the original 3D feature map.

    Input:
        windows:
            (B * num_windows, window_volume, C)

    Output:
        (B, C, D, H, W)
    """

    depth, height, width = spatial_shape

    windows_per_depth = depth // window_size
    windows_per_height = height // window_size
    windows_per_width = width // window_size

    x = windows.view(
        batch_size,
        windows_per_depth,
        windows_per_height,
        windows_per_width,
        window_size,
        window_size,
        window_size,
        channels,
    )

    x = x.permute(
        0,
        7,
        1,
        4,
        2,
        5,
        3,
        6,
    ).contiguous()

    x = x.view(
        batch_size,
        channels,
        depth,
        height,
        width,
    )

    return x


# ==============================================================================
# Feed Forward Network
# ==============================================================================


class MLPBlock(nn.Module):
    """
    Feed-forward block used after local attention.
    """

    def __init__(
        self,
        channels: int,
        mlp_ratio: float,
        dropout: float,
    ):

        super().__init__()

        hidden_channels = int(channels * mlp_ratio)

        self.network = nn.Sequential(
            nn.Linear(
                channels,
                hidden_channels,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_channels,
                channels,
            ),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)


# ==============================================================================
# Local 3D Multi-Head Attention Block
# ==============================================================================


class Local3DMultiHeadAttentionBlock(nn.Module):
    """
    Memory-efficient local 3D multi-head attention.

    Attention operates independently inside each 3D window.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        dropout: float,
        mlp_ratio: float,
    ):

        super().__init__()

        if channels % num_heads != 0:

            raise ValueError(
                f"Channels ({channels}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.channels = channels

        self.num_heads = num_heads

        self.window_size = window_size

        self.norm1 = nn.LayerNorm(channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(channels)

        self.mlp = MLPBlock(
            channels=channels,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input:
            (B, C, D, H, W)

        Output:
            (B, C, D, H, W)
        """

        batch_size = x.shape[0]

        windows, spatial_shape = partition_windows(
            x,
            self.window_size,
        )

        normalized = self.norm1(windows)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )

        windows = windows + attention_output

        windows = windows + self.mlp(self.norm2(windows))

        output = reverse_windows(
            windows=windows,
            window_size=self.window_size,
            spatial_shape=spatial_shape,
            batch_size=batch_size,
            channels=self.channels,
        )

        return output


# ==============================================================================
# Feature Projection
# ==============================================================================


class FeatureProjection(nn.Module):
    """
    Lightweight projection layer for each feature level.
    """

    def __init__(
        self,
        channels: int,
    ):

        super().__init__()

        self.projection = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=1,
                bias=True,
            ),
            nn.BatchNorm3d(channels),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.projection(x)


# ==============================================================================
# Single Feature Attention Module
# ==============================================================================


class FeatureAttentionModule(nn.Module):
    """
    Projection + local 3D multi-head attention.
    """

    def __init__(
        self,
        channels: int,
        config: MultiHeadAttentionConfig,
    ):

        super().__init__()

        self.projection = FeatureProjection(channels)

        self.attention = Local3DMultiHeadAttentionBlock(
            channels=channels,
            num_heads=config.num_heads,
            window_size=config.window_size,
            dropout=config.dropout,
            mlp_ratio=config.mlp_ratio,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.projection(x)

        x = self.attention(x)

        return x


# ==============================================================================
# 3D Multi-Head Attention Fusion
# ==============================================================================


class MultiHeadAttentionFusion(nn.Module):
    """
    Multi-scale 3D multi-head attention fusion.

    Each feature level is processed independently using
    local 3D attention.

    Input:
        List/Tuple of feature tensors.

    Example:
        [
            (B, 16, 32, 32, 32),
            (B, 32, 16, 16, 16),
            (B, 64, 8, 8, 8),
            (B, 128, 4, 4, 4),
        ]

    Output:
        Same feature resolutions and channel counts.
    """

    def __init__(
        self,
        config: MultiHeadAttentionConfig | None = None,
    ):

        super().__init__()

        if config is None:

            config = MultiHeadAttentionConfig()

        self.config = config

        self.channels = config.channels

        self.num_heads = config.num_heads

        self.window_size = config.window_size

        self.feature_levels = nn.ModuleList(
            [
                FeatureAttentionModule(
                    channels=channels,
                    config=config,
                )
                for channels in config.channels
            ]
        )

        LOGGER.info("3D multi-head attention fusion initialized.")

        LOGGER.info(
            "Feature levels : %d",
            len(self.channels),
        )

        LOGGER.info(
            "Channels       : %s",
            self.channels,
        )

        LOGGER.info(
            "Attention heads: %d",
            self.num_heads,
        )

        LOGGER.info(
            "Window size    : %d",
            self.window_size,
        )

    # ==========================================================================
    # Forward
    # ==========================================================================

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Process all feature levels.

        Parameters
        ----------
        features:
            Sequence of 3D feature maps.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            Processed feature maps.
        """

        if len(features) != len(self.feature_levels):

            raise ValueError(
                "Unexpected number of feature levels. "
                f"Expected {len(self.feature_levels)}, "
                f"received {len(features)}."
            )

        outputs = []

        for index, feature in enumerate(features):

            expected_channels = self.channels[index]

            if feature.ndim != 5:

                raise ValueError(
                    f"Feature {index + 1} must have shape "
                    f"(B, C, D, H, W). "
                    f"Received {tuple(feature.shape)}."
                )

            if feature.shape[1] != expected_channels:

                raise ValueError(
                    f"Feature {index + 1} expected "
                    f"{expected_channels} channels. "
                    f"Received {feature.shape[1]}."
                )

            output = self.feature_levels[index](feature)

            outputs.append(output)

        return tuple(outputs)

    # ==========================================================================
    # Summary
    # ==========================================================================

    def summary(self) -> None:
        """
        Print model configuration and parameter count.
        """

        parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        LOGGER.info("=" * 70)

        LOGGER.info("3D Multi-Head Attention Fusion Summary")

        LOGGER.info("=" * 70)

        LOGGER.info(
            "Feature Levels      : %d",
            len(self.channels),
        )

        LOGGER.info(
            "Channels            : %s",
            self.channels,
        )

        LOGGER.info(
            "Attention Heads     : %d",
            self.num_heads,
        )

        LOGGER.info(
            "Window Size         : %d",
            self.window_size,
        )

        LOGGER.info("Attention Type      : Local 3D Window Attention")

        LOGGER.info(
            "Trainable Parameters: %s",
            f"{parameters:,}",
        )

        LOGGER.info("=" * 70)


# ==============================================================================
# Model Factory
# ==============================================================================


def create_multi_head_attention(
    device: torch.device | None = None,
) -> MultiHeadAttentionFusion:
    """
    Create the multi-head attention fusion module.
    """

    if device is None:

        device = get_device()

    config = MultiHeadAttentionConfig(
        channels=(16, 32, 64, 128),
        num_heads=4,
        window_size=4,
        dropout=0.0,
        mlp_ratio=2.0,
    )

    model = MultiHeadAttentionFusion(config)

    model = model.to(device)

    return model


# ==============================================================================
# Test Data
# ==============================================================================


def create_test_features(
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    """
    Create feature maps matching the project's backbone outputs.
    """

    features = (
        torch.randn(
            1,
            16,
            32,
            32,
            32,
            device=device,
        ),
        torch.randn(
            1,
            32,
            16,
            16,
            16,
            device=device,
        ),
        torch.randn(
            1,
            64,
            8,
            8,
            8,
            device=device,
        ),
        torch.randn(
            1,
            128,
            4,
            4,
            4,
            device=device,
        ),
    )

    return features


# ==============================================================================
# Test Model
# ==============================================================================


def test_model() -> None:
    """
    Complete built-in test.

    The test verifies:

    1. Model initialization.
    2. GPU execution.
    3. Four feature levels.
    4. Correct output shapes.
    5. Correct dtype.
    6. No global-attention CUDA OOM.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s : %(message)s",
    )

    device = get_device()

    LOGGER.info("Initializing memory-efficient attention model...")

    model = create_multi_head_attention(device=device)

    model.eval()

    model.summary()

    LOGGER.info("Running multi-head attention test...")

    features = create_test_features(device=device)

    input_shapes = [tuple(feature.shape) for feature in features]

    LOGGER.info(
        "Input Feature 1 Shape : %s",
        input_shapes[0],
    )

    LOGGER.info(
        "Input Feature 2 Shape : %s",
        input_shapes[1],
    )

    LOGGER.info(
        "Input Feature 3 Shape : %s",
        input_shapes[2],
    )

    LOGGER.info(
        "Input Feature 4 Shape : %s",
        input_shapes[3],
    )

    # --------------------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------------------

    with torch.inference_mode():

        outputs = model(features)

    # --------------------------------------------------------------------------
    # Validate output count
    # --------------------------------------------------------------------------

    if len(outputs) != 4:

        raise RuntimeError(
            "Expected 4 output feature levels. " f"Received {len(outputs)}."
        )

    # --------------------------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------------------------

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

    for index, (
        output,
        expected_shape,
    ) in enumerate(zip(outputs, expected_shapes)):

        LOGGER.info(
            "Fused Feature %d Shape : %s",
            index + 1,
            tuple(output.shape),
        )

        if tuple(output.shape) != expected_shape:

            raise RuntimeError(
                f"Feature {index + 1} shape mismatch. "
                f"Expected {expected_shape}, "
                f"received {tuple(output.shape)}."
            )

        if output.dtype != torch.float32:

            raise RuntimeError(
                f"Feature {index + 1} has incorrect dtype. "
                f"Expected torch.float32, "
                f"received {output.dtype}."
            )

    # --------------------------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------------------------

    parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    # --------------------------------------------------------------------------
    # CUDA memory information
    # --------------------------------------------------------------------------

    if device.type == "cuda":

        allocated = torch.cuda.memory_allocated(device) / (1024**2)

        reserved = torch.cuda.memory_reserved(device) / (1024**2)

        LOGGER.info(
            "CUDA Allocated Memory : %.2f MB",
            allocated,
        )

        LOGGER.info(
            "CUDA Reserved Memory  : %.2f MB",
            reserved,
        )

    # --------------------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------------------

    print()

    print("=" * 70)

    print("3D Multi-Head Attention Fusion Test Passed")

    print("=" * 70)

    for index, output in enumerate(outputs):

        print(f"Fused Feature {index + 1} : " f"{tuple(output.shape)}")

    print(f"Parameters            : {parameters:,}")

    print(f"Device                : {device}")

    print("Attention             : Local 3D Window")

    print("Window Size           : 4 x 4 x 4")

    print("=" * 70)


# ==============================================================================
# Main
# ==============================================================================


if __name__ == "__main__":

    test_model()
