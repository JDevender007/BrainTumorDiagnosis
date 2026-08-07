"""
==============================================================================
Multi-Scale 3D MLP
==============================================================================

Purpose:
    Combines multi-scale 3D feature maps into a compact feature vector.

Input:
    Feature 1: (B, 16, 32, 32, 32)
    Feature 2: (B, 32, 16, 16, 16)
    Feature 3: (B, 64,  8,  8,  8)
    Feature 4: (B, 128, 4,  4,  4)

Output:
    (B, 128)

Run:
    python models\classification\mlp.py
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn

# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s",
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Device
# ==============================================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    logger.info(
        "Using CUDA device: %s",
        torch.cuda.get_device_name(0),
    )
else:
    DEVICE = torch.device("cpu")
    logger.info("Using CPU device")


# ==============================================================================
# Multi-Scale 3D MLP
# ==============================================================================


class MultiScale3DMLP(nn.Module):
    """
    Multi-scale 3D feature aggregation followed by an MLP.

    Each feature level is globally average pooled.
    The pooled feature vectors are concatenated.
    The combined vector is passed through a fully connected network.
    """

    def __init__(
        self,
        input_channels: Sequence[int] = (16, 32, 64, 128),
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_channels = tuple(input_channels)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.combined_channels = sum(self.input_channels)

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.mlp = nn.Sequential(
            nn.Linear(
                self.combined_channels,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
            nn.LayerNorm(output_dim),
        )

        logger.info("Multi-scale 3D MLP initialized.")
        logger.info(
            "Input channels : %s",
            self.input_channels,
        )
        logger.info(
            "Combined channels : %d",
            self.combined_channels,
        )
        logger.info(
            "Hidden dimension : %d",
            self.hidden_dim,
        )
        logger.info(
            "Output dimension : %d",
            self.output_dim,
        )

    # ==========================================================================
    # Forward
    # ==========================================================================

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features:
            Sequence of 3D feature maps.

        Returns
        -------
        torch.Tensor
            Feature vector with shape (B, output_dim).
        """

        if len(features) != len(self.input_channels):
            raise ValueError(
                f"Expected {len(self.input_channels)} feature levels, "
                f"received {len(features)}."
            )

        pooled_features = []

        batch_size = None

        for index, (feature, expected_channels) in enumerate(
            zip(features, self.input_channels)
        ):
            if feature.ndim != 5:
                raise ValueError(
                    f"Feature {index + 1} must have shape "
                    f"(B, C, D, H, W). "
                    f"Received {tuple(feature.shape)}."
                )

            if feature.shape[1] != expected_channels:
                raise ValueError(
                    f"Feature {index + 1} has "
                    f"{feature.shape[1]} channels. "
                    f"Expected {expected_channels}."
                )

            if batch_size is None:
                batch_size = feature.shape[0]

            if feature.shape[0] != batch_size:
                raise ValueError("All feature levels must have the same batch size.")

            pooled = self.global_pool(feature)

            pooled = pooled.flatten(1)

            pooled_features.append(pooled)

        combined = torch.cat(
            pooled_features,
            dim=1,
        )

        output = self.mlp(combined)

        return output


# ==============================================================================
# Parameter Counter
# ==============================================================================


def count_parameters(
    model: nn.Module,
) -> int:
    """
    Return the number of trainable parameters.
    """

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


# ==============================================================================
# Model Summary
# ==============================================================================


def print_model_summary(
    model: MultiScale3DMLP,
) -> None:
    """
    Print model configuration.
    """

    logger.info("=" * 70)
    logger.info("Multi-Scale 3D MLP Summary")
    logger.info("=" * 70)

    logger.info(
        "Feature Levels       : %d",
        len(model.input_channels),
    )

    logger.info(
        "Input Channels       : %s",
        model.input_channels,
    )

    logger.info(
        "Combined Channels    : %d",
        model.combined_channels,
    )

    logger.info(
        "Hidden Dimension     : %d",
        model.hidden_dim,
    )

    logger.info(
        "Output Dimension     : %d",
        model.output_dim,
    )

    logger.info(
        "Trainable Parameters : %s",
        f"{count_parameters(model):,}",
    )

    logger.info("=" * 70)


# ==============================================================================
# Test Features
# ==============================================================================


def create_test_features(
    device: torch.device,
) -> list[torch.Tensor]:
    """
    Create synthetic multi-scale features for testing.
    """

    return [
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
    ]


# ==============================================================================
# Built-In Test
# ==============================================================================


def test_model() -> None:
    """
    Test the MLP using synthetic GPU or CPU data.
    """

    logger.info("Initializing multi-scale MLP...")

    model = MultiScale3DMLP(
        input_channels=(16, 32, 64, 128),
        hidden_dim=256,
        output_dim=128,
        dropout=0.2,
    )

    model = model.to(DEVICE)

    model.eval()

    print_model_summary(model)

    logger.info("Creating test features...")

    features = create_test_features(DEVICE)

    for index, feature in enumerate(
        features,
        start=1,
    ):
        logger.info(
            "Feature %d Shape : %s",
            index,
            tuple(feature.shape),
        )

    # --------------------------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------------------------

    with torch.no_grad():
        output = model(features)

    logger.info(
        "MLP Output Shape : %s",
        tuple(output.shape),
    )

    # --------------------------------------------------------------------------
    # Output shape test
    # --------------------------------------------------------------------------

    expected_shape = (
        1,
        128,
    )

    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            "Incorrect output shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(output.shape)}."
        )

    # --------------------------------------------------------------------------
    # Device test
    #
    # Compare device.type rather than comparing:
    #
    # cuda
    #
    # against:
    #
    # cuda:0
    #
    # Both represent the same CUDA device.
    # --------------------------------------------------------------------------

    if DEVICE.type == "cuda":

        if output.device.type != "cuda":
            raise RuntimeError(
                "Incorrect output device. " f"Expected CUDA, received {output.device}."
            )

    else:

        if output.device.type != "cpu":
            raise RuntimeError(
                "Incorrect output device. " f"Expected CPU, received {output.device}."
            )

    # --------------------------------------------------------------------------
    # Dtype test
    # --------------------------------------------------------------------------

    if output.dtype != torch.float32:
        raise RuntimeError(
            "Incorrect output dtype. "
            f"Expected torch.float32, "
            f"received {output.dtype}."
        )

    # --------------------------------------------------------------------------
    # Numerical stability test
    # --------------------------------------------------------------------------

    if not torch.isfinite(output).all():
        raise RuntimeError("MLP output contains NaN or Inf values.")

    # --------------------------------------------------------------------------
    # CUDA memory
    # --------------------------------------------------------------------------

    if torch.cuda.is_available():

        allocated = torch.cuda.memory_allocated() / (1024**2)

        reserved = torch.cuda.memory_reserved() / (1024**2)

        logger.info(
            "CUDA Allocated Memory : %.2f MB",
            allocated,
        )

        logger.info(
            "CUDA Reserved Memory  : %.2f MB",
            reserved,
        )

    # --------------------------------------------------------------------------
    # Success
    # --------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("Multi-Scale 3D MLP Test Passed")
    print("=" * 70)

    print(f"Output Shape : {tuple(output.shape)}")

    print(f"Parameters   : {count_parameters(model):,}")

    print(f"Device       : {output.device}")

    print(f"Dtype        : {output.dtype}")

    print("=" * 70)


# ==============================================================================
# Main
# ==============================================================================


if __name__ == "__main__":
    test_model()
