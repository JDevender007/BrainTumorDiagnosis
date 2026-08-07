"""
==============================================================================
Swin UNETR Segmentation Model

Author  : Devender J.
Project : Hybrid Brain Tumor Diagnosis

Purpose:
    3D brain tumor segmentation using Swin UNETR.

Input:
    4 MRI modalities
    T1n, T1c, T2w, T2f

Input Shape:
    (B, 4, 128, 128, 128)

Output:
    4 segmentation classes

Classes:
    0 -> Background
    1 -> NCR / NET
    2 -> ED
    3 -> ET
==============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

LOGGER = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================


@dataclass(slots=True)
class SwinUNETRConfig:
    """
    Configuration for the Swin UNETR segmentation model.
    """

    in_channels: int = 4

    out_channels: int = 4

    feature_size: int = 24

    use_checkpoint: bool = True

    spatial_size: Tuple[int, int, int] = (
        128,
        128,
        128,
    )


# ==============================================================================
# Swin UNETR Model
# ==============================================================================


class SwinUNETRModel(nn.Module):
    """
    3D Swin UNETR model for BraTS brain tumor segmentation.
    """

    def __init__(
        self,
        config: SwinUNETRConfig | None = None,
    ) -> None:

        super().__init__()

        if config is None:
            config = SwinUNETRConfig()

        self.config = config

        self.model = SwinUNETR(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            feature_size=config.feature_size,
            use_checkpoint=config.use_checkpoint,
            spatial_dims=3,
        )

        LOGGER.info("Swin UNETR initialized.")
        LOGGER.info(
            "Input channels : %d",
            config.in_channels,
        )
        LOGGER.info(
            "Output classes : %d",
            config.out_channels,
        )
        LOGGER.info(
            "Feature size   : %d",
            config.feature_size,
        )
        LOGGER.info(
            "Input size     : %s",
            config.spatial_size,
        )

    # ==============================================================================
    # Forward
    # ==============================================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:
                Tensor with shape:

                (B, 4, H, W, D)

        Returns:
            Segmentation logits with shape:

                (B, 4, H, W, D)
        """

        self._validate_input(x)

        return self.model(x)

    # ==============================================================================
    # Input Validation
    # ==============================================================================

    def _validate_input(
        self,
        x: torch.Tensor,
    ) -> None:
        """
        Validate model input.
        """

        if not isinstance(x, torch.Tensor):

            raise TypeError(f"Expected torch.Tensor, " f"got {type(x).__name__}.")

        if x.ndim != 5:

            raise ValueError(
                "Expected input with 5 dimensions "
                "(B, C, H, W, D). "
                f"Received shape: {tuple(x.shape)}."
            )

        if x.shape[1] != self.config.in_channels:

            raise ValueError(
                f"Expected {self.config.in_channels} "
                f"input channels. "
                f"Received {x.shape[1]}."
            )

        if not torch.is_floating_point(x):

            raise TypeError(
                "Expected floating-point input. " f"Received dtype: {x.dtype}."
            )

    # ==============================================================================
    # Parameter Count
    # ==============================================================================

    def parameter_count(self) -> int:
        """
        Return total number of trainable parameters.
        """

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    # ==============================================================================
    # Model Summary
    # ==============================================================================

    def summary(self) -> None:
        """
        Log basic model information.
        """

        total_parameters = self.parameter_count()

        LOGGER.info("=" * 70)
        LOGGER.info("Swin UNETR Model Summary")
        LOGGER.info("=" * 70)

        LOGGER.info(
            "Input Channels   : %d",
            self.config.in_channels,
        )

        LOGGER.info(
            "Output Classes   : %d",
            self.config.out_channels,
        )

        LOGGER.info(
            "Feature Size     : %d",
            self.config.feature_size,
        )

        LOGGER.info(
            "Input Volume     : %s",
            self.config.spatial_size,
        )

        LOGGER.info(
            "Trainable Params : %d",
            total_parameters,
        )

        LOGGER.info(
            "Checkpointing    : %s",
            self.config.use_checkpoint,
        )

        LOGGER.info("=" * 70)


# ==============================================================================
# Factory
# ==============================================================================


class SwinUNETRFactory:
    """
    Factory for creating Swin UNETR models.
    """

    @staticmethod
    def create(
        in_channels: int = 4,
        out_channels: int = 4,
        feature_size: int = 24,
        use_checkpoint: bool = True,
    ) -> SwinUNETRModel:

        config = SwinUNETRConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )

        return SwinUNETRModel(config)


# ==============================================================================
# Device Utility
# ==============================================================================


def get_device() -> torch.device:
    """
    Select CUDA when available, otherwise CPU.
    """

    if torch.cuda.is_available():

        device = torch.device("cuda")

        LOGGER.info(
            "Using CUDA device: %s",
            torch.cuda.get_device_name(0),
        )

        return device

    LOGGER.info("CUDA unavailable. Using CPU.")

    return torch.device("cpu")


# ==============================================================================
# Model Test
# ==============================================================================


def test_model() -> None:
    """
    Run a basic Swin UNETR forward-pass test.

    A smaller volume and feature size are used during testing
    to reduce GPU memory usage.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s : %(message)s",
    )

    device = get_device()

    # --------------------------------------------------------------------------
    # Test configuration
    # --------------------------------------------------------------------------

    test_config = SwinUNETRConfig(
        in_channels=4,
        out_channels=4,
        feature_size=12,
        use_checkpoint=True,
        spatial_size=(
            64,
            64,
            64,
        ),
    )

    # --------------------------------------------------------------------------
    # Create model
    # --------------------------------------------------------------------------

    model = SwinUNETRModel(test_config)

    model = model.to(device)

    model.eval()

    model.summary()

    # --------------------------------------------------------------------------
    # Create dummy MRI volume
    # --------------------------------------------------------------------------

    input_tensor = torch.randn(
        1,
        4,
        64,
        64,
        64,
        dtype=torch.float32,
        device=device,
    )

    LOGGER.info(
        "Input Shape : %s",
        tuple(input_tensor.shape),
    )

    # --------------------------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------------------------

    with torch.no_grad():

        output = model(input_tensor)

    # --------------------------------------------------------------------------
    # Check output
    # --------------------------------------------------------------------------

    LOGGER.info(
        "Output Shape : %s",
        tuple(output.shape),
    )

    LOGGER.info(
        "Output Dtype : %s",
        output.dtype,
    )

    LOGGER.info(
        "Output Device : %s",
        output.device,
    )

    expected_shape = (
        1,
        test_config.out_channels,
        64,
        64,
        64,
    )

    if tuple(output.shape) != expected_shape:

        raise RuntimeError(
            "Unexpected output shape. "
            f"Expected {expected_shape}, "
            f"received {tuple(output.shape)}."
        )

    if not torch.isfinite(output).all():

        raise RuntimeError("Model output contains NaN or Inf values.")

    # --------------------------------------------------------------------------
    # Test successful
    # --------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("Swin UNETR Test Passed")
    print("=" * 70)
    print(f"Input Shape  : {tuple(input_tensor.shape)}")
    print(f"Output Shape : {tuple(output.shape)}")
    print(f"Parameters   : {model.parameter_count():,}")
    print(f"Device       : {device}")
    print("=" * 70)


# ==============================================================================
# Main
# ==============================================================================


if __name__ == "__main__":

    test_model()
