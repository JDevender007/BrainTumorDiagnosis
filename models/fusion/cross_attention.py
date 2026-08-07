"""
==============================================================================
Hybrid Brain Tumor Diagnosis
3D Cross Attention Fusion Module

Author  : Devender J.
Project : Hybrid Brain Tumor Diagnosis

Purpose:
    Fuse multi-scale 3D features from ConvNeXt V2 and Swin Transformer V2
    using cross-attention.

Expected feature format:
    (B, C, D, H, W)

Example:
    ConvNeXt V2:
        (B, 16, 32, 32, 32)
        (B, 32, 16, 16, 16)
        (B, 64,  8,  8,  8)
        (B, 128, 4,  4,  4)

    Swin V2:
        (B, 16, 32, 32, 32)
        (B, 32, 16, 16, 16)
        (B, 64,  8,  8,  8)
        (B, 128, 4,  4,  4)

==============================================================================
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


# ==============================================================================
# Cross Attention
# ==============================================================================


class CrossAttention3D(nn.Module):
    """
    Multi-head cross-attention for 3D feature maps.

    Query:
        Feature map from one backbone.

    Key / Value:
        Feature map from the second backbone.

    Input:
        (B, C, D, H, W)

    Output:
        (B, C, D, H, W)
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be greater than zero.")

        if num_heads <= 0:
            raise ValueError("num_heads must be greater than zero.")

        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible "
                f"by num_heads ({num_heads})."
            )

        self.channels = channels
        self.num_heads = num_heads

        self.norm_query = nn.LayerNorm(channels)

        self.norm_context = nn.LayerNorm(channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.output_projection = nn.Linear(
            channels,
            channels,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query:
                Tensor with shape (B, C, D, H, W).

            context:
                Tensor with shape (B, C, D, H, W).

        Returns:
            Tensor with shape (B, C, D, H, W).
        """

        if query.ndim != 5:
            raise ValueError(
                "Query must have shape "
                "(B, C, D, H, W). "
                f"Received {tuple(query.shape)}."
            )

        if context.ndim != 5:
            raise ValueError(
                "Context must have shape "
                "(B, C, D, H, W). "
                f"Received {tuple(context.shape)}."
            )

        if query.shape[0] != context.shape[0]:
            raise ValueError("Query and context batch sizes must match.")

        if query.shape[1] != self.channels:
            raise ValueError(
                f"Query channels must be {self.channels}. "
                f"Received {query.shape[1]}."
            )

        if context.shape[1] != self.channels:
            raise ValueError(
                f"Context channels must be {self.channels}. "
                f"Received {context.shape[1]}."
            )

        if query.shape[2:] != context.shape[2:]:
            raise ValueError(
                "Query and context spatial dimensions "
                "must match. "
                f"Query: {tuple(query.shape[2:])}, "
                f"Context: {tuple(context.shape[2:])}."
            )

        batch_size = query.shape[0]
        channels = query.shape[1]

        depth = query.shape[2]
        height = query.shape[3]
        width = query.shape[4]

        # ----------------------------------------------------------------------
        # Convert:
        #
        # (B, C, D, H, W)
        #
        # to:
        #
        # (B, N, C)
        #
        # where N = D * H * W
        # ----------------------------------------------------------------------

        query_tokens = query.flatten(start_dim=2).transpose(
            1,
            2,
        )

        context_tokens = context.flatten(start_dim=2).transpose(
            1,
            2,
        )

        # ----------------------------------------------------------------------
        # Normalization
        # ----------------------------------------------------------------------

        query_tokens = self.norm_query(query_tokens)

        context_tokens = self.norm_context(context_tokens)

        # ----------------------------------------------------------------------
        # Cross Attention
        #
        # Query comes from one backbone.
        # Key and Value come from the other backbone.
        # ----------------------------------------------------------------------

        attended, _ = self.attention(
            query=query_tokens,
            key=context_tokens,
            value=context_tokens,
            need_weights=False,
        )

        attended = self.output_projection(attended)

        attended = self.dropout(attended)

        # ----------------------------------------------------------------------
        # Residual connection
        # ----------------------------------------------------------------------

        attended = attended + query_tokens

        # ----------------------------------------------------------------------
        # Convert:
        #
        # (B, N, C)
        #
        # back to:
        #
        # (B, C, D, H, W)
        # ----------------------------------------------------------------------

        output = attended.transpose(
            1,
            2,
        ).reshape(
            batch_size,
            channels,
            depth,
            height,
            width,
        )

        return output


# ==============================================================================
# Bidirectional Cross Attention
# ==============================================================================


class BidirectionalCrossAttention3D(nn.Module):
    """
    Bidirectional cross-attention between ConvNeXt V2 and Swin V2 features.

    Two attention operations are performed:

        ConvNeXt queries Swin
        Swin queries ConvNeXt

    The resulting features are combined using a learnable projection.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        self.channels = channels

        self.convnext_to_swin = CrossAttention3D(
            channels=channels,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.swin_to_convnext = CrossAttention3D(
            channels=channels,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.fusion = nn.Sequential(
            nn.Conv3d(
                channels * 2,
                channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm3d(channels),
            nn.GELU(),
        )

    def forward(
        self,
        convnext_feature: torch.Tensor,
        swin_feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            convnext_feature:
                (B, C, D, H, W)

            swin_feature:
                (B, C, D, H, W)

        Returns:
            Fused feature:
                (B, C, D, H, W)
        """

        if convnext_feature.shape != swin_feature.shape:
            raise ValueError(
                "ConvNeXt and Swin features must have "
                "identical shapes. "
                f"ConvNeXt: {tuple(convnext_feature.shape)}, "
                f"Swin: {tuple(swin_feature.shape)}."
            )

        convnext_attended = self.convnext_to_swin(
            query=convnext_feature,
            context=swin_feature,
        )

        swin_attended = self.swin_to_convnext(
            query=swin_feature,
            context=convnext_feature,
        )

        fused = torch.cat(
            [
                convnext_attended,
                swin_attended,
            ],
            dim=1,
        )

        return self.fusion(fused)


# ==============================================================================
# Multi-Scale Cross Attention
# ==============================================================================


class MultiScaleCrossAttention(nn.Module):
    """
    Applies bidirectional cross-attention to all four feature levels.

    Default channels:
        Level 1: 16
        Level 2: 32
        Level 3: 64
        Level 4: 128
    """

    def __init__(
        self,
        channels: Sequence[int] = (
            16,
            32,
            64,
            128,
        ),
        num_heads: Sequence[int] = (
            4,
            4,
            4,
            4,
        ),
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        if len(channels) != len(num_heads):
            raise ValueError(
                "channels and num_heads must have " "the same number of levels."
            )

        self.channels = tuple(channels)

        self.num_levels = len(self.channels)

        self.blocks = nn.ModuleList()

        for channel, heads in zip(
            channels,
            num_heads,
        ):

            self.blocks.append(
                BidirectionalCrossAttention3D(
                    channels=channel,
                    num_heads=heads,
                    dropout=dropout,
                )
            )

    def forward(
        self,
        convnext_features: Sequence[torch.Tensor],
        swin_features: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Args:
            convnext_features:
                List of multi-scale ConvNeXt features.

            swin_features:
                List of multi-scale Swin features.

        Returns:
            List of fused multi-scale features.
        """

        if len(convnext_features) != self.num_levels:
            raise ValueError(
                "Unexpected number of ConvNeXt feature levels. "
                f"Expected {self.num_levels}, "
                f"received {len(convnext_features)}."
            )

        if len(swin_features) != self.num_levels:
            raise ValueError(
                "Unexpected number of Swin feature levels. "
                f"Expected {self.num_levels}, "
                f"received {len(swin_features)}."
            )

        fused_features = []

        for index, block in enumerate(self.blocks):

            fused = block(
                convnext_features[index],
                swin_features[index],
            )

            fused_features.append(fused)

        return fused_features


# ==============================================================================
# Main Cross Attention Module
# ==============================================================================


class CrossAttentionFusion(nn.Module):
    """
    Main cross-attention fusion interface.

    This module receives features from the two existing backbones:

        ConvNeXt V2 3D
        Swin Transformer V2 3D

    and produces four fused feature maps.
    """

    def __init__(
        self,
        channels: Sequence[int] = (
            16,
            32,
            64,
            128,
        ),
        num_heads: Sequence[int] = (
            4,
            4,
            4,
            4,
        ),
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        self.fusion = MultiScaleCrossAttention(
            channels=channels,
            num_heads=num_heads,
            dropout=dropout,
        )

        LOGGER.info("3D cross-attention fusion initialized.")

    def forward(
        self,
        convnext_features: Sequence[torch.Tensor],
        swin_features: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:

        return self.fusion(
            convnext_features,
            swin_features,
        )

    def summary(self) -> None:

        LOGGER.info("=" * 70)

        LOGGER.info("3D Cross Attention Fusion Summary")

        LOGGER.info("=" * 70)

        LOGGER.info(
            "Feature Levels : %d",
            self.fusion.num_levels,
        )

        LOGGER.info(
            "Channels       : %s",
            self.fusion.channels,
        )

        LOGGER.info("Attention      : Bidirectional")

        LOGGER.info("Input Format   : (B, C, D, H, W)")

        LOGGER.info("=" * 70)


# ==============================================================================
# Parameter Count
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
# Test
# ==============================================================================


def test_model() -> None:
    """
    Test the complete multi-scale cross-attention module.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s : %(message)s",
    )

    # --------------------------------------------------------------------------
    # Device
    # --------------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":

        LOGGER.info(
            "Using CUDA device: %s",
            torch.cuda.get_device_name(0),
        )

    else:

        LOGGER.info("Using CPU device.")

    # --------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------

    model = CrossAttentionFusion(
        channels=(
            16,
            32,
            64,
            128,
        ),
        num_heads=(
            4,
            4,
            4,
            4,
        ),
        dropout=0.0,
    ).to(device)

    model.eval()

    model.summary()

    # --------------------------------------------------------------------------
    # Test Feature Shapes
    #
    # These match the feature shapes produced by your tested Swin V2 model.
    # --------------------------------------------------------------------------

    spatial_sizes = (
        (32, 32, 32),
        (16, 16, 16),
        (8, 8, 8),
        (4, 4, 4),
    )

    channels = (
        16,
        32,
        64,
        128,
    )

    convnext_features = []
    swin_features = []

    for channel, spatial_size in zip(
        channels,
        spatial_sizes,
    ):

        convnext_features.append(
            torch.randn(
                1,
                channel,
                *spatial_size,
                device=device,
            )
        )

        swin_features.append(
            torch.randn(
                1,
                channel,
                *spatial_size,
                device=device,
            )
        )

    # --------------------------------------------------------------------------
    # Run Test
    # --------------------------------------------------------------------------

    LOGGER.info("Running cross-attention test...")

    with torch.no_grad():

        fused_features = model(
            convnext_features,
            swin_features,
        )

    # --------------------------------------------------------------------------
    # Print Results
    # --------------------------------------------------------------------------

    for index, feature in enumerate(
        fused_features,
        start=1,
    ):

        LOGGER.info(
            "Fused Feature %d Shape : %s",
            index,
            tuple(feature.shape),
        )

    parameters = count_parameters(model)

    # --------------------------------------------------------------------------
    # Assertions
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

    for feature, expected_shape in zip(
        fused_features,
        expected_shapes,
    ):

        assert tuple(feature.shape) == expected_shape

        assert torch.isfinite(feature).all()

    # --------------------------------------------------------------------------
    # Final Test Output
    # --------------------------------------------------------------------------

    print()

    print("=" * 70)
    print("3D Cross Attention Fusion Test Passed")
    print("=" * 70)

    for index, feature in enumerate(
        fused_features,
        start=1,
    ):

        print(f"Fused Feature {index} : " f"{tuple(feature.shape)}")

    print(
        "Parameters            :",
        f"{parameters:,}",
    )

    print(
        "Device                :",
        device,
    )

    print("=" * 70)


# ==============================================================================
# Main
# ==============================================================================


if __name__ == "__main__":

    test_model()
