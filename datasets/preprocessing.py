"""
==============================================================================
BraTS 2024 Preprocessing

Author : Devender J.
Project : Hybrid Brain Tumor Diagnosis
==============================================================================

This module performs:

• Validation
• Intensity Clipping
• Z-Score Normalization
• Min-Max Normalization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================


@dataclass(slots=True)
class PreprocessingConfig:

    clip_percentile_lower: float = 0.5

    clip_percentile_upper: float = 99.5

    epsilon: float = 1e-8


# ==============================================================================
# Preprocessor
# ==============================================================================


class MRIProcessor:
    """
    MRI preprocessing utilities.
    """

    def __init__(
        self,
        config: PreprocessingConfig,
    ):

        self.config = config

    # ==========================================================================
    # Validate Volume
    # ==========================================================================

    @staticmethod
    def validate_volume(
        volume: np.ndarray,
    ) -> None:

        if not isinstance(volume, np.ndarray):

            raise TypeError("Volume must be numpy array.")

        if volume.ndim != 3:

            raise ValueError(f"Expected 3D volume. Got {volume.shape}")

    # ==========================================================================
    # Intensity Clipping
    # ==========================================================================

    def intensity_clipping(
        self,
        volume: np.ndarray,
    ) -> np.ndarray:

        self.validate_volume(volume)

        lower = np.percentile(
            volume,
            self.config.clip_percentile_lower,
        )

        upper = np.percentile(
            volume,
            self.config.clip_percentile_upper,
        )

        volume = np.clip(
            volume,
            lower,
            upper,
        )

        return volume.astype(np.float32)

    # ==========================================================================
    # Z Score
    # ==========================================================================

    def z_score_normalization(
        self,
        volume: np.ndarray,
    ) -> np.ndarray:

        self.validate_volume(volume)

        mask = volume > 0

        if np.sum(mask) == 0:

            return volume

        mean = volume[mask].mean()

        std = volume[mask].std()

        std = max(
            std,
            self.config.epsilon,
        )

        volume = volume.astype(np.float32)

        volume[mask] = (volume[mask] - mean) / std

        return volume

    # ==========================================================================
    # Min Max
    # ==========================================================================

    def min_max_normalization(
        self,
        volume: np.ndarray,
    ) -> np.ndarray:

        self.validate_volume(volume)

        minimum = volume.min()

        maximum = volume.max()

        denominator = maximum - minimum

        denominator = max(
            denominator,
            self.config.epsilon,
        )

        volume = (volume - minimum) / denominator

        return volume.astype(np.float32)

    # ==========================================================================
    # Center Crop
    # ==========================================================================

    @staticmethod
    def center_crop(
        volume: np.ndarray,
        output_size: Tuple[int, int, int],
    ) -> np.ndarray:

        MRIProcessor.validate_volume(volume)

        h, w, d = volume.shape

        oh, ow, od = output_size

        start_h = max((h - oh) // 2, 0)
        start_w = max((w - ow) // 2, 0)
        start_d = max((d - od) // 2, 0)

        end_h = start_h + oh
        end_w = start_w + ow
        end_d = start_d + od

        return volume[
            start_h:end_h,
            start_w:end_w,
            start_d:end_d,
        ]

    # ==========================================================================
    # Foreground Crop
    # ==========================================================================

    @staticmethod
    def foreground_crop(
        volume: np.ndarray,
    ) -> np.ndarray:

        MRIProcessor.validate_volume(volume)

        foreground = np.where(volume > 0)

        if len(foreground[0]) == 0:
            return volume

        x_min = foreground[0].min()
        x_max = foreground[0].max()

        y_min = foreground[1].min()
        y_max = foreground[1].max()

        z_min = foreground[2].min()
        z_max = foreground[2].max()

        return volume[
            x_min : x_max + 1,
            y_min : y_max + 1,
            z_min : z_max + 1,
        ]

    # ==========================================================================
    # Zero Padding
    # ==========================================================================

    @staticmethod
    def pad_volume(
        volume: np.ndarray,
        target_size: Tuple[int, int, int],
    ) -> np.ndarray:

        MRIProcessor.validate_volume(volume)

        h, w, d = volume.shape

        th, tw, td = target_size

        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)
        pad_d = max(td - d, 0)

        pad_before_h = pad_h // 2
        pad_after_h = pad_h - pad_before_h

        pad_before_w = pad_w // 2
        pad_after_w = pad_w - pad_before_w

        pad_before_d = pad_d // 2
        pad_after_d = pad_d - pad_before_d

        volume = np.pad(
            volume,
            (
                (pad_before_h, pad_after_h),
                (pad_before_w, pad_after_w),
                (pad_before_d, pad_after_d),
            ),
            mode="constant",
            constant_values=0,
        )

        return volume

    # ==========================================================================
    # Resize
    # ==========================================================================

    @staticmethod
    def resize_volume(
        volume: np.ndarray,
        target_size: Tuple[int, int, int],
    ) -> np.ndarray:

        from scipy.ndimage import zoom

        MRIProcessor.validate_volume(volume)

        zoom_factor = (
            target_size[0] / volume.shape[0],
            target_size[1] / volume.shape[1],
            target_size[2] / volume.shape[2],
        )

        resized = zoom(
            volume,
            zoom_factor,
            order=1,
        )

        return resized.astype(np.float32)

    # ==========================================================================
    # Shape Validator
    # ==========================================================================

    @staticmethod
    def validate_shape(
        volume: np.ndarray,
        expected: Tuple[int, int, int],
    ) -> None:

        if volume.shape != expected:

            raise RuntimeError(f"Expected {expected}" f"\nFound {volume.shape}")

    # ==========================================================================
    # Complete Preprocessing Pipeline
    # ==========================================================================

    def preprocess(
        self,
        volume: np.ndarray,
        target_size: Tuple[int, int, int] = (128, 128, 128),
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        """

        self.validate_volume(volume)

        volume = self.intensity_clipping(volume)

        volume = self.z_score_normalization(volume)

        volume = self.foreground_crop(volume)

        volume = self.pad_volume(
            volume,
            target_size,
        )

        volume = self.resize_volume(
            volume,
            target_size,
        )

        self.validate_shape(
            volume,
            target_size,
        )

        return volume

    # ==========================================================================
    # Process Multi-Modal MRI
    # ==========================================================================

    def preprocess_modalities(
        self,
        modalities: dict,
        target_size: Tuple[int, int, int] = (128, 128, 128),
    ) -> dict:
        """
        Preprocess all MRI modalities.
        """

        processed = {}

        for name, volume in modalities.items():

            LOGGER.info(
                "Processing %s",
                name,
            )

            processed[name] = self.preprocess(
                volume,
                target_size,
            )

        return processed

    # ==========================================================================
    # Stack Modalities
    # ==========================================================================

    @staticmethod
    def stack_modalities(
        modalities: dict,
    ) -> np.ndarray:
        """
        Stack MRI modalities into
        C x H x W x D
        """

        return np.stack(
            [
                modalities["t1n"],
                modalities["t1c"],
                modalities["t2w"],
                modalities["t2f"],
            ],
            axis=0,
        ).astype(np.float32)

    # ==========================================================================
    # Process Segmentation Mask
    # ==========================================================================

    @staticmethod
    def preprocess_mask(
        mask: np.ndarray,
        target_size: Tuple[int, int, int] = (
            128,
            128,
            128,
        ),
    ) -> np.ndarray:

        mask = MRIProcessor.foreground_crop(mask)

        mask = MRIProcessor.pad_volume(
            mask,
            target_size,
        )

        from scipy.ndimage import zoom

        zoom_factor = (
            target_size[0] / mask.shape[0],
            target_size[1] / mask.shape[1],
            target_size[2] / mask.shape[2],
        )

        mask = zoom(
            mask,
            zoom_factor,
            order=0,
        )

        return mask.astype(np.uint8)

    # ==========================================================================
    # Batch Processing
    # ==========================================================================

    def preprocess_batch(
        self,
        batch: list,
    ) -> list:

        output = []

        for volume in batch:

            output.append(self.preprocess(volume))

        return output

    # ==========================================================================
    # Dataset Statistics
    # ==========================================================================

    @staticmethod
    def statistics(
        volume: np.ndarray,
    ) -> dict:

        return {
            "shape": volume.shape,
            "dtype": volume.dtype,
            "min": float(volume.min()),
            "max": float(volume.max()),
            "mean": float(volume.mean()),
            "std": float(volume.std()),
        }

    # ==========================================================================
    # Print Statistics
    # ==========================================================================

    def summary(
        self,
        volume: np.ndarray,
    ) -> None:

        stats = self.statistics(volume)

        LOGGER.info("===================================")

        LOGGER.info("MRI Volume Summary")

        LOGGER.info("===================================")

        for key, value in stats.items():

            LOGGER.info(
                "%s : %s",
                key,
                value,
            )

        LOGGER.info("===================================")
