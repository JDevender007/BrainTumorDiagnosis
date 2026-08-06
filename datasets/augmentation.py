"""
==============================================================================
BraTS 2024 Data Augmentation

Author : Devender J.
Project : Hybrid Brain Tumor Diagnosis

Training, Validation and Inference Augmentation Pipeline
Built using MONAI.
==============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

LOGGER = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================


@dataclass(slots=True)
class AugmentationConfig:
    """
    Configuration for augmentation pipeline.
    """

    flip_probability: float = 0.50

    rotate_probability: float = 0.30

    affine_probability: float = 0.30

    gaussian_noise_probability: float = 0.20

    gaussian_smooth_probability: float = 0.20

    intensity_shift_probability: float = 0.20

    intensity_scale_probability: float = 0.20


# ==============================================================================
# Training Augmentation
# ==============================================================================


class TrainAugmentation:
    """
    Training augmentation pipeline.
    """

    def __init__(
        self,
        config: AugmentationConfig,
    ):

        self.config = config

        self.transforms = self.build()

    # --------------------------------------------------------------------------

    def build(self):

        LOGGER.info("Building training augmentation pipeline...")

        return Compose(
            [
                EnsureTyped(
                    keys=["image", "mask"],
                ),
                RandFlipd(
                    keys=["image", "mask"],
                    spatial_axis=0,
                    prob=self.config.flip_probability,
                ),
                RandFlipd(
                    keys=["image", "mask"],
                    spatial_axis=1,
                    prob=self.config.flip_probability,
                ),
                RandFlipd(
                    keys=["image", "mask"],
                    spatial_axis=2,
                    prob=self.config.flip_probability,
                ),
                RandRotate90d(
                    keys=["image", "mask"],
                    prob=self.config.rotate_probability,
                    max_k=3,
                ),
                RandAffined(
                    keys=["image", "mask"],
                    prob=self.config.affine_probability,
                    rotate_range=(
                        0.20,
                        0.20,
                        0.20,
                    ),
                    scale_range=(
                        0.10,
                        0.10,
                        0.10,
                    ),
                    mode=(
                        "bilinear",
                        "nearest",
                    ),
                    padding_mode="border",
                ),
                RandGaussianNoised(
                    keys=["image"],
                    prob=self.config.gaussian_noise_probability,
                    mean=0.0,
                    std=0.10,
                ),
                RandGaussianSmoothd(
                    keys=["image"],
                    prob=self.config.gaussian_smooth_probability,
                    sigma_x=(0.5, 1.0),
                    sigma_y=(0.5, 1.0),
                    sigma_z=(0.5, 1.0),
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=self.config.intensity_shift_probability,
                ),
                RandScaleIntensityd(
                    keys=["image"],
                    factors=0.10,
                    prob=self.config.intensity_scale_probability,
                ),
            ]
        )

    # --------------------------------------------------------------------------
    # Apply Augmentation
    # --------------------------------------------------------------------------

    def __call__(self, image, mask):

        if mask.ndim == 3:
            mask = mask[None]

        sample = {
            "image": image,
            "mask": mask,
        }

        sample = self.transforms(sample)

        sample["mask"] = sample["mask"].squeeze(0)

        return sample

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------

    def summary(self):

        LOGGER.info("=" * 60)

        LOGGER.info("Training Augmentation")

        LOGGER.info("=" * 60)

        LOGGER.info(self.transforms)

        LOGGER.info("=" * 60)


# ==============================================================================
# Validation Augmentation
# ==============================================================================


class ValidationAugmentation:
    """
    Validation pipeline.

    Only converts NumPy arrays into MONAI MetaTensor.
    No random augmentation is applied.
    """

    def __init__(self):

        self.transforms = Compose(
            [
                EnsureTyped(
                    keys=["image", "mask"],
                ),
            ]
        )

    def __call__(
        self,
        image,
        mask,
    ):

        sample = {
            "image": image,
            "mask": mask,
        }

        return self.transforms(sample)


# ==============================================================================
# Inference Augmentation
# ==============================================================================


class InferenceAugmentation:
    """
    Inference pipeline.
    """

    def __init__(self):

        self.transforms = Compose(
            [
                EnsureTyped(
                    keys=["image"],
                ),
            ]
        )

    def __call__(
        self,
        image,
    ):

        sample = {
            "image": image,
        }

        sample = self.transforms(sample)

        return sample


# ==============================================================================
# Demonstration
# ==============================================================================


if __name__ == "__main__":

    import numpy as np

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s : %(message)s",
    )

    image = np.random.rand(
        4,
        128,
        128,
        128,
    ).astype(np.float32)

    mask = np.random.randint(
        0,
        4,
        (
            128,
            128,
            128,
        ),
    ).astype(np.uint8)

    trainer = TrainAugmentation(AugmentationConfig())

    trainer.summary()

    sample = trainer(
        image,
        mask,
    )

    print()

    print("=" * 60)

    print("Training Output")

    print("=" * 60)

    print(type(sample))

    print(type(sample["image"]))

    print(type(sample["mask"]))

    print()

    print(sample["image"].shape)

    print(sample["mask"].shape)

    print()

    print(sample["image"].dtype)

    print(sample["mask"].dtype)

    print("=" * 60)
