"""
==============================================================================
BraTS 2024 Transform Pipeline

Author : Devender J.
Project : Hybrid Brain Tumor Diagnosis

Transforms for preprocessing, augmentation and tensor conversion.
==============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from datasets.augmentation import (
    AugmentationConfig,
    InferenceAugmentation,
    TrainAugmentation,
    ValidationAugmentation,
)
from datasets.preprocessing import MRIProcessor, PreprocessingConfig

LOGGER = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================


@dataclass(slots=True)
class TransformConfig:
    """
    Configuration for transform pipeline.
    """

    image_size: Tuple[int, int, int] = (128, 128, 128)

    training: bool = True


# ==============================================================================
# Base Transform
# ==============================================================================


class BaseTransform:
    """
    Base transform used by training,
    validation and inference.
    """

    def __init__(
        self,
        config: TransformConfig,
    ):

        self.config = config

        self.processor = MRIProcessor(PreprocessingConfig())

    # --------------------------------------------------------------------------
    # Tensor Conversion
    # --------------------------------------------------------------------------

    @staticmethod
    def to_tensor(
        image,
    ) -> torch.Tensor:

        if isinstance(image, torch.Tensor):
            return image.float()

        return torch.from_numpy(image).float()

    # --------------------------------------------------------------------------

    @staticmethod
    def mask_to_tensor(
        mask,
    ) -> torch.Tensor:

        if isinstance(mask, torch.Tensor):
            return mask.long()

        return torch.from_numpy(mask).long()

    # --------------------------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------------------------

    def preprocess(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
    ):

        modality_names = (
            "t1n",
            "t1c",
            "t2w",
            "t2f",
        )

        processed = {}

        for index, modality in enumerate(modality_names):

            processed[modality] = self.processor.preprocess(
                image[index],
                self.config.image_size,
            )

        image = self.processor.stack_modalities(processed)

        if mask is None:

            return image

        mask = self.processor.preprocess_mask(
            mask,
            self.config.image_size,
        )

        return image, mask

    # --------------------------------------------------------------------------
    # Base Call
    # --------------------------------------------------------------------------

    def __call__(
        self,
        image,
        mask,
    ):

        image, mask = self.preprocess(
            image,
            mask,
        )

        return {
            "image": self.to_tensor(image),
            "mask": self.mask_to_tensor(mask),
        }


# ==============================================================================
# Training Transform
# ==============================================================================


class TrainTransform(BaseTransform):
    """
    Training transform.
    Applies preprocessing followed by data augmentation.
    """

    def __init__(
        self,
        config: TransformConfig,
    ):

        super().__init__(config)

        self.augmentation = TrainAugmentation(AugmentationConfig())

    def __call__(
        self,
        image,
        mask,
    ):

        image, mask = self.preprocess(
            image,
            mask,
        )

        sample = self.augmentation(
            image,
            mask,
        )

        return {
            "image": sample["image"].float(),
            "mask": sample["mask"].long(),
        }


# ==============================================================================
# Validation Transform
# ==============================================================================


class ValidationTransform(BaseTransform):
    """
    Validation transform.
    Only preprocessing is performed.
    """

    def __init__(
        self,
        config: TransformConfig,
    ):

        super().__init__(config)

        self.augmentation = ValidationAugmentation()

    def __call__(
        self,
        image,
        mask,
    ):

        image, mask = self.preprocess(
            image,
            mask,
        )

        sample = self.augmentation(
            image,
            mask,
        )

        return {
            "image": sample["image"].float(),
            "mask": sample["mask"].long(),
        }


# ==============================================================================
# Inference Transform
# ==============================================================================


class InferenceTransform(BaseTransform):
    """
    Transform used during inference.
    """

    def __init__(
        self,
        config: TransformConfig,
    ):

        super().__init__(config)

        self.augmentation = InferenceAugmentation()

    def __call__(
        self,
        image,
    ):

        image = self.preprocess(
            image=image,
            mask=None,
        )

        sample = self.augmentation(
            image,
        )

        return sample["image"].float()


# ==============================================================================
# Factory
# ==============================================================================


class TransformFactory:
    """
    Factory for creating transform pipelines.
    """

    @staticmethod
    def create(
        training: bool = True,
    ):

        config = TransformConfig(
            training=training,
        )

        if training:

            LOGGER.info("Creating training transform...")

            return TrainTransform(
                config,
            )

        LOGGER.info("Creating validation transform...")

        return ValidationTransform(
            config,
        )


# ==============================================================================
# Transform Manager
# ==============================================================================


class TransformManager:
    """
    Manages all transform pipelines.
    """

    def __init__(self):

        self.train_transform = TrainTransform(TransformConfig(training=True))

        self.validation_transform = ValidationTransform(TransformConfig(training=False))

        self.inference_transform = InferenceTransform(TransformConfig(training=False))

    # --------------------------------------------------------------------------

    def training(self):

        return self.train_transform

    # --------------------------------------------------------------------------

    def validation(self):

        return self.validation_transform

    # --------------------------------------------------------------------------

    def inference(self):

        return self.inference_transform

    # --------------------------------------------------------------------------

    def summary(self):

        LOGGER.info("=" * 60)
        LOGGER.info("Transform Pipeline")
        LOGGER.info("=" * 60)
        LOGGER.info("Training   : Ready")
        LOGGER.info("Validation : Ready")
        LOGGER.info("Inference  : Ready")
        LOGGER.info(
            "Image Size : %s",
            self.train_transform.config.image_size,
        )
        LOGGER.info("=" * 60)


# ==============================================================================
# Utility Functions
# ==============================================================================


def verify_sample(
    sample: Dict,
):

    if "image" not in sample:
        raise KeyError("Sample does not contain 'image'.")

    image = sample["image"]

    if not isinstance(
        image,
        torch.Tensor,
    ):
        raise TypeError("Image must be a torch.Tensor.")

    if image.ndim != 4:
        raise RuntimeError(f"Expected image shape (C,H,W,D), got {image.shape}")

    if "mask" in sample:

        mask = sample["mask"]

        if not isinstance(
            mask,
            torch.Tensor,
        ):
            raise TypeError("Mask must be a torch.Tensor.")

        if mask.ndim not in (3, 4):
            raise RuntimeError(
                f"Expected mask shape (H,W,D) or (1,H,W,D), got {mask.shape}"
            )

    return True


# ==============================================================================
# Demo
# ==============================================================================


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s : %(message)s",
    )

    image = np.random.rand(
        4,
        240,
        240,
        155,
    ).astype(np.float32)

    mask = np.random.randint(
        0,
        4,
        (
            240,
            240,
            155,
        ),
    ).astype(np.uint8)

    manager = TransformManager()

    manager.summary()

    transform = manager.training()

    sample = transform(
        image,
        mask,
    )

    verify_sample(
        sample,
    )

    print()

    print("=" * 60)

    print("Transform Test")

    print("=" * 60)

    print(type(sample["image"]))
    print(type(sample["mask"]))

    print()

    print(sample["image"].shape)
    print(sample["mask"].shape)

    print()

    print(sample["image"].dtype)
    print(sample["mask"].dtype)

    print("=" * 60)
