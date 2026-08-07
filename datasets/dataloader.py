"""
==============================================================================
BraTS 2024 DataLoader

Author  : Devender J.
Project : Hybrid Brain Tumor Diagnosis

Creates training and validation DataLoaders for BraTS 2024.
==============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split

from datasets.brats_dataset import BraTSDataset, DatasetConfig
from datasets.transforms import TransformFactory

LOGGER = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================


@dataclass(slots=True)
class DataLoaderConfig:

    dataset_root: str

    batch_size: int = 1

    num_workers: int = 0

    validation_split: float = 0.20

    shuffle: bool = True

    pin_memory: bool = False

    persistent_workers: bool = False

    random_seed: int = 42


# ==============================================================================
# BraTS Data Module
# ==============================================================================


class BraTSDataModule:

    def __init__(
        self,
        config: DataLoaderConfig,
    ) -> None:

        self.config = config

        self.dataset_root = Path(config.dataset_root)

        self.full_dataset: Optional[BraTSDataset] = None

        self.train_dataset = None

        self.validation_dataset = None

    # ==========================================================================
    # Setup
    # ==========================================================================

    def setup(self) -> None:

        LOGGER.info("Loading BraTS Dataset...")

        # ----------------------------------------------------------------------
        # Training Transform
        # ----------------------------------------------------------------------

        LOGGER.info("Creating training transform...")

        train_transform = TransformFactory.create(training=True)

        # ----------------------------------------------------------------------
        # Dataset Configuration
        # ----------------------------------------------------------------------

        dataset_config = DatasetConfig(dataset_root=str(self.dataset_root))

        # ----------------------------------------------------------------------
        # Full Dataset
        # ----------------------------------------------------------------------

        self.full_dataset = BraTSDataset(
            config=dataset_config,
            transform=train_transform,
        )

        total_size = len(self.full_dataset)

        # ----------------------------------------------------------------------
        # Validation Size
        # ----------------------------------------------------------------------

        validation_size = int(total_size * self.config.validation_split)

        # Ensure validation set exists
        # when the dataset has enough samples.
        if total_size > 1:

            validation_size = max(
                1,
                validation_size,
            )

            validation_size = min(
                validation_size,
                total_size - 1,
            )

        else:

            validation_size = 0

        train_size = total_size - validation_size

        # ----------------------------------------------------------------------
        # Split Dataset
        # ----------------------------------------------------------------------

        generator = torch.Generator()

        generator.manual_seed(self.config.random_seed)

        if validation_size > 0:

            self.train_dataset, self.validation_dataset = random_split(
                self.full_dataset,
                [
                    train_size,
                    validation_size,
                ],
                generator=generator,
            )

        else:

            self.train_dataset = self.full_dataset

            self.validation_dataset = None

        LOGGER.info("Dataset Loaded Successfully")

        LOGGER.info(
            "Total Samples      : %d",
            total_size,
        )

        LOGGER.info(
            "Training Samples   : %d",
            len(self.train_dataset),
        )

        LOGGER.info(
            "Validation Samples : %d",
            (
                len(self.validation_dataset)
                if self.validation_dataset is not None
                else 0
            ),
        )

    # ==========================================================================
    # Training DataLoader
    # ==========================================================================

    def train_dataloader(self) -> DataLoader:

        if self.train_dataset is None:

            raise RuntimeError("DataModule has not been setup. " "Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=(
                self.config.persistent_workers and self.config.num_workers > 0
            ),
        )

    # ==========================================================================
    # Validation DataLoader
    # ==========================================================================

    def val_dataloader(self) -> DataLoader:

        if self.validation_dataset is None:

            raise RuntimeError("Validation dataset is not available.")

        return DataLoader(
            self.validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=(
                self.config.persistent_workers and self.config.num_workers > 0
            ),
        )

    # ==========================================================================
    # Alias
    # ==========================================================================

    def validation_dataloader(self) -> DataLoader:

        return self.val_dataloader()

    # ==========================================================================
    # Summary
    # ==========================================================================

    def summary(self) -> None:

        LOGGER.info("===================================")

        LOGGER.info("BraTS DataLoader Summary")

        LOGGER.info("===================================")

        LOGGER.info(
            "Dataset Root : %s",
            self.dataset_root,
        )

        LOGGER.info(
            "Batch Size   : %d",
            self.config.batch_size,
        )

        LOGGER.info(
            "Workers      : %d",
            self.config.num_workers,
        )

        LOGGER.info(
            "Validation   : %.2f",
            self.config.validation_split,
        )

        LOGGER.info("===================================")


# ==============================================================================
# Test
# ==============================================================================


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s : %(message)s",
    )

    dataset_root = Path(__file__).resolve().parents[1] / "data" / "raw" / "BraTS2024"

    config = DataLoaderConfig(
        dataset_root=str(dataset_root),
        batch_size=1,
        num_workers=0,
        validation_split=0.20,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    datamodule = BraTSDataModule(config)

    datamodule.setup()

    datamodule.summary()

    train_loader = datamodule.train_dataloader()

    LOGGER.info(
        "Training Batches : %d",
        len(train_loader),
    )

    if datamodule.validation_dataset is not None:

        val_loader = datamodule.val_dataloader()

        LOGGER.info(
            "Validation Batches : %d",
            len(val_loader),
        )
