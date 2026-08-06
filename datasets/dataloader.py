"""
==============================================================================
BraTS 2024 DataLoader

Author : Devender J.
Project : Hybrid Brain Tumor Diagnosis

PyTorch DataLoader for BraTS 2024
==============================================================================
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from datasets.brats_dataset import BraTSDataset
from datasets.transforms import TransformFactory

LOGGER = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================


@dataclass(slots=True)
class DataLoaderConfig:
    """
    DataLoader configuration.
    """

    dataset_root: str

    batch_size: int = 1

    num_workers: int = 4

    validation_split: float = 0.20

    shuffle: bool = True

    pin_memory: bool = True

    persistent_workers: bool = True

    prefetch_factor: int = 2

    random_seed: int = 42


# ==============================================================================
# Worker Seed
# ==============================================================================


def seed_worker(worker_id: int):

    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)

    random.seed(worker_seed)


# ==============================================================================
# BraTS Data Module
# ==============================================================================


class BraTSDataModule:

    def __init__(
        self,
        config: DataLoaderConfig,
    ):

        self.config = config

        self.train_dataset = None

        self.validation_dataset = None

        self.test_dataset = None

        self.generator = torch.Generator().manual_seed(self.config.random_seed)

    # --------------------------------------------------------------------------
    # Prepare Dataset
    # --------------------------------------------------------------------------

    def setup(self):

        LOGGER.info("Loading BraTS Dataset...")

        full_dataset = BraTSDataset(
            dataset_root=Path(self.config.dataset_root),
            transform=TransformFactory.create(training=True),
        )

        total_samples = len(full_dataset)

        validation_size = int(total_samples * self.config.validation_split)

        train_size = total_samples - validation_size

        self.train_dataset, self.validation_dataset = random_split(
            full_dataset,
            [train_size, validation_size],
            generator=self.generator,
        )

        LOGGER.info("Dataset Loaded Successfully")

        LOGGER.info("Total Samples      : %d", total_samples)

        LOGGER.info("Training Samples  : %d", train_size)

        LOGGER.info("Validation Samples: %d", validation_size)

    # --------------------------------------------------------------------------
    # Training DataLoader
    # --------------------------------------------------------------------------

    def train_dataloader(self):

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            worker_init_fn=seed_worker,
            generator=self.generator,
        )

    # --------------------------------------------------------------------------
    # Validation DataLoader
    # --------------------------------------------------------------------------

    def validation_dataloader(self):

        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            worker_init_fn=seed_worker,
            generator=self.generator,
        )

    # --------------------------------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------------------------------

    def test_dataloader(self):

        if self.test_dataset is None:

            self.test_dataset = BraTSDataset(
                dataset_root=Path(self.config.dataset_root),
                transform=TransformFactory.create(
                    training=False,
                ),
            )

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            worker_init_fn=seed_worker,
            generator=self.generator,
        )

    # --------------------------------------------------------------------------
    # Dataset Summary
    # --------------------------------------------------------------------------

    def summary(self):

        LOGGER.info("=" * 60)
        LOGGER.info("BraTS DataLoader Summary")
        LOGGER.info("=" * 60)
        LOGGER.info("Dataset Root      : %s", self.config.dataset_root)
        LOGGER.info("Batch Size        : %d", self.config.batch_size)
        LOGGER.info("Workers           : %d", self.config.num_workers)
        LOGGER.info("Validation Split  : %.2f", self.config.validation_split)
        LOGGER.info("Pin Memory        : %s", self.config.pin_memory)
        LOGGER.info("Shuffle           : %s", self.config.shuffle)
        LOGGER.info("=" * 60)


# ==============================================================================
# Demo
# ==============================================================================


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s : %(message)s",
    )

    # --------------------------------------------------------------------------
    # Update this path to your BraTS 2024 Training Dataset
    # --------------------------------------------------------------------------

    DATASET_ROOT = r"D:\Datasets\BraTS2024"

    config = DataLoaderConfig(
        dataset_root=DATASET_ROOT,
        batch_size=1,
        num_workers=4,
        validation_split=0.20,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
    )

    datamodule = BraTSDataModule(config)

    datamodule.setup()

    datamodule.summary()

    train_loader = datamodule.train_dataloader()

    print()

    print("=" * 70)
    print("Testing Training DataLoader")
    print("=" * 70)

    batch = next(iter(train_loader))

    print("Image Shape :", batch["image"].shape)
    print("Mask Shape  :", batch["mask"].shape)

    print()

    print("Image Dtype :", batch["image"].dtype)
    print("Mask Dtype  :", batch["mask"].dtype)

    print()

    print("Image Device :", batch["image"].device)
    print("Mask Device  :", batch["mask"].device)

    print()

    print("Batch Size :", batch["image"].shape[0])

    print("=" * 70)

    LOGGER.info("DataLoader test completed successfully.")
