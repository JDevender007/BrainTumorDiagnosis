"""
==============================================================================
BraTS 2024 DataLoader Test
==============================================================================

Run:

python test_dataloader.py
==============================================================================
"""

import logging
from pathlib import Path

import torch

from datasets.dataloader import BraTSDataModule, DataLoaderConfig

# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s",
)


# ==============================================================================
# Dataset Path
# ==============================================================================

DATASET_ROOT = r"D:\Projects\BrainTumorDiagnosis\data\raw\BraTS2024"


# ==============================================================================
# Check Dataset
# ==============================================================================

dataset_path = Path(DATASET_ROOT)

if not dataset_path.exists():

    raise FileNotFoundError(f"Dataset not found:\n{dataset_path}")

if not dataset_path.is_dir():

    raise NotADirectoryError(f"Dataset path is not a directory:\n{dataset_path}")


print("=" * 70)
print("Dataset Found:")
print(dataset_path)
print("=" * 70)


# ==============================================================================
# Configuration
# ==============================================================================

config = DataLoaderConfig(
    dataset_root=DATASET_ROOT,
    batch_size=1,
    num_workers=0,
    validation_split=0.20,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
    random_seed=42,
)


# ==============================================================================
# Create DataModule
# ==============================================================================

datamodule = BraTSDataModule(config)


# ==============================================================================
# Setup
# ==============================================================================

print()
print("Loading Dataset...")
print()

datamodule.setup()


# ==============================================================================
# Summary
# ==============================================================================

datamodule.summary()


# ==============================================================================
# Training DataLoader
# ==============================================================================

train_loader = datamodule.train_dataloader()


print()
print("=" * 70)
print("Training DataLoader")
print("=" * 70)

print(
    "Number of Training Batches:",
    len(train_loader),
)


# ==============================================================================
# Read First Training Batch
# ==============================================================================

print()
print("Reading First Training Batch...")
print()


batch = next(iter(train_loader))


# ==============================================================================
# Batch Information
# ==============================================================================

print("=" * 70)
print("Batch Test Results")
print("=" * 70)


print()
print("Batch Keys:")
print(batch.keys())


print()
print("Image Shape:")
print(batch["image"].shape)


print()
print("Mask Shape:")
print(batch["mask"].shape)


print()
print("Image Type:")
print(type(batch["image"]))


print()
print("Mask Type:")
print(type(batch["mask"]))


print()
print("Image Dtype:")
print(batch["image"].dtype)


print()
print("Mask Dtype:")
print(batch["mask"].dtype)


print()
print("Image Device:")
print(batch["image"].device)


print()
print("Mask Device:")
print(batch["mask"].device)


# ==============================================================================
# Patient ID
# ==============================================================================

if "patient_id" in batch:

    print()
    print("Patient ID:")
    print(batch["patient_id"])


# ==============================================================================
# Validation DataLoader
# ==============================================================================

if datamodule.validation_dataset is not None:

    val_loader = datamodule.val_dataloader()

    print()
    print("=" * 70)
    print("Validation DataLoader")
    print("=" * 70)

    print(
        "Number of Validation Batches:",
        len(val_loader),
    )


# ==============================================================================
# Final Validation
# ==============================================================================

image = batch["image"]
mask = batch["mask"]


if not isinstance(
    image,
    torch.Tensor,
):

    raise TypeError("Image is not a PyTorch tensor.")


if not isinstance(
    mask,
    torch.Tensor,
):

    raise TypeError("Mask is not a PyTorch tensor.")


if image.ndim != 5:

    raise RuntimeError(f"Expected image shape " f"(B,C,H,W,D), got {image.shape}")


if mask.ndim != 4:

    raise RuntimeError(f"Expected mask shape " f"(B,H,W,D), got {mask.shape}")


if image.shape[1] != 4:

    raise RuntimeError(f"Expected 4 MRI modalities, " f"got {image.shape[1]}")


if image.shape[2:] != mask.shape[1:]:

    raise RuntimeError("Image and mask spatial dimensions do not match.")


# ==============================================================================
# Success
# ==============================================================================

print()
print("=" * 70)
print("DataLoader Test Passed Successfully")
print("=" * 70)

print()
print("Final Image Shape :", image.shape)
print("Final Mask Shape  :", mask.shape)
print("MRI Channels      :", image.shape[1])
print("Image Dtype       :", image.dtype)
print("Mask Dtype        :", mask.dtype)
print("Device            :", image.device)

print()
print("=" * 70)
