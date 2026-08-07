"""
==============================================================================
Test Script for transforms.py
==============================================================================

Run:

python test_transforms.py

Expected:
- No errors
- Image Tensor
- Mask Tensor
- Correct Shapes
==============================================================================
"""

import logging

import numpy as np
import torch

from datasets.transforms import TransformFactory, TransformManager, verify_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s",
)

# --------------------------------------------------------------------------
# Dummy BraTS Sample
# --------------------------------------------------------------------------

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

print("=" * 70)
print("Original Data")
print("=" * 70)

print("Image Shape :", image.shape)
print("Mask Shape  :", mask.shape)

print()

# --------------------------------------------------------------------------
# Factory Test
# --------------------------------------------------------------------------

print("=" * 70)
print("Testing TransformFactory")
print("=" * 70)

transform = TransformFactory.create(
    training=True,
)

sample = transform(
    image,
    mask,
)

verify_sample(sample)

print("Factory Test Passed")

print()

# --------------------------------------------------------------------------
# Manager Test
# --------------------------------------------------------------------------

print("=" * 70)
print("Testing TransformManager")
print("=" * 70)

manager = TransformManager()

manager.summary()

sample = manager.training()(
    image,
    mask,
)

verify_sample(sample)

print()

print("=" * 70)
print("Output")
print("=" * 70)

print(type(sample["image"]))
print(type(sample["mask"]))

print()

print(sample["image"].shape)
print(sample["mask"].shape)

print()

print(sample["image"].dtype)
print(sample["mask"].dtype)

print()

print("Image Tensor :", isinstance(sample["image"], torch.Tensor))
print("Mask Tensor  :", isinstance(sample["mask"], torch.Tensor))

print()

print("Image Min :", sample["image"].min().item())
print("Image Max :", sample["image"].max().item())

print()

print("All tests passed successfully.")

print("=" * 70)
print("=" * 70)
