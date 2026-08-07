import numpy as np

from datasets.augmentation import AugmentationConfig, TrainAugmentation

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
        1,
        128,
        128,
        128,
    ),
).astype(np.uint8)

augment = TrainAugmentation(AugmentationConfig())

sample = augment(image, mask)

sample["mask"] = sample["mask"].squeeze(0)

print("=" * 60)

print("Image Type :", type(sample["image"]))

print("Mask Type  :", type(sample["mask"]))

print()

print("Image Shape :", sample["image"].shape)

print("Mask Shape  :", sample["mask"].shape)

print()

print("Image Dtype :", sample["image"].dtype)

print("Mask Dtype  :", sample["mask"].dtype)

print("=" * 60)
