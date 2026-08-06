import numpy as np

from datasets.transforms import TransformFactory

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

transform = TransformFactory.create(training=True)

sample = transform(
    image,
    mask,
)

print(sample["image"].shape)

print(sample["mask"].shape)
