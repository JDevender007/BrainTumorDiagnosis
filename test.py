from pathlib import Path

from datasets.brats_dataset import BraTSDataset, DatasetConfig

config = DatasetConfig(
    dataset_root=Path("data/raw/BraTS2024"),
    use_segmentation=True,
)

dataset = BraTSDataset(config)

dataset.summary()

print(dataset)

print()

sample = dataset[0]

print("Patient :", sample["patient_id"])

print("Image Shape :", sample["image"].shape)

print("Mask Shape :", sample["mask"].shape)

print()

print("Image Type :", sample["image"].dtype)

print("Mask Type :", sample["mask"].dtype)
