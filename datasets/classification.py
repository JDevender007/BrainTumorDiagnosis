from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import logging
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from datasets.normalization import build_classification_transform


LOGGER = logging.getLogger(__name__)

CLASS_NAMES: tuple[str, ...] = (
    "Glioma",
    "Meningioma",
    "Pituitary",
    "No Tumor",
)

CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


@dataclass(frozen=True)
class ClassificationSample:
    image_path: Path
    class_name: str
    class_index: int
    split: str
    # Phase 3 (λ_seg sweep, docs/TODO.md A): optional aligned binary mask.
    # When set, the dataset yields (image, label, mask) triples and the
    # trainer feeds seg_targets to MultiTaskLoss (activates L_seg, D2).
    mask_path: Path | None = None


@dataclass(frozen=True)
class ClassificationDataConfig:
    data_directory: str | Path = "data/raw/classification"
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 16
    num_workers: int = 0
    validation_split: float = 0.2
    seed: int = 42


ClassificationDataLoaderConfig = ClassificationDataConfig


def _normalise_class_name(name: str) -> str:
    value = name.strip().lower()

    mapping = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "pituitary": "Pituitary",
        "no tumor": "No Tumor",
        "notumor": "No Tumor",
        "no_tumor": "No Tumor",
    }

    if value not in mapping:
        raise ValueError(f"Unknown classification class directory: {name}")

    return mapping[value]


def _find_class_directory(
    split_directory: Path,
    class_name: str,
) -> Path | None:
    if not split_directory.exists():
        return None

    expected = class_name.lower().replace(" ", "")

    for directory in split_directory.iterdir():
        if not directory.is_dir():
            continue

        current = directory.name.lower().replace(" ", "")

        if current == expected:
            return directory

    return None


def _collect_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _find_split_directory(
    root: Path,
    split_name: str,
) -> Path | None:
    if not root.exists():
        return None

    expected = split_name.lower().replace(" ", "")

    for directory in root.iterdir():
        if not directory.is_dir():
            continue

        current = directory.name.lower().replace(" ", "")

        if current == expected:
            return directory

    return None


def discover_classification_samples(
    data_directory: str | Path = "data/raw/classification",
) -> list[ClassificationSample]:
    root = Path(data_directory)

    if not root.exists():
        raise FileNotFoundError(
            f"Classification dataset directory does not exist: {root}"
        )

    samples: list[ClassificationSample] = []

    split_directories = {
        "train": _find_split_directory(root, "train"),
        "test": _find_split_directory(root, "test"),
    }

    for split_name, split_directory in split_directories.items():

        if split_directory is None:
            LOGGER.warning(
                "Classification split directory does not exist: %s",
                root / split_name,
            )
            continue

        for class_name in CLASS_NAMES:

            class_directory = _find_class_directory(
                split_directory,
                class_name,
            )

            if class_directory is None:
                LOGGER.warning(
                    "Classification class directory does not exist: %s",
                    split_directory / class_name,
                )
                continue

            images = _collect_images(class_directory)

            LOGGER.info(
                "%-12s %-6s : %d images",
                split_name,
                class_name,
                len(images),
            )

            for image_path in images:
                samples.append(
                    ClassificationSample(
                        image_path=image_path,
                        class_name=class_name,
                        class_index=CLASS_TO_INDEX[class_name],
                        split=split_name,
                    )
                )

    if not samples:
        raise RuntimeError(f"No classification images found in: {root}")

    return samples


def get_classification_samples(
    data_directory: str | Path = "data/raw/classification",
) -> list[ClassificationSample]:
    return discover_classification_samples(data_directory)


def split_classification_samples(
    samples: Sequence[ClassificationSample],
    validation_split: float = 0.2,
    seed: int = 42,
) -> tuple[
    list[ClassificationSample],
    list[ClassificationSample],
]:
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1.")

    samples = list(samples)

    if not samples:
        raise ValueError("No classification samples provided.")

    train_samples = []
    validation_samples = []

    rng = random.Random(seed)

    for class_index in range(len(CLASS_NAMES)):

        class_samples = [
            sample for sample in samples if sample.class_index == class_index
        ]

        rng.shuffle(class_samples)

        validation_count = max(
            1,
            int(len(class_samples) * validation_split),
        )

        validation_samples.extend(class_samples[:validation_count])

        train_samples.extend(class_samples[validation_count:])

    rng.shuffle(train_samples)
    rng.shuffle(validation_samples)

    return train_samples, validation_samples


class BrainTumorClassificationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[ClassificationSample],
        image_size: tuple[int, int] = (224, 224),
        transform=None,
    ) -> None:

        self.samples = list(samples)

        if not self.samples:
            raise ValueError("BrainTumorClassificationDataset received no samples.")

        if transform is None:
            transform = build_classification_transform(image_size)

        self.transform = transform
        self.image_size = tuple(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        sample = self.samples[index]

        image = Image.open(sample.image_path).convert("RGB")

        image = self.transform(image)

        label = torch.tensor(
            sample.class_index,
            dtype=torch.long,
        )

        if sample.mask_path is None:
            return image, label

        # Phase 3 λ_seg sweep: aligned binary mask -> (B=) (H, W) long map.
        # Deterministic NEAREST resize mirrors the image transform's Resize
        # (no random augmentation in the canonical pipeline) so image/mask
        # stay pixel-aligned. BRISC PNG masks are anti-aliased (0..255 edge
        # values), so anything >=127 counts as tumor; already-binary masks
        # (max <= 1) pass through unchanged.
        mask_img = Image.open(sample.mask_path)
        if mask_img.mode != "L":
            mask_img = mask_img.convert("L")
        nearest = getattr(Image, "Resampling", Image).NEAREST
        mask_img = mask_img.resize(self.image_size, nearest)
        mask_arr = np.array(mask_img, dtype=np.int64)
        if mask_arr.max() > 1:
            mask_arr = (mask_arr >= 127).astype(np.int64)
        return image, label, torch.from_numpy(mask_arr)


ClassificationDataset = BrainTumorClassificationDataset


def create_classification_dataloaders(
    data_directory: str | Path = "data/raw/classification",
    config: ClassificationDataConfig | None = None,
) -> tuple[DataLoader, DataLoader]:

    if config is None:
        config = ClassificationDataConfig(data_directory=data_directory)

    samples = discover_classification_samples(config.data_directory)

    train_samples = [sample for sample in samples if sample.split.lower() == "train"]

    test_samples = [sample for sample in samples if sample.split.lower() == "test"]

    if not train_samples:
        raise RuntimeError("No training classification images found.")

    if not test_samples:
        raise RuntimeError("No test classification images found.")

    transform = build_classification_transform(config.image_size)

    train_dataset = BrainTumorClassificationDataset(
        train_samples,
        image_size=config.image_size,
        transform=transform,
    )

    test_dataset = BrainTumorClassificationDataset(
        test_samples,
        image_size=config.image_size,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def create_classification_train_val_dataloaders(
    data_directory: str | Path = "data/raw/classification",
    config: ClassificationDataConfig | None = None,
    validation_split: float = 0.2,
    validation_seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from the official
    training split only.

    The validation set is held out from the training data with a
    deterministic per-class stratification. The official Test
    split is NEVER used for validation and remains untouched for
    final evaluation.
    """

    if config is None:
        config = ClassificationDataConfig(data_directory=data_directory)

    samples = discover_classification_samples(config.data_directory)

    train_samples = [sample for sample in samples if sample.split.lower() == "train"]

    if not train_samples:
        raise RuntimeError("No training classification images found.")

    train_samples, validation_samples = split_classification_samples(
        train_samples,
        validation_split=validation_split,
        seed=validation_seed,
    )

    transform = build_classification_transform(config.image_size)

    train_dataset = BrainTumorClassificationDataset(
        train_samples,
        image_size=config.image_size,
        transform=transform,
    )

    validation_dataset = BrainTumorClassificationDataset(
        validation_samples,
        image_size=config.image_size,
        transform=transform,
    )

    from datasets.dataloaders import make_dataloader

    train_loader = make_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=validation_seed,
    )

    validation_loader = make_dataloader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, validation_loader


def create_classification_dataloaders_from_directory(
    data_directory: str | Path = "data/raw/classification",
    batch_size: int = 16,
    image_size: tuple[int, int] = (224, 224),
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:

    config = ClassificationDataConfig(
        data_directory=data_directory,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
    )

    return create_classification_dataloaders(
        data_directory=data_directory,
        config=config,
    )


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s | %(levelname)s | " "%(name)s | %(message)s"),
    )

    LOGGER.info("=" * 78)
    LOGGER.info("BrainTumorDiagnosis Classification Dataset Test")
    LOGGER.info("=" * 78)

    config = ClassificationDataConfig()

    LOGGER.info(
        "Searching classification dataset in: %s",
        config.data_directory,
    )

    try:
        samples = discover_classification_samples(config.data_directory)

        LOGGER.info(
            "Total classification samples: %d",
            len(samples),
        )

        train_samples = [
            sample for sample in samples if sample.split.lower() == "train"
        ]

        test_samples = [sample for sample in samples if sample.split.lower() == "test"]

        LOGGER.info(
            "Training images: %d",
            len(train_samples),
        )

        LOGGER.info(
            "Test images: %d",
            len(test_samples),
        )

        for class_name in CLASS_NAMES:

            train_count = sum(
                sample.class_name == class_name for sample in train_samples
            )

            test_count = sum(sample.class_name == class_name for sample in test_samples)

            LOGGER.info(
                "%-12s | Train: %5d | Test: %5d",
                class_name,
                train_count,
                test_count,
            )

        train_loader, test_loader = create_classification_dataloaders(config=config)

        LOGGER.info(
            "Train batches: %d",
            len(train_loader),
        )

        LOGGER.info(
            "Test batches: %d",
            len(test_loader),
        )

        images, labels = next(iter(train_loader))

        LOGGER.info(
            "Test batch image shape: %s",
            tuple(images.shape),
        )

        LOGGER.info(
            "Test batch label shape: %s",
            tuple(labels.shape),
        )

        LOGGER.info("Classification dataset test completed successfully.")

    except Exception:
        LOGGER.exception("Classification dataset test failed.")
        raise

    LOGGER.info("=" * 78)


if __name__ == "__main__":
    main()


__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_INDEX",
    "ClassificationSample",
    "ClassificationDataConfig",
    "ClassificationDataLoaderConfig",
    "BrainTumorClassificationDataset",
    "ClassificationDataset",
    "discover_classification_samples",
    "get_classification_samples",
    "split_classification_samples",
    "create_classification_dataloaders",
    "create_classification_dataloaders_from_directory",
]
