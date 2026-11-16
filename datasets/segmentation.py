"""
Patch-based BraTS segmentation dataset.

Builds 3D patches for segmentation training with:

    - per-modality normalization of nonzero voxels (z-score)
    - foreground-biased random patch sampling
    - random spatial flips

BraTS 2024 post-treatment label schema:

    Label 0: background
    Label 1: non-enhancing tumor core (NETC)
    Label 2: surrounding non-enhancing FLAIR hyperintensity (SNFH / edema)
    Label 3: enhancing tumor (ET)
    Label 4: resection cavity (RC)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.brats import (
    BraTSSample,
    MRI_MODALITIES,
    discover_brats_cases,
    load_nifti,
)


LOGGER = logging.getLogger(__name__)

FOREGROUND_LABELS = (1, 2, 3, 4)


@dataclass(frozen=True)
class SegmentationDatasetConfig:
    """Configuration for the patch-based segmentation dataset."""

    patch_size: tuple[int, int, int] = (96, 96, 96)
    samples_per_case: int = 8
    foreground_ratio: float = 0.6
    augment: bool = True


def _nonzero_zscore(
    volume: np.ndarray,
) -> np.ndarray:
    """
    Z-score normalize nonzero voxels per volume.

    Background (zero) voxels are excluded from the statistic and
    kept at zero afterwards.

    Delegates to the canonical implementation in
    datasets.preprocessing.normalize_volume.
    """

    from datasets.preprocessing import normalize_volume

    return normalize_volume(
        volume,
        clip_percentiles=None,
        mask_background=True,
    )


def _crop_patch(
    array: np.ndarray,
    center: Sequence[int],
    patch_size: Sequence[int],
    has_channels: bool = False,
) -> np.ndarray:
    """
    Crop an edge-clamped spatial patch around a center.

    Args:
        array: Volume with shape [D, H, W] or [C, D, H, W].
        center: Spatial center coordinates (depth, height, width).
        patch_size: Patch size per spatial dimension.
        has_channels: Whether the first axis holds channels.

    Falls back to edge padding when the patch is larger than
    the volume along a dimension.
    """
    working = array

    center = list(center)

    spatial_offset = 1 if has_channels else 0

    for dim in range(spatial_offset, spatial_offset + 3):
        size = working.shape[dim]
        patch = patch_size[dim - spatial_offset]

        if patch > size:
            pad_before = patch // 2

            pad_width = [(0, 0)] * working.ndim
            pad_width[dim] = (
                pad_before,
                patch - pad_before,
            )

            working = np.pad(
                working,
                pad_width,
                mode="edge",
            )

            center[dim - spatial_offset] += pad_before

    slices = [slice(None)] * working.ndim

    for dim in range(spatial_offset, spatial_offset + 3):
        patch = patch_size[dim - spatial_offset]

        working_size = working.shape[dim]

        start = max(0, center[dim - spatial_offset] - patch // 2)

        stop = min(working_size, start + patch)

        start = stop - patch

        slices[dim] = slice(start, stop)

    return working[tuple(slices)]


def load_normalized_case(
    case: BraTSSample,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one case as normalized image volume and label map.

    Returns:
        image: [4, D, H, W] float32
        segmentation: [D, H, W] int64
    """
    volumes = []

    for modality in MRI_MODALITIES:
        path = getattr(case, modality)

        volume = load_nifti(path)

        volumes.append(_nonzero_zscore(volume))

    image = np.stack(
        volumes,
        axis=0,
    )

    if case.segmentation is None:
        raise ValueError(
            f"Segmentation file missing for case {case.case_id}"
        )

    segmentation = load_nifti(
        case.segmentation,
    )

    segmentation = np.asarray(
        segmentation,
        dtype=np.int64,
    )

    if segmentation.shape != image.shape[1:]:
        raise ValueError(
            "Segmentation shape does not match modalities for "
            f"case {case.case_id}: "
            f"{tuple(segmentation.shape)} vs {tuple(image.shape[1:])}"
        )

    unknown_labels = np.setdiff1d(
        np.unique(segmentation),
        [0, 1, 2, 3, 4],
    )

    if unknown_labels.size > 0:
        raise ValueError(
            "Unexpected label values "
            f"{unknown_labels.tolist()} in case {case.case_id}"
        )

    return image, segmentation


class SegmentationDataset(Dataset):
    """
    Patch-based segmentation dataset for BraTS 2024.
    """

    def __init__(
        self,
        cases: Sequence[BraTSSample],
        config: SegmentationDatasetConfig | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if not cases:
            raise ValueError("Segmentation dataset requires at least one case.")

        self.cases = list(cases)

        self.config = config if config is not None else SegmentationDatasetConfig()

        self.patch_size = tuple(self.config.patch_size)

        if len(self.patch_size) != 3:
            raise ValueError(f"patch_size must have three entries: {self.patch_size}")

        self.seed = seed

        self._generator = np.random.default_rng(seed)

        self._cache_case_id: str | None = None
        self._cache_image: np.ndarray | None = None
        self._cache_segmentation: np.ndarray | None = None
        self._cache_foreground: tuple[int, int, int, int, int, int] | None = None

    def __len__(self) -> int:
        return len(self.cases) * self.config.samples_per_case

    def reseed(
        self,
        epoch: int,
    ) -> None:
        """
        Reset the sampling generator so every epoch is reproducible.
        """
        self._generator = np.random.default_rng(self.seed + epoch)

    def _get_case_arrays(
        self,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int, int, int]]:
        case = self.cases[index // self.config.samples_per_case]

        if case.case_id != self._cache_case_id:
            image, segmentation = load_normalized_case(case)

            self._cache_case_id = case.case_id
            self._cache_image = image
            self._cache_segmentation = segmentation

            foreground_indices = np.argwhere(
                np.isin(
                    segmentation,
                    FOREGROUND_LABELS,
                )
            )

            if foreground_indices.size == 0:
                self._cache_foreground = (0, 0, 0, *image.shape[1:])
            else:
                lower = foreground_indices.min(axis=0)
                upper = foreground_indices.max(axis=0)

                self._cache_foreground = tuple(
                    int(value) for value in (*lower, *upper)
                )

        return (
            self._cache_image,
            self._cache_segmentation,
            self._cache_foreground,
        )

    @staticmethod
    def _random_center(
        rng: np.random.Generator,
        shape: Sequence[int],
        foreground: tuple[int, int, int, int, int, int],
    ) -> tuple[int, int, int]:
        depth_low, height_low, width_low, depth_high, height_high, width_high = (
            foreground
        )

        center = (
            int(rng.integers(depth_low, depth_high + 1)),
            int(rng.integers(height_low, height_high + 1)),
            int(rng.integers(width_low, width_high + 1)),
        )

        return center

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        image, segmentation, foreground = self._get_case_arrays(index)

        shape = image.shape[1:]

        if self._generator.random() < self.config.foreground_ratio:
            center = self._random_center(
                self._generator,
                shape,
                foreground,
            )
        else:
            center = (
                int(self._generator.integers(0, shape[0])),
                int(self._generator.integers(0, shape[1])),
                int(self._generator.integers(0, shape[2])),
            )

        image_patch = _crop_patch(
            image,
            center,
            self.patch_size,
            has_channels=True,
        )

        segmentation_patch = _crop_patch(
            segmentation,
            center,
            self.patch_size,
        )

        if self.config.augment:
            image_patch, segmentation_patch = _random_flips(
                self._generator,
                image_patch,
                segmentation_patch,
            )

        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(
                    image_patch,
                    dtype=np.float32,
                )
            ),
            "label": torch.from_numpy(
                np.ascontiguousarray(
                    segmentation_patch,
                    dtype=np.int64,
                )
            ),
        }


def _random_flips(
    rng: np.random.Generator,
    image: np.ndarray,
    segmentation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the same random spatial flips to image and segmentation.
    """
    for axis in (1, 2, 3):
        if rng.random() < 0.5:
            image = np.flip(
                image,
                axis=axis,
            )
            segmentation = np.flip(
                segmentation,
                axis=axis - 1,
            )

    return image, segmentation


def load_split_cases(
    manifest_path: str | Path,
    split_name: str,
) -> list[BraTSSample]:
    """
    Load BraTS cases belonging to one manifest split.

    Cases are re-discovered and filtered by the exact case ids
    recorded in the manifest.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    if "root_dir" not in manifest:
        raise ValueError(
            f"Manifest {manifest_path} does not record a root directory."
        )

    splits = manifest.get("splits", {})

    if split_name not in splits:
        raise ValueError(
            f"Split '{split_name}' not found in manifest. "
            f"Available: {sorted(splits.keys())}"
        )

    split_entries = splits[split_name]

    expected_case_ids = {entry["case_id"] for entry in split_entries}

    all_cases = discover_brats_cases(
        manifest["root_dir"],
    )

    selected = [
        case
        for case in all_cases
        if case.case_id in expected_case_ids
    ]

    missing = sorted(expected_case_ids - {case.case_id for case in selected})

    if missing:
        raise RuntimeError(
            f"Manifest split '{split_name}' references "
            f"{len(missing)} cases that could not be found: {missing[:5]}"
        )

    LOGGER.info(
        "Loaded %d cases for split '%s' from manifest %s",
        len(selected),
        split_name,
        manifest_path.name,
    )

    return selected


def create_synthetic_brats_cases(
    target_dir: str | Path,
    num_cases: int = 3,
    volume_shape: Sequence[int] = (48, 64, 56),
    seed: int = 7,
) -> list[BraTSSample]:
    """
    Create small synthetic NIfTI BraTS cases for self-tests.

    Each case contains a blob of enhancing tumor (label 3) with a
    small resection cavity (label 4) inside it.
    """
    import nibabel as nib

    target_dir = Path(target_dir)

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(seed)

    shape = tuple(volume_shape)

    cases: list[BraTSSample] = []

    for case_number in range(num_cases):
        name = f"BraTS-GLI-{case_number + 1:05d}-100"
        case_dir = target_dir / name
        case_dir.mkdir(parents=True)

        segmentation = np.zeros(
            shape,
            dtype=np.int16,
        )

        depth, height, width = shape

        blob = (
            slice(depth // 5, depth // 2),
            slice(height // 4, height // 2 + height // 8),
            slice(width // 5, width // 2),
        )

        segmentation[blob] = 3

        cavity = (
            slice(
                depth // 5 + depth // 10,
                depth // 5 + depth // 10 + 4,
            ),
            slice(
                height // 4 + 4,
                height // 4 + 8,
            ),
            slice(
                width // 5 + 4,
                width // 5 + 8,
            ),
        )

        segmentation[cavity] = 4

        paths: dict[str, str] = {}

        for modality in MRI_MODALITIES:
            volume = rng.normal(
                size=shape,
            ).astype(np.float32)

            volume[segmentation > 0] += 40.0

            path = case_dir / f"{name}-{modality}.nii"

            nib.save(
                nib.Nifti1Image(
                    volume,
                    np.eye(4),
                ),
                str(path),
            )

            paths[modality] = str(path)

        seg_path = case_dir / f"{name}-seg.nii"

        nib.save(
            nib.Nifti1Image(
                segmentation,
                np.eye(4),
            ),
            str(seg_path),
        )

        cases.append(
            BraTSSample(
                case_id=name,
                case_dir=case_dir,
                t1n=paths["t1n"],
                t1c=paths["t1c"],
                t2w=paths["t2w"],
                t2f=paths["t2f"],
                segmentation=seg_path,
            )
        )

    return cases


def _test() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    import tempfile

    temp_dir = tempfile.TemporaryDirectory()

    cases = create_synthetic_brats_cases(
        Path(temp_dir.name),
        num_cases=3,
        volume_shape=(48, 64, 56),
    )

    config = SegmentationDatasetConfig(
        patch_size=(16, 16, 16),
        samples_per_case=12,
        foreground_ratio=0.6,
    )

    dataset = SegmentationDataset(
        cases,
        config=config,
        seed=1,
    )

    LOGGER.info("Dataset length: %d", len(dataset))

    foreground_patches = 0

    for index in range(len(dataset)):
        sample = dataset[index]

        assert tuple(sample["image"].shape) == (4, 16, 16, 16)
        assert tuple(sample["label"].shape) == (16, 16, 16)
        assert sample["label"].dtype == torch.long

        if torch.any(sample["label"] > 0):
            foreground_patches += 1

    LOGGER.info(
        "Foreground patches: %d / %d",
        foreground_patches,
        len(dataset),
    )

    assert foreground_patches > 0

    dataset.reseed(0)

    first_sample = dataset[0]

    dataset.reseed(0)

    second_sample = dataset[0]

    assert bool(
        torch.equal(
            first_sample["image"],
            second_sample["image"],
        )
    )

    LOGGER.info("Reproducibility check passed.")

    temp_dir.cleanup()

    LOGGER.info("Segmentation dataset test passed.")


if __name__ == "__main__":
    _test()