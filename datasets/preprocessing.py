"""
MRI preprocessing utilities for BrainTumorDiagnosis.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch


MRI_MODALITIES = (
    "t1n",
    "t1c",
    "t2w",
    "t2f",
)


def load_nifti(
    path: str | Path,
) -> np.ndarray:
    """Load a NIfTI image as float32."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    image = nib.load(str(path))

    return np.asarray(
        image.dataobj,
        dtype=np.float32,
    )


def normalize_volume(
    volume: np.ndarray,
    clip_percentiles: tuple[float, float] | None = None,
    mask_background: bool = True,
) -> np.ndarray:
    """
    Canonical z-score normalization for brain MRI volumes.

    Parameters:
        clip_percentiles:
            Optional (low, high) percentiles applied to the voxel
            intensities before z-scoring.

        mask_background:
            If True (default), zero (background) voxels are excluded
            from the statistics and remain zero after normalization.
            If False, all finite voxels contribute to the statistics.

    This is the single reference implementation used across the
    project (datasets, preprocessing, training and inference).
    """

    volume = np.asarray(
        volume,
        dtype=np.float32,
    )

    if mask_background:
        mask = (volume != 0) & np.isfinite(volume)
    else:
        mask = np.isfinite(volume)

    if not np.any(mask):
        return np.zeros_like(
            volume,
            dtype=np.float32,
        )

    values = volume[mask]

    if clip_percentiles is not None:
        low, high = clip_percentiles

        low_value, high_value = np.percentile(
            values,
            [low, high],
        )

        volume = np.clip(
            volume,
            low_value,
            high_value,
        )

        values = volume[mask]

    mean = float(values.mean())

    std = float(values.std())

    normalized = np.zeros_like(
        volume,
        dtype=np.float32,
    )

    if std < 1e-8:
        normalized[mask] = values - mean
    else:
        normalized[mask] = (values - mean) / std

    normalized[~np.isfinite(normalized)] = 0.0

    return normalized


def preprocess_modalities(
    paths: list[str | Path],
) -> torch.Tensor:
    """
    Load and normalize all four MRI modalities.

    Returns:
        [4, D, H, W]
    """

    if len(paths) != 4:
        raise ValueError("Exactly four MRI modalities are required.")

    volumes: list[np.ndarray] = []

    reference_shape: tuple[int, ...] | None = None

    for path in paths:

        volume = load_nifti(path)

        if reference_shape is None:
            reference_shape = volume.shape

        elif volume.shape != reference_shape:
            raise ValueError(
                "MRI modalities have different shapes: "
                f"{reference_shape} vs {volume.shape}"
            )

        volume = normalize_volume(volume)

        volumes.append(volume)

    stacked = np.stack(
        volumes,
        axis=0,
    )

    return torch.from_numpy(
        np.ascontiguousarray(
            stacked,
            dtype=np.float32,
        )
    )
