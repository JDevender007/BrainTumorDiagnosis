"""Single source of truth for classification input normalization.

Two normalizers live here:

1. ``IMAGENET_MEAN`` / ``IMAGENET_STD`` + ``imagenet_normalize()`` /
   ``build_classification_transform()`` — the ImageNet statistics that every
   classification loader in this repo currently uses (grayscale MRI slices are
   converted to 3-channel RGB, then normalized per channel).  These values are
   the ones quoted in TAMNet_Paper.docx ("mean = [0.485, 0.456, 0.406],
   std = [0.229, 0.224, 0.225]").  Importing them from here instead of
   retyping the literals keeps training, evaluation and inference bitwise
   consistent.

2. ``percentile_zscore()`` — MRI-native slice normalization (clip to the
   [low, high] percentile range, then z-score), matching the approach already
   used for the 3-D segmentation volumes in ``datasets/preprocessing.py``.
   It is provided for Phase 3 experiments and is deliberately NOT wired into
   any training path yet: changing normalization invalidates previously
   reported numbers, which is only allowed together with a full re-run.

Nothing in this module changes existing behaviour: the factories reproduce
exactly the ``Resize -> ToTensor -> Normalize`` pipeline that was previously
written out by hand in ``datasets/classification.py`` and
``scripts/run_research_experiments.py``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Canonical ImageNet channel statistics (RGB order).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE: tuple[int, int] = (224, 224)


def imagenet_normalize():
    """Return a torchvision Normalize transform with the repo's constants.

    torchvision is imported lazily so that the pure-numpy ``percentile_zscore``
    stays usable in environments without it.
    """
    from torchvision import transforms

    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def build_classification_transform(
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
):
    """The canonical classification input pipeline: Resize -> ToTensor -> Normalize.

    This is byte-for-byte the pipeline that was previously duplicated in
    ``datasets/classification.py`` (3 sites) and
    ``scripts/run_research_experiments.py`` (2 sites).
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor(),
            imagenet_normalize(),
        ]
    )


def percentile_zscore(
    array: np.ndarray,
    low: float = 1.0,
    high: float = 99.0,
    mask: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Clip to ``[low, high]`` percentile then z-score an MRI slice/volume.

    Args:
        array: 2-D or 3-D image array (any dtype, cast to float32).
        low, high: percentiles used for clipping (default 1-99, the standard
            choice for MRI intensity outlier suppression).
        mask: optional boolean mask; statistics are computed on masked voxels
            only and voxels outside the mask are set to 0 (mirrors
            ``datasets.preprocessing.normalize_volume``).
        eps: minimum std treated as non-degenerate.

    Returns:
        float32 array of the same shape; constant input yields all zeros
        instead of NaN/inf.
    """
    values = np.asarray(array, dtype=np.float32)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        selected = values[mask]
        if selected.size == 0:
            return np.zeros_like(values)
        low_value, high_value = np.percentile(selected, [low, high])
        clipped = np.clip(values, low_value, high_value)
        selected = clipped[mask]
        mean = float(selected.mean())
        std = float(selected.std())
        out = np.zeros_like(clipped)
        out[mask] = (selected - mean) / (std if std > eps else 1.0)
        out[~np.isfinite(out)] = 0.0
        return out.astype(np.float32)

    low_value, high_value = np.percentile(values, [low, high])
    clipped = np.clip(values, low_value, high_value).astype(np.float32)
    mean = float(clipped.mean())
    std = float(clipped.std())
    out = (clipped - mean) / (std if std > eps else 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DEFAULT_IMAGE_SIZE",
    "imagenet_normalize",
    "build_classification_transform",
    "percentile_zscore",
]
