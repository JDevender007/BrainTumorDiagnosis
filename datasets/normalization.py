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
   Wired into a training path only through the explicit item-G retrain switches
   (``scripts/phase3_train.py --per-image-norm``, exposed here as
   ``PerImageZScore`` / ``normalize="per_image"``); the canonical pipeline and
   every previously reported number keep ImageNet statistics unchanged, so no
   published value moves unless a dedicated new run is trained.

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


class PerImageZScore:
    """Per-image, per-channel MRI-native normalisation for 2-D slices.

    Input is a float tensor ``[0, 1]`` with shape ``(C, H, W)`` (i.e. applied
    right after ``ToTensor``).  Each channel is clipped to its own
    ``[low, high]`` percentile range and then z-scored with ``percentile_zscore``
    — the same routine the 3-D segmentation path uses, so the item-G retrain
    (docs/TODO.md) is MRI-native rather than ImageNet-statistics based.

    Present but unused by the canonical pipeline: it only enters a run when
    the trainer is invoked with ``--per-image-norm``, which writes a new run
    directory and never rewrites a published number.
    """

    def __init__(self, low: float = 1.0, high: float = 99.0) -> None:
        self.low = low
        self.high = high

    def __call__(self, tensor):
        import torch

        channels = [
            torch.from_numpy(percentile_zscore(ch.numpy(), self.low, self.high))
            for ch in tensor
        ]
        return torch.stack(channels, dim=0)


def _normalizer(normalize: str):
    """Return the last transform step for the requested normalisation mode."""
    if normalize == "imagenet":
        return imagenet_normalize()
    if normalize == "per_image":
        return PerImageZScore()
    raise ValueError(f"unknown normalisation mode: {normalize!r} (use 'imagenet' or 'per_image')")


def build_classification_transform(
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
    normalize: str = "imagenet",
):
    """The classification input pipeline: Resize -> ToTensor -> Normalize.

    Defaults reproduce byte-for-byte the canonical pipeline that was
    previously duplicated in ``datasets/classification.py`` (3 sites) and
    ``scripts/run_research_experiments.py`` (2 sites).  ``normalize`` is
    additive: ``"imagenet"`` (default) or ``"per_image"`` (item-G retrain).
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor(),
            _normalizer(normalize),
        ]
    )


def build_classification_augment_transform(
    image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
    normalize: str = "imagenet",
    brightness: float = 0.3,
    contrast: float = 0.3,
    p: float = 0.8,
):
    """Training-only pipeline with intensity augmentation (item G retrain).

    ``Resize -> ToTensor -> RandomApply(ColorJitter(brightness, contrast)) ->
    Normalize``.  Only intensity is perturbed (no geometry), and the
    augmentation is applied *before* normalisation so the normaliser sees the
    corrupted pixel values.  Never use this for validation/test/eval — those
    must stay deterministic via ``build_classification_transform``.
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(tuple(image_size)),
            transforms.ToTensor(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=brightness, contrast=contrast)], p=p
            ),
            _normalizer(normalize),
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
    "PerImageZScore",
    "imagenet_normalize",
    "build_classification_transform",
    "build_classification_augment_transform",
    "percentile_zscore",
]
