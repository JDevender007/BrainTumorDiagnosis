"""Tests for datasets.normalization (Phase 1 data-integrity hardening).

Run with:
  .venv/Scripts/python -m pytest tests/test_normalization.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from datasets.normalization import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PerImageZScore,
    build_classification_augment_transform,
    build_classification_transform,
    imagenet_normalize,
    percentile_zscore,
)

# The values quoted verbatim in TAMNet_Paper.docx.
PAPER_MEAN = [0.485, 0.456, 0.406]
PAPER_STD = [0.229, 0.224, 0.225]


class TestImageNetConstants:
    def test_constants_match_paper(self) -> None:
        assert list(IMAGENET_MEAN) == PAPER_MEAN
        assert list(IMAGENET_STD) == PAPER_STD

    def test_normalize_transform_uses_constants(self) -> None:
        transform = imagenet_normalize()
        assert list(transform.mean) == PAPER_MEAN
        assert list(transform.std) == PAPER_STD

    def test_transform_reproduces_handwritten_pipeline(self) -> None:
        """The factory must equal the old inline Resize/ToTensor/Normalize."""
        from torchvision import transforms

        old = transforms.Compose(
            [
                transforms.Resize(DEFAULT_IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=PAPER_MEAN, std=PAPER_STD),
            ]
        )
        new = build_classification_transform()
        rng = np.random.default_rng(0)
        image = Image.fromarray(
            rng.integers(0, 256, (64, 80, 3), dtype=np.uint8), "RGB"
        )
        torch_old = old(image)
        torch_new = new(image)
        assert torch_old.shape == torch_new.shape == (3, 224, 224)
        assert torch_old.equal(torch_new)

    def test_custom_image_size(self) -> None:
        transform = build_classification_transform((128, 96))
        image = Image.new("RGB", (200, 160), color=(10, 200, 30))
        tensor = transform(image)
        assert tuple(tensor.shape) == (3, 128, 96)


class TestPercentileZscore:
    def test_constant_input_yields_zeros_not_nan(self) -> None:
        out = percentile_zscore(np.full((16, 16), 7.5, dtype=np.float32))
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.0)

    def test_clips_outliers_and_zero_centres(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(100.0, 10.0, size=(128, 128)).astype(np.float32)
        data[0, 0] = 1e6  # extreme outlier
        out = percentile_zscore(data, low=1.0, high=99.0)
        assert np.all(np.isfinite(out))
        assert abs(float(out.mean())) < 0.5
        # the outlier must have been clipped: max z-score is bounded
        assert float(np.abs(out).max()) < 10.0

    def test_mask_path(self) -> None:
        data = np.zeros((32, 32), dtype=np.float32)
        data[8:24, 8:24] = np.linspace(0, 100, 16 * 16, dtype=np.float32).reshape(
            16, 16
        )
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        out = percentile_zscore(data, mask=mask)
        assert np.allclose(out[~mask], 0.0)
        inside = out[mask]
        assert np.all(np.isfinite(inside))
        assert abs(float(inside.mean())) < 1e-4

    def test_empty_mask_returns_zeros(self) -> None:
        out = percentile_zscore(np.ones((8, 8)), mask=np.zeros((8, 8), bool))
        assert np.allclose(out, 0.0)

    def test_accepts_int_input(self) -> None:
        out = percentile_zscore(np.arange(100, dtype=np.int16).reshape(10, 10))
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))


class TestItemGRetrainTransforms:
    """docs/TODO.md item G: per-image normalisation + intensity augmentation."""

    def test_default_pipeline_is_unchanged(self) -> None:
        """No argument may ever alter the canonical ImageNet pipeline."""
        import torch

        image = Image.new("RGB", (200, 160), color=(10, 200, 30))
        canonical = build_classification_transform()
        explicit = build_classification_transform(normalize="imagenet")
        assert torch.equal(canonical(image), explicit(image))

    def test_per_image_mode_zero_centres_each_channel(self) -> None:
        import torch

        rng = np.random.default_rng(7)
        tensor = torch.from_numpy(
            rng.random((3, 224, 224), dtype=np.float32)
        )
        out = PerImageZScore()(tensor)
        assert tuple(out.shape) == (3, 224, 224)
        assert torch.isfinite(out).all()
        for ch in out:
            assert abs(float(ch.mean())) < 1e-4
            assert abs(float(ch.std()) - 1.0) < 0.05

    def test_per_image_transform_differs_from_imagenet(self) -> None:
        import torch

        rng = np.random.default_rng(0)
        image = Image.fromarray(
            rng.integers(0, 256, (64, 80, 3), dtype=np.uint8), "RGB"
        )
        a = build_classification_transform(normalize="imagenet")(image)
        b = build_classification_transform(normalize="per_image")(image)
        assert not torch.allclose(a, b)

    def test_unknown_normalisation_rejected(self) -> None:
        with pytest.raises(ValueError, match="normalisation"):
            build_classification_transform(normalize="magic")

    def test_augment_pipeline_shape_and_determinism_when_seeded(self) -> None:
        import torch

        image = Image.new("RGB", (200, 160), color=(120, 120, 120))
        tf = build_classification_augment_transform()
        torch.manual_seed(123)
        a = tf(image)
        torch.manual_seed(123)
        b = tf(image)
        assert tuple(a.shape) == (3, 224, 224)
        assert torch.equal(a, b)  # same seed -> same augmentation


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
