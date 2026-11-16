"""Tests for the unified classification evaluator."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.classification_evaluator import (
    calculate_metrics,
    resolve_checkpoint,
)


def test_calculate_metrics_perfect() -> None:
    labels = [0, 1, 2, 3]

    probabilities = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    metrics = calculate_metrics(
        labels,
        labels,
        probabilities,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_weighted"] == pytest.approx(1.0)
    assert metrics["roc_auc_ovr"] == pytest.approx(1.0)
    assert metrics["test_samples"] == 4

    assert metrics["confusion_matrix"][0] == [1, 0, 0, 0]


def test_calculate_metrics_without_probabilities() -> None:
    labels = [0, 1, 1, 0]
    predictions = [0, 1, 0, 0]

    metrics = calculate_metrics(
        labels,
        predictions,
        probabilities=None,
    )

    assert "roc_auc_ovr" not in metrics
    assert metrics["accuracy"] == pytest.approx(0.75)


def test_resolve_checkpoint_explicit(tmp_path) -> None:
    explicit = tmp_path / "my_checkpoint.pt"

    explicit.touch()

    resolved = resolve_checkpoint("densenet121", str(explicit))

    assert resolved == explicit


def test_resolve_checkpoint_defaults_to_best(monkeypatch) -> None:
    candidate = Path("checkpoints/classification/densenet121/best.pt")

    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: str(self) == str(candidate),
    )

    resolved = resolve_checkpoint("densenet121", None)

    assert resolved == candidate


def test_resolve_checkpoint_missing_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: False,
    )

    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("densenet121", None)