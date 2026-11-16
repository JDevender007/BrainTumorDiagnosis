"""Tests for the persistent training manager."""

from __future__ import annotations

import json

import pytest

from training.training_manager import (
    BASELINE_BACKBONES,
    BackboneProgress,
    _ensure_state_shape,
    _fresh_state,
    compute_backbone_percentage,
    compute_project_percentage,
    config_file_for,
    load_state,
    progress_file_for,
    save_state,
    stop_flag_for,
)


def test_fresh_state_shape() -> None:
    state = _fresh_state()

    _ensure_state_shape(state)

    assert set(state["backbones"]) == set(BASELINE_BACKBONES)

    entry = state["backbones"]["densenet121"]

    assert entry["status"] == "pending"
    assert entry["epochs_done"] == 0
    assert entry["total_epochs"] == 30


def test_state_roundtrip(tmp_path) -> None:
    path = tmp_path / "state.json"

    state = _fresh_state()

    state["backbones"]["densenet121"]["epochs_done"] = 15
    state["backbones"]["densenet121"]["status"] = "running"

    save_state(state, path)

    loaded = load_state(path)

    assert loaded["backbones"]["densenet121"]["epochs_done"] == 15
    assert loaded["backbones"]["densenet121"]["status"] == "running"


def test_load_state_missing_returns_fresh(tmp_path) -> None:
    state = load_state(tmp_path / "missing.json")

    assert state["version"] == 1


def test_backbone_percentage() -> None:
    state = _fresh_state()

    state["backbones"]["densenet121"]["total_epochs"] = 30
    state["backbones"]["densenet121"]["epochs_done"] = 15

    assert compute_backbone_percentage(state, "densenet121") == pytest.approx(0.5)

    assert compute_backbone_percentage(state, "resnet50") == 0.0

    state["backbones"]["densenet121"]["epochs_done"] = 60

    assert compute_backbone_percentage(state, "densenet121") == 1.0


def test_project_percentage_partial() -> None:
    state = _fresh_state()

    state["backbones"]["densenet121"]["epochs_done"] = 30
    state["backbones"]["densenet121"]["total_epochs"] = 30

    project = compute_project_percentage(state)

    expected = 0.60 / len(BASELINE_BACKBONES)

    assert project["baselines"] == pytest.approx(expected)

    assert 0.0 < project["total"] < 1.0


def test_project_percentage_complete() -> None:
    state = _fresh_state()

    for name in BASELINE_BACKBONES:
        state["backbones"][name]["epochs_done"] = 30
        state["backbones"][name]["total_epochs"] = 30
        state["backbones"][name]["status"] = "done"

    state["project"]["segmentation_trained"] = True
    state["project"]["ssl_pretrained"] = True
    state["project"]["xai_done"] = True
    state["project"]["evals_done"] = list(BASELINE_BACKBONES)

    project = compute_project_percentage(state)

    assert project["total"] == pytest.approx(1.0)


def test_path_helpers() -> None:
    assert stop_flag_for("densenet121").name == "densenet121.stop"
    assert progress_file_for("densenet121").name == "progress.json"
    assert config_file_for("densenet121").name == "densenet121.yaml"


def test_backbone_progress_dataclass_defaults() -> None:
    entry = BackboneProgress(backbone="cnn2d")

    assert entry.status == "pending"
    assert entry.epochs_done == 0