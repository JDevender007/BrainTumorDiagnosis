"""Tests for dataset discovery and patient-level splitting."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from datasets.brats import (
    BraTSSample,
    extract_patient_id,
    group_cases_by_patient,
    split_brats_cases_by_patient,
)
from datasets.classification import _normalise_class_name, _collect_images


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (16, 16), "white").save(path)


def test_normalise_class_name() -> None:
    assert _normalise_class_name("No Tumor") == "No Tumor"
    assert _normalise_class_name("no tumor") == "No Tumor"
    assert _normalise_class_name("Glioma") == "Glioma"

    with pytest.raises(ValueError):
        _normalise_class_name("unknown")


def test_collect_images_case_insensitive(tmp_path) -> None:
    _make_image(tmp_path / "No Tumor" / "a.jpg")
    _make_image(tmp_path / "glioma" / "b.jpg")

    images = _collect_images(tmp_path)

    assert len(images) == 2


def test_extract_patient_id() -> None:
    assert extract_patient_id("BraTS-GLI-02062-100") == "BraTS-GLI-02062"
    assert extract_patient_id("BraTS-GLI-02062-101") == "BraTS-GLI-02062"

    assert extract_patient_id("unusual-name") == "unusual-name"


def _sample(case_id: str) -> BraTSSample:
    base = Path(".") / case_id

    return BraTSSample(
        case_id=case_id,
        case_dir=base,
        t1n=base / f"{case_id}-t1n.nii",
        t1c=base / f"{case_id}-t1c.nii",
        t2w=base / f"{case_id}-t2w.nii",
        t2f=base / f"{case_id}-t2f.nii",
    )


def test_group_cases_by_patient() -> None:
    cases = [
        _sample("BraTS-GLI-0001-100"),
        _sample("BraTS-GLI-0001-101"),
        _sample("BraTS-GLI-0002-100"),
    ]

    groups = group_cases_by_patient(cases)

    assert set(groups) == {"BraTS-GLI-0001", "BraTS-GLI-0002"}
    assert len(groups["BraTS-GLI-0001"]) == 2


def test_patient_level_split_keeps_patients_together() -> None:
    cases = []

    for patient in range(6):
        for exam in range(2):
            cases.append(
                _sample(f"BraTS-GLI-{patient:04d}-{100 + exam}")
            )

    train, validation, test = split_brats_cases_by_patient(
        cases,
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_patients = {extract_patient_id(c.case_id) for c in train}
    validation_patients = {extract_patient_id(c.case_id) for c in validation}
    test_patients = {extract_patient_id(c.case_id) for c in test}

    assert len(train) + len(validation) + len(test) == len(cases)

    assert train_patients.isdisjoint(validation_patients)
    assert train_patients.isdisjoint(test_patients)
    assert validation_patients.isdisjoint(test_patients)

    assert len(train) == len(train_patients) * 2
    assert len(test) == len(test_patients) * 2


def test_patient_level_split_rejects_bad_ratios() -> None:
    cases = [_sample("BraTS-GLI-0001-100")]

    with pytest.raises(ValueError):
        split_brats_cases_by_patient(
            cases,
            train_ratio=0.5,
            validation_ratio=0.5,
            test_ratio=0.5,
        )


def test_patient_level_split_rejects_empty() -> None:
    with pytest.raises(ValueError):
        split_brats_cases_by_patient([])