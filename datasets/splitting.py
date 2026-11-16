"""
BraTS split management.

Creates, saves, loads and validates patient-level split manifests
so every experiment uses exactly the same data folds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from datasets.brats import (
    BraTSSample,
    discover_brats_cases,
    extract_patient_id,
    split_brats_cases_by_patient,
)


LOGGER = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SplitManifestConfig:
    """Configuration for creating a split manifest."""

    root_dir: str | Path = "data/raw/BraTS2024"
    output_path: str | Path = "data/splits/brats/split_manifest.json"
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    require_segmentation: bool = True


def _entry(
    case: BraTSSample,
) -> dict[str, str]:
    return {
        "case_id": case.case_id,
        "patient_id": extract_patient_id(case.case_id),
        "case_dir": str(case.case_dir),
    }


def build_split_manifest(
    config: SplitManifestConfig | None = None,
) -> dict[str, Any]:
    """
    Build a patient-level split manifest for the BraTS dataset.

    Returns:
        Manifest dictionary containing the exact per-case assignment
        for each split together with dataset statistics.
    """
    if config is None:
        config = SplitManifestConfig()

    root = Path(config.root_dir)

    cases = discover_brats_cases(
        root,
        require_segmentation=config.require_segmentation,
    )

    (
        train_cases,
        validation_cases,
        test_cases,
    ) = split_brats_cases_by_patient(
        cases,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    splits: dict[str, list[dict[str, str]]] = {
        "train": [_entry(case) for case in train_cases],
        "validation": [_entry(case) for case in validation_cases],
        "test": [_entry(case) for case in test_cases],
    }

    patients = {
        split_name: sorted(
            {entry["patient_id"] for entry in entries}
        )
        for split_name, entries in splits.items()
    }

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": "BraTS2024",
        "split_level": "patient",
        "root_dir": str(root),
        "seed": config.seed,
        "ratios": {
            "train": config.train_ratio,
            "validation": config.validation_ratio,
            "test": config.test_ratio,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "statistics": {
            "total_cases": len(cases),
            "total_patients": len(patients["train"]) + len(patients["validation"]) + len(patients["test"]),
            "cases_per_split": {
                split_name: len(entries)
                for split_name, entries in splits.items()
            },
            "patients_per_split": {
                split_name: len(patient_ids)
                for split_name, patient_ids in patients.items()
            },
        },
        "splits": splits,
    }

    return manifest


def save_split_manifest(
    manifest: dict[str, Any],
    output_path: str | Path | None = None,
) -> Path:
    """
    Save a split manifest as JSON.
    """
    path = Path(output_path) if output_path is not None else Path(
        "data/splits/brats/split_manifest.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=4,
        )

    LOGGER.info("Split manifest saved: %s", path)

    return path


def load_split_manifest(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load a split manifest from JSON.
    """
    manifest_path = Path(path)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    return manifest


def validate_split_manifest(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate a split manifest.

    Checks performed:
        - schema version
        - required top-level keys
        - every case appears in exactly one split
        - every patient appears in exactly one split
        - per-split statistics match the actual split sizes

    Returns:
        Validation report with a ``valid`` flag.
    """
    checks: dict[str, Any] = {}

    checks["schema_version"] = manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION

    required_keys = {
        "dataset",
        "split_level",
        "seed",
        "ratios",
        "statistics",
        "splits",
    }

    checks["required_keys"] = required_keys.issubset(manifest.keys())

    splits = manifest.get("splits", {})

    known_splits = {"train", "validation", "test"}

    checks["split_names"] = known_splits.issubset(splits.keys())

    case_splits: dict[str, set[str]] = {}
    patient_splits: dict[str, set[str]] = {}
    duplicate_cases: list[str] = []
    duplicate_patients: list[str] = []

    split_size_mismatch: list[str] = []

    for split_name, entries in splits.items():

        if not isinstance(entries, list):
            checks["split_names"] = False
            continue

        for entry in entries:
            case_id = entry["case_id"]
            patient_id = entry["patient_id"]

            case_splits.setdefault(case_id, set()).add(split_name)

            patient_splits.setdefault(patient_id, set()).add(split_name)

    checks["no_duplicate_cases"] = all(
        len(split_names) == 1 for split_names in case_splits.values()
    )

    checks["no_duplicate_patients"] = all(
        len(split_names) == 1 for split_names in patient_splits.values()
    )

    checks["duplicate_cases"] = [
        case_id
        for case_id, split_names in case_splits.items()
        if len(split_names) > 1
    ]

    checks["duplicate_patients"] = [
        patient_id
        for patient_id, split_names in patient_splits.items()
        if len(split_names) > 1
    ]

    statistics = manifest.get("statistics", {})

    cases_per_split = statistics.get("cases_per_split", {})
    patients_per_split = statistics.get("patients_per_split", {})

    for split_name in known_splits:
        entries = splits.get(split_name, [])

        if not isinstance(entries, list):
            continue

        actual_cases = len(entries)
        actual_patients = len(
            {entry["patient_id"] for entry in entries}
        )

        expected_cases = cases_per_split.get(split_name)
        expected_patients = patients_per_split.get(split_name)

        if expected_cases is not None and expected_cases != actual_cases:
            split_size_mismatch.append(
                f"{split_name}: cases {expected_cases} != {actual_cases}"
            )

        if expected_patients is not None and expected_patients != actual_patients:
            split_size_mismatch.append(
                f"{split_name}: patients {expected_patients} != {actual_patients}"
            )

    checks["statistics_consistent"] = not split_size_mismatch
    checks["statistics_mismatch"] = split_size_mismatch

    valid = all(
        checks[key]
        for key in (
            "schema_version",
            "required_keys",
            "split_names",
            "no_duplicate_cases",
            "no_duplicate_patients",
            "statistics_consistent",
        )
    )

    report: dict[str, Any] = {
        "valid": bool(valid),
        "total_cases": len(case_splits),
        "total_patients": len(patient_splits),
        "checks": checks,
    }

    return report


def create_and_save_manifest(
    config: SplitManifestConfig | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """
    Create, save and validate a split manifest in one step.

    Returns:
        Tuple of (manifest, saved path, validation report).
    """
    if config is None:
        config = SplitManifestConfig()

    manifest = build_split_manifest(config)

    path = save_split_manifest(
        manifest,
        config.output_path,
    )

    validation = validate_split_manifest(manifest)

    if not validation["valid"]:
        raise RuntimeError(
            "Split manifest validation failed: "
            f"{validation['checks']}"
        )

    LOGGER.info(
        "Split manifest validated: %d cases, %d patients, seed %d.",
        validation["total_cases"],
        validation["total_patients"],
        manifest["seed"],
    )

    return (
        manifest,
        path,
        validation,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    LOGGER.info("=" * 78)
    LOGGER.info("BraTS Split Manifest Test")
    LOGGER.info("=" * 78)

    manifest, path, validation = create_and_save_manifest()

    statistics = manifest["statistics"]

    LOGGER.info(
        "Total patients: %d | Total cases: %d",
        statistics["total_patients"],
        statistics["total_cases"],
    )

    LOGGER.info("Cases per split: %s", statistics["cases_per_split"])

    LOGGER.info("Patients per split: %s", statistics["patients_per_split"])

    LOGGER.info("Manifest: %s", path)

    LOGGER.info("Validation: valid=%s", validation["valid"])

    loaded = load_split_manifest(path)

    assert loaded == manifest

    LOGGER.info("Manifest round-trip: OK")

    LOGGER.info("=" * 78)
    LOGGER.info("Split manifest test passed.")
    LOGGER.info("=" * 78)


if __name__ == "__main__":
    main()