"""
BraTS2024 dataset validation report.

Scans every case in the raw dataset and reports:

    - modality shapes and voxel spacings (header only)
    - segmentation label set and class-wise voxel counts
    - whole-tumor / tumor-core / enhancing-tumor statistics
    - patient structure (exams per patient)
    - cases that look suspicious or malformed

Output:
    outputs/dataset_reports/brats_validation.json
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from datasets.brats import (
    MRI_MODALITIES,
    discover_brats_cases,
    extract_patient_id,
    load_nifti,
)


LOGGER = logging.getLogger(__name__)

LABEL_NAMES = {
    "0": "background",
    "1": "non_enhancing_tumor_core",
    "2": "edema_snfh",
    "3": "enhancing_tumor",
    "4": "resection_cavity",
}

KNOWN_LABELS = {"0", "1", "2", "3", "4"}

REPORT_PATH = Path("outputs/dataset_reports/brats_validation.json")


def _header_info(
    path: Path | None,
) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None

    image = nib.load(str(path))

    zooms = image.header.get_zooms()

    return {
        "shape": tuple(int(value) for value in image.shape[:3]),
        "spacing": tuple(float(value) for value in zooms[:3]),
        "dtype": str(image.get_data_dtype()),
    }


def _label_counts(
    mask: np.ndarray,
) -> dict[str, int]:
    values, counts = np.unique(mask, return_counts=True)

    return {
        str(int(value)): int(count)
        for value, count in zip(values, counts)
    }


def _scan_case(
    case,
) -> dict[str, Any] | None:
    entry: dict[str, Any] = {
        "case_id": case.case_id,
        "patient_id": extract_patient_id(case.case_id),
        "case_dir": str(case.case_dir),
    }

    shapes: dict[str, tuple[int, ...]] = {}

    for modality in MRI_MODALITIES:
        path = getattr(case, modality)

        info = _header_info(path)

        if info is None:
            entry[f"{modality}_error"] = "missing file"
            continue

        entry[modality] = info

        shapes[modality] = info["shape"]

    unique_shapes = set(shapes.values())

    entry["modality_shapes_match"] = len(unique_shapes) <= 1

    if case.segmentation is None:
        entry["segmentation_error"] = "missing file"
        return entry

    segmentation_info = _header_info(case.segmentation)

    if segmentation_info is None:
        entry["segmentation_error"] = "missing file"
        return entry

    entry["segmentation"] = segmentation_info

    try:
        mask = load_nifti(
            case.segmentation,
            dtype=np.float32,
        )
    except Exception as exc:
        entry["segmentation_error"] = f"load failed: {exc}"
        return entry

    expected_shape = next(iter(shapes.values()), None)

    if expected_shape is not None and mask.shape != expected_shape:
        entry["segmentation_shape_mismatch"] = {
            "segmentation": tuple(mask.shape),
            "modalities": expected_shape,
        }

    counts = _label_counts(mask)

    unknown_labels = [label for label in counts if label not in KNOWN_LABELS]

    if unknown_labels:
        entry["unknown_labels"] = unknown_labels

    entry["label_counts"] = counts

    enhancing = counts.get("3", 0)

    core = counts.get("1", 0) + enhancing

    whole = core + counts.get("2", 0)

    entry["region_voxels"] = {
        "whole_tumor": int(whole),
        "tumor_core": int(core),
        "enhancing_tumor": int(enhancing),
        "resection_cavity": int(counts.get("4", 0)),
    }

    return entry


def build_validation_report(
    root_dir: str | Path = "data/raw/BraTS2024",
) -> dict[str, Any]:
    cases = discover_brats_cases(root_dir)

    entries: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        entry = _scan_case(case)

        if entry is not None:
            entries.append(entry)

        if index % 25 == 0:
            LOGGER.info("Scanned %d/%d cases.", index, len(cases))

    shape_counter: Counter = Counter()

    spacing_by_modality: dict[str, list[tuple[float, ...]]] = {
        modality: [] for modality in MRI_MODALITIES
    }

    label_counter: Counter = Counter()

    region_totals: dict[str, int] = {
        "whole_tumor": 0,
        "tumor_core": 0,
        "enhancing_tumor": 0,
        "resection_cavity": 0,
    }

    patient_exams: Counter = Counter()

    cases_with_tumor = 0

    errors: list[str] = []
    anomalies: list[str] = []

    for entry in entries:
        modality_shapes = [
            entry[modality]["shape"]
            for modality in MRI_MODALITIES
            if modality in entry and modality not in ("segmentation",)
        ]

        for shape in modality_shapes:
            shape_counter[shape] += 1

        for modality in MRI_MODALITIES:
            if modality in entry:
                spacing_by_modality[modality].append(entry[modality]["spacing"])

        label_counter.update(entry.get("label_counts", {}))

        regions = entry.get("region_voxels", {})

        for region in region_totals:
            region_totals[region] += regions.get(region, 0)

        patient_exams[entry["patient_id"]] += 1

        if entry.get("region_voxels", {}).get("whole_tumor", 0) > 0:
            cases_with_tumor += 1

        has_error = (
            any(f"{modality}_error" in entry for modality in MRI_MODALITIES)
            or "segmentation_error" in entry
            or not entry.get("modality_shapes_match", True)
            or "segmentation_shape_mismatch" in entry
        )

        if has_error:
            errors.append(entry["case_id"])

        if entry.get("unknown_labels") or not entry.get("label_counts"):
            anomalies.append(entry["case_id"])

    if len(entries) == 0:
        raise RuntimeError("No BraTS cases found for validation.")

    report: dict[str, Any] = {
        "dataset": "BraTS2024",
        "root_dir": str(root_dir),
        "cases": len(entries),
        "patients": len(patient_exams),
        "exams_per_patient": {
            "min": min(patient_exams.values()),
            "max": max(patient_exams.values()),
            "mean": sum(patient_exams.values()) / len(patient_exams),
            "patient_with_most_exams": patient_exams.most_common(1)[0],
        },
        "modality_shape_distribution": {
            str(shape): count for shape, count in shape_counter.most_common()
        },
        "modality_spacing": {
            modality: {
                "first": list(spacing_by_modality[modality][0])
                if spacing_by_modality[modality]
                else None,
                "distinct": sorted(
                    set(spacing_by_modality[modality])
                )[:10],
                "distinct_count": len(set(spacing_by_modality[modality])),
            }
            for modality in MRI_MODALITIES
        },
        "label_voxel_totals": {
            LABEL_NAMES.get(label, f"label_{label}"): count
            for label, count in sorted(label_counter.items())
        },
        "region_voxel_totals": region_totals,
        "cases_with_tumor": cases_with_tumor,
        "cases_without_tumor": len(entries) - cases_with_tumor,
        "error_cases": errors,
        "anomaly_cases": anomalies,
        "case_entries": entries,
    }

    return report


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    LOGGER.info("=" * 78)
    LOGGER.info("BraTS2024 Dataset Validation")
    LOGGER.info("=" * 78)

    report = build_validation_report()

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    LOGGER.info("Report saved: %s", REPORT_PATH)

    LOGGER.info("Cases    : %d", report["cases"])
    LOGGER.info("Patients : %d", report["patients"])
    LOGGER.info(
        "Exams per patient: min=%d max=%d",
        report["exams_per_patient"]["min"],
        report["exams_per_patient"]["max"],
    )
    LOGGER.info(
        "Modality shapes : %s",
        report["modality_shape_distribution"],
    )
    LOGGER.info(
        "Label voxel totals: %s",
        report["label_voxel_totals"],
    )
    LOGGER.info(
        "Region voxel totals: %s",
        report["region_voxel_totals"],
    )
    LOGGER.info(
        "Cases with tumor: %d / %d",
        report["cases_with_tumor"],
        report["cases"],
    )
    LOGGER.info(
        "Error cases   : %d",
        len(report["error_cases"]),
    )
    LOGGER.info(
        "Anomaly cases : %d",
        len(report["anomaly_cases"]),
    )

    if report["error_cases"]:
        LOGGER.warning(
            "Error case IDs (missing/mismatched files): %s",
            report["error_cases"],
        )

    if report["anomaly_cases"]:
        LOGGER.warning(
            "Anomaly case IDs (unexpected labels): %s",
            report["anomaly_cases"],
        )

    LOGGER.info("=" * 78)


if __name__ == "__main__":
    main()