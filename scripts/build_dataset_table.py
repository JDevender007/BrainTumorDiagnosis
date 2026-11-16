#!/usr/bin/env python
"""Build results/dataset_table.csv — one row per dataset split used (or
present) in the TAMNet_Paper.docx experiments.

Image counts and per-class counts are read from disk by this script; source /
citation / modality metadata that is not derivable from the repo is either
taken from an on-disk manifest (BRISC, BraTS split) or marked [VERIFY] when it
still depends on the open FILL questions in docs/AUDIT_REPORT.md section 9.

Usage:
  .venv/Scripts/python scripts/build_dataset_table.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "results" / "dataset_table.csv"

CLASS_DIR_ALIASES = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "pituitary tumor": "pituitary",
    "no tumor": "no_tumor",
    "notumor": "no_tumor",
    "no_tumor": "no_tumor",
}

# Static metadata. Every [VERIFY] ties to an open FILL in docs/AUDIT_REPORT.md.
ROWS = [
    {
        "dataset": "internal_classification",
        "split": "train",
        "path": "data/raw/classification/Train",
        "version": "[VERIFY] unknown",
        "source": "[VERIFY] not BraTS; Kaggle-style filenames; 839 Te-* files "
        "copied from a Kaggle Testing set (see results/leakage_report.json)",
        "citation": "[CITATION NEEDED]",
        "modality": "MRI, 2D slices (JPEG/PNG)",
        "plane": "[VERIFY] not recorded",
        "role": "classifier training (paper internal train)",
        "mask_available": "no",
        "notes": "label provenance open FILL #1",
    },
    {
        "dataset": "internal_classification",
        "split": "test",
        "path": "data/raw/classification/Test",
        "version": "[VERIFY] unknown (same source as train)",
        "source": "[VERIFY] same as train",
        "citation": "[CITATION NEEDED]",
        "modality": "MRI, 2D slices (JPEG/PNG)",
        "plane": "[VERIFY] not recorded",
        "role": "internal test set (headline 96.45%)",
        "mask_available": "no",
        "notes": "used once per final config (audit rule)",
    },
    {
        "dataset": "kaggle_brain_tumor",
        "split": "testing",
        "path": "data/external/kaggle_brain_tumor/Testing",
        "version": "[VERIFY] FILL #2 exact Kaggle dataset name/link",
        "source": "Kaggle (Te-* / Tr-* naming)",
        "citation": "[CITATION NEEDED] likely 'Brain Tumor MRI Dataset' [VERIFY]",
        "modality": "MRI, 2D slices (JPEG)",
        "plane": "[VERIFY] not recorded",
        "role": "external test (paper 99.16%)",
        "mask_available": "no",
        "notes": "CONTAMINATED: overlaps internal train (results/leakage_report.json)",
    },
    {
        "dataset": "kaggle_brain_tumor",
        "split": "training",
        "path": "data/external/kaggle_brain_tumor/Training",
        "version": "[VERIFY] FILL #2",
        "source": "Kaggle (Tr-* naming); local copy lacks the pituitary folder",
        "citation": "[CITATION NEEDED]",
        "modality": "MRI, 2D slices (JPEG)",
        "plane": "[VERIFY] not recorded",
        "role": "not used by paper runs",
        "mask_available": "no",
        "notes": "incomplete local copy (no pituitary)",
    },
    {
        "dataset": "BRISC2025",
        "split": "train",
        "path": "data/external/brisc2025/brisc2025/classification_task/train",
        "version": "BRISC 2025 (manifest on disk)",
        "source": "BRain tumor Image Segmentation & Classification dataset",
        "citation": "Fateh et al., arXiv:2506.14318, 2025",
        "modality": "T1 MRI, 2D slices (JPEG, 512x512)",
        "plane": "axial / coronal / sagittal (in filename)",
        "role": "external train (Phase 3 cross-dataset matrix)",
        "mask_available": "yes (segmentation_task/masks)",
        "notes": "publisher sha256 manifest checked by leakage_check.py",
    },
    {
        "dataset": "BRISC2025",
        "split": "test",
        "path": "data/external/brisc2025/brisc2025/classification_task/test",
        "version": "BRISC 2025 (manifest on disk)",
        "source": "BRain tumor Image Segmentation & Classification dataset",
        "citation": "Fateh et al., arXiv:2506.14318, 2025",
        "modality": "T1 MRI, 2D slices (JPEG, 512x512)",
        "plane": "axial / coronal / sagittal (in filename)",
        "role": "external test (paper 97.70%)",
        "mask_available": "yes (segmentation_task/masks)",
        "notes": "",
    },
    {
        "dataset": "BraTS2024",
        "split": "patient-level 70/15/15 (manifest seed 42)",
        "path": "data/raw/BraTS2024",
        "version": "BraTS 2024 GLI subset (local copy)",
        "source": "Medical Image Computing and Computer-Assisted Intervention "
        "BraTS 2024 challenge",
        "citation": "[VERIFY] BraTS 2024 / Bakas et al. citation missing in paper",
        "modality": "3D multimodal MRI (t1c, t1n, t2f, t2w) NIfTI",
        "plane": "volumetric (3D)",
        "role": "segmentation branch evaluation (paper WT/TC/ET Dice)",
        "mask_available": "yes (seg/)",
        "notes": "39 patients, 200 case dirs; split_manifest.json is "
        "split_level=patient — the only patient-level split in the repo",
    },
]


def count_split(path: Path) -> tuple[int, dict[str, int], int]:
    """Return (total_images, per-class counts, n_jpeg_png)."""
    totals: dict[str, int] = {}
    total = 0
    if not path.is_dir():
        return 0, {}, 0
    for entry in sorted(path.iterdir()):
        if not entry.is_dir():
            continue
        key = CLASS_DIR_ALIASES.get(entry.name.strip().lower(), entry.name.lower())
        n = sum(
            1
            for f in entry.rglob("*")
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        totals[key] = totals.get(key, 0) + n
        total += n
    # files not under a class dir
    loose = sum(
        1
        for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if loose:
        totals["(no class dir)"] = loose
        total += loose
    return total, totals, total


def brats_metadata(path: Path) -> dict:
    manifest_path = path.parent / "splits" / "brats" / "split_manifest.json"
    if not manifest_path.is_file():
        manifest_path = ROOT / "data" / "splits" / "brats" / "split_manifest.json"
    meta: dict = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        meta["split_level"] = manifest.get("split_level")
        meta["seed"] = manifest.get("seed")
        meta["ratios"] = manifest.get("ratios") or {
            k: manifest.get(k) for k in ("train_ratio", "val_ratio", "test_ratio")
        }
        meta["splits"] = {
            name: {
                "cases": len(entry.get("cases", [])),
                "patients": len(entry.get("patients", entry.get("patient_ids", []))),
            }
            for name, entry in manifest.items()
            if isinstance(entry, dict) and "cases" in entry
        }
    n_cases = sum(1 for d in path.iterdir() if d.is_dir()) if path.is_dir() else 0
    patients = set()
    if path.is_dir():
        for d in path.iterdir():
            if d.is_dir():
                parts = d.name.split("-")
                if len(parts) >= 3:
                    patients.add("-".join(parts[:3]))
    meta["n_case_dirs"] = n_cases
    meta["n_patients"] = len(patients)
    return meta


def git_metadata() -> dict:
    meta: dict = {"commit": None}
    try:
        meta["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        pass
    return meta


def main() -> int:
    out_rows = []
    brats_meta = None
    for spec in ROWS:
        path = ROOT / spec["path"]
        total, per_class, _ = count_split(path)
        row = {
            "dataset": spec["dataset"],
            "split": spec["split"],
            "path": spec["path"],
            "version": spec["version"],
            "source": spec["source"],
            "citation": spec["citation"],
            "modality": spec["modality"],
            "plane": spec["plane"],
            "n_images": total,
            "class_glioma": per_class.get("glioma", 0),
            "class_meningioma": per_class.get("meningioma", 0),
            "class_pituitary": per_class.get("pituitary", 0),
            "class_no_tumor": per_class.get("no_tumor", 0),
            "role": spec["role"],
            "mask_available": spec["mask_available"],
            "notes": spec["notes"],
        }
        if spec["dataset"] == "BraTS2024":
            brats_meta = brats_metadata(path)
            row["n_images"] = brats_meta.get("n_case_dirs", total)
            row["notes"] += f" | manifest: {json.dumps(brats_meta, default=str)}"
        out_rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    # sidecar with provenance of the run
    sidecar = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/build_dataset_table.py",
        "git": git_metadata(),
        "brats_manifest": brats_meta,
        "open_fills": [
            "FILL #1: internal dataset provenance / label origin",
            "FILL #2: exact Kaggle dataset name/link",
            "FILL #3: 3D->2D conversion and whether BraTS was used for classification",
        ],
    }
    sidecar_path = OUT_CSV.with_name("dataset_table_meta.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print(f"wrote {OUT_CSV.relative_to(ROOT)} ({len(out_rows)} rows)")
    for row in out_rows:
        print(
            f"  {row['dataset']:<24} {row['split']:<42} n={row['n_images']:>6} "
            f"gli={row['class_glioma']} men={row['class_meningioma']} "
            f"pit={row['class_pituitary']} no={row['class_no_tumor']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
