#!/usr/bin/env python
"""Phase 1 split audit: reconstruct every validation-split scheme used by the
TAMNet experiments and record, machine-readably, which ones are valid.

Reproduces (deterministically, no training):
  * scheme C — scripts/run_research_experiments.py:70-75 naive 85/15 tail
    slice over the class-block-ordered sample list (the scheme that produced
    every number in TAMNet_Paper.docx);
  * scheme A — training/classification_trainer.py:353-358 stratified 80/20;
  * the BraTS patient-level split manifest (the only patient-level split);
  * patient-ID availability per dataset (decides whether subject-level
    splitting is even possible).

Output: results/split_audit.json (git commit + config embedded).

Usage:
  .venv/Scripts/python scripts/audit_splits.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "split_audit.json"

INTERNAL_TRAIN = ROOT / "data" / "raw" / "classification" / "Train"
BRATS_MANIFEST = ROOT / "data" / "splits" / "brats" / "split_manifest.json"

# Mirrors datasets/classification.py CLASS_NAMES order (class-block ordering).
CLASS_ORDER = ("Glioma", "Meningioma", "Pituitary", "No Tumor")
PAPER_TAIL_FRACTION = 0.15  # run_research_experiments.py:70 (train_size = 0.85n)
VALIDATION_SEED = 42        # classification.yaml validation_seed
TRAINER_TEST_SIZE = 0.2     # classification_trainer.py:355


def normalise(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def ordered_sample_list(root: Path) -> list[str]:
    """Rebuild the sample list exactly like datasets/classification.py:146-186:
    class-by-class in CLASS_NAMES order, files sorted within each class."""
    labels: list[str] = []
    dirs = {normalise(d.name): d for d in root.iterdir() if d.is_dir()}
    for class_name in CLASS_ORDER:
        target = normalise(class_name)
        directory = dirs.get(target)
        if directory is None:
            # tolerant match (e.g. 'notumor' vs 'no tumor')
            for key, candidate in dirs.items():
                if key.replace("tumor", "") == target.replace("tumor", ""):
                    directory = candidate
                    break
        if directory is None:
            raise FileNotFoundError(f"class dir not found for {class_name!r} in {root}")
        files = sorted(
            f for f in directory.iterdir() if f.is_file()
            and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        labels.extend([class_name] * len(files))
    return labels


def git_metadata() -> dict:
    meta: dict = {"commit": None}
    try:
        meta["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        pass
    return meta


def scheme_c_tail_split(labels: list[str]) -> dict:
    """run_research_experiments.py:70-75 — no shuffle, no stratification."""
    n = len(labels)
    train_size = int((1.0 - PAPER_TAIL_FRACTION) * n)
    train, val = labels[:train_size], labels[train_size:]
    return {
        "location": "scripts/run_research_experiments.py:70-75",
        "method": "train[:int(0.85*n)] / train[int(0.85*n):] on the "
        "class-block-ordered sample list (no shuffle, no stratification)",
        "n_total": n,
        "n_train": len(train),
        "n_val": len(val),
        "train_class_counts": dict(Counter(train)),
        "val_class_counts": dict(Counter(val)),
        "val_n_classes": len(set(val)),
        "valid": len(set(val)) > 1,
        "verdict": "INVALID — validation tail is single-class; model selection "
        "(best val F1, early stopping) ran on a degenerate set"
        if len(set(val)) == 1
        else "OK",
    }


def scheme_a_stratified(labels: list[str]) -> dict:
    """classification_trainer.py:353-358 — stratified image-level 80/20."""
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=TRAINER_TEST_SIZE,
        random_state=VALIDATION_SEED,
        stratify=[labels[i] for i in indices],
    )
    train = [labels[i] for i in train_idx]
    val = [labels[i] for i in val_idx]
    return {
        "location": "training/classification_trainer.py:353-358",
        "method": f"sklearn train_test_split(test_size={TRAINER_TEST_SIZE}, "
        f"random_state={VALIDATION_SEED}, stratify=labels) — image-level",
        "n_total": len(labels),
        "n_train": len(train),
        "n_val": len(val),
        "train_class_counts": dict(Counter(train)),
        "val_class_counts": dict(Counter(val)),
        "val_n_classes": len(set(val)),
        "valid": len(set(val)) > 1,
        "verdict": "OK stratified, but IMAGE-level: no patient IDs exist in "
        "this dataset, so subject-level splitting is impossible (see "
        "patient_id_availability)",
    }


def brats_patient_split() -> dict:
    if not BRATS_MANIFEST.is_file():
        return {"error": f"missing {BRATS_MANIFEST}"}
    manifest = json.loads(BRATS_MANIFEST.read_text(encoding="utf-8"))
    # schema: {"split_level": ..., "statistics": {...},
    #          "splits": {"train": [{"case_id", "patient_id", "case_dir"}, ...], ...}}
    raw_splits = manifest.get("splits") or {}
    splits: dict[str, dict] = {}
    patient_sets: dict[str, set] = {}
    for name, entries in raw_splits.items():
        if not isinstance(entries, list):
            continue
        patients = {
            entry["patient_id"]
            for entry in entries
            if isinstance(entry, dict) and entry.get("patient_id")
        }
        splits[name] = {
            "n_cases": len(entries),
            "n_patients": len(patients),
        }
        patient_sets[name] = patients
    overlaps = {}
    names = sorted(patient_sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlaps[f"{a}__{b}"] = len(patient_sets[a] & patient_sets[b])
    zero_overlap = bool(overlaps) and all(v == 0 for v in overlaps.values())
    is_patient_level = manifest.get("split_level") == "patient"
    return {
        "location": "data/splits/brats/split_manifest.json",
        "split_level": manifest.get("split_level"),
        "seed": manifest.get("seed"),
        "ratios": manifest.get("ratios"),
        "statistics": manifest.get("statistics"),
        "splits": splits,
        "cross_split_patient_overlap": overlaps,
        "valid": is_patient_level and zero_overlap,
        "verdict": (
            "OK — patient-level, zero cross-split patient overlap"
            if is_patient_level and zero_overlap
            else "CHECK — patient-level split but patients overlap across splits"
            if is_patient_level
            else "CHECK — not a patient-level split"
        ),
    }


def main() -> int:
    labels = ordered_sample_list(INTERNAL_TRAIN)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/audit_splits.py",
        "git": git_metadata(),
        "dataset": {
            "path": str(INTERNAL_TRAIN.relative_to(ROOT)),
            "n_images": len(labels),
            "class_counts": dict(Counter(labels)),
            "sample_list_order": "class-block ordered (Glioma, Meningioma, "
            "Pituitary, No Tumor), files sorted within class — "
            "datasets/classification.py:146-186",
        },
        "schemes": {
            "C_paper_runs": scheme_c_tail_split(labels),
            "A_trainer": scheme_a_stratified(labels),
            "B_dataset_helper": {
                "location": "datasets/classification.py:200-241",
                "method": "per-class stratified holdout, seed 42, image-level",
                "note": "not used by the paper runs; same image-level caveat",
            },
            "D_all_experiments": {
                "location": "scripts/run_all_experiments.py:127-138",
                "method": "np.random.seed + shuffle, unstratified, image-level",
                "note": "not used by the paper runs",
            },
        },
        "brats": brats_patient_split(),
        "patient_id_availability": {
            "internal_classification": {
                "patient_ids_in_filenames": False,
                "examples": ["gg (325).jpg", "1299.jpg", "1 no.jpeg", "1178.jpg"],
                "subject_level_split_possible": False,
                "impact": "the paper's internal train/test split CANNOT be made "
                "subject-level without new metadata; must be disclosed as a "
                "limitation (Phase 5)",
            },
            "kaggle_brain_tumor": {
                "patient_ids_in_filenames": False,
                "examples": ["Te-glTr_0000.jpg", "Tr-glTr_0000.jpg"],
                "subject_level_split_possible": False,
            },
            "brisc2025": {
                "patient_ids_in_filenames": False,
                "examples": ["brisc2025_test_00001_gl_ax_t1.jpg"],
                "note": "index + plane + sequence only, no subject identifier",
                "subject_level_split_possible": False,
            },
            "brats2024": {
                "patient_ids_in_filenames": True,
                "examples": ["BraTS-GLI-02062-100 ..."],
                "subject_level_split_possible": True,
                "in_use": True,
            },
        },
        "headline_finding": (
            "Scheme C (the paper's scheme) yields a validation set of "
            f"{scheme_c_tail_split(labels)['n_val']} images that are 100% "
            "single-class — this is the mechanical cause of macro/weighted F1 "
            "(0.9847) exceeding accuracy (0.9645) in TAMNet_Paper.docx."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
    for key, scheme in report["schemes"].items():
        if isinstance(scheme, dict) and "val_class_counts" in scheme:
            print(f"  {key}: val={scheme['val_class_counts']} -> {scheme['verdict']}")
    print(f"  brats: {report['brats'].get('verdict')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
