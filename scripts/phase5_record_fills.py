"""Record the three Phase 5 FILL resolutions into ``results/dataset_table.csv``.

Run once after ``scripts/phase5_fill_evidence.py``. Only rewrites the cells that
the evidence actually settles; every field that stays unknown keeps its
``[VERIFY]`` / ``[CITATION NEEDED]`` marker so no provenance is invented.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "dataset_table.csv"

FILL1_NOTES = (
    "FILL#1 RESOLVED (partial): 11148 images / 10223 unique basenames; 4463 (43.7%) "
    "basename-match the canonical Kaggle zip. Filename families: Tr-* 3624, X_###_XX_ 3439, "
    "plain-numeric 1713, other 1013, Te-* 839, 'gg (N)' 363, 'image(N)' 123, 'N no' 34. "
    "490 basenames occur in >1 class folder but 0 are byte-identical (benign name collisions, "
    "no label conflict). No patient/subject IDs in any filename. Label ORIGIN still "
    "[CITATION NEEDED]: no download script, manifest, README or citation exists in the repo. "
    "See results/fill_evidence.json"
)

FILL2_SOURCE = (
    "Kaggle archive on disk: data/external/kaggle_brain_tumor/brain-tumor-mri-dataset.zip "
    "(7023 entries = Training 5712 + Testing 1311)"
)
FILL2_CITATION = (
    "Kaggle 'Brain Tumor MRI Dataset' (Te-*/Tr-* naming, 4 classes, 7023 images) "
    "[VERIFY owner slug + URL against the Kaggle page before camera-ready]"
)
FILL2_TEST_NOTES = (
    "CONTAMINATED: overlaps internal train (results/leakage_report.json). "
    "FILL#2 RESOLVED (partial): zip per-directory counts are Testing 300/306/405/300 = 1311 "
    "and Training 1321/1339/1595/1457 = 5712, i.e. exactly the canonical 7023-image release. "
    "Local *extracted* Testing/ folder matches the zip (1311); the URL itself is [VERIFY]. "
    "See results/fill_evidence.json"
)
FILL2_TRAIN_NOTES = (
    "FILL#2 RESOLVED (partial): the local Training/ folder is an INCOMPLETE extraction of "
    "brain-tumor-mri-dataset.zip - it has glioma 1321 / meningioma 1339 / notumor 1221 and "
    "NO pituitary folder, against 1321/1339/1595/1457 in the zip. The zip itself is complete "
    "(5712). not used by paper runs. See results/fill_evidence.json"
)

FILL3_SOURCE = (
    "BraTS 2024 GLI subset (local copy); SEGMENTATION ONLY - contributes no classification "
    "images (FILL#3)"
)
FILL3_NOTES = (
    "FILL#3 RESOLVED: 1000 NIfTI volumes (t1c 200 / t1n 200 / t2f 200 / t2w 200) + 200 seg "
    "targets, 200 case dirs, 39 patients; split_manifest.json is split_level=patient, seed 42, "
    "ratios 0.7/0.15/0.15 - the only patient-level split in the repo. BraTS carries NO pituitary "
    "and NO no-tumour label, so it cannot supply the 4-class labels; no 3D->2D conversion exists "
    "anywhere in the classification path. Paper WT/TC/ET 0.9005/0.5803/0.6010 belong to the "
    "standalone 3-D U-Net at checkpoints/segmentation/unet3d (finding D3), not to TAMNet's aux "
    "branch. Exact BraTS task/subset + slice rule remain [VERIFY]. "
    "See results/fill_evidence.json"
)


def main() -> int:
    with CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fieldnames = list(rows[0].keys())

    touched = []
    for r in rows:
        key = (r["dataset"], r["split"])
        if key == ("internal_classification", "train"):
            r["notes"] = FILL1_NOTES
            touched.append("internal/train notes")
        elif key == ("internal_classification", "test"):
            r["notes"] = (
                "used once per final config (audit rule); same unresolved label origin as train "
                "([CITATION NEEDED], FILL#1)"
            )
            touched.append("internal/test notes")
        elif key == ("kaggle_brain_tumor", "testing"):
            r["source"] = FILL2_SOURCE
            r["citation"] = FILL2_CITATION
            r["notes"] = FILL2_TEST_NOTES
            touched.append("kaggle/testing source+citation+notes")
        elif key == ("kaggle_brain_tumor", "training"):
            r["source"] = FILL2_SOURCE
            r["citation"] = FILL2_CITATION
            r["notes"] = FILL2_TRAIN_NOTES
            touched.append("kaggle/training source+citation+notes")
        elif key == ("BraTS2024", r["split"]) and r["dataset"] == "BraTS2024":
            r["source"] = FILL3_SOURCE
            r["notes"] = FILL3_NOTES
            touched.append("BraTS source+notes")

    tmp = CSV.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(CSV)
    print(f"updated {CSV}")
    for t in touched:
        print("  -", t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
