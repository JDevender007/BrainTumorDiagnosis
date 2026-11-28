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
    "Kaggle 'Brain Tumor MRI Dataset', M. Nickparvar - "
    "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset "
    "(Te-*/Tr-* naming, 4 classes, 7023 images); owner slug + URL verified "
    "against the live dataset page 2026-11-28"
)
FILL2_TEST_NOTES = (
    "CONTAMINATED: overlaps internal train (results/leakage_report.json). "
    "FILL#2 RESOLVED (partial): zip per-directory counts are Testing 300/306/405/300 = 1311 "
    "and Training 1321/1339/1595/1457 = 5712, i.e. exactly the canonical 7023-image release. "
    "Local *extracted* Testing/ folder matches the zip (1311); URL verified 2026-11-28. "
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
    "branch. Exact local release id + slice rule remain [VERIFY]. "
    "See results/fill_evidence.json"
)
FILL3_CITATION = (
    "de Verdier et al., 'The 2024 Brain Tumor Segmentation (BraTS) Challenge: Glioma "
    "Segmentation on Post-treatment MRI,' arXiv:2405.18368, 2024 - the {NETC, SNFH, ET, RC} "
    "schema in datasets/segmentation.py is that challenge's label set; the exact local release "
    "identifier is not recorded in the repo"
)


def main() -> int:
    with CSV.open(newline="", encoding="utf-8") as fh:
        raw = list(csv.reader(fh))
    header, body = raw[0], raw[1:]
    role_idx = header.index("role")

    # One legacy row was written with an unquoted comma inside `role`
    # ("... (headline, clean protocol)"), which shifted every later column.
    # Re-join it on read so the row round-trips cleanly.
    rows, repaired = [], 0
    for r in body:
        if len(r) > len(header):
            # an unquoted comma inside `role` shifted the later columns
            r = r[:role_idx] + [f"{r[role_idx]},{r[role_idx + 1]}"] + r[role_idx + 2:]
            repaired += 1
        if len(r) > len(header):
            print(f"WARNING: row with {len(r)} fields vs {len(header)} header "
                  f"fields ({r[0]}/{r[1]}) - extra fields dropped")
        rows.append(dict(zip(header, r)))

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
            # this row lost the tail of `role` to a bare comma in an earlier
            # run of this script; put the fields back where they belong
            r["role"] = ("BraTS 3-D segmentation only; WT/TC/ET Dice belong "
                         "to the standalone 3-D U-Net, not TAMNet")
            r["mask_available"] = "yes (seg/)"
            r["source"] = FILL3_SOURCE
            r["citation"] = FILL3_CITATION
            r["notes"] = FILL3_NOTES
            touched.append("BraTS role+mask+source+citation+notes")

    tmp = CSV.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(CSV)
    print(f"updated {CSV} ({repaired} row(s) re-quoted)")
    for t in touched:
        print("  -", t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
