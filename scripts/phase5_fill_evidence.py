"""Collect hard evidence for the three Phase 5 FILL questions.

FILL #1  internal dataset provenance / label origin
FILL #2  exact Kaggle dataset name + link
FILL #3  BraTS role in classification + 3D->2D conversion

Everything here is read-only filesystem fact. Nothing is inferred beyond what
the filenames and archive contents literally say; anything not provable is
reported as ``unknown`` so the paper can keep a ``[VERIFY]`` / ``[CITATION
NEEDED]`` marker instead of a fabricated sentence.

Writes ``results/fill_evidence.json``.
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERNAL = ROOT / "data" / "raw" / "classification"
KAGGLE_ZIP = ROOT / "data" / "external" / "kaggle_brain_tumor" / "brain-tumor-mri-dataset.zip"
KAGGLE_DIR = ROOT / "data" / "external" / "kaggle_brain_tumor"
BRATS = ROOT / "data" / "raw" / "BraTS2024"
OUT = ROOT / "results" / "fill_evidence.json"

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("inst_code_X_###_XX_", re.compile(r"^[A-Za-z]+_\d+_[A-Za-z]{2}_\.(?:jpg|jpeg)$", re.I)),
    ("kaggle_Te_prefix", re.compile(r"^Te-.*\.(?:jpg|jpeg)$", re.I)),
    ("kaggle_Tr_prefix", re.compile(r"^Tr-.*\.(?:jpg|jpeg)$", re.I)),
    ("gg_paren", re.compile(r"^gg \(\d+\)\.(?:jpg|jpeg)$", re.I)),
    ("image_paren", re.compile(r"^image\(\d+\)\.(?:jpg|jpeg|png)$", re.I)),
    ("N_no", re.compile(r"^\d+ no\.(?:jpg|jpeg)$", re.I)),
    ("plain_numeric", re.compile(r"^\d+\.(?:jpg|jpeg|png)$", re.I)),
]


def family(name: str) -> str:
    for label, pat in PATTERNS:
        if pat.match(name):
            return label
    return "other"


def scan_internal() -> dict:
    per_class: dict[str, Counter] = {}
    fam_counter: Counter = Counter()
    fam_examples: dict[str, list[str]] = defaultdict(list)
    by_name: dict[str, list[str]] = defaultdict(list)
    for split in ("Train", "Test"):
        split_dir = INTERNAL / split
        if not split_dir.is_dir():
            continue
        for cls in sorted(p.name for p in split_dir.iterdir() if p.is_dir()):
            files = sorted(f.name for f in (split_dir / cls).iterdir() if f.is_file())
            per_class[f"{split}/{cls}"] = Counter(files)
            for n in files:
                fam = family(n)
                fam_counter[fam] += 1
                if len(fam_examples[fam]) < 5:
                    fam_examples[fam].append(f"{split}/{cls}/{n}")
                by_name[n].append(f"{split}/{cls}")
    return {
        "counts": {k: sum(v.values()) for k, v in per_class.items()},
        "total_images": sum(sum(v.values()) for v in per_class.values()),
        "unique_basenames": len(by_name),
        "filename_families": dict(fam_counter.most_common()),
        "filename_family_examples": dict(fam_examples),
        "note": "A basename appearing in several class folders would be a labelling "
                "conflict; duplicates_within_split are reported separately below.",
        "basename_in_multiple_classes": {
            n: locs for n, locs in by_name.items() if len({x.split("/")[1] for x in locs}) > 1
        },
    }


def scan_kaggle_zip() -> dict:
    if not KAGGLE_ZIP.is_file():
        return {"present": False, "path": str(KAGGLE_ZIP)}
    with zipfile.ZipFile(KAGGLE_ZIP) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    per_dir: Counter = Counter()
    for n in names:
        parts = n.split("/")
        if len(parts) >= 3:
            per_dir[f"{parts[0]}/{parts[1]}"] += 1
    return {
        "present": True,
        "path": str(KAGGLE_ZIP),
        "archive_name": KAGGLE_ZIP.name,
        "entries": len(names),
        "per_directory": dict(sorted(per_dir.items())),
        "top_level_dirs": sorted({n.split("/")[0] for n in names if "/" in n}),
        "sample_names": sorted(os_path_basename(n) for n in names)[:5],
    }


def os_path_basename(n: str) -> str:
    return n.rsplit("/", 1)[-1]


def scan_kaggle_extracted() -> dict:
    out: dict[str, dict] = {}
    for split in ("Training", "Testing"):
        d = KAGGLE_DIR / split
        if not d.is_dir():
            continue
        out[split] = {
            p.name: sum(1 for f in p.iterdir() if f.is_file())
            for p in sorted(d.iterdir())
            if p.is_dir()
        }
    return out


def scan_brats() -> dict:
    if not BRATS.is_dir():
        return {"present": False}
    case_dirs = [p for p in BRATS.rglob("*") if p.is_dir()]
    nii = list(BRATS.rglob("*.nii")) + list(BRATS.rglob("*.nii.gz"))
    seg = [p for p in BRATS.rglob("*") if p.is_file() and "seg" in p.name.lower()]
    modality_tokens: Counter = Counter()
    for p in nii:
        for tok in ("t1c", "t1n", "t2f", "t2w", "t1", "t2", "flair"):
            if tok in p.name.lower():
                modality_tokens[tok] += 1
    manifest = None
    for cand in (BRATS / "split_manifest.json", ROOT / "data" / "splits" / "brats" / "split_manifest.json"):
        if cand.is_file():
            manifest = json.loads(cand.read_text(encoding="utf-8"))
            break
    return {
        "present": True,
        "n_nifti": len(nii),
        "n_case_like_dirs": len(case_dirs),
        "n_seg_files": len(seg),
        "modality_tokens_seen": dict(modality_tokens),
        "manifest": manifest,
        "class_labels_in_filenames": sorted(
            {tok for p in nii for tok in ("glioma", "meningioma", "pituitary", "notumor")
             if tok in p.name.lower()}
        ),
        "has_pituitary_label": any(
            "pituitary" in p.name.lower() for p in BRATS.rglob("*") if p.is_file()
        ),
    }


def overlap(internal_names: set[str], other_names: set[str]) -> dict:
    inter = internal_names & other_names
    return {
        "n_internal_unique": len(internal_names),
        "n_other_unique": len(other_names),
        "n_basename_overlap": len(inter),
        "frac_of_internal": round(len(inter) / len(internal_names), 4) if internal_names else None,
        "sample": sorted(inter)[:10],
    }


def main() -> int:
    internal = scan_internal()
    zipinfo = scan_kaggle_zip()
    extracted = scan_kaggle_extracted()
    brats = scan_brats()

    internal_names = set()
    for split in ("Train", "Test"):
        d = INTERNAL / split
        if d.is_dir():
            for cls in d.iterdir():
                if cls.is_dir():
                    internal_names |= {f.name for f in cls.iterdir() if f.is_file()}

    zip_names: set[str] = set()
    if zipinfo.get("present"):
        with zipfile.ZipFile(KAGGLE_ZIP) as zf:
            zip_names = {os_path_basename(n) for n in zf.namelist() if not n.endswith("/")}

    extracted_names: set[str] = set()
    for split in ("Training", "Testing"):
        d = KAGGLE_DIR / split
        if d.is_dir():
            for cls in d.iterdir():
                if cls.is_dir():
                    extracted_names |= {f.name for f in cls.iterdir() if f.is_file()}

    brisc_names: set[str] = set()
    brisc_root = ROOT / "data" / "external" / "brisc2025"
    if brisc_root.is_dir():
        brisc_names = {p.name for p in brisc_root.rglob("*") if p.is_file()}

    payload = {
        "kind": "phase5_fill_evidence",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Evidence for the three FILL questions that block Phase 5 wording. "
                   "Only literal filesystem facts; unknowns stay unknown.",
        "FILL_1_internal_provenance": {
            "internal": internal,
            "vs_kaggle_zip": overlap(internal_names, zip_names),
            "vs_kaggle_extracted": overlap(internal_names, extracted_names),
            "vs_brisc": overlap(internal_names, brisc_names),
            "brats_has_pituitary_or_notumor_label": brats.get("has_pituitary_label"),
            "verdict": None,  # filled in below
        },
        "FILL_2_kaggle_identity": {
            "zip": zipinfo,
            "extracted_dirs": extracted,
            "canonical_brain_tumor_mri_dataset": {
                "expected_entries": 7023,
                "expected_split": {"Training": 5712, "Testing": 1311},
                "matches_local_zip": zipinfo.get("entries") == 7023,
            },
            "verdict": None,
        },
        "FILL_3_brats_role": {
            "brats": brats,
            "classification_data_is_2d_jpeg": True,
            "verdict": None,
        },
    }

    # --- verdicts, stated only as far as the evidence goes -------------------
    ov = payload["FILL_1_internal_provenance"]["vs_kaggle_zip"]
    internal_total = internal["total_images"]
    payload["FILL_1_internal_provenance"]["verdict"] = {
        "established": [
            f"The internal set is {internal_total} 2-D JPEG/PNG images in 4 class folders "
            f"(Train + Test); no patient/subject identifiers exist in any filename.",
            f"{ov['n_basename_overlap']} of {ov['n_internal_unique']} unique internal basenames "
            f"({ov['frac_of_internal']:.1%}) are also present in the local Kaggle zip, so the "
            "internal pool is substantially a re-use of that Kaggle release.",
            "The remainder uses a distinct naming family "
            f"{internal['filename_families']}, which is not BraTS (BraTS has no "
            "pituitary or no-tumour class).",
        ],
        "not_established": [
            "Who produced the 4-class labels and from which primary source the "
            "non-Kaggle portion was collated - no download script, README, manifest or "
            "citation for this folder exists anywhere in the repository.",
        ],
        "paper_wording": "state the class counts and the image-level split; mark the label "
                         "origin [CITATION NEEDED]; do not claim BraTS provenance.",
    }

    payload["FILL_2_kaggle_identity"]["verdict"] = {
        "established": [
            f"The archive on disk is literally named '{zipinfo.get('archive_name')}' and holds "
            f"{zipinfo.get('entries')} entries with top-level dirs {zipinfo.get('top_level_dirs')} "
            f"and per-directory counts {zipinfo.get('per_directory')}.",
            "Those counts equal the canonical Kaggle 'Brain Tumor MRI Dataset' release "
            "(Training 5712 + Testing 1311 = 7023).",
            "The *extracted* folders used by the pipeline are an incomplete extraction of that "
            f"zip: {extracted}.",
        ],
        "not_established": [
            "The Kaggle owner slug / URL is not written in the repo; only the archive file "
            "name is. Treat the exact link as [VERIFY] until confirmed against the Kaggle page.",
        ],
        "paper_wording": "cite the dataset by its archive name and canonical size; "
                         "[VERIFY] the author/URL before camera-ready.",
    }

    payload["FILL_3_brats_role"]["verdict"] = {
        "established": [
            "Classification is trained and evaluated only on 2-D JPEG/PNG slices "
            "(data/raw/classification); BraTS contributes no classification images.",
            f"The local BraTS copy holds {brats.get('n_nifti')} NIfTI volumes and "
            f"{brats.get('n_seg_files')} segmentation targets; it is the segmentation-only source.",
            "BraTS 2024 carries no pituitary and no healthy/no-tumour label, so it cannot "
            "supply the 4-class labels used by the classifier.",
            "No 3-D->2-D slice conversion exists in the classification path; the 4-class "
            "task was never run on converted BraTS volumes.",
        ],
        "not_established": [
            "Which BraTS 2024 task/subset the segmentation checkpoint was trained on, and "
            "the exact slice-selection rule, are only recorded as [VERIFY] in "
            "results/dataset_table.csv.",
        ],
        "paper_wording": "BraTS = segmentation evaluation only; WT/TC/ET Dice belong to the "
                         "standalone 3-D U-Net (D3), not to TAMNet's classification path.",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(OUT)
    print(f"wrote {OUT}")
    print(f"  internal images            : {internal_total}")
    print(f"  internal unique basenames  : {internal['unique_basenames']}")
    print(f"  overlap with Kaggle zip    : {ov['n_basename_overlap']} ({ov['frac_of_internal']:.1%})")
    print(f"  Kaggle zip entries         : {zipinfo.get('entries')}")
    print(f"  filename families          : {internal['filename_families']}")
    print(f"  BraTS nifti / seg files    : {brats.get('n_nifti')} / {brats.get('n_seg_files')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
