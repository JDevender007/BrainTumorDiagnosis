"""Ingest the standalone 3-D U-Net's WT/TC/ET record into ``results/``.

Finding D3 attributes the paper's WT/TC/ET Dice (0.9005 / 0.5803 / 0.6010) to a
*standalone 3-D U-Net* trained on BraTS volumes, not to TAMNet's auxiliary head.
That claim was carried in the paper without a per-case record under ``results/``
— the report lived in ``outputs/metrics/`` where no downstream script looked.

This script is the bridge and the provenance gate:

  1. read ``outputs/metrics/segmentation_evaluation_test.json``;
  2. require 45 test cases, WT/TC/ET keys on every case, and aggregates that
     reproduce from the per-case values (fail closed otherwise);
  3. require the BraTS split manifest to be the patient-level one (no case in
     two splits) so the eval is patient-held-out;
  4. write ``results/brats_unet3d_wt_tc_et.json`` with sha256 of the source
     report, the manifest, the checkpoint, the script and the git commit.

Usage:
    python -X utf8 scripts/phase5_ingest_unet3d.py
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "metrics" / "segmentation_evaluation_test.json"
MANIFEST = ROOT / "data" / "splits" / "brats" / "split_manifest.json"
CKPT = (ROOT / "checkpoints" / "segmentation" / "unet3d" / "best"
        / "unet3d_segmentation.pt")
OUT = ROOT / "results" / "brats_unet3d_wt_tc_et.json"

REGIONS = ("WT", "TC", "ET")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    for p in (SRC, MANIFEST, CKPT):
        if not p.is_file():
            return fail(f"missing input {p.relative_to(ROOT)}")

    report = json.loads(SRC.read_text(encoding="utf-8-sig"))
    per_case = report.get("per_case") or []
    agg = report.get("aggregate") or {}

    if len(per_case) != 45:
        return fail(f"expected 45 test cases, got {len(per_case)}")

    # every case must carry all three classic BraTS regions
    for c in per_case:
        rd = c.get("region_dice") or {}
        missing = [r for r in REGIONS if not isinstance(rd.get(r), (int, float))]
        if missing:
            return fail(f"{c.get('case_id')}: missing region dice {missing}")

    # aggregates must reproduce from the per-case values (ddof=0, as stored)
    for r in REGIONS:
        vals = [c["region_dice"][r] for c in per_case]
        mean = statistics.fmean(vals)
        sd = statistics.pstdev(vals)
        if abs(mean - agg[f"{r}_dice_mean"]) > 1e-9:
            return fail(f"{r} mean {mean} != stored {agg[f'{r}_dice_mean']}")
        if abs(sd - agg[f"{r}_dice_std"]) > 1e-9:
            return fail(f"{r} std {sd} != stored {agg[f'{r}_dice_std']}")

    # patient-level split: no case or patient in two splits, 45 cases in test
    man = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    if man.get("split_level") != "patient":
        return fail(f"split_level is {man.get('split_level')!r}, not 'patient'")
    ids: dict[str, str] = {}
    patients: dict[str, str] = {}
    for name, cases in (man.get("splits") or {}).items():
        for entry in cases or []:
            cid = entry.get("case_id") if isinstance(entry, dict) else entry
            pid = (entry.get("patient_id") if isinstance(entry, dict) else entry)
            if cid in ids and ids[cid] != name:
                return fail(f"{cid} appears in both {ids[cid]} and {name}")
            ids[cid] = name
            if pid in patients and patients[pid] != name:
                return fail(f"patient {pid} crosses splits "
                            f"({patients[pid]} vs {name})")
            patients[pid] = name
    test_ids = {c["case_id"] for c in per_case}
    held = {cid for cid, name in ids.items() if name == "test"}
    if test_ids != held:
        return fail(f"report cases != manifest test split "
                    f"(only {len(test_ids & held)} of {len(held)} match)")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                capture_output=True, text=True,
                                encoding="utf-8", errors="replace"
                                ).stdout.strip()
    except OSError:
        commit = "unknown"

    out = {
        "kind": "brats_unet3d_wt_tc_et",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "standalone 3-D U-Net (configs/segmentation.yaml, 5 channels, "
                 "base_channels 16, patch 96^3, sliding-window inference)",
        "attribution": "Finding D3: these are NOT TAMNet's auxiliary head. They "
                       "are the numbers the legacy manuscript attributed to TAMNet.",
        "dataset": "BraTS 2024, patient-held-out test split "
                   "(129 train / 26 validation / 45 test cases)",
        "n_cases": len(per_case),
        "aggregate": {r: {"mean": agg[f"{r}_dice_mean"],
                           "std_ddof0": agg[f"{r}_dice_std"]} for r in REGIONS},
        "mean_wt_tc_et_dice": agg.get("mean_wt_tc_et_dice"),
        "per_case": [{"case_id": c["case_id"],
                      **{r: c["region_dice"][r] for r in REGIONS}}
                     for c in per_case],
        "provenance": {
            "source_report": str(SRC.relative_to(ROOT)),
            "source_sha256": sha256(SRC),
            "split_manifest": str(MANIFEST.relative_to(ROOT)),
            "split_manifest_sha256": sha256(MANIFEST),
            "split_level": man.get("split_level"),
            "checkpoint": str(CKPT.relative_to(ROOT)),
            "checkpoint_sha256": sha256(CKPT),
            "ingest_script": "scripts/phase5_ingest_unet3d.py",
            "ingest_script_sha256": sha256(Path(__file__)),
            "git_commit": commit,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(per_case)} cases)")
    for r in REGIONS:
        a = out["aggregate"][r]
        print(f"  {r}: {a['mean']:.4f} ± {a['std_ddof0']:.4f} (ddof=0, "
              f"ddof=1 = {statistics.stdev([c[r] for c in out['per_case']]):.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
