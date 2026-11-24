"""External-validation driver: the defensible replacement for the paper's 99.16 / 97.70.

Why this exists
---------------
`results/phase3/external_dedup_manifest.json` shows that 67.8 % of the Kaggle
*Testing* images and 56.9 % of the BRISC *test* images are near-duplicates
(pHash<=6 / dHash<=4 union) of images in the clean train split. The legacy paper
numbers 99.16 % (Kaggle) and 97.70 % (BRISC) were produced without that check,
so they are mostly in-distribution.

This script evaluates a finished run on each external set twice:
  * ``full``   - every image (comparable to the legacy number, reproduces the leak)
  * ``dedup``  - only images that are NOT near-duplicates of clean train (Phase 3)

Reported value = mean over the supplied seeds, sample SD (ddof=1). n=1 seeds get
no SD. The ``dedup`` numbers are the ones the paper may claim as external
generalisation; the ``full`` numbers are reported next to them only to quantify
how much of the legacy result was leakage.

Usage:
    python -X utf8 scripts/phase3_external_eval.py --dry-run
    python -X utf8 scripts/phase3_external_eval.py
    python -X utf8 scripts/phase3_external_eval.py --runs tamnet_clean_s42
    python -X utf8 scripts/phase3_external_eval.py --datasets kaggle_testing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "phase3" / "runs"
SUMMARY = ROOT / "results" / "phase3" / "external_eval_summary.json"

DEFAULT_DATASETS = ("kaggle_testing", "brisc_test")
HEADLINE_PREFIX = "tamnet_clean_s"


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def metrics_path(run_dir: Path, dataset: str, dedup: bool) -> Path:
    suffix = "_dedup" if dedup else ""
    return run_dir / f"eval_{dataset}_external{suffix}" / "metrics.json"


def default_runs() -> list[Path]:
    """The headline configuration's seeds (tamnet_clean_s42 ... s46)."""
    out = []
    for d in sorted(RUNS.iterdir()) if RUNS.is_dir() else []:
        if not d.is_dir() or not d.name.startswith(HEADLINE_PREFIX):
            continue
        # s42+mc / s43-tonly etc. are ablations, not the headline config
        if "+" in d.name or "-" in d.name or d.name.endswith("budget"):
            continue
        if (d / "training_result.json").is_file():
            out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run directory names (default: headline seeds s42..s46)")
    ap.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if a metrics.json already exists")
    args = ap.parse_args()

    if args.runs:
        run_dirs = [RUNS / r for r in args.runs]
    else:
        run_dirs = default_runs()
    missing = [d for d in run_dirs if not (d / "best.pt").is_file()]
    if missing:
        for d in missing:
            print(f"MISSING best.pt: {d.name}")
        return 2

    jobs = []
    for d in run_dirs:
        for dataset in args.datasets:
            for dedup in (True, False):
                dest = metrics_path(d, dataset, dedup)
                if dest.is_file() and not args.force:
                    continue
                jobs.append((d, dataset, dedup))

    print(f"external_eval: {len(run_dirs)} runs x {len(args.datasets)} datasets "
          f"x 2 variants -> {len(jobs)} evaluations to run")
    for d, dataset, dedup in jobs:
        print(f"  {d.name:<28} {dataset:<16} {'dedup' if dedup else 'full'}")
    if args.dry_run or not jobs:
        if jobs:
            print("\n(dry run - nothing executed)")
        else:
            print("nothing to do; aggregate only")
    if args.dry_run:
        return 0

    records = []
    for i, (d, dataset, dedup) in enumerate(jobs, 1):
        cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "phase3_evaluate.py"),
               "--run-dir", str(d), "--dataset", dataset,
               "--batch-size", str(args.batch_size),
               "--num-workers", str(args.num_workers)]
        if dedup:
            cmd.append("--dedup")
        t0 = time.time()
        print(f"[{i}/{len(jobs)}] {d.name} {dataset} {'dedup' if dedup else 'full'} ...",
              flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        mp = metrics_path(d, dataset, dedup)
        m = read_json(mp) or {}
        ok = proc.returncode == 0 and bool(m)
        print(f"  {'ok' if ok else 'FAILED'}  n={m.get('n')} acc={m.get('accuracy')} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if not ok:
            print((proc.stderr or "")[-600:], flush=True)
        records.append({
            "run": d.name, "dataset": dataset, "dedup": dedup, "ok": ok,
            "n": m.get("n"), "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"), "weighted_f1": m.get("weighted_f1"),
            "seconds": round(time.time() - t0, 1),
        })

    # ---- aggregate over all on-disk evaluations (not only this run's jobs) ----
    agg: dict[str, dict] = {}
    failed = [r for r in records if not r["ok"]]
    for dataset in args.datasets:
        for dedup in (True, False):
            key = f"{dataset}{'_dedup' if dedup else '_full'}"
            accs, f1s, ns = [], [], []
            for d in run_dirs:
                m = read_json(metrics_path(d, dataset, dedup))
                if not m or m.get("accuracy") is None:
                    continue
                accs.append(float(m["accuracy"]))
                f1s.append(float(m["macro_f1"]))
                ns.append(int(m["n"]))
            if not accs:
                continue
            a = np.asarray(accs)
            agg[key] = {
                "datasets": dataset,
                "dedup": dedup,
                "n_seeds": len(a),
                "n_images_per_seed": ns[0] if len(set(ns)) == 1 else ns,
                "accuracy_mean": float(a.mean()),
                "accuracy_std": float(a.std(ddof=1)) if len(a) > 1 else None,
                "macro_f1_mean": float(np.mean(f1s)),
                "macro_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else None,
                "per_seed": {d.name: read_json(metrics_path(d, dataset, dedup))["accuracy"]
                             for d in run_dirs
                             if read_json(metrics_path(d, dataset, dedup))},
            }

    previous = read_json(SUMMARY) or {}
    payload = {
        "kind": "phase3_external_eval",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "headline_rule": "mean over seeds; sample SD (ddof=1); n=1 -> no SD",
        "note": "'dedup' = leakage-free images only (see external_dedup_manifest.json); "
                "'full' = every image, comparable to the legacy paper numbers",
        "runs": [d.name for d in run_dirs],
        "aggregate": agg,
        "this_batch": records,
        "history": (previous.get("history") or [])[-20:],
    }
    payload["history"].append({
        "created_at": payload["created_at"],
        "n_jobs": len(jobs),
        "n_ok": len(records) - len(failed),
    })
    SUMMARY.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for key, a in agg.items():
        sd = "" if a["accuracy_std"] is None else f" ± {a['accuracy_std']:.4f}"
        print(f"{key:<26} acc {a['accuracy_mean']:.4f}{sd} "
              f"(seeds={a['n_seeds']}, n={a['n_images_per_seed']})")
    print(f"summary -> {SUMMARY.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
