"""Aggregate per-seed ``phase3_evaluate`` metrics into mean±std summaries.

The headline rule (docs/TODO.md Phase 3): **mean over seeds, never the best
seed**. This script reads each run's ``eval_<dataset>_<split>/metrics.json``
(written by ``scripts/phase3_evaluate.py``), computes mean / sample std
(ddof=1) / min / max for accuracy, macro-F1 and weighted-F1, and writes one
JSON with per-seed values, source paths and full provenance (git, command,
timestamp). Every paper number produced by this script therefore traces back
to a single evaluation artifact per seed.

Usage:
    python scripts/phase3_aggregate.py \
        --runs results/phase3/runs/tamnet_clean_s42 ... \
        --eval-dir eval_internal_test \
        --out results/phase3/stage1_test_summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METRICS = ("accuracy", "macro_f1", "weighted_f1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 multi-seed aggregator")
    p.add_argument("--runs", nargs="+", required=True, help="run directories")
    p.add_argument(
        "--eval-dir",
        default="eval_internal_test",
        help="evaluation sub-directory inside each run (default eval_internal_test)",
    )
    p.add_argument("--out", required=True, help="summary JSON to write")
    return p.parse_args()


def git_info() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT, check=True,
        ).stdout.strip()
        dirty = (
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=ROOT, check=True,
            ).stdout.strip()
            != ""
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:  # noqa: BLE001 - provenance must never crash the run
        return {"commit": None, "dirty": None}


def main() -> int:
    args = parse_args()

    per_seed: list[dict] = []
    for run in sorted(args.runs):
        run_dir = Path(run)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        metrics_path = run_dir / args.eval_dir / "metrics.json"
        if not metrics_path.exists():
            raise SystemExit(f"missing evaluation: {metrics_path}")
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        prov = m.get("provenance", {})
        per_seed.append(
            {
                "run": run_dir.name,
                "seed": prov.get("seed"),
                "checkpoint_best_epoch": m.get("checkpoint_best_epoch"),
                "checkpoint_val_f1": m.get("checkpoint_val_f1"),
                "n": m["n"],
                **{k: m[k] for k in METRICS},
                "source": metrics_path.relative_to(ROOT).as_posix(),
                "evaluated_at": prov.get("evaluated_at"),
                "checkpoint_sha256": prov.get("checkpoint_sha256"),
            }
        )

    if len(per_seed) < 2:
        raise SystemExit("need >= 2 seeds for mean±std")

    summary = {
        metric: {
            "mean": statistics.fmean(s[metric] for s in per_seed),
            # sample std (ddof=1) - stated explicitly so the paper can cite it
            "std_sample_ddof1": statistics.stdev(s[metric] for s in per_seed),
            "min": min(s[metric] for s in per_seed),
            "max": max(s[metric] for s in per_seed),
        }
        for metric in METRICS
    }

    out = {
        "kind": "phase3_multiseed_test_summary",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "command": " ".join(sys.argv),
        "eval_dir": args.eval_dir,
        "n_seeds": len(per_seed),
        "headline_rule": "mean over seeds (never best seed); std = sample ddof=1",
        "per_seed": per_seed,
        "summary": summary,
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"WROTE {out_path.relative_to(ROOT)}", flush=True)
    for metric in METRICS:
        s = summary[metric]
        print(
            f"  {metric:14s} {s['mean']:.4f} ± {s['std_sample_ddof1']:.4f} "
            f"(min {s['min']:.4f}, max {s['max']:.4f})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
