"""Evaluate every completed Phase 3 run on the internal TEST split, once each.

Phase 3 runs are saved with ``training_result.json`` but the TEST split is only
touched in ``scripts/phase3_evaluate.py``. This driver walks the runs directory,
skips anything already evaluated (``eval_internal_test/metrics.json`` exists) and
evaluates the rest one at a time.

The TEST split must be used once per final configuration, so the default is to
refuse re-evaluating a run that already has metrics unless --force is passed.

Usage:
    python -X utf8 scripts/phase3_evaluate_all.py --dry-run
    python -X utf8 scripts/phase3_evaluate_all.py
    python -X utf8 scripts/phase3_evaluate_all.py --only tamnet_clean_s42 densenet121_clean_s42
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUNS = Path("results/phase3/runs")
SUMMARY = Path("results/phase3/eval_all_summary.json")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def candidates(only: list[str] | None, force: bool) -> list[Path]:
    out = []
    if not RUNS.is_dir():
        return out
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        if only and d.name not in only:
            continue
        if not (d / "training_result.json").is_file():
            continue
        has_eval = (d / "eval_internal_test" / "metrics.json").is_file()
        if has_eval and not force:
            continue
        out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-evaluate runs that already have TEST metrics (breaks "
                         "the 'test set used once' rule - only for a documented reason)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=1)
    args = ap.parse_args()

    todo = candidates(args.only, args.force)
    done_already = sum(
        1 for d in RUNS.iterdir()
        if d.is_dir() and (d / "eval_internal_test" / "metrics.json").is_file()
    )
    print(f"evaluate_all: {len(todo)} to evaluate, {done_already} already have TEST metrics")
    for d in todo:
        print(f"  {d.name}")
    if args.dry_run or not todo:
        return 0

    records = []
    for i, d in enumerate(todo, 1):
        cmd = [sys.executable, "-X", "utf8", "scripts/phase3_evaluate.py",
               "--run-dir", str(d), "--split", "test",
               "--batch-size", str(args.batch_size),
               "--num-workers", str(args.num_workers)]
        t0 = time.time()
        print(f"[{i}/{len(todo)}] {d.name} ...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        ok = proc.returncode == 0 and (d / "eval_internal_test" / "metrics.json").is_file()
        m = read_json(d / "eval_internal_test" / "metrics.json") or {}
        rec = {
            "run": d.name,
            "ok": ok,
            "returncode": proc.returncode,
            "seconds": round(time.time() - t0, 1),
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "weighted_f1": m.get("weighted_f1"),
        }
        if not ok:
            rec["stderr_tail"] = (proc.stderr or "")[-800:]
            print(f"  FAILED: {rec['stderr_tail'][-300:]}", flush=True)
        else:
            print(f"  ok  acc {rec['accuracy']:.4f}  macroF1 {rec['macro_f1']:.4f} "
                  f"({rec['seconds']}s)", flush=True)
        records.append(rec)

    payload = {
        "kind": "phase3_eval_all",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_evaluated": len(records),
        "n_ok": sum(1 for r in records if r["ok"]),
        "records": records,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"evaluate_all: {payload['n_ok']}/{len(records)} ok -> {SUMMARY}")
    return 0 if payload["n_ok"] == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
