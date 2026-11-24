"""Stream Phase 3 training progress until it finishes (read-only).

Prints the monitor's status block every time the underlying state changes, plus a
one-line heartbeat every few minutes so the caller can see it is alive. Never
touches the trainers, the GPU, or any results file - it only reads
``monitor_latest.txt`` / ``monitor_log.csv`` / the runs directory.

Usage:
    python -X utf8 scripts/phase3_watch.py --minutes 25
    python -X utf8 scripts/phase3_watch.py --until-done
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATEST = ROOT / "results" / "phase3" / "monitor_latest.txt"
LOG = ROOT / "results" / "phase3" / "monitor_log.csv"
RUNS = ROOT / "results" / "phase3" / "runs"


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8-sig")
    except OSError:
        return ""


def completed_runs() -> int:
    if not RUNS.is_dir():
        return 0
    return sum(1 for d in RUNS.iterdir() if d.is_dir() and (d / "training_result.json").is_file())


def live_runs() -> list[str]:
    """Runs that have a progress.json but no training_result.json yet."""
    out = []
    if not RUNS.is_dir():
        return out
    for d in RUNS.iterdir():
        if not d.is_dir():
            continue
        if (d / "progress.json").is_file() and not (d / "training_result.json").is_file():
            out.append(d.name)
    return sorted(out)


def eta_table() -> dict[str, dict]:
    """Latest epoch_s / epoch / ETA per run from the monitor CSV."""
    if not LOG.is_file():
        return {}
    latest: dict[str, dict] = {}
    try:
        with LOG.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                latest[row.get("run", "")] = row
    except OSError:
        return {}
    out = {}
    for run, row in latest.items():
        try:
            ep = int(row["epoch"])
            total = int(row.get("epochs_total") or 30)
            eps = float(row["epoch_s"]) if row.get("epoch_s") else 0.0
        except (KeyError, ValueError):
            continue
        left = max(0, total - ep)
        out[run] = {
            "epoch": ep,
            "total": total,
            "left": left,
            "epoch_s": eps,
            "eta_min": round(eps * left / 60.0, 1) if eps else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=25.0)
    ap.add_argument("--until-done", action="store_true",
                    help="run until no trainer is live and the queue is empty")
    ap.add_argument("--interval", type=float, default=15.0)
    args = ap.parse_args()

    t_end = time.time() + args.minutes * 60
    last = None
    last_beat = 0.0
    idle_checks = 0

    while True:
        now = time.time()
        body = read_text(LATEST).strip()
        if body != last:
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n===== {stamp} =====")
            print(body if body else "(no monitor output yet)")
            last = body
            last_beat = now
        elif now - last_beat > 180:
            stamp = datetime.now().strftime("%H:%M:%S")
            live = live_runs()
            print(f"[{stamp}] heartbeat · {len(live)} live · "
                  f"{completed_runs()} completed runs on disk")
            last_beat = now

        live = live_runs()
        if not live:
            idle_checks += 1
            if idle_checks >= 4:
                print("\nNo live trainers and no queued runs remain.")
                break
        else:
            idle_checks = 0

        if not args.until_done and now >= t_end:
            print(f"\n(window of {args.minutes:g} min elapsed)")
            break
        time.sleep(args.interval)

    table = eta_table()
    if table:
        print("\nlast known ETA per run (from monitor_log.csv):")
        for run, r in sorted(table.items()):
            eta = "done" if r["left"] == 0 else (
                f"{r['eta_min']:.0f} min" if r["eta_min"] is not None else "?")
            print(f"  {run:<28} ep{r['epoch']:>2}/{r['total']}  {r['epoch_s']:.0f}s/ep  ETA {eta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
