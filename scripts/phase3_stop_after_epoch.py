"""Graceful scheduled stop for Phase 3 training (midnight cutoff).

Why this exists
---------------
Phase 3 runs on a laptop and must be stopped overnight without losing work.
The trainer writes ``checkpoint_last.pt`` *before* it rewrites
``progress.json`` (see training/multitask_trainer.py), so as soon as an epoch
number appears in ``progress.json`` the checkpoint for that epoch is already
fully on disk. Waiting for that event therefore gives a loss-free stop:
at most the currently-running epoch is discarded.

What it does
------------
1. Snapshot every run whose ``progress.json`` says ``status == running``.
2. Poll until each of those runs has advanced one epoch (or finished, or
   died), with a timeout.
3. Terminate the queue drivers and their trainer children.
4. Verify each run still has a readable ``checkpoint_last.pt`` + ``progress.json``.
5. Write ``results/phase3/shutdown_<stamp>.json`` with the per-run state and
   the exact resume command for tomorrow.

Nothing is deleted, nothing is overwritten except the new shutdown record.

Usage
-----
    python -X utf8 scripts/phase3_stop_after_epoch.py
    python -X utf8 scripts/phase3_stop_after_epoch.py --dry-run
    python -X utf8 scripts/phase3_stop_after_epoch.py --timeout-min 25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

RUNS_DIR = Path("results/phase3/runs")
PHASE3_DIR = Path("results/phase3")

QUEUES = [
    ("queue_stage2g.json", "queue_status_2g.json"),
    ("queue_stage2h.json", "queue_status_2h.json"),
    ("queue_stage2i.json", "queue_status_2i.json"),
    ("queue_stage3a.json", "queue_status_3a.json"),
    ("queue_stage3b.json", "queue_status_3b.json"),
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def live_run_names() -> set[str]:
    """Basenames of the run directories that a live trainer is writing to."""
    names: set[str] = set()
    for p in python_procs("phase3_train"):
        try:
            cmdline = p.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        for i, tok in enumerate(cmdline):
            if tok == "--out" and i + 1 < len(cmdline):
                names.add(Path(cmdline[i + 1].replace("\\", "/")).name)
            elif tok.startswith("--out="):
                names.add(Path(tok.split("=", 1)[1].replace("\\", "/")).name)
    return names


def active_runs() -> dict[str, dict]:
    """Runs that are genuinely training right now.

    ``progress.json`` alone is not enough: abandoned experiments from earlier
    today (``*_partial``) still say ``status == running`` but have no live
    process. A run counts only if a ``phase3_train.py --out <that dir>``
    process is actually alive.
    """
    live = live_run_names()
    out: dict[str, dict] = {}
    if not RUNS_DIR.is_dir():
        return out
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if d.name not in live:
            continue
        if (d / "training_result.json").is_file():
            continue
        prog = read_json(d / "progress.json")
        if not prog or prog.get("status") != "running":
            continue
        out[d.name] = {
            "dir": str(d),
            "epoch": prog.get("epoch"),
            "epochs_total": prog.get("epochs_total"),
            "best_val_f1": prog.get("best_val_f1"),
            "best_epoch": prog.get("best_epoch"),
        }
    return out


def python_procs(needle: str) -> list[psutil.Process]:
    found = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info["name"] or "").lower()
            if name not in ("python.exe", "python3.exe", "pythonw.exe"):
                continue
            cmdline = " ".join(p.info["cmdline"] or [])
            if needle in cmdline:
                found.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def kill_tree(p: psutil.Process) -> int:
    """Kill a process and its children (trainers first, driver last)."""
    killed = 0
    try:
        children = p.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in children:
        try:
            child.kill()
            killed += 1
        except psutil.Error:
            pass
    try:
        p.kill()
        killed += 1
    except psutil.Error:
        pass
    return killed


def epoch_advanced(name: str, snap: dict) -> bool:
    d = Path(snap["dir"])
    if (d / "training_result.json").is_file():
        return True
    prog = read_json(d / "progress.json")
    if not prog:
        return False
    if prog.get("status") in ("completed", "early_stopped"):
        return True
    cur = prog.get("epoch")
    old = snap.get("epoch")
    return isinstance(cur, int) and isinstance(old, int) and cur > old


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout-min", type=float, default=25.0,
                    help="give up waiting for a clean epoch boundary after N minutes")
    ap.add_argument("--poll-s", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen, change nothing")
    args = ap.parse_args()

    snap = active_runs()
    if not snap:
        print("STOP: no active runs (nothing marked 'running' without a result).", flush=True)
        return 0

    print(f"STOP: waiting for a clean epoch boundary on {len(snap)} run(s):", flush=True)
    for name, s in snap.items():
        print(f"  {name}: at epoch {s['epoch']}/{s['epochs_total']} "
              f"(best {s['best_val_f1']}@{s['best_epoch']})", flush=True)

    if args.dry_run:
        print("STOP: --dry-run, would wait for the next epoch boundary then stop. "
              "Nothing changed.", flush=True)
        return 0

    pending = set(snap)
    deadline = time.time() + args.timeout_min * 60
    while pending:
        for name in list(pending):
            if epoch_advanced(name, snap[name]):
                pending.discard(name)
                print(f"  clean boundary reached: {name}", flush=True)
        if not pending:
            break
        if not python_procs("phase3_train"):
            print("  no trainer alive any more - stopping early", flush=True)
            break
        if time.time() > deadline:
            print(f"  timed out after {args.timeout_min} min - stopping anyway", flush=True)
            break
        time.sleep(args.poll_s)

    drivers = python_procs("phase3_queue")
    trainers = python_procs("phase3_train")
    print(f"STOP: terminating {len(drivers)} queue driver(s) "
          f"and {len(trainers)} trainer(s)", flush=True)

    killed = 0
    for p in trainers:
        try:
            p.kill()
            killed += 1
        except psutil.Error:
            pass
    for p in drivers:
        killed += kill_tree(p)

    time.sleep(3)

    records = []
    ok = True
    for name, s in snap.items():
        d = Path(s["dir"])
        ckpt = d / "checkpoint_last.pt"
        prog = read_json(d / "progress.json")
        result = read_json(d / "training_result.json")
        entry = {
            "run": name,
            "snapshot_epoch": s["epoch"],
            "checkpoint_exists": ckpt.is_file(),
            "checkpoint_bytes": ckpt.stat().st_size if ckpt.is_file() else 0,
            "progress_epoch": (prog or {}).get("epoch"),
            "progress_status": (prog or {}).get("status"),
            "best_val_f1": (prog or {}).get("best_val_f1"),
            "best_epoch": (prog or {}).get("best_epoch"),
            "finished": result is not None,
        }
        if not entry["checkpoint_exists"]:
            ok = False
        records.append(entry)
        print(f"  {name}: checkpoint={entry['checkpoint_exists']} "
              f"({entry['checkpoint_bytes'] / 1e6:.1f} MB) "
              f"epoch={entry['progress_epoch']}/{s['epochs_total']} "
              f"best={entry['best_val_f1']}@{entry['best_epoch']}", flush=True)

    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    resume_cmds = [
        f"python -X utf8 scripts\\phase3_queue.py --queue results\\phase3\\{q} "
        f"--status-file results\\phase3\\{st}"
        for q, st in QUEUES
    ]
    record = {
        "kind": "phase3_scheduled_shutdown",
        "created_at": utc_now().isoformat(timespec="seconds"),
        "reason": "user-scheduled midnight cutoff",
        "killed_processes": killed,
        "all_checkpoints_ok": ok,
        "runs": records,
        "resume": {
            "note": "relaunch with the same queue files; completed runs are skipped "
                    "via training_result.json and in-flight runs resume from "
                    "checkpoint_last.pt at the exact epoch",
            "commands": resume_cmds,
        },
    }
    out = PHASE3_DIR / f"shutdown_{stamp}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"STOP: wrote {out}", flush=True)
    print(f"STOP: all checkpoints ok = {ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
