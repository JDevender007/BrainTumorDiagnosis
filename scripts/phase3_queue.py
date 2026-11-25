"""Sequential queue runner for Phase 3 training runs.

Reads a queue JSON (list of {"name", "args"} entries), runs
``scripts/phase3_train.py`` for each one, and skips entries whose run
directory already contains ``training_result.json`` (crash-safe resume).

Progress reporting (user protocol 2026-11-19):
  - every run writes its own ``progress.json`` (per epoch) via the trainer;
  - this driver maintains ``results/phase3/queue_status.json`` with the
    currently-running name, completed list and failures, so an external
    watcher can report status every few minutes;
  - each run's stdout is teed to ``results/phase3/logs/<name>.log``.

Fail-stop: a non-zero exit records the failure and stops the queue (no silent
skips). Usage:

    python scripts/phase3_queue.py --queue results/phase3/queue_stage1.json
    python scripts/phase3_queue.py --queue ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "results" / "phase3" / "runs"
LOGS_DIR = ROOT / "results" / "phase3" / "logs"
DEFAULT_STATUS_PATH = ROOT / "results" / "phase3" / "queue_status.json"

# Windows only: spawn every trainer with NO console of its own. The
# 2026-11-25 logon task killed all 3 trainers ~60 s after start with
# "forrtl: error (200): program aborting due to window-CLOSE event" -
# the Task Scheduler console was closed while the trainers were still
# attached to it. stdout/stderr are redirected to the run log anyway,
# so a console is never needed (and DETACHED_PROCESS also shields the
# trainers from Ctrl+C meant for the driver).
NO_CONSOLE = (
    subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    if sys.platform == "win32"
    else 0
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(status_path: Path, payload: dict) -> None:
    payload["updated_at"] = now()
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(status_path)


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 3 sequential queue runner")
    p.add_argument("--queue", required=True, help="queue JSON path")
    p.add_argument(
        "--status-file",
        default=None,
        help="status JSON path (default results/phase3/queue_status.json); "
        "set per queue when running multiple queues in parallel",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="optional wall-clock stop guard for this driver session",
    )
    args = p.parse_args()

    status_path = DEFAULT_STATUS_PATH
    if args.status_file:
        status_path = Path(args.status_file)
        if not status_path.is_absolute():
            status_path = ROOT / status_path

    queue_path = Path(args.queue)
    if not queue_path.is_absolute():
        queue_path = ROOT / queue_path
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    status: dict = {
        "queue": str(queue_path.relative_to(ROOT)),
        "started_at": now(),
        "current": None,
        "completed": [],
        "failed": None,
        "total_entries": len(queue),
    }
    # resume bookkeeping: count pre-existing completed runs
    for entry in queue:
        if (RUNS_DIR / entry["name"] / "training_result.json").exists():
            status["completed"].append(entry["name"])

    if args.dry_run:
        for entry in queue:
            state = (
                "DONE"
                if entry["name"] in status["completed"]
                else "PENDING"
            )
            print(f"  [{state:7s}] {entry['name']}: {' '.join(entry['args'])}")
        return 0

    write_status(status_path, status)
    t0 = time.time()
    for entry in queue:
        name = entry["name"]
        if name in status["completed"]:
            print(f"SKIP (already done): {name}", flush=True)
            continue
        if args.max_hours is not None and (time.time() - t0) / 3600 > args.max_hours:
            print("max-hours guard reached - stopping queue", flush=True)
            break

        status["current"] = name
        status["current_started_at"] = now()
        write_status(status_path, status)
        print(f"START {name} @ {status['current_started_at']}", flush=True)

        log_path = LOGS_DIR / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"# {name} started {status['current_started_at']}\n")
            log.flush()
            rc = subprocess.run(
                [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "phase3_train.py"), *entry["args"]],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=NO_CONSOLE,
            ).returncode

        if rc != 0:
            status["failed"] = {"name": name, "exit_code": rc, "log": str(log_path.relative_to(ROOT))}
            status["current"] = None
            write_status(status_path, status)
            print(f"FAIL {name} (exit {rc}) - see {log_path}", flush=True)
            return rc

        status["completed"].append(name)
        status["current"] = None
        write_status(status_path, status)
        print(f"DONE {name} ({len(status['completed'])}/{status['total_entries']})", flush=True)

    status["current"] = None
    status["finished_at"] = now()
    write_status(status_path, status)
    print(f"QUEUE COMPLETE: {len(status['completed'])}/{status['total_entries']} runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
