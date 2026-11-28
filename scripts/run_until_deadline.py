"""Wall-clock deadline supervisor for the item-G retrain queue.

Runs the same idempotent stages as ``scripts/phase3_retrain_iaug.py`` but stops
all compute at ``--deadline`` (default 23:00 local time) instead of running to
completion, so the machine can be shut down on schedule:

    for each seed 42..46:  train (30 epochs, GPU-serial) -> TEST eval (once)
    then:                  multi-seed aggregate -> 27x3 corruption MRRS

Every stage is resumable, so re-running the supervisor (or
``run_until_11pm.bat``) continues exactly where it stopped:

* training auto-resumes from ``checkpoint_last.pt`` (written every epoch)
* evaluation is skipped when ``eval_internal_test/metrics.json`` exists
* aggregate / robustness are skipped when their JSON is newer than the metrics

Launch:
    run_until_11pm.bat                 (deadline 23:00, editable in the bat)
    python -X utf8 scripts/run_until_deadline.py --deadline 23:00 --seeds 42 43 44 45 46

Exit codes: 0 = all stages finished or stopped at the deadline (resumable),
1 = a stage failed (nothing is deleted; fix and re-run).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "phase3" / "runs"
SUMMARY = ROOT / "results" / "phase3" / "intensity_aug_retrain_summary.json"
ROBUSTNESS = ROOT / "results" / "phase3" / "robustness_iaug.json"
GROUP_LABEL = "tamnet_clean+iaugpn"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
POLL_SECONDS = 5


def run_name(seed: int) -> str:
    return f"tamnet_clean_s{seed}+iaug+pnorm"


def run_dir(seed: int) -> Path:
    return RUNS / run_name(seed)


def metrics_path(seed: int) -> Path:
    return run_dir(seed) / "eval_internal_test" / "metrics.json"


# --------------------------------------------------------------------------- #
# step definitions (same commands as scripts/phase3_retrain_iaug.py)
# --------------------------------------------------------------------------- #
class Step:
    def __init__(self, label: str, cmd: list[str], done) -> None:
        self.label = label
        self.cmd = cmd
        self.done = done


def train_cmd(seed: int) -> list[str]:
    return [
        sys.executable, "-X", "utf8", "scripts/phase3_train.py",
        "--model", "tamnet",
        "--seed", str(seed),
        "--protocol", "clean",
        "--epochs", "30",
        "--batch-size", "8",
        "--num-workers", "0",
        "--intensity-aug",
        "--per-image-norm",
    ]


def eval_cmd(seed: int) -> list[str]:
    return [
        sys.executable, "-X", "utf8", "scripts/phase3_evaluate.py",
        "--run-dir", str(run_dir(seed).relative_to(ROOT)),
        "--split", "test",
        "--dataset", "internal",
        "--batch-size", "16",
        "--num-workers", "0",
    ]


def aggregate_cmd(seeds: tuple[int, ...]) -> list[str]:
    return [
        sys.executable, "-X", "utf8", "scripts/phase3_aggregate.py",
        "--runs", *[str(run_dir(s).relative_to(ROOT)) for s in seeds],
        "--eval-dir", "eval_internal_test",
        "--out", str(SUMMARY.relative_to(ROOT)),
    ]


def robustness_cmd(seeds: tuple[int, ...]) -> list[str]:
    return [
        sys.executable, "-X", "utf8", "scripts/phase3_robustness.py",
        "--runs", *[run_name(s) for s in seeds],
        "--intensity-runs",
        "--out", str(ROBUSTNESS),
        "--group-label", GROUP_LABEL,
    ]


def newest_metrics_mtime(seeds: tuple[int, ...]) -> float:
    stamps = [metrics_path(s).stat().st_mtime for s in seeds if metrics_path(s).exists()]
    return max(stamps) if stamps else 0.0


def json_fresh(path: Path, seeds: tuple[int, ...]) -> bool:
    if not path.exists():
        return False
    newest = newest_metrics_mtime(seeds)
    return newest > 0 and path.stat().st_mtime >= newest


def build_steps(seeds: tuple[int, ...]) -> list[Step]:
    steps: list[Step] = []
    for seed in seeds:
        steps.append(Step(
            f"train s{seed}",
            train_cmd(seed),
            (lambda s=seed: (run_dir(s) / "training_result.json").exists()),
        ))
        steps.append(Step(
            f"eval  s{seed}",
            eval_cmd(seed),
            (lambda s=seed: metrics_path(s).exists()),
        ))
    all_eval_done = lambda: all(metrics_path(s).exists() for s in seeds)  # noqa: E731
    steps.append(Step(
        "aggregate",
        aggregate_cmd(seeds),
        lambda: all_eval_done() and json_fresh(SUMMARY, seeds),
    ))
    steps.append(Step(
        "robustness",
        robustness_cmd(seeds),
        lambda: all_eval_done() and json_fresh(ROBUSTNESS, seeds),
    ))
    return steps


# --------------------------------------------------------------------------- #
def resolve_deadline(spec: str) -> datetime:
    hour, minute = (int(x) for x in spec.split(":"))
    now = datetime.now()
    deadline = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)  # already past today -> next occurrence
    return deadline


def terminate_tree(pid: int) -> None:
    """Kill a child and anything it spawned (the trainer), never deleting work."""
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True, text=True,
    )


def current_epoch(seed: int) -> str:
    progress = run_dir(seed) / "progress.json"
    try:
        d = json.loads(progress.read_text(encoding="utf-8"))
        return f"epoch {d.get('epoch')}  train_loss {d.get('train_loss'):.4f}"
    except Exception:  # noqa: BLE001 - progress is cosmetic
        return "starting"


def main() -> int:
    ap = argparse.ArgumentParser(description="item-G queue with a wall-clock deadline")
    ap.add_argument("--deadline", default="23:00", help="local HH:MM at which all compute stops")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    args = ap.parse_args()

    seeds = tuple(args.seeds)
    try:
        deadline = resolve_deadline(args.deadline)
    except ValueError:
        print(f"invalid --deadline {args.deadline!r}: expected HH:MM in 00:00..23:59", flush=True)
        return 2
    remaining = (deadline - datetime.now()).total_seconds() / 3600
    print(f"deadline : {deadline:%Y-%m-%d %H:%M} local ({remaining:.1f} h from now)", flush=True)
    print(f"seeds    : {', '.join(map(str, seeds))} | GPU-serial, 30 epochs, batch 8", flush=True)
    print(f"resumable: training -> checkpoint_last.pt; eval/aggregate -> skip-if-present", flush=True)

    steps = build_steps(seeds)
    last_report = 0.0
    current_step = ""
    current_seed: int | None = None

    for step in steps:
        if step.done():
            print(f"skip  {step.label} (artifact present)", flush=True)
            continue
        if datetime.now() >= deadline:
            print(f"\nSTOP  deadline reached before {step.label} - nothing running", flush=True)
            print("      re-run run_until_11pm.bat to continue", flush=True)
            return 0

        print(f"\nSTART {step.label} at {datetime.now():%H:%M:%S}", flush=True)
        proc = subprocess.Popen(step.cmd, cwd=ROOT)
        current_step = step.label
        current_seed = None
        if step.label.startswith("train s"):
            current_seed = int(step.label.split("s")[-1])

        while proc.poll() is None:
            now_dt = datetime.now()
            if now_dt >= deadline:
                print(f"\nSTOP  deadline reached inside {step.label}; terminating (resumable)", flush=True)
                terminate_tree(proc.pid)
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                print("      re-run run_until_11pm.bat to resume", flush=True)
                return 0
            if time.time() - last_report > 60:
                last_report = time.time()
                left = (deadline - now_dt).total_seconds() / 3600
                detail = f" | {current_epoch(current_seed)}" if current_seed is not None else ""
                print(f"  ...   {current_step}{detail} | {left:.2f} h left", flush=True)
            time.sleep(POLL_SECONDS)

        if proc.returncode != 0:
            print(f"FAIL  {step.label} exited {proc.returncode}; fix and re-run", flush=True)
            return 1
        print(f"done  {step.label} at {datetime.now():%H:%M:%S}", flush=True)

    print(f"\nALL DONE at {datetime.now():%H:%M:%S}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
