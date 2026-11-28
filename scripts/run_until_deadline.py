"""Wall-clock deadline supervisor for the item-G retrain queue.

Stages (identical commands to ``scripts/phase3_retrain_iaug.py``):

    train s42..s46 (30 epochs, batch 8, --num-workers 0)   [parallel-safe]
    TEST eval once per seed                                [serial]
    multi-seed aggregate -> 27x3 corruption MRRS           [serial]

Two rules enforced here:

* **Deadline** — every child is killed at ``--deadline`` (default 23:00), so the
  machine can be shut down on schedule.  Training resumes from
  ``checkpoint_last.pt`` (written every epoch), finished evals/aggregates are
  skipped, so re-running continues exactly where it stopped.
* **Parallelism** — ``--parallel N`` runs up to N *trainings* at once
  (documented repo max is 3; the 15 GB host and 4 GB RTX 3050 mean N=2 is the
  safe default, N=3 risks the Win32 1455 paging cliff).  Evaluations and
  aggregate/robustness always run alone, after the trainings.

Child output is redirected to ``logs/run_until_11pm.<step>.log``; the console
shows one status line per minute with each active run's epoch.

Launch:
    run_until_11pm.bat                 (deadline 23:00, parallel 1)
    run_until_11pm.bat 23:00 2         (2 trainings at a time)
    python -X utf8 scripts/run_until_deadline.py --deadline 23:00 --parallel 2

Exit codes: 0 = finished or stopped at the deadline (resumable), 1 = at least
one stage failed (nothing is deleted; re-run to retry).
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
# step definitions
# --------------------------------------------------------------------------- #
class Step:
    def __init__(self, label: str, cmd: list[str], done, seed: int | None = None) -> None:
        self.label = label
        self.cmd = cmd
        self.done = done
        self.seed = seed


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


def build_steps(seeds: tuple[int, ...]) -> tuple[list[Step], list[Step]]:
    trains = [
        Step(
            f"train s{s}",
            train_cmd(s),
            (lambda s=s: (run_dir(s) / "training_result.json").exists()),
            seed=s,
        )
        for s in seeds
    ]
    all_eval_done = lambda: all(metrics_path(s).exists() for s in seeds)  # noqa: E731
    post: list[Step] = [
        Step(
            f"eval s{s}",
            eval_cmd(s),
            (lambda s=s: metrics_path(s).exists()),
            seed=s,
        )
        for s in seeds
    ]
    post.append(Step(
        "aggregate",
        aggregate_cmd(seeds),
        lambda: all_eval_done() and json_fresh(SUMMARY, seeds),
    ))
    post.append(Step(
        "robustness",
        robustness_cmd(seeds),
        lambda: all_eval_done() and json_fresh(ROBUSTNESS, seeds),
    ))
    return trains, post


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


def epoch_status(seed: int | None) -> str:
    if seed is None:
        return ""
    try:
        d = json.loads((run_dir(seed) / "progress.json").read_text(encoding="utf-8"))
        return f"ep {d.get('epoch')}/30 loss {d.get('train_loss'):.4f}"
    except Exception:  # noqa: BLE001 - progress is cosmetic
        return "starting"


class Supervisor:
    def __init__(self, deadline: datetime, parallel: int) -> None:
        self.deadline = deadline
        self.parallel = parallel
        self.active: dict[subprocess.Popen, Step] = {}
        self.failed: list[str] = []
        self.last_report = 0.0
        (ROOT / "logs").mkdir(exist_ok=True)

    # -- deadline / process helpers ---------------------------------------
    def expired(self) -> bool:
        return datetime.now() >= self.deadline

    def start(self, step: Step) -> None:
        log = ROOT / "logs" / f"run_until_11pm.{step.label.replace(' ', '_')}.log"
        fh = log.open("a", encoding="utf-8")
        fh.write(f"\n===== {step.label} started {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        fh.flush()
        print(f"START {step.label} (child log: {log.relative_to(ROOT)})", flush=True)
        proc = subprocess.Popen(step.cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        self.active[proc] = step

    def kill_all(self) -> None:
        for proc, step in list(self.active.items()):
            print(f"STOP  {step.label}: terminating (resumable)", flush=True)
            terminate_tree(proc.pid)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.active.clear()

    def report(self) -> None:
        if time.time() - self.last_report < 60:
            return
        self.last_report = time.time()
        left = (self.deadline - datetime.now()).total_seconds() / 3600
        parts = [f"{step.label} {epoch_status(step.seed)}" for step in self.active.values()]
        print(f"  ...   {' | '.join(parts)} | {left:.2f} h left", flush=True)

    # -- phases -----------------------------------------------------------
    def run_trains(self, trains: list[Step]) -> None:
        idx = 0
        while idx < len(trains) or self.active:
            while idx < len(trains) and trains[idx].done():
                print(f"skip  {trains[idx].label} (artifact present)", flush=True)
                idx += 1
            while (
                idx < len(trains)
                and len(self.active) < self.parallel
                and not trains[idx].done()
                and not self.expired()
            ):
                self.start(trains[idx])
                idx += 1
            if not self.active:
                if idx >= len(trains) or self.expired():
                    break
                continue
            if self.expired():
                self.kill_all()
                return
            for proc in [p for p in self.active if p.poll() is not None]:
                step = self.active.pop(proc)
                rc = proc.returncode
                if rc == 0:
                    print(f"done  {step.label} at {datetime.now():%H:%M:%S}", flush=True)
                else:
                    print(f"FAIL  {step.label} exited {rc} (see its child log)", flush=True)
                    self.failed.append(step.label)
            self.report()
            time.sleep(POLL_SECONDS)

    def run_post(self, post: list[Step]) -> bool:
        """Serial post-steps. Returns False if stopped by the deadline."""
        for step in post:
            if step.done():
                print(f"skip  {step.label} (artifact present)", flush=True)
                continue
            if step.label.startswith("eval") and step.seed is not None:
                if not (run_dir(step.seed) / "best.pt").exists():
                    print(f"skip  {step.label} (training incomplete)", flush=True)
                    continue
            if self.expired():
                print(f"STOP  deadline reached before {step.label}", flush=True)
                return False
            self.start(step)
            while self.active:
                if self.expired():
                    self.kill_all()
                    return False
                for proc in [p for p in self.active if p.poll() is not None]:
                    s = self.active.pop(proc)
                    if proc.returncode == 0:
                        print(f"done  {s.label} at {datetime.now():%H:%M:%S}", flush=True)
                    else:
                        print(f"FAIL  {s.label} exited {proc.returncode}", flush=True)
                        self.failed.append(s.label)
                self.report()
                time.sleep(POLL_SECONDS)
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description="item-G queue with a wall-clock deadline")
    ap.add_argument("--deadline", default="23:00", help="local HH:MM at which all compute stops")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="concurrent trainings (repo max 3; 2 is safe on this 15 GB host, "
        "3 risks the Win32 1455 paging cliff)",
    )
    args = ap.parse_args()

    if args.parallel < 1 or args.parallel > 3:
        print("--parallel must be 1..3 (documented max is 3 concurrent trainings)", flush=True)
        return 2

    seeds = tuple(args.seeds)
    try:
        deadline = resolve_deadline(args.deadline)
    except ValueError:
        print(f"invalid --deadline {args.deadline!r}: expected HH:MM in 00:00..23:59", flush=True)
        return 2

    remaining = (deadline - datetime.now()).total_seconds() / 3600
    print(f"deadline : {deadline:%Y-%m-%d %H:%M} local ({remaining:.1f} h from now)", flush=True)
    print(f"seeds    : {', '.join(map(str, seeds))} | parallel trainings: {args.parallel}", flush=True)
    print("resumable: training -> checkpoint_last.pt; eval/aggregate -> skip-if-present", flush=True)

    trains, post = build_steps(seeds)
    sup = Supervisor(deadline, args.parallel)

    sup.run_trains(trains)
    stopped = sup.expired()
    if not stopped:
        stopped = not sup.run_post(post)

    if sup.failed:
        print(f"\nFAILED stages: {', '.join(sup.failed)} - nothing was deleted; re-run to retry",
              flush=True)
        return 1
    if stopped:
        print("\nSTOP  deadline reached - nothing running; re-run to resume", flush=True)
        return 0
    print(f"\nALL DONE at {datetime.now():%H:%M:%S}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
