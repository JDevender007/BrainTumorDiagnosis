"""docs/TODO.md item G — serial retrain queue: intensity aug + per-image norm.

Trains TAMNet on the clean split with:

    --intensity-aug     train-only ColorJitter(brightness=0.3, contrast=0.3, p=0.8)
    --per-image-norm    percentile(1-99) z-score per image instead of ImageNet stats

One seed at a time (GPU-serial: 4 GB RTX 3050), 30 epochs, batch 8,
``--num-workers 0`` — identical to the published protocol except the two input
switches, so any difference is attributable to them alone.

Each stage is skipped when its artifact already exists, so the queue can be
killed (shutdown rule) and restarted without redoing work:

    train   -> results/phase3/runs/tamnet_clean_s{seed}+iaug+pnorm/training_result.json
    eval    -> .../eval_internal_test/metrics.json          (TEST touched once)
    aggregate -> results/phase3/intensity_aug_retrain_summary.json
    robustness -> results/phase3/robustness_iaug.json       (27 corruptions x 3 severities)

Usage:
    python -X utf8 scripts/phase3_retrain_iaug.py --seeds 42 43 44 45 46
    python -X utf8 scripts/phase3_retrain_iaug.py --stage aggregate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "phase3" / "runs"
SUMMARY = ROOT / "results" / "phase3" / "intensity_aug_retrain_summary.json"
ROBUSTNESS = ROOT / "results" / "phase3" / "robustness_iaug.json"
GROUP_LABEL = "tamnet_clean+iaugpn"
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def run_name(seed: int) -> str:
    return f"tamnet_clean_s{seed}+iaug+pnorm"


def run_dir(seed: int) -> Path:
    return RUNS / run_name(seed)


def sh(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"  ok in {time.time() - t0:.0f}s", flush=True)


def train(seed: int) -> None:
    if (run_dir(seed) / "training_result.json").exists():
        print(f"skip train s{seed}: training_result.json exists", flush=True)
        return
    sh([
        sys.executable, "-X", "utf8", "scripts/phase3_train.py",
        "--model", "tamnet",
        "--seed", str(seed),
        "--protocol", "clean",
        "--epochs", "30",
        "--batch-size", "8",
        "--num-workers", "0",
        "--intensity-aug",
        "--per-image-norm",
    ])


def evaluate(seed: int) -> None:
    metrics = run_dir(seed) / "eval_internal_test" / "metrics.json"
    if metrics.exists():
        print(f"skip eval s{seed}: metrics.json exists", flush=True)
        return
    if not (run_dir(seed) / "best.pt").exists():
        raise SystemExit(f"missing best.pt for s{seed} — train first")
    sh([
        sys.executable, "-X", "utf8", "scripts/phase3_evaluate.py",
        "--run-dir", str(run_dir(seed).relative_to(ROOT)),
        "--split", "test",
        "--dataset", "internal",
        "--batch-size", "16",
        "--num-workers", "0",
    ])


def aggregate(seeds: tuple[int, ...]) -> None:
    dirs = [str(run_dir(s).relative_to(ROOT)) for s in seeds]
    sh([
        sys.executable, "-X", "utf8", "scripts/phase3_aggregate.py",
        "--runs", *dirs,
        "--eval-dir", "eval_internal_test",
        "--out", str(SUMMARY.relative_to(ROOT)),
    ])


def robustness(seeds: tuple[int, ...]) -> None:
    if ROBUSTNESS.exists():
        print(f"skip robustness: {ROBUSTNESS.relative_to(ROOT)} exists", flush=True)
        return
    dirs = [run_name(s) for s in seeds]
    sh([
        sys.executable, "-X", "utf8", "scripts/phase3_robustness.py",
        "--runs", *dirs,
        "--intensity-runs",
        "--out", str(ROBUSTNESS),
        "--group-label", GROUP_LABEL,
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description="item-G serial retrain queue")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument(
        "--stage",
        choices=("all", "train", "eval", "aggregate", "robustness"),
        default="all",
    )
    args = ap.parse_args()
    seeds = tuple(args.seeds)

    t0 = time.time()
    if args.stage in ("all", "train", "eval"):
        for seed in seeds:
            if args.stage in ("all", "train"):
                train(seed)
            if args.stage in ("all", "eval"):
                evaluate(seed)
    if args.stage in ("all", "aggregate"):
        aggregate(seeds)
    if args.stage in ("all", "robustness"):
        robustness(seeds)

    print(f"\nqueue finished in {(time.time() - t0) / 3600:.2f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
