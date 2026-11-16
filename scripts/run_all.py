"""
Master orchestrator: run all experiment phases in sequence.
Handles: training -> multi-seed -> ablation -> figures -> tables
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PYTHON = str(Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe")
ROOT = str(Path(__file__).resolve().parent.parent)


def run_phase(name: str, script: str, args: list[str] | None = None):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    cmd = [PYTHON, "-u", str(Path(ROOT) / script)]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  WARNING: {name} exited with code {result.returncode}")
    return result.returncode == 0


def main():
    start = time.time()
    print("="*70)
    print("  MASTER EXPERIMENT PIPELINE")
    print("="*70)

    success = True

    if not run_phase("Phase 1-7: Full Experiment", "scripts/run_research_experiments.py"):
        success = False

    if not run_phase("Phase 8: Multi-Seed Validation", "scripts/run_multi_seed.py"):
        success = False

    if not run_phase("Phase 9: Ablation Study", "scripts/run_ablation.py"):
        success = False

    run_phase("Phase 10: Generate Figures", "scripts/generate_figures.py")
    run_phase("Phase 11: Generate Tables", "scripts/generate_tables.py")

    elapsed = time.time() - start
    hours = elapsed / 3600

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE ({'SUCCESS' if success else 'PARTIAL'})")
    print(f"  Total time: {hours:.1f} hours")
    print(f"  Results: outputs/research_experiments/")
    print(f"  Figures: outputs/research_experiments/figures/")
    print(f"  Tables:  outputs/research_experiments/tables/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
