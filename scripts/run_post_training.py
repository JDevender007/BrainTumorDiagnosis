"""
Post-training pipeline: runs after run_research_experiments.py completes.
Chains: aggregate -> multi-seed -> ablation -> figures -> tables -> paper
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")


def run_phase(name: str, script: str, args: list[str] | None = None):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    cmd = [PYTHON, "-u", str(ROOT / script)]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  WARNING: {name} exited with code {result.returncode}")
    return result.returncode == 0


def main():
    start = time.time()
    print("="*70)
    print("  POST-TRAINING PIPELINE")
    print("="*70)

    run_phase("Aggregate Results", "scripts/aggregate_results.py")

    run_phase("Multi-Seed Validation (3 seeds)", "scripts/run_multi_seed.py")

    run_phase("Ablation Study (10 experiments)", "scripts/run_ablation.py")

    run_phase("Grad-CAM Visualization", "scripts/generate_gradcam_figures.py")

    run_phase("Regenerate Figures (16)", "scripts/generate_figures.py")
    run_phase("Regenerate Tables (12)", "scripts/generate_tables.py")

    run_phase("Generate LaTeX Manuscript", "scripts/generate_paper.py")

    elapsed = time.time() - start
    hours = elapsed / 3600
    print(f"\n{'='*70}")
    print(f"  POST-TRAINING COMPLETE ({hours:.1f}h)")
    print(f"  Results: outputs/research_experiments/")
    print(f"  Figures: outputs/research_experiments/figures/")
    print(f"  Tables:  outputs/research_experiments/tables/")
    print(f"  Paper:   outputs/research_experiments/paper/main.tex")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
