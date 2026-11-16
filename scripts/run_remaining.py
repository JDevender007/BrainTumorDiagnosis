"""Run remaining phases: ablation + multi-seed + final regeneration."""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")

def run(name, script):
    print(f"\n{'='*70}\n  {name}\n{'='*70}", flush=True)
    r = subprocess.run([PYTHON, "-u", str(ROOT / script)], cwd=str(ROOT), text=True)
    if r.returncode != 0:
        print(f"  WARNING: {name} exit code {r.returncode}", flush=True)

if __name__ == "__main__":
    t0 = time.time()
    run("Ablation Study (10 experiments)", "scripts/run_ablation.py")
    run("Multi-Seed Validation (3 seeds)", "scripts/run_multi_seed.py")
    run("Regenerate Tables", "scripts/generate_tables.py")
    run("Regenerate Paper", "scripts/generate_paper.py")
    h = (time.time() - t0) / 3600
    print(f"\nDone in {h:.1f}h", flush=True)
