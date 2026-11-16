"""
Aggregate all experiment results into a single all_experiment_results.json.
Reads from: baseline/, proposed_model/, multi_seed/, ablation_study/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path("outputs/research_experiments")


def safe_load(path: Path) -> dict | list | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def aggregate():
    results = {}

    baseline_dir = OUTPUT_DIR / "baseline"
    if baseline_dir.exists():
        hist = safe_load(baseline_dir / "training_history.json")
        if hist:
            if isinstance(hist, list):
                results["baseline"] = {"training_history": hist, "final_accuracy": hist[-1].get("val_acc", hist[-1].get("val_accuracy", 0))}
            else:
                results["baseline"] = hist

    proposed_dir = OUTPUT_DIR / "proposed_model"
    if proposed_dir.exists():
        hist = safe_load(proposed_dir / "training_history.json")
        if hist:
            if isinstance(hist, list):
                results["proposed_model"] = {"training_history": hist}
            else:
                results["proposed_model"] = hist

    eval_dir = OUTPUT_DIR / "evaluation"
    if eval_dir.exists():
        for f in eval_dir.glob("*.json"):
            key = f.stem
            data = safe_load(f)
            if data:
                results[key] = data

    ms_dir = OUTPUT_DIR / "multi_seed"
    if ms_dir.exists():
        all_seeds = safe_load(ms_dir / "all_seed_results.json")
        if all_seeds:
            results["multi_seed"] = all_seeds

    ablation_dir = OUTPUT_DIR / "ablation_study"
    if ablation_dir.exists():
        ablation_results = safe_load(ablation_dir / "ablation_results.json")
        if ablation_results:
            results["ablation"] = ablation_results

    smoke_dir = OUTPUT_DIR.parent / "research_experiments_smoke"
    if smoke_dir.exists():
        smoke = safe_load(smoke_dir / "smoke_results.json")
        if smoke:
            results["smoke_test"] = smoke

    out_path = OUTPUT_DIR / "all_experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Aggregated results saved to {out_path}")
    return results


if __name__ == "__main__":
    aggregate()
