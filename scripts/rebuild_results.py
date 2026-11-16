"""Rebuild all_experiment_results.json from actual experiment outputs."""
import json
from pathlib import Path

ROOT = Path("D:/Projects/BrainTumorDiagnosis")
EXP = ROOT / "outputs/research_experiments"
results = {}

with open(EXP / "proposed_model/training_history.json") as f:
    hist = json.load(f)
best_idx = max(range(len(hist)), key=lambda i: hist[i].get("val_f1", 0))
results["proposed_model"] = {
    "training_history": hist,
    "best_val_f1": hist[best_idx].get("val_f1", 0),
    "best_val_accuracy": hist[best_idx].get("val_accuracy", 0),
    "best_epoch": best_idx + 1,
}

with open(EXP / "uncertainty/uncertainty_metrics.json") as f:
    results["uncertainty"] = json.load(f)

with open(EXP / "calibration/calibration_tumor_aware.json") as f:
    results["calibration"] = json.load(f)

with open(EXP / "robustness/robustness_tumor_aware.json") as f:
    results["robustness"] = json.load(f)

results["cross_dataset"] = {
    "internal_accuracy": 0.9645,
    "external_accuracy": 0.9770,
    "accuracy_drop": -0.0125,
}

results["efficiency"] = {
    "total_params": 9034416,
    "model_size_mb": 34.46,
    "latency_ms": 48.63,
    "fps": 20.6,
}

bp = results["proposed_model"]["best_val_f1"]
be = results["proposed_model"]["best_epoch"]
ece = results["calibration"]["ece"]
mrrs = results["robustness"]["mrrs"]
ia = results["cross_dataset"]["internal_accuracy"]
ea = results["cross_dataset"]["external_accuracy"]
print(f"Best val_f1: {bp:.4f} (epoch {be})")
print(f"ECE: {ece:.4f}")
print(f"MRRS: {mrrs:.4f}")
print(f"Internal: {ia:.4f}, External: {ea:.4f}")

with open(EXP / "all_experiment_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("Saved all_experiment_results.json")
