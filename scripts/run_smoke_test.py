"""Quick smoke test: 3 epochs for each phase, single seed."""
import sys
sys.path.insert(0, ".")

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.seed import set_seed
from utils.device import get_device

set_seed(42)
device = get_device()
print(f"Device: {device}")

output_dir = Path("outputs/research_experiments_smoke")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "uncertainty").mkdir(exist_ok=True)
(output_dir / "calibration").mkdir(exist_ok=True)
(output_dir / "robustness").mkdir(exist_ok=True)
(output_dir / "proposed_model").mkdir(exist_ok=True)

# --- Load data ---
from scripts.run_research_experiments import load_classification_data, load_external_data
print("Loading data...")
train_loader, val_loader, test_loader = load_classification_data("data/raw/classification", (224, 224), 8)
print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
external_loader = load_external_data("data/external/brisc2025/brisc2025/classification_task/test", (224, 224), 8)
print(f"  External: {len(external_loader.dataset)}")

# --- Phase 1: Baseline (3 epochs) ---
print("\n=== PHASE 1: Baseline (3 epochs) ===")
from training.classification_trainer import ClassificationTrainingConfig, train_classification_model
baseline_config = ClassificationTrainingConfig(
    backbone="densenet121",
    epochs=3,
    batch_size=8,
    image_size=(224, 224),
    learning_rate=0.0001,
    patience=5,
    model_directory=str(output_dir / "baseline"),
    result_directory=str(output_dir / "baseline_results"),
)
baseline_result = train_classification_model(baseline_config)
print(f"  Best val acc: {baseline_result.best_validation_accuracy:.4f} at epoch {baseline_result.best_epoch}")

# --- Phase 2: TumorAwareMultiTaskNet (3 epochs) ---
print("\n=== PHASE 2: Proposed Model (3 epochs) ===")
from training.multitask_trainer import MultiTaskTrainingConfig, train_multitask_model
proposed_config = MultiTaskTrainingConfig(
    backbone="densenet121",
    pretrained=True,
    epochs=3,
    batch_size=8,
    image_size=(224, 224),
    lr=0.0001,
    patience=7,
    seg_loss_weight=0.5,
    output_dir=str(output_dir / "proposed_model"),
)
proposed_result = train_multitask_model(proposed_config, train_loader, val_loader, device)
print(f"  Best val F1: {proposed_result.best_val_f1:.4f} at epoch {proposed_result.best_epoch}")

# --- Phase 3: Uncertainty ---
print("\n=== PHASE 3: Uncertainty ===")
from evaluation.uncertainty import compute_uncertainty_metrics
from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig
model_config = TumorAwareConfig(backbone="densenet121")
model = TumorAwareMultiTaskNet(model_config).to(device)
checkpoint = torch.load(output_dir / "proposed_model" / "best.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
unc_metrics = compute_uncertainty_metrics(model, test_loader, device, num_passes=5)
with open(output_dir / "uncertainty" / "uncertainty_metrics.json", "w") as f:
    json.dump(unc_metrics, f, indent=2, default=str)
print(f"  ECE: {unc_metrics['ece']:.4f}, Brier: {unc_metrics['brier']:.4f}")

# --- Phase 4: Calibration ---
print("\n=== PHASE 4: Calibration ===")
from evaluation.calibration import evaluate_calibration
cal_result = evaluate_calibration(model, test_loader, device, model_name="tumor_aware", output_dir=output_dir / "calibration")
print(f"  ECE: {cal_result.ece:.4f}, MCE: {cal_result.mce:.4f}")

# --- Phase 5: Robustness ---
print("\n=== PHASE 5: Robustness ===")
from evaluation.advanced_robustness import run_robustness_evaluation, CorruptionConfig
rob_result = run_robustness_evaluation(model, test_loader, device, model_name="tumor_aware", output_dir=output_dir / "robustness", config=CorruptionConfig())
print(f"  MRRS: {rob_result['mrrs']:.4f}")

# --- Phase 6: Efficiency ---
print("\n=== PHASE 6: Efficiency ===")
from evaluation.efficiency_analysis import analyze_efficiency
eff_result = analyze_efficiency(model, "TumorAwareMultiTaskNet", test_loader, device, accuracy=proposed_result.best_val_f1, f1=proposed_result.best_val_f1)
print(f"  Params: {eff_result.total_params/1e6:.2f}M, Latency: {eff_result.latency_ms:.2f}ms, FPS: {eff_result.fps:.1f}")

# --- Phase 7: Cross-dataset ---
print("\n=== PHASE 7: Cross-Dataset ===")
from evaluation.cross_dataset_analysis import analyze_domain_shift
cross_result = analyze_domain_shift(model, "TumorAwareMultiTaskNet", test_loader, external_loader, device)
print(f"  Internal: {cross_result.internal_accuracy:.4f}, External: {cross_result.external_accuracy:.4f}, Drop: {cross_result.accuracy_drop:.4f}")

# --- Save all ---
all_results = {
    "baseline": {"best_accuracy": baseline_result.best_validation_accuracy, "best_epoch": baseline_result.best_epoch},
    "proposed": {"best_f1": proposed_result.best_val_f1, "best_epoch": proposed_result.best_epoch},
    "uncertainty": unc_metrics,
    "calibration": {"ece": cal_result.ece, "mce": cal_result.mce, "brier": cal_result.brier, "nll": cal_result.nll},
    "robustness": {"mrrs": rob_result["mrrs"], "clean_accuracy": rob_result["clean_accuracy"]},
    "efficiency": {"params": eff_result.total_params, "size_mb": eff_result.model_size_mb, "latency_ms": eff_result.latency_ms, "fps": eff_result.fps},
    "cross_dataset": {"internal_acc": cross_result.internal_accuracy, "external_acc": cross_result.external_accuracy, "drop": cross_result.accuracy_drop},
}
with open(output_dir / "smoke_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n{'='*70}")
print("SMOKE TEST COMPLETE — all 7 phases passed!")
print(f"Results saved to: {output_dir}")
