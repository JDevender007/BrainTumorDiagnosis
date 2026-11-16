"""Recreate A-C results from completed training and run D-H."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_all_experiments import *
from utils.seed import set_seed
import torch

def main():
    device = get_device()
    print(f"Device: {device}")
    t_start = time.time()
    print(f"Start: {time.strftime('%H:%M:%S')}")

    set_seed(42)
    train_loader, val_loader, test_loader = load_data(
        "data/raw/classification", batch_size=8, seed=42
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    results = {}
    partial_path = Path("paper/results/ablation_all.json")
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    all_configs = list(ABLATION_CONFIGS.items())

    for i, (exp_name, exp_cfg) in enumerate(all_configs):
        key = f"{exp_name}_seed42"
        ckpt_path = Path("paper/results/ablation") / key / "best.pt"
        hist_path = Path("paper/results/ablation") / key / "training_history.json"

        if ckpt_path.exists() and hist_path.exists():
            print(f"\n[{i+1}/8] {exp_name}: LOADING from checkpoint")
            set_seed(42)
            model = exp_cfg["builder"]()
            param_count = sum(p.numel() for p in model.parameters())
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)

            test_result = evaluate_model(model, test_loader, device, key)
            eff = measure_efficiency(model, device, n_runs=10)

            with open(hist_path) as f:
                history = json.load(f)

            entry = {
                "experiment": exp_name,
                "description": exp_cfg["desc"],
                "seed": 42,
                "params": param_count,
                "train": {
                    "best_val_f1": ckpt.get("val_f1", 0),
                    "best_epoch": ckpt.get("epoch", 0),
                    "total_time": 0,
                },
                "test": {
                    "accuracy": test_result.accuracy,
                    "balanced_accuracy": test_result.balanced_accuracy,
                    "macro_f1": test_result.macro_f1,
                    "weighted_f1": test_result.weighted_f1,
                    "roc_auc": test_result.roc_auc,
                    "pr_auc": test_result.pr_auc,
                    "mcc": test_result.mcc,
                    "kappa": test_result.kappa,
                    "per_class_f1": test_result.per_class_f1,
                    "sensitivity": test_result.sensitivity,
                    "specificity": test_result.specificity,
                },
                "efficiency": eff,
                "confusion_matrix": test_result.confusion_matrix,
                "roc_data": test_result.roc_data,
                "pr_data": test_result.pr_data,
            }
            results[key] = entry
            print(f"  acc={test_result.accuracy:.4f} f1={test_result.macro_f1:.4f} auc={test_result.roc_auc:.4f}")
        else:
            t0 = time.time()
            print(f"\n[{i+1}/8] {exp_name}: TRAINING from scratch")
            print(f"  Start: {time.strftime('%H:%M:%S')}")

            set_seed(42)
            model = exp_cfg["builder"]()
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  Params: {param_count/1e6:.2f}M")

            config = TrainConfig(
                model_name=key,
                epochs=3,
                seed=42,
                output_dir="paper/results/ablation",
                use_turma_loss=exp_cfg["use_turma_loss"],
            )

            train_result = train_model(model, config, train_loader, val_loader, device)
            ckpt = torch.load(train_result["checkpoint"], map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

            test_result = evaluate_model(model, test_loader, device, key)
            eff = measure_efficiency(model, device, n_runs=10)

            entry = {
                "experiment": exp_name,
                "description": exp_cfg["desc"],
                "seed": 42,
                "params": param_count,
                "train": {
                    "best_val_f1": train_result["best_val_f1"],
                    "best_epoch": train_result["best_epoch"],
                    "total_time": train_result["total_time"],
                },
                "test": {
                    "accuracy": test_result.accuracy,
                    "balanced_accuracy": test_result.balanced_accuracy,
                    "macro_f1": test_result.macro_f1,
                    "weighted_f1": test_result.weighted_f1,
                    "roc_auc": test_result.roc_auc,
                    "pr_auc": test_result.pr_auc,
                    "mcc": test_result.mcc,
                    "kappa": test_result.kappa,
                    "per_class_f1": test_result.per_class_f1,
                    "sensitivity": test_result.sensitivity,
                    "specificity": test_result.specificity,
                },
                "efficiency": eff,
                "confusion_matrix": test_result.confusion_matrix,
                "roc_data": test_result.roc_data,
                "pr_data": test_result.pr_data,
            }
            results[key] = entry
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.0f}s | acc={test_result.accuracy:.4f} f1={test_result.macro_f1:.4f} auc={test_result.roc_auc:.4f}")

        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved ({len(results)}/8)")

    total = time.time() - t_start
    print(f"\nABLATION COMPLETE at {time.strftime('%H:%M:%S')}")
    print(f"Total time: {total/60:.1f} minutes")

    print("\n--- ABLATION SUMMARY ---")
    for key in sorted(results.keys()):
        res = results[key]
        t = res["test"]
        print(f"  {res['experiment']:35s} acc={t['accuracy']:.4f} f1={t['macro_f1']:.4f} auc={t['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
