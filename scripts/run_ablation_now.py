"""Run ablation study: 8 configs x 3 epochs, seed 42."""
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
    configs = list(ABLATION_CONFIGS.items())

    for i, (exp_name, exp_cfg) in enumerate(configs):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(configs)}] {exp_name}: {exp_cfg['desc']}")
        print(f"  Start: {time.strftime('%H:%M:%S')}")

        set_seed(42)
        model = exp_cfg["builder"]()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Params: {param_count/1e6:.2f}M")

        config = TrainConfig(
            model_name=f"{exp_name}_seed42",
            epochs=3,
            seed=42,
            output_dir="paper/results/ablation",
            use_turma_loss=exp_cfg["use_turma_loss"],
        )

        train_result = train_model(model, config, train_loader, val_loader, device)

        ckpt = torch.load(train_result["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        test_result = evaluate_model(model, test_loader, device, f"{exp_name}_seed42")
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
        results[f"{exp_name}_seed42"] = entry

        out_path = Path("paper/results/ablation") / f"{exp_name}_seed42.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(entry, f, indent=2)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s | acc={test_result.accuracy:.4f} f1={test_result.macro_f1:.4f} auc={test_result.roc_auc:.4f}")

    with open("paper/results/ablation_all.json", "w") as f:
        json.dump(results, f, indent=2)

    total = time.time() - t_start
    print(f"\nABLATION COMPLETE at {time.strftime('%H:%M:%S')}")
    print(f"Total time: {total/60:.1f} minutes")

    print("\n--- SUMMARY ---")
    for key, res in sorted(results.items()):
        t = res["test"]
        print(f"  {res['experiment']:35s} acc={t['accuracy']:.4f} f1={t['macro_f1']:.4f} auc={t['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
