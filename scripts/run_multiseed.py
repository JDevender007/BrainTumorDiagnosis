"""Multi-seed validation: 3 models x 3 seeds x 10 epochs."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_all_experiments import *
from utils.seed import set_seed
import torch

MODELS = {
    "A_baseline_densenet121": {
        "desc": "Baseline: DenseNet121",
        "builder": lambda: build_baseline("densenet121"),
        "use_turma_loss": False,
    },
    "D_tumor_aware": {
        "desc": "TumorAwareMultiTaskNet",
        "builder": lambda: build_tumor_aware("densenet121"),
        "use_turma_loss": False,
    },
    "E_turma_full": {
        "desc": "TURMA-Net Full",
        "builder": lambda: build_turma_net("densenet121", use_atgfm=True, use_consistency=True, use_uncertainty_weights=True),
        "use_turma_loss": True,
    },
}

SEEDS = [42, 123, 456]

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

    results_path = Path("paper/results/multi_seed_all.json")
    results = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing multi-seed results")

    total_runs = len(MODELS) * len(SEEDS)
    run_count = 0

    for model_name, model_cfg in MODELS.items():
        for seed in SEEDS:
            run_count += 1
            key = f"{model_name}_seed{seed}"
            if key in results:
                print(f"\n[{run_count}/{total_runs}] [SKIP] {key}")
                continue

            t0 = time.time()
            print(f"\n{'='*60}")
            print(f"[{run_count}/{total_runs}] {model_name} seed={seed}")
            print(f"  Start: {time.strftime('%H:%M:%S')}")

            set_seed(seed)
            model = model_cfg["builder"]()
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  Params: {param_count/1e6:.2f}M")

            config = TrainConfig(
                model_name=key,
                epochs=5,
                seed=seed,
                output_dir="paper/results/multi_seed",
                use_turma_loss=model_cfg["use_turma_loss"],
            )

            train_result = train_model(model, config, train_loader, val_loader, device)

            ckpt = torch.load(train_result["checkpoint"], map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

            test_result = evaluate_model(model, test_loader, device, key)
            eff = measure_efficiency(model, device, n_runs=10)

            entry = {
                "model_name": model_name,
                "description": model_cfg["desc"],
                "seed": seed,
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
                "y_true": test_result.y_true,
                "y_pred": test_result.y_pred,
                "y_probs": test_result.y_probs,
            }
            results[key] = entry

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.0f}s | acc={test_result.accuracy:.4f} f1={test_result.macro_f1:.4f} auc={test_result.roc_auc:.4f}")

    total = time.time() - t_start
    print(f"\nMULTI-SEED COMPLETE at {time.strftime('%H:%M:%S')}")
    print(f"Total time: {total/60:.1f} minutes")

    print("\n--- MULTI-SEED SUMMARY ---")
    for model_name in MODELS:
        model_results = {k: v for k, v in results.items() if v["model_name"] == model_name}
        if model_results:
            accs = [v["test"]["accuracy"] for v in model_results.values()]
            f1s = [v["test"]["macro_f1"] for v in model_results.values()]
            aucs = [v["test"]["roc_auc"] for v in model_results.values()]
            print(f"  {model_name}:")
            print(f"    Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f} (range: {min(accs):.4f}-{max(accs):.4f})")
            print(f"    Macro F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} (range: {min(f1s):.4f}-{max(f1s):.4f})")
            print(f"    ROC-AUC:  {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} (range: {min(aucs):.4f}-{max(aucs):.4f})")


if __name__ == "__main__":
    main()
