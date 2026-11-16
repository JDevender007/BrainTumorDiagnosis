"""Full experiment suite on GPU: ablation + multi-seed + evaluation + figures."""
import sys, json, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_all_experiments import *
from utils.seed import set_seed
import torch
import numpy as np

def main():
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    t_start = time.time()
    print(f"Start: {time.strftime('%H:%M:%S')}")

    set_seed(42)
    train_loader, val_loader, test_loader = load_data(
        "data/raw/classification", batch_size=8, seed=42
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # =========================================================================
    # PHASE 1: ABLATION (8 configs x 3 epochs)
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: ABLATION STUDY (8 configs x 3 epochs)")
    print("="*70)

    ablation_results = {}
    ablation_path = Path("paper/results/ablation_all.json")
    if ablation_path.exists():
        with open(ablation_path) as f:
            ablation_results = json.load(f)

    for i, (exp_name, exp_cfg) in enumerate(ABLATION_CONFIGS.items()):
        key = f"{exp_name}_seed42"
        if key in ablation_results:
            print(f"  [SKIP] {exp_name}")
            continue

        t0 = time.time()
        print(f"\n[{i+1}/8] {exp_name}: {exp_cfg['desc']}")
        set_seed(42)
        model = exp_cfg["builder"]().to(device)
        param_count = sum(p.numel() for p in model.parameters())
        config = TrainConfig(model_name=key, epochs=3, seed=42,
                             output_dir="paper/results/ablation", use_turma_loss=exp_cfg["use_turma_loss"])
        train_result = train_model(model, config, train_loader, val_loader, device)
        ckpt = torch.load(train_result["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_result = evaluate_model(model, test_loader, device, key)
        eff = measure_efficiency(model, device, n_runs=20)
        entry = {
            "experiment": exp_name, "description": exp_cfg["desc"], "seed": 42, "params": param_count,
            "train": {"best_val_f1": train_result["best_val_f1"], "best_epoch": train_result["best_epoch"],
                      "total_time": train_result["total_time"]},
            "test": {
                "accuracy": test_result.accuracy, "balanced_accuracy": test_result.balanced_accuracy,
                "macro_f1": test_result.macro_f1, "weighted_f1": test_result.weighted_f1,
                "roc_auc": test_result.roc_auc, "pr_auc": test_result.pr_auc,
                "mcc": test_result.mcc, "kappa": test_result.kappa,
                "per_class_f1": test_result.per_class_f1,
                "sensitivity": test_result.sensitivity, "specificity": test_result.specificity,
            },
            "efficiency": eff, "confusion_matrix": test_result.confusion_matrix,
            "roc_data": test_result.roc_data, "pr_data": test_result.pr_data,
        }
        ablation_results[key] = entry
        with open(ablation_path, "w") as f:
            json.dump(ablation_results, f, indent=2)
        elapsed = time.time() - t0
        print(f"  Done {elapsed:.0f}s | acc={test_result.accuracy:.4f} f1={test_result.macro_f1:.4f} auc={test_result.roc_auc:.4f}")

    # =========================================================================
    # PHASE 2: MULTI-SEED (3 models x 3 seeds x 5 epochs)
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 2: MULTI-SEED (3 models x 3 seeds x 5 epochs)")
    print("="*70)

    ms_results = {}
    ms_path = Path("paper/results/multi_seed_all.json")
    if ms_path.exists():
        with open(ms_path) as f:
            ms_results = json.load(f)

    SEEDS = [42, 123, 456]
    KEY_MODELS = {
        "A_baseline_densenet121": {"desc": "Baseline: DenseNet121",
            "builder": lambda: build_baseline("densenet121"), "use_turma_loss": False},
        "D_tumor_aware": {"desc": "TumorAwareMultiTaskNet",
            "builder": lambda: build_tumor_aware("densenet121"), "use_turma_loss": False},
        "E_turma_full": {"desc": "TURMA-Net Full",
            "builder": lambda: build_turma_net("densenet121", use_atgfm=True, use_consistency=True, use_uncertainty_weights=True),
            "use_turma_loss": True},
    }

    run_num = 0
    total_ms = len(KEY_MODELS) * len(SEEDS)
    for model_name, model_cfg in KEY_MODELS.items():
        for seed in SEEDS:
            run_num += 1
            key = f"{model_name}_seed{seed}"
            if key in ms_results:
                print(f"  [SKIP] {key}")
                continue
            t0 = time.time()
            print(f"\n[{run_num}/{total_ms}] {model_name} seed={seed}")
            set_seed(seed)
            model = model_cfg["builder"]().to(device)
            param_count = sum(p.numel() for p in model.parameters())
            config = TrainConfig(model_name=key, epochs=5, seed=seed,
                                 output_dir="paper/results/multi_seed", use_turma_loss=model_cfg["use_turma_loss"])
            train_result = train_model(model, config, train_loader, val_loader, device)
            ckpt = torch.load(train_result["checkpoint"], map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            test_result = evaluate_model(model, test_loader, device, key)
            eff = measure_efficiency(model, device, n_runs=20)
            entry = {
                "model_name": model_name, "description": model_cfg["desc"], "seed": seed,
                "params": param_count,
                "train": {"best_val_f1": train_result["best_val_f1"], "best_epoch": train_result["best_epoch"],
                          "total_time": train_result["total_time"]},
                "test": {
                    "accuracy": test_result.accuracy, "balanced_accuracy": test_result.balanced_accuracy,
                    "macro_f1": test_result.macro_f1, "weighted_f1": test_result.weighted_f1,
                    "roc_auc": test_result.roc_auc, "pr_auc": test_result.pr_auc,
                    "mcc": test_result.mcc, "kappa": test_result.kappa,
                    "per_class_f1": test_result.per_class_f1,
                    "sensitivity": test_result.sensitivity, "specificity": test_result.specificity,
                },
                "efficiency": eff, "confusion_matrix": test_result.confusion_matrix,
                "roc_data": test_result.roc_data, "pr_data": test_result.pr_data,
                "y_true": test_result.y_true, "y_pred": test_result.y_pred, "y_probs": test_result.y_probs,
            }
            ms_results[key] = entry
            with open(ms_path, "w") as f:
                json.dump(ms_results, f, indent=2)
            elapsed = time.time() - t0
            print(f"  Done {elapsed:.0f}s | acc={test_result.accuracy:.4f} f1={test_result.macro_f1:.4f}")

    # =========================================================================
    # PHASE 3: STATISTICAL COMPARISON
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 3: STATISTICAL COMPARISON")
    print("="*70)

    stats = {}
    comparisons = [
        ("A_baseline_densenet121_seed42", "D_tumor_aware_seed42", "Baseline vs TumorAware"),
        ("A_baseline_densenet121_seed42", "E_turma_full_seed42", "Baseline vs TURMA-Net"),
        ("D_tumor_aware_seed42", "E_turma_full_seed42", "TumorAware vs TURMA-Net"),
    ]

    for ka, kb, desc in comparisons:
        if ka in ablation_results and kb in ablation_results:
            ya = np.array(ablation_results[ka]["y_true"]) if "y_true" in ablation_results[ka] else None
            ypa = np.array(ablation_results[ka]["y_probs"]) if "y_probs" in ablation_results[ka] else None
            yb = np.array(ablation_results[kb]["y_true"]) if "y_true" in ablation_results[kb] else None
            ypb = np.array(ablation_results[kb]["y_probs"]) if "y_probs" in ablation_results[kb] else None
            if ya is not None and ypa is not None and yb is not None and ypb is not None:
                bootstrap = paired_bootstrap_test(ya, ypa, ypb)
                effect = compute_effect_size(ya, ypa, ypb)
                stats[desc] = {"bootstrap": bootstrap, "effect_size": effect}
                print(f"  {desc}: diff={bootstrap['original_diff']:+.4f} p={bootstrap['p_value']:.4f} d={effect['cohens_d']:.3f} ({effect['effect_interpretation']})")
            else:
                print(f"  {desc}: SKIPPED (no y_true/y_probs)")
        else:
            print(f"  {desc}: SKIPPED (missing results)")

    with open("paper/results/statistical_comparison.json", "w") as f:
        json.dump(stats, f, indent=2)

    # =========================================================================
    # PHASE 4: EXTERNAL VALIDATION
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 4: EXTERNAL VALIDATION")
    print("="*70)

    ext_results = {}
    external_datasets = {
        "BRISC": "data/external/brisc2025/brisc2025/classification_task/test",
        "Kaggle": "data/external/kaggle_brain_tumor/Testing",
    }

    key_checkpoints = {
        "A_baseline_densenet121": ("A_baseline_densenet121_seed42", lambda: build_baseline("densenet121")),
        "D_tumor_aware": ("D_tumor_aware_seed42", lambda: build_tumor_aware("densenet121")),
        "E_turma_full": ("E_turma_full_seed42", lambda: build_turma_net("densenet121", use_atgfm=True, use_consistency=True, use_uncertainty_weights=True)),
    }

    for ext_name, ext_root in external_datasets.items():
        if not Path(ext_root).exists():
            print(f"  {ext_name}: NOT FOUND")
            continue
        ext_loader = load_external(ext_root, batch_size=8)
        print(f"\n  {ext_name}: {len(ext_loader.dataset)} samples")
        for model_name, (ckpt_key, builder) in key_checkpoints.items():
            ckpt_path = Path("paper/results/multi_seed") / ckpt_key / "best.pt"
            if not ckpt_path.exists():
                ckpt_path = Path("paper/results/ablation") / ckpt_key / "best.pt"
            if not ckpt_path.exists():
                print(f"    {model_name}: NO CHECKPOINT")
                continue
            set_seed(42)
            model = builder().to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            ext_result = evaluate_model(model, ext_loader, device, f"{model_name}_{ext_name}")
            k = f"{model_name}_{ext_name}"
            ext_results[k] = {
                "model": model_name, "dataset": ext_name,
                "accuracy": ext_result.accuracy, "macro_f1": ext_result.macro_f1,
                "roc_auc": ext_result.roc_auc, "per_class_f1": ext_result.per_class_f1,
                "confusion_matrix": ext_result.confusion_matrix,
            }
            print(f"    {model_name}: acc={ext_result.accuracy:.4f} f1={ext_result.macro_f1:.4f} auc={ext_result.roc_auc:.4f}")

    with open("paper/results/external_validation.json", "w") as f:
        json.dump(ext_results, f, indent=2)

    # =========================================================================
    # PHASE 5: GENERATE FIGURES AND TABLES
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 5: GENERATE FIGURES AND TABLES")
    print("="*70)

    import importlib
    spec = importlib.util.spec_from_file_location("gen", "scripts/generate_all_figures.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    all_results_combined = {**ablation_results, **ms_results}
    gen.generate_figures(all_results_combined, ms_results)
    gen.generate_tables(ablation_results)

    total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE at {time.strftime('%H:%M:%S')}")
    print(f"Total time: {total/60:.1f} minutes ({total/3600:.1f} hours)")
    print(f"{'='*70}")

    print("\n--- ABLATION SUMMARY ---")
    for key in sorted(ablation_results.keys()):
        r = ablation_results[key]
        t = r["test"]
        print(f"  {r['experiment']:35s} acc={t['accuracy']:.4f} f1={t['macro_f1']:.4f} auc={t['roc_auc']:.4f}")

    if ms_results:
        print("\n--- MULTI-SEED SUMMARY ---")
        for model_name in KEY_MODELS:
            mrs = {k: v for k, v in ms_results.items() if v.get("model_name") == model_name}
            if mrs:
                accs = [v["test"]["accuracy"] for v in mrs.values()]
                f1s = [v["test"]["macro_f1"] for v in mrs.values()]
                aucs = [v["test"]["roc_auc"] for v in mrs.values()]
                print(f"  {model_name}:")
                print(f"    Acc: {np.mean(accs):.4f}+/-{np.std(accs):.4f}  F1: {np.mean(f1s):.4f}+/-{np.std(f1s):.4f}  AUC: {np.mean(aucs):.4f}+/-{np.std(aucs):.4f}")

    if ext_results:
        print("\n--- EXTERNAL VALIDATION ---")
        for k, v in ext_results.items():
            print(f"  {k}: acc={v['accuracy']:.4f} f1={v['macro_f1']:.4f} auc={v['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
