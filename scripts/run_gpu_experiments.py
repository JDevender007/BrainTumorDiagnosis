"""Full GPU experiments with epoch-by-epoch progress."""
import sys, json, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_all_experiments import *
from utils.seed import set_seed
import torch
import numpy as np

PHASE = sys.argv[1] if len(sys.argv) > 1 else "ablation"

def elapsed_str(t):
    m, s = divmod(int(time.time() - t), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"

def train_with_epoch_log(model, config, train_loader, val_loader, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    if config.use_turma_loss:
        from models.novel.losses import TurmaNetLoss
        criterion = TurmaNetLoss(num_classes=4, use_uncertainty_weights=True)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = torch.amp.GradScaler("cuda")
    output_dir = Path(config.output_dir) / config.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    t_start = time.time()

    for epoch in range(config.epochs):
        t_epoch = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                if config.use_turma_loss:
                    outputs = model(images, return_seg=True)
                else:
                    outputs = model(images)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                if config.use_turma_loss and isinstance(outputs, dict) and "seg_logits" in outputs:
                    loss, _ = criterion(logits, labels, outputs["seg_logits"], None)
                else:
                    loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                preds = logits.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)

        epoch_sec = time.time() - t_epoch
        total_elapsed = elapsed_str(t_start)

        marker = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            marker = " *BEST*"
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "val_f1": val_f1, "val_accuracy": val_acc}, output_dir / "best.pt")
        else:
            patience_counter += 1

        print(f"    Ep {epoch+1:2d}/{config.epochs} | loss={avg_loss:.4f} val_f1={val_f1:.4f} val_acc={val_acc:.4f} | {epoch_sec:.0f}s | total={total_elapsed}{marker}", flush=True)

        history.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_f1": val_f1, "val_accuracy": val_acc})

        if patience_counter >= config.patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    torch.save({"epoch": history[-1]["epoch"], "model_state_dict": model.state_dict()}, output_dir / "last.pt")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {"best_val_f1": best_val_f1, "best_epoch": best_epoch, "total_time": time.time() - t_start,
            "history": history, "checkpoint": str(output_dir / "best.pt")}


def run_ablation_gpu():
    device = get_device()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    t_global = time.time()

    set_seed(42)
    train_loader, val_loader, test_loader = load_data("data/raw/classification", batch_size=8, seed=42)
    print(f"Data: train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    results = {}
    partial = Path("paper/results/ablation_all.json")
    if partial.exists():
        results = json.load(open(partial))

    remaining = [(k, v) for k, v in ABLATION_CONFIGS.items() if f"{k}_seed42" not in results]
    total = len(ABLATION_CONFIGS)
    done = total - len(remaining)

    print(f"\n{'='*70}")
    print(f"ABLATION: {done}/{total} done, {len(remaining)} to train (10 epochs each)")
    print(f"{'='*70}\n")

    for i, (exp_name, exp_cfg) in enumerate(remaining):
        key = f"{exp_name}_seed42"
        t0 = time.time()
        print(f"[{done+i+1}/{total}] {exp_name}: {exp_cfg['desc']}")
        print(f"  Started: {time.strftime('%H:%M:%S')}")

        set_seed(42)
        model = exp_cfg["builder"]().to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {params/1e6:.2f}M")

        config = TrainConfig(model_name=key, epochs=10, seed=42,
                             output_dir="paper/results/ablation", use_turma_loss=exp_cfg["use_turma_loss"])
        train_result = train_with_epoch_log(model, config, train_loader, val_loader, device)

        ckpt = torch.load(train_result["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_result = evaluate_model(model, test_loader, device, key)
        eff = measure_efficiency(model, device, n_runs=20)

        entry = {
            "experiment": exp_name, "description": exp_cfg["desc"], "seed": 42, "params": params,
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
        results[key] = entry
        with open(partial, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  RESULT: acc={test_result.accuracy:.4f} f1={test_result.macro_f1:.4f} auc={test_result.roc_auc:.4f}")
        print(f"  Finished: {time.strftime('%H:%M:%S')} ({elapsed_str(t0)})")
        print(f"  Progress: {done+i+1}/{total} complete, total elapsed: {elapsed_str(t_global)}\n")

    print(f"\n{'='*70}")
    print(f"ABLATION COMPLETE ({elapsed_str(t_global)})")
    print(f"{'='*70}")
    print(f"\n{'Model':35s} {'Acc':>8s} {'F1':>8s} {'AUC':>8s}")
    print("-" * 63)
    for key in sorted(results.keys()):
        r = results[key]
        t = r["test"]
        print(f"  {r['experiment']:33s} {t['accuracy']:.4f} {t['macro_f1']:.4f} {t['roc_auc']:.4f}")


def run_multiseed_gpu():
    device = get_device()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    t_global = time.time()

    set_seed(42)
    train_loader, val_loader, test_loader = load_data("data/raw/classification", batch_size=8, seed=42)

    KEY_MODELS = {
        "A_baseline_densenet121": {"desc": "Baseline: DenseNet121",
            "builder": lambda: build_baseline("densenet121"), "use_turma_loss": False},
        "D_tumor_aware": {"desc": "TumorAwareMultiTaskNet",
            "builder": lambda: build_tumor_aware("densenet121"), "use_turma_loss": False},
        "E_turma_full": {"desc": "TURMA-Net Full",
            "builder": lambda: build_turma_net("densenet121", use_atgfm=True, use_consistency=True, use_uncertainty_weights=True),
            "use_turma_loss": True},
    }
    SEEDS = [42, 123, 456]

    ms_results = {}
    ms_path = Path("paper/results/multi_seed_all.json")
    if ms_path.exists():
        ms_results = json.load(open(ms_path))

    total = len(KEY_MODELS) * len(SEEDS)
    done = sum(1 for mn in KEY_MODELS for s in SEEDS if f"{mn}_seed{s}" in ms_results)

    print(f"\n{'='*70}")
    print(f"MULTI-SEED: {done}/{total} done, {total-done} to train (10 epochs each)")
    print(f"{'='*70}\n")

    run_num = 0
    for model_name, model_cfg in KEY_MODELS.items():
        for seed in SEEDS:
            run_num += 1
            key = f"{model_name}_seed{seed}"
            if key in ms_results:
                print(f"[{run_num}/{total}] [SKIP] {key}")
                continue

            t0 = time.time()
            print(f"\n[{run_num}/{total}] {model_name} seed={seed}")
            print(f"  Started: {time.strftime('%H:%M:%S')}")
            set_seed(seed)
            model = model_cfg["builder"]().to(device)
            params = sum(p.numel() for p in model.parameters())
            config = TrainConfig(model_name=key, epochs=10, seed=seed,
                                 output_dir="paper/results/multi_seed", use_turma_loss=model_cfg["use_turma_loss"])
            train_result = train_with_epoch_log(model, config, train_loader, val_loader, device)

            ckpt = torch.load(train_result["checkpoint"], map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            test_result = evaluate_model(model, test_loader, device, key)
            eff = measure_efficiency(model, device, n_runs=20)

            entry = {
                "model_name": model_name, "description": model_cfg["desc"], "seed": seed, "params": params,
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
            print(f"  RESULT: acc={test_result.accuracy:.4f} f1={test_result.macro_f1:.4f}")
            print(f"  Finished: {time.strftime('%H:%M:%S')} ({elapsed_str(t0)})\n")

    print(f"\nMULTI-SEED COMPLETE ({elapsed_str(t_global)})")
    for mn in KEY_MODELS:
        mrs = {k: v for k, v in ms_results.items() if v.get("model_name") == mn}
        if mrs:
            accs = [v["test"]["accuracy"] for v in mrs.values()]
            f1s = [v["test"]["macro_f1"] for v in mrs.values()]
            aucs = [v["test"]["roc_auc"] for v in mrs.values()]
            print(f"  {mn}:")
            print(f"    Acc:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
            print(f"    F1:   {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
            print(f"    AUC:  {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")


def run_external_gpu():
    device = get_device()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    t_global = time.time()

    set_seed(42)
    _, _, _ = load_data("data/raw/classification", batch_size=8, seed=42)

    ext_results = {}
    datasets = {
        "BRISC": "data/external/brisc2025/brisc2025/classification_task/test",
        "Kaggle": "data/external/kaggle_brain_tumor/Testing",
    }
    models_to_eval = {
        "A_baseline_densenet121": ("A_baseline_densenet121_seed42", lambda: build_baseline("densenet121")),
        "D_tumor_aware": ("D_tumor_aware_seed42", lambda: build_tumor_aware("densenet121")),
        "E_turma_full": ("E_turma_full_seed42", lambda: build_turma_net("densenet121", use_atgfm=True, use_consistency=True, use_uncertainty_weights=True)),
    }

    for ext_name, ext_root in datasets.items():
        if not Path(ext_root).exists():
            print(f"\n  {ext_name}: NOT FOUND at {ext_root}")
            continue
        ext_loader = load_external(ext_root, batch_size=8)
        print(f"\n{'='*70}")
        print(f"EXTERNAL: {ext_name} ({len(ext_loader.dataset)} samples)")
        print(f"{'='*70}")

        for model_name, (ckpt_key, builder) in models_to_eval.items():
            ckpt_path = Path("paper/results/multi_seed") / ckpt_key / "best.pt"
            if not ckpt_path.exists():
                ckpt_path = Path("paper/results/ablation") / ckpt_key / "best.pt"
            if not ckpt_path.exists():
                print(f"  {model_name}: NO CHECKPOINT")
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
            print(f"  {model_name:30s} acc={ext_result.accuracy:.4f} f1={ext_result.macro_f1:.4f} auc={ext_result.roc_auc:.4f}")

    with open("paper/results/external_validation.json", "w") as f:
        json.dump(ext_results, f, indent=2)
    print(f"\nExternal validation complete ({elapsed_str(t_global)})")


if __name__ == "__main__":
    print(f"GPU Experiment Runner")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Phase: {PHASE}")
    print()

    if PHASE == "ablation":
        run_ablation_gpu()
    elif PHASE == "multiseed":
        run_multiseed_gpu()
    elif PHASE == "external":
        run_external_gpu()
    elif PHASE == "all":
        run_ablation_gpu()
        run_multiseed_gpu()
        run_external_gpu()

        print(f"\n{'='*70}")
        print("REGENERATING FIGURES AND TABLES")
        print(f"{'='*70}")
        import importlib
        spec = importlib.util.spec_from_file_location("gen", "scripts/generate_all_figures.py")
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        ab = json.load(open("paper/results/ablation_all.json")) if Path("paper/results/ablation_all.json").exists() else {}
        ms = json.load(open("paper/results/multi_seed_all.json")) if Path("paper/results/multi_seed_all.json").exists() else {}
        gen.generate_figures({**ab, **ms}, ms)
        gen.generate_tables(ab)
