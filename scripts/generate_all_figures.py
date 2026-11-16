"""Generate all publication figures and tables from saved ablation data."""
import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

CLASS_NAMES = ["glioma", "meningioma", "no tumor", "pituitary"]

def _name(r, key=""):
    return r.get("experiment", r.get("model_name", str(key)))

def load_ablation():
    with open("paper/results/ablation_all.json") as f:
        return json.load(f)

def load_multiseed():
    p = Path("paper/results/multi_seed_all.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

def generate_figures(results, ms_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "figure.dpi": 150, "figure.facecolor": "white",
    })
    figures_dir = Path("paper/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Fig 1: Ablation Accuracy & F1 ---
    fig, ax = plt.subplots(figsize=(14, 5))
    names = []
    accs = []
    f1s = []
    for key in sorted(results.keys()):
        r = results[key]
        names.append(_name(r, key).replace("_", " ").replace("seed42", "").strip())
        accs.append(r["test"]["accuracy"])
        f1s.append(r["test"]["macro_f1"])
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#3498db", edgecolor="white")
    bars2 = ax.bar(x + w/2, f1s, w, label="Macro F1", color="#e74c3c", edgecolor="white")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Accuracy and Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis="y", alpha=0.3)
    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontsize=7)
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig1_ablation_accuracy_f1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Fig 1: Ablation Accuracy & F1")

    # --- Fig 2: Confusion Matrices ---
    key_models = ["A_baseline_densenet121_seed42", "d_tumor_aware_seed42", "e_turma_full_seed42"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, mk in zip(axes, key_models):
        if mk in results:
            cm = np.array(results[mk]["confusion_matrix"])
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            short = _name(results[mk], mk).replace("_", " ").replace("seed42", "").strip()
            ax.set_title(short, fontsize=10)
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=9)
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(["Glio", "Meno", "NoT", "Pit"], fontsize=8)
            ax.set_yticklabels(["Glio", "Meno", "NoT", "Pit"], fontsize=8)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig2_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Fig 2: Confusion Matrices")

    # --- Fig 3: ROC Curves ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    idx_to_name = {str(i): cn for i, cn in enumerate(CLASS_NAMES)}
    for ax, mk in zip(axes, key_models):
        if mk in results:
            roc = results[mk]["roc_data"]
            for idx_str in roc["fpr"]:
                cn = idx_to_name.get(idx_str, idx_str)
                fpr = np.array(roc["fpr"][idx_str])
                tpr = np.array(roc["tpr"][idx_str])
                auc_key = cn if cn in roc["auc"] else idx_str
                auc_val = roc["auc"].get(auc_key, 0)
                ci = int(idx_str) if idx_str.isdigit() else 0
                ax.plot(fpr, tpr, color=colors[ci], label=f"{cn} ({auc_val:.3f})", linewidth=1.5)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
            short = _name(results[mk], mk).replace("_", " ").replace("seed42", "").strip()
            ax.set_title(short, fontsize=10)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig3_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Fig 3: ROC Curves")

    # --- Fig 4: Per-class F1 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    model_keys = sorted(results.keys())
    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    for i, mk in enumerate(model_keys[:3]):
        pc = results[mk]["test"]["per_class_f1"]
        vals = [pc.get(cn, 0) for cn in CLASS_NAMES]
        short = _name(results[mk], mk).replace("_", " ").replace("seed42", "").strip()[:20]
        ax.bar(x + i*width, vals, width, label=short)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend(fontsize=8)
    ax.set_ylim([0, 1.05])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig4_per_class_f1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Fig 4: Per-Class F1")

    # --- Fig 5: Efficiency ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    unique_models = {}
    for key in sorted(results.keys()):
        r = results[key]
        mname = _name(r, key).split("_seed")[0]
        if mname not in unique_models:
            unique_models[mname] = r
    labels = [k.replace("_", " ")[:25] for k in unique_models.keys()]
    params = [v["efficiency"]["params"]/1e6 for v in unique_models.values()]
    lats = [v["efficiency"]["latency_ms"] for v in unique_models.values()]
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    axes[0].barh(labels, params, color=colors)
    axes[0].set_xlabel("Parameters (M)")
    axes[0].set_title("Model Size")
    axes[1].barh(labels, lats, color=colors)
    axes[1].set_xlabel("Latency (ms, CPU)")
    axes[1].set_title("Inference Speed")
    plt.tight_layout()
    plt.savefig(figures_dir / "fig5_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Fig 5: Efficiency")

    # --- Fig 6: Ablation Delta ---
    baseline_key = "C_baseline_efficientnet_seed42"
    baseline_acc = results.get(baseline_key, {}).get("test", {}).get("accuracy", 0.9682)
    baseline_f1 = results.get(baseline_key, {}).get("test", {}).get("macro_f1", 0.9693)
    turma_keys = [k for k in sorted(results.keys()) if "turma" in k.lower() or "tumor_aware" in k.lower()]
    fig, ax = plt.subplots(figsize=(10, 5))
    names2 = []
    d_accs = []
    d_f1s = []
    for mk in turma_keys:
        r = results[mk]
        names2.append(_name(r, mk).replace("_", " ").replace("seed42", "").strip())
        d_accs.append(r["test"]["accuracy"] - baseline_acc)
        d_f1s.append(r["test"]["macro_f1"] - baseline_f1)
    x2 = np.arange(len(names2))
    ax.bar(x2 - 0.2, d_accs, 0.35, label="Δ Accuracy", color="#3498db")
    ax.bar(x2 + 0.2, d_f1s, 0.35, label="Δ F1", color="#e74c3c")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Difference vs Best Baseline (EfficientNetV2-S)")
    ax.set_title("Multi-Task Models: Performance Delta vs Best Baseline")
    ax.set_xticks(x2)
    ax.set_xticklabels(names2, fontsize=8, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "fig6_ablation_delta.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Fig 6: Ablation Delta")

    print(f"\nAll figures saved to {figures_dir}/")


def generate_tables(results):
    tables_dir = Path("paper/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # --- Table 1: Main comparison ---
    lines = ["| Model | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | ROC-AUC | PR-AUC | MCC | Params (M) | Latency (ms) |",
             "|-------|----------|--------------|----------|-------------|---------|--------|-----|------------|--------------|"]
    for key in sorted(results.keys()):
        r = results[key]
        t = r["test"]
        e = r["efficiency"]
        lines.append(
            f"| {_name(r, key)} | {t['accuracy']:.4f} | {t['balanced_accuracy']:.4f} | "
            f"{t['macro_f1']:.4f} | {t['weighted_f1']:.4f} | {t['roc_auc']:.4f} | "
            f"{t['pr_auc']:.4f} | {t['mcc']:.4f} | {e['params']/1e6:.2f} | {e['latency_ms']:.1f} |"
        )
    with open(tables_dir / "table1_main_comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  Table 1: Main Comparison")

    # --- Table 2: Per-class F1 ---
    lines2 = ["| Model | Glioma | Meningioma | No Tumor | Pituitary |",
              "|-------|--------|------------|----------|-----------|"]
    for key in sorted(results.keys()):
        r = results[key]
        pc = r["test"]["per_class_f1"]
        lines2.append(
            f"| {_name(r, key)} | {pc.get('glioma', 0):.4f} | {pc.get('meningioma', 0):.4f} | "
            f"{pc.get('no tumor', 0):.4f} | {pc.get('pituitary', 0):.4f} |"
        )
    with open(tables_dir / "table2_per_class_f1.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines2))
    print("  Table 2: Per-Class F1")

    # --- Table 3: Ablation Delta ---
    baseline_key = "C_baseline_efficientnet_seed42"
    b_acc = results.get(baseline_key, {}).get("test", {}).get("accuracy", 0)
    b_f1 = results.get(baseline_key, {}).get("test", {}).get("macro_f1", 0)
    lines3 = ["| Configuration | Accuracy | Macro F1 | Delta Accuracy | Delta F1 |",
              "|---------------|----------|----------|----------------|----------|"]
    for key in sorted(results.keys()):
        r = results[key]
        t = r["test"]
        lines3.append(f"| {_name(r, key)} | {t['accuracy']:.4f} | {t['macro_f1']:.4f} | {t['accuracy']-b_acc:+.4f} | {t['macro_f1']-b_f1:+.4f} |")
    with open(tables_dir / "table3_ablation_delta.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines3))
    print("  Table 3: Ablation Delta")

    print(f"\nAll tables saved to {tables_dir}/")


if __name__ == "__main__":
    results = load_ablation()
    ms = load_multiseed()
    print(f"Loaded {len(results)} ablation + {len(ms)} multi-seed results")
    print("\nGenerating figures...")
    generate_figures(results, ms)
    print("\nGenerating tables...")
    generate_tables(results)
    print("\nDone!")
