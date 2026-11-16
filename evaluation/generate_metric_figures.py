"""
Generate comprehensive metric visualizations for all backbones.
Saves charts to outputs/comprehensive_metrics/figures/
"""
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path("outputs/comprehensive_metrics/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
BACKBONE_NAMES = {
    "cnn2d": "CNN-2D",
    "densenet121": "DenseNet121",
    "resnet50": "ResNet50",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_t": "Swin-T",
    "vit_b_16": "ViT-B/16",
}
COLORS = {
    "cnn2d": "#2ecc71",
    "densenet121": "#3498db",
    "resnet50": "#e74c3c",
    "efficientnet_v2_s": "#f39c12",
    "convnext_tiny": "#9b59b6",
    "swin_t": "#1abc9c",
    "vit_b_16": "#e67e22",
}


def load_results():
    all_path = Path("outputs/comprehensive_metrics/all_backbones_comprehensive.json")
    if not all_path.exists():
        print("Run evaluation/comprehensive_metrics.py first")
        return {}
    with open(all_path) as f:
        return json.load(f)


def get_internal(r):
    return r.get("internal_test", {})


def get_external(r):
    return r.get("external_test_brisc2025", {})


def get_inference(r):
    return r.get("inference_speed", {})


def get_robustness(r):
    return r.get("robustness", {})


def plot_internal_vs_external_accuracy(results):
    fig, ax = plt.subplots(figsize=(12, 6))
    backbones = [b for b in results if b != "metadata"]
    names = [BACKBONE_NAMES.get(b, b) for b in backbones]
    internal = [get_internal(results[b])["accuracy"] * 100 for b in backbones]
    external = [get_external(results[b])["accuracy"] * 100 for b in backbones]

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, internal, width, label="Internal Test (1,857)", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width/2, external, width, label="BRISC 2025 External (1,000)", color="#e74c3c", alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Classification Accuracy: Internal vs External Test Set", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(50, 102)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_accuracy_internal_vs_external.png", dpi=150)
    plt.close()
    print("Saved: 01_accuracy_internal_vs_external.png")


def plot_comprehensive_metrics_radar(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    backbones = [b for b in results if b != "metadata"]

    for idx, (get_fn, title) in enumerate([(get_internal, "Internal Test Set"), (get_external, "BRISC 2025 External")]):
        ax = axes[idx]
        metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "specificity_macro", "roc_auc_macro", "matthews_corrcoef", "cohen_kappa"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "ROC-AUC", "MCC", "Kappa"]

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        for backbone in backbones:
            data = get_fn(results[backbone])
            values = [data.get(m, 0) for m in metrics]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=1.5, label=BACKBONE_NAMES.get(backbone, backbone),
                    color=COLORS.get(backbone, '#333'), markersize=4)
            ax.fill(angles, values, alpha=0.05, color=COLORS.get(backbone, '#333'))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0.4, 1.05)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        if idx == 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)

    plt.suptitle("Comprehensive Metrics Comparison - Radar Charts", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_radar_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_radar_comparison.png")


def plot_f1_heatmap(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    backbones = [b for b in results if b != "metadata"]
    names = [BACKBONE_NAMES.get(b, b) for b in backbones]

    internal_f1 = []
    external_f1 = []
    for backbone in backbones:
        int_per_class = get_internal(results[backbone]).get("per_class", {})
        ext_per_class = get_external(results[backbone]).get("per_class", {})
        internal_f1.append([int_per_class.get(c, {}).get("f1", 0) for c in CLASS_NAMES])
        external_f1.append([ext_per_class.get(c, {}).get("f1", 0) for c in CLASS_NAMES])

    data = np.array(internal_f1 + external_f1)
    split_labels = names + [n + " (Ext)" for n in names]

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax.set_yticks(range(len(split_labels)))
    ax.set_yticklabels(split_labels, fontsize=9)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.3f}", ha='center', va='center', fontsize=9,
                    color='black' if data[i,j] > 0.6 else 'white', fontweight='bold')

    ax.set_title("Per-Class F1-Score Heatmap", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='F1-Score', shrink=0.8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_f1_heatmap.png", dpi=150)
    plt.close()
    print("Saved: 03_f1_heatmap.png")


def plot_noise_robustness(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    noise_levels = [0.0, 0.05, 0.1, 0.2]
    noise_labels = ["Clean", "σ=0.05", "σ=0.1", "σ=0.2"]
    backbones = [b for b in results if b != "metadata"]

    for idx, (get_fn, title) in enumerate([(get_internal, "Internal Test Set"), (get_external, "BRISC 2025 External")]):
        ax = axes[idx]
        for backbone in backbones:
            accs = []
            for noise in noise_levels:
                if noise == 0.0:
                    accs.append(get_fn(results[backbone])["accuracy"] * 100)
                else:
                    rob = get_robustness(results[backbone])
                    key = f"noise_std_{noise}"
                    accs.append(rob.get(key, {}).get("accuracy", 0) * 100)
            ax.plot(noise_labels, accs, 'o-', linewidth=2, markersize=6,
                    label=BACKBONE_NAMES.get(backbone, backbone), color=COLORS.get(backbone, '#333'))

        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        if idx == 1:
            ax.legend(loc='lower left', fontsize=8)

    plt.suptitle("Robustness to Gaussian Noise", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_noise_robustness.png", dpi=150)
    plt.close()
    print("Saved: 04_noise_robustness.png")


def plot_inference_speed(results):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    backbones = [b for b in results if b != "metadata"]
    names = [BACKBONE_NAMES.get(b, b) for b in backbones]
    latencies = [get_inference(results[b])["mean_ms"] for b in backbones]
    fps = [get_inference(results[b])["fps"] for b in backbones]
    params = [results[b]["model_info"]["total_params"] / 1e6 for b in backbones]

    x = np.arange(len(names))
    bars = ax1.bar(x, latencies, 0.6, color=[COLORS.get(b, '#333') for b in backbones], alpha=0.85)
    for bar, lat in zip(bars, latencies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f"{lat:.1f}ms", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel("Latency (ms)", fontsize=11)
    ax1.set_title("Inference Speed & Model Size", fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, params, 'D-', color='navy', linewidth=2, markersize=8, label='Parameters (M)')
    ax2.set_ylabel("Parameters (Millions)", fontsize=11, color='navy')
    ax2.tick_params(axis='y', labelcolor='navy')

    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines2, labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_inference_speed.png", dpi=150)
    plt.close()
    print("Saved: 05_inference_speed.png")


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    backbones = [b for b in results if b != "metadata"]

    for idx, backbone in enumerate(backbones):
        row, col = divmod(idx, 4)
        ax = axes[row][col]
        cm = np.array(get_internal(results[backbone]).get("confusion_matrix", [[0]*4]*4))
        if cm.sum() == 0:
            ax.set_visible(False)
            continue

        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(BACKBONE_NAMES.get(backbone, backbone), fontsize=11, fontweight='bold')
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(["G", "M", "P", "N"], fontsize=9)
        ax.set_yticklabels(["G", "M", "P", "N"], fontsize=9)

        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i,j]}", ha='center', va='center', fontsize=8,
                        color='white' if cm_norm[i,j] > 0.5 else 'black')

    axes[1][3].set_visible(False)
    plt.suptitle("Confusion Matrices (Internal Test Set)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_confusion_matrices.png", dpi=150)
    plt.close()
    print("Saved: 06_confusion_matrices.png")


def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    backbones = ["cnn2d", "densenet121", "resnet50", "efficientnet_v2_s", "convnext_tiny", "swin_t", "vit_b_16"]

    for backbone in backbones:
        history_path = Path(f"outputs/{backbone}_training/history.json")
        if not history_path.exists():
            continue
        with open(history_path) as f:
            history = json.load(f)

        val_acc = [e.get("val_acc", 0) for e in history]
        epochs = range(1, len(val_acc) + 1)
        axes[0].plot(epochs, val_acc, 'o-', linewidth=1.5, markersize=3,
                     label=BACKBONE_NAMES.get(backbone, backbone), color=COLORS.get(backbone, '#333'))

    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Validation Accuracy (%)", fontsize=11)
    axes[0].set_title("Validation Accuracy Curves", fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for backbone in backbones:
        history_path = Path(f"outputs/{backbone}_training/history.json")
        if not history_path.exists():
            continue
        with open(history_path) as f:
            history = json.load(f)

        val_loss = [e.get("val_loss", 0) for e in history]
        epochs = range(1, len(val_loss) + 1)
        axes[1].plot(epochs, val_loss, 'o-', linewidth=1.5, markersize=3,
                     label=BACKBONE_NAMES.get(backbone, backbone), color=COLORS.get(backbone, '#333'))

    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Validation Loss", fontsize=11)
    axes[1].set_title("Validation Loss Curves", fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training History - All Backbones", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_training_curves.png", dpi=150)
    plt.close()
    print("Saved: 07_training_curves.png")


def plot_metric_comparison_bar(results):
    metrics_to_plot = [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision"),
        ("recall_macro", "Recall"),
        ("f1_macro", "F1-Score"),
        ("specificity_macro", "Specificity"),
        ("roc_auc_macro", "ROC-AUC"),
        ("pr_auc_macro", "PR-AUC"),
        ("matthews_corrcoef", "MCC"),
        ("cohen_kappa", "Kappa"),
        ("log_loss", "LogLoss"),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    backbones = [b for b in results if b != "metadata"]
    names = [BACKBONE_NAMES.get(b, b) for b in backbones]
    x = np.arange(len(names))

    for idx, (metric, label) in enumerate(metrics_to_plot):
        ax = axes[idx // 5][idx % 5]
        internal_vals = [get_internal(results[b]).get(metric, 0) for b in backbones]
        external_vals = [get_external(results[b]).get(metric, 0) for b in backbones]

        if metric == "log_loss":
            internal_vals = [min(v, 2) for v in internal_vals]
            external_vals = [min(v, 2) for v in external_vals]

        width = 0.35
        ax.bar(x - width/2, internal_vals, width, label="Internal", color="#3498db", alpha=0.85)
        ax.bar(x + width/2, external_vals, width, label="External", color="#e74c3c", alpha=0.85)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.grid(axis='y', alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    plt.suptitle("All Metrics Comparison - Internal vs External", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_all_metrics_comparison.png", dpi=150)
    plt.close()
    print("Saved: 08_all_metrics_comparison.png")


def plot_efficiency_frontier(results):
    fig, ax = plt.subplots(figsize=(10, 7))
    backbones = [b for b in results if b != "metadata"]

    for backbone in backbones:
        fps = get_inference(results[backbone])["fps"]
        acc = get_internal(results[backbone])["accuracy"] * 100
        params = results[backbone]["model_info"]["total_params"] / 1e6
        size_mb = results[backbone]["model_info"]["model_size_mb"]

        ax.scatter(fps, acc, s=200, color=COLORS.get(backbone, '#333'), alpha=0.85, edgecolors='black', linewidth=1)
        ax.annotate(f"{BACKBONE_NAMES.get(backbone, backbone)}\n{params:.0f}M / {size_mb:.0f}MB",
                   (fps, acc), fontsize=8, ha='center', va='bottom',
                   xytext=(0, 10), textcoords='offset points')

    ax.set_xlabel("Inference FPS (higher = faster)", fontsize=12)
    ax.set_ylabel("Internal Test Accuracy (%)", fontsize=12)
    ax.set_title("Efficiency Frontier: Accuracy vs Speed", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 600)
    ax.set_ylim(55, 100)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_efficiency_frontier.png", dpi=150)
    plt.close()
    print("Saved: 09_efficiency_frontier.png")


def plot_summary_table(results):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')

    backbones = [b for b in results if b != "metadata"]
    col_labels = ["Backbone", "Params", "Size", "FPS",
                  "Int Acc", "Int F1", "Ext Acc", "Ext F1",
                  "ROC-AUC", "MCC", "Kappa"]

    cell_data = []
    for backbone in backbones:
        r = results[backbone]
        cell_data.append([
            BACKBONE_NAMES.get(backbone, backbone),
            f"{r['model_info']['total_params']/1e6:.1f}M",
            f"{r['model_info']['model_size_mb']:.0f}MB",
            f"{get_inference(r)['fps']:.0f}",
            f"{get_internal(r)['accuracy']*100:.1f}%",
            f"{get_internal(r)['f1_macro']:.3f}",
            f"{get_external(r)['accuracy']*100:.1f}%",
            f"{get_external(r)['f1_macro']:.3f}",
            f"{get_internal(r)['roc_auc_macro']:.4f}",
            f"{get_internal(r)['matthews_corrcoef']:.3f}",
            f"{get_internal(r)['cohen_kappa']:.3f}",
        ])

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#34495e')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        cell.set_edgecolor('#bdc3c7')

    ax.set_title("Comprehensive Results Summary Table", fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 10_summary_table.png")


def plot_per_class_f1_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    backbones = [b for b in results if b != "metadata"]
    names = [BACKBONE_NAMES.get(b, b) for b in backbones]
    x = np.arange(len(names))
    width = 0.2

    for idx, (get_fn, title) in enumerate([(get_internal, "Internal Test Set"), (get_external, "BRISC 2025 External")]):
        ax = axes[idx]
        for i, cls in enumerate(CLASS_NAMES):
            f1s = [get_fn(results[b]).get("per_class", {}).get(cls, {}).get("f1", 0) for b in backbones]
            ax.bar(x + i*width - 1.5*width, f1s, width, label=cls, alpha=0.85)

        ax.set_ylabel("F1-Score", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0.3, 1.05)

    plt.suptitle("Per-Class F1-Score by Backbone", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "11_per_class_f1.png", dpi=150)
    plt.close()
    print("Saved: 11_per_class_f1.png")


if __name__ == "__main__":
    print("Generating metric visualizations...")
    print("=" * 60)

    results = load_results()
    if not results:
        exit(1)

    plot_internal_vs_external_accuracy(results)
    plot_comprehensive_metrics_radar(results)
    plot_f1_heatmap(results)
    plot_noise_robustness(results)
    plot_inference_speed(results)
    plot_confusion_matrices(results)
    plot_training_curves()
    plot_metric_comparison_bar(results)
    plot_efficiency_frontier(results)
    plot_summary_table(results)
    plot_per_class_f1_comparison(results)

    print("=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
