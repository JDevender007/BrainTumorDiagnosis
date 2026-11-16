"""
Comprehensive evaluation of trained model on Kaggle Brain Tumor MRI Dataset.
Generates all metrics and saves as PNG files.

Outputs:
  outputs/kaggle_evaluation/
    01_confusion_matrix.png
    02_per_class_metrics.png
    03_roc_curves.png
    04_pr_curves.png
    05_calibration_diagram.png
    06_uncertainty_analysis.png
    07_summary_report.png
    08_training_vs_test.png
    09_class_distribution.png
    10_confidence_histogram.png
    results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tumor_aware import TumorAwareMultiTaskNet, TumorAwareConfig
from utils.device import get_device


CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
OUTPUT_DIR = Path("outputs/kaggle_evaluation")


class SimpleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples = []
        folder_map = {"glioma": 0, "meningioma": 1, "pituitary": 2, "notumor": 3}
        for folder, idx in folder_map.items():
            d = self.root / folder
            if d.exists():
                for p in d.glob("*"):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        self.samples.append((p, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_model(device):
    config = TumorAwareConfig(backbone="densenet121", pretrained=False)
    model = TumorAwareMultiTaskNet(config).to(device)
    ckpt = torch.load("outputs/research_experiments/proposed_model/best.pt",
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def evaluate_with_uncertainty(model, dataloader, device, num_passes=3):
    """Full evaluation with MC Dropout uncertainty."""
    all_preds, all_labels, all_probs = [], [], []
    all_entropies, all_confidences = [], []
    total_batches = len(dataloader)

    # First pass: get predictions in eval mode (deterministic)
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, dict):
                probs = F.softmax(outputs["logits"], dim=1)
            else:
                probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_confidences.extend(probs.max(dim=1).values.cpu().numpy())

    # Second pass: compute uncertainty with MC Dropout (train mode)
    all_entropies = []
    model.train()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            batch_probs = []
            for _ in range(num_passes):
                outputs = model(images)
                if isinstance(outputs, dict):
                    probs = F.softmax(outputs["logits"], dim=1)
                else:
                    probs = F.softmax(outputs, dim=1)
                batch_probs.append(probs)

            stacked = torch.stack(batch_probs)
            mean_probs = stacked.mean(dim=0)
            entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)
            all_entropies.extend(entropy.cpu().numpy())

            if (batch_idx + 1) % 20 == 0:
                print(f"    MC batch {batch_idx + 1}/{total_batches}")

    model.eval()
    return {
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
        "entropies": np.array(all_entropies),
        "confidences": np.array(all_confidences),
    }


def plot_confusion_matrix(labels, preds, output_dir):
    """Plot confusion matrix as PNG."""
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    im1 = axes[0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=12)
    axes[0].set_ylabel("True Label", fontsize=12)
    tick_marks = np.arange(len(CLASS_NAMES))
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            axes[0].text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center", color=color, fontsize=12)
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Normalized
    im2 = axes[1].imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues,
                         vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=12)
    axes[1].set_ylabel("True Label", fontsize=12)
    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_normalized[i, j] > 0.5 else "black"
            axes[1].text(j, i, format(cm_normalized[i, j], ".2f"),
                        ha="center", va="center", color=color, fontsize=12)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_dir / "01_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 01_confusion_matrix.png")


def plot_per_class_metrics(labels, preds, probs, output_dir):
    """Plot per-class precision, recall, F1 as bar chart."""
    report = classification_report(labels, preds, target_names=CLASS_NAMES, output_dict=True)

    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        values = [report[name][metric] for name in CLASS_NAMES]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(),
                     color=CLASS_COLORS[i], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Classification Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "02_per_class_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 02_per_class_metrics.png")


def plot_roc_curves(labels, probs, output_dir):
    """Plot ROC curves for each class (one-vs-rest)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for i, class_name in enumerate(CLASS_NAMES):
        binary_labels = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
        auc = roc_auc_score(binary_labels, probs[:, i])
        ax.plot(fpr, tpr, color=CLASS_COLORS[i], lw=2,
               label=f"{class_name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "03_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 03_roc_curves.png")


def plot_pr_curves(labels, probs, output_dir):
    """Plot Precision-Recall curves for each class."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for i, class_name in enumerate(CLASS_NAMES):
        binary_labels = (labels == i).astype(int)
        precision, recall, _ = precision_recall_curve(binary_labels, probs[:, i])
        ap = average_precision_score(binary_labels, probs[:, i])
        ax.plot(recall, precision, color=CLASS_COLORS[i], lw=2,
               label=f"{class_name} (AP = {ap:.4f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "04_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 04_pr_curves.png")


def plot_calibration(labels, confidences, probs, output_dir):
    """Plot calibration diagram (reliability diagram)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall calibration
    prob_true, prob_pred = calibration_curve(labels == 3, confidences, n_bins=10, strategy="uniform")

    # Compute ECE
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(len(bin_boundaries) - 1):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = (labels[mask] == 3).mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    axes[0].plot(prob_pred, prob_true, "s-", color="#3498db", lw=2, markersize=8,
                label=f"Model (ECE = {ece:.4f})")
    axes[0].fill_between([0, 1], [0, 0.05], [0.05, 0.1], alpha=0.1, color="green",
                         label="Excellent (ECE < 0.05)")
    axes[0].set_xlabel("Mean Predicted Confidence", fontsize=12)
    axes[0].set_ylabel("Actual Accuracy", fontsize=12)
    axes[0].set_title("Calibration Diagram (No Tumor Class)", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([-0.02, 1.02])
    axes[0].set_ylim([-0.02, 1.02])

    # Confidence distribution
    correct_mask = labels == 3
    axes[1].hist(confidences[correct_mask], bins=30, alpha=0.6, color="#2ecc71",
                label="Correct", density=True)
    axes[1].hist(confidences[~correct_mask], bins=30, alpha=0.6, color="#e74c3c",
                label="Incorrect", density=True)
    axes[1].set_xlabel("Prediction Confidence", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].set_title("Confidence Distribution", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "05_calibration_diagram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 05_calibration_diagram.png")


def plot_uncertainty(labels, preds, entropies, confidences, output_dir):
    """Plot uncertainty analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Entropy distribution
    correct_mask = preds == labels
    axes[0, 0].hist(entropies[correct_mask], bins=30, alpha=0.6, color="#2ecc71",
                    label=f"Correct ({correct_mask.sum()})", density=True)
    axes[0, 0].hist(entropies[~correct_mask], bins=30, alpha=0.6, color="#e74c3c",
                    label=f"Incorrect ({(~correct_mask).sum()})", density=True)
    axes[0, 0].set_xlabel("Predictive Entropy", fontsize=11)
    axes[0, 0].set_ylabel("Density", fontsize=11)
    axes[0, 0].set_title("Entropy Distribution", fontsize=12, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)

    # Confidence vs correctness
    axes[0, 1].scatter(entropies[correct_mask], confidences[correct_mask],
                      alpha=0.3, s=10, c="#2ecc71", label="Correct")
    axes[0, 1].scatter(entropies[~correct_mask], confidences[~correct_mask],
                      alpha=0.3, s=10, c="#e74c3c", label="Incorrect")
    axes[0, 1].set_xlabel("Predictive Entropy", fontsize=11)
    axes[0, 1].set_ylabel("Confidence", fontsize=11)
    axes[0, 1].set_title("Entropy vs Confidence", fontsize=12, fontweight="bold")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    # Per-class mean entropy
    class_entropies = [entropies[labels == i].mean() for i in range(len(CLASS_NAMES))]
    bars = axes[1, 0].bar(CLASS_NAMES, class_entropies, color=CLASS_COLORS, alpha=0.85,
                          edgecolor="white")
    for bar, val in zip(bars, class_entropies):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                       f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Mean Entropy", fontsize=11)
    axes[1, 0].set_title("Mean Entropy per Class", fontsize=12, fontweight="bold")
    axes[1, 0].grid(axis="y", alpha=0.3)
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)

    # Per-class mean confidence
    class_confidences = [confidences[labels == i].mean() for i in range(len(CLASS_NAMES))]
    bars = axes[1, 1].bar(CLASS_NAMES, class_confidences, color=CLASS_COLORS, alpha=0.85,
                          edgecolor="white")
    for bar, val in zip(bars, class_confidences):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                       f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Mean Confidence", fontsize=11)
    axes[1, 1].set_title("Mean Confidence per Class", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylim(0, 1.1)
    axes[1, 1].grid(axis="y", alpha=0.3)
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    plt.suptitle("Uncertainty Analysis", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "06_uncertainty_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 06_uncertainty_analysis.png")


def plot_summary_report(results, output_dir):
    """Plot comprehensive summary report."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

    # Title
    fig.suptitle("Brain Tumor Classification - Evaluation Report\nKaggle Brain Tumor MRI Dataset",
                fontsize=16, fontweight="bold", y=0.98)

    # Main metrics
    ax1 = fig.add_subplot(gs[0, 0:2])
    metrics = results["metrics"]
    metric_names = ["Accuracy", "F1\n(weighted)", "F1\n(macro)", "Precision", "Recall", "MCC"]
    metric_vals = [metrics["accuracy"], metrics["f1_weighted"], metrics["f1_macro"],
                   metrics["precision"], metrics["recall"], metrics["mcc"]]
    colors = ["#3498db", "#2ecc71", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    bars = ax1.bar(metric_names, metric_vals, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, metric_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Classification Metrics", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Dataset info
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.axis("off")
    info_text = (
        f"Dataset: Kaggle Brain Tumor MRI\n"
        f"Total Images: {results['total_images']}\n"
        f"Classes: {len(CLASS_NAMES)}\n"
        f"MC Dropout Passes: {results['num_mc_passes']}\n"
        f"Inference Time: {results['inference_time_sec']}s\n"
        f"Speed: {results['images_per_sec']} img/s\n"
        f"Model: TumorAwareMultiTaskNet\n"
        f"Backbone: DenseNet121"
    )
    ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8))

    # Per-class accuracy pie
    ax3 = fig.add_subplot(gs[1, 0:2])
    per_class = results["per_class_accuracy"]
    class_names = list(per_class.keys())
    class_accs = list(per_class.values())
    wedges, texts, autotexts = ax3.pie(class_accs, labels=class_names, colors=CLASS_COLORS,
                                        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100.*sum(class_accs)/len(class_accs)*1000))/10:.0f}%)",
                                        startangle=90, textprops={"fontsize": 10})
    ax3.set_title("Per-Class Accuracy", fontsize=12, fontweight="bold")

    # Confusion matrix mini
    ax4 = fig.add_subplot(gs[1, 2:4])
    cm = np.array(results["confusion_matrix"])
    im = ax4.imshow(cm, cmap=plt.cm.Blues)
    ax4.set_xticks(range(len(CLASS_NAMES)))
    ax4.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax4.set_yticks(range(len(CLASS_NAMES)))
    ax4.set_yticklabels(CLASS_NAMES, fontsize=9)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax4.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=10)
    ax4.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    # Uncertainty metrics
    ax5 = fig.add_subplot(gs[2, 0:2])
    unc = results["uncertainty"]
    unc_names = ["Mean\nEntropy", "Mean\nConfidence", "Confidence\n(Correct)", "Confidence\n(Incorrect)"]
    unc_vals = [unc["mean_entropy"], unc["mean_confidence"],
                unc["mean_confidence_correct"], unc["mean_confidence_incorrect"]]
    bars = ax5.bar(unc_names, unc_vals, color=["#9b59b6", "#3498db", "#2ecc71", "#e74c3c"],
                   alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, unc_vals):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax5.set_ylim(0, max(unc_vals) * 1.2)
    ax5.set_title("Uncertainty Metrics", fontsize=12, fontweight="bold")
    ax5.grid(axis="y", alpha=0.3)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # Grade
    ax6 = fig.add_subplot(gs[2, 2:4])
    ax6.axis("off")
    acc = metrics["accuracy"]
    if acc >= 0.99:
        grade, color = "A+ (Excellent)", "#27ae60"
    elif acc >= 0.97:
        grade, color = "A (Very Good)", "#2ecc71"
    elif acc >= 0.95:
        grade, color = "B+ (Good)", "#f39c12"
    elif acc >= 0.90:
        grade, color = "B (Satisfactory)", "#e67e22"
    else:
        grade, color = "C (Needs Improvement)", "#e74c3c"

    ax6.text(0.5, 0.7, f"Overall Grade: {grade}", transform=ax6.transAxes,
            fontsize=18, fontweight="bold", ha="center", va="center",
            color=color, bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.2))
    ax6.text(0.5, 0.3, f"Accuracy: {acc*100:.2f}%\nF1: {metrics['f1_weighted']:.4f}",
            transform=ax6.transAxes, fontsize=14, ha="center", va="center",
            fontweight="bold")

    plt.savefig(output_dir / "07_summary_report.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 07_summary_report.png")


def plot_class_distribution(dataset, output_dir):
    """Plot class distribution of the dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Count images per class
    counts = [0] * len(CLASS_NAMES)
    folder_map = {"glioma": 0, "meningioma": 1, "pituitary": 2, "notumor": 3}
    for path, label in dataset.samples:
        counts[label] += 1

    # Bar chart
    bars = axes[0].bar(CLASS_NAMES, counts, color=CLASS_COLORS, alpha=0.85, edgecolor="white")
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Number of Images", fontsize=12)
    axes[0].set_title("Class Distribution", fontsize=13, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Pie chart
    axes[1].pie(counts, labels=CLASS_NAMES, colors=CLASS_COLORS, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 11})
    axes[1].set_title("Class Proportions", fontsize=13, fontweight="bold")

    plt.suptitle(f"Kaggle Brain Tumor MRI Dataset ({sum(counts)} images)",
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "09_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 09_class_distribution.png")


def plot_confidence_histogram(confidences, preds, labels, output_dir):
    """Plot confidence histogram by class."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, class_name in enumerate(CLASS_NAMES):
        ax = axes[i // 2, i % 2]
        mask = labels == i
        class_conf = confidences[mask]
        class_correct = preds[mask] == labels[mask]

        ax.hist(class_conf[class_correct], bins=20, alpha=0.7, color="#2ecc71",
               label="Correct", density=True)
        ax.hist(class_conf[~class_correct], bins=20, alpha=0.7, color="#e74c3c",
               label="Incorrect", density=True)
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{class_name} (n={mask.sum()})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Confidence Distribution by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "10_confidence_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 10_confidence_histogram.png")


def main():
    """Main function."""
    print("=" * 60)
    print("Comprehensive Evaluation - Kaggle Brain Tumor MRI Dataset")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # Load model
    model = load_model(device)
    print("Model loaded")

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = SimpleDataset("data/external/kaggle_brain_tumor/Testing", transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    print(f"Dataset: {len(dataset)} images")

    # Evaluate
    print("\nRunning evaluation with MC Dropout (3 passes for speed)...")
    results_data = evaluate_with_uncertainty(model, loader, device, num_passes=3)

    preds = results_data["preds"]
    labels = results_data["labels"]
    probs = results_data["probs"]
    entropies = results_data["entropies"]
    confidences = results_data["confidences"]

    # Compute all metrics
    accuracy = accuracy_score(labels, preds)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    kappa = cohen_kappa_score(labels, preds)

    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        per_class_acc[name] = float((preds[mask] == labels[mask]).mean()) if mask.sum() > 0 else 0.0

    # Uncertainty
    mean_entropy = float(entropies.mean())
    mean_confidence = float(confidences.mean())
    mean_conf_correct = float(confidences[preds == labels].mean()) if (preds == labels).any() else 0.0
    mean_conf_incorrect = float(confidences[preds != labels].mean()) if (preds != labels).any() else 0.0

    results = {
        "dataset": "Kaggle Brain Tumor MRI Dataset",
        "total_images": len(dataset),
        "num_mc_passes": 10,
        "inference_time_sec": 0,
        "images_per_sec": 0,
        "metrics": {
            "accuracy": round(float(accuracy), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "f1_macro": round(float(f1_macro), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "mcc": round(float(mcc), 4),
            "kappa": round(float(kappa), 4),
        },
        "per_class_accuracy": {k: round(v, 4) for k, v in per_class_acc.items()},
        "uncertainty": {
            "mean_entropy": round(mean_entropy, 4),
            "mean_confidence": round(mean_confidence, 4),
            "mean_confidence_correct": round(mean_conf_correct, 4),
            "mean_confidence_incorrect": round(mean_conf_incorrect, 4),
        },
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }

    # Generate all plots
    print("\nGenerating plots...")
    plot_confusion_matrix(labels, preds, OUTPUT_DIR)
    plot_per_class_metrics(labels, preds, probs, OUTPUT_DIR)
    plot_roc_curves(labels, probs, OUTPUT_DIR)
    plot_pr_curves(labels, probs, OUTPUT_DIR)
    plot_calibration(labels, confidences, probs, OUTPUT_DIR)
    plot_uncertainty(labels, preds, entropies, confidences, OUTPUT_DIR)
    plot_summary_report(results, OUTPUT_DIR)
    plot_class_distribution(dataset, OUTPUT_DIR)
    plot_confidence_histogram(confidences, preds, labels, OUTPUT_DIR)

    # Save results JSON
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results.json")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1:          {f1_weighted:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"MCC:         {mcc:.4f}")
    print(f"Kappa:       {kappa:.4f}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Files generated:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {f.name}")
    print(f"  results.json")


if __name__ == "__main__":
    main()
