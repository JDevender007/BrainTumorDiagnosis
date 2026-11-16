"""
Calibration analysis for brain tumor classification.

Implements:
- Expected Calibration Error (ECE)
- Brier Score
- Maximum Calibration Error (MCE)
- Reliability diagrams
- Temperature scaling
- Per-model calibration comparison
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class CalibrationResult:
    ece: float
    mce: float
    brier: float
    nll: float
    temperature: float
    mean_confidence: float
    mean_accuracy: float
    num_samples: int
    bin_data: dict = field(default_factory=dict)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Maximum Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

        if in_bin.any():
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            mce = max(mce, abs(avg_accuracy - avg_confidence))

    return float(mce)


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier Score."""
    num_classes = probs.shape[1]
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Negative Log-Likelihood."""
    eps = 1e-10
    log_probs = np.log(probs + eps)
    nll = -np.mean(log_probs[np.arange(len(labels)), labels])
    return float(nll)


def get_reliability_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Get data for reliability diagram."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            bin_confidences.append(float(confidences[in_bin].mean()))
            bin_accuracies.append(float(accuracies[in_bin].mean()))
            bin_counts.append(int(in_bin.sum()))
        else:
            bin_confidences.append(float((bin_boundaries[i] + bin_boundaries[i + 1]) / 2))
            bin_accuracies.append(0.0)
            bin_counts.append(0)

    return {
        "bin_boundaries": bin_boundaries.tolist(),
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "ece": compute_ece(probs, labels, n_bins),
        "mce": compute_mce(probs, labels, n_bins),
    }


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    output_path: Path,
    n_bins: int = 15,
):
    """Plot reliability diagram for a single model."""
    data = get_reliability_data(probs, labels, n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bin_boundaries = data["bin_boundaries"]
    bin_confidences = data["bin_confidences"]
    bin_accuracies = data["bin_accuracies"]
    bin_counts = data["bin_counts"]
    total = sum(bin_counts)

    bar_width = 0.8 * (bin_boundaries[1] - bin_boundaries[0])

    for i in range(len(bin_confidences)):
        if bin_counts[i] > 0:
            x = bin_confidences[i]
            ax.bar(x, bin_accuracies[i], width=bar_width, alpha=0.7, color="steelblue", edgecolor="black")
            ax.bar(x, bin_confidences[i] - bin_accuracies[i], bottom=bin_accuracies[i],
                   width=bar_width, alpha=0.3, color="salmon", edgecolor="black")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Reliability Diagram - {model_name}\nECE={data['ece']:.4f}, MCE={data['mce']:.4f}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax2 = axes[1]
    non_zero = [c for c in bin_counts if c > 0]
    bin_centers = [(bin_boundaries[i] + bin_boundaries[i + 1]) / 2 for i in range(len(bin_counts)) if bin_counts[i] > 0]
    ax2.bar(bin_centers, non_zero, width=bar_width, alpha=0.7, color="steelblue", edgecolor="black")
    ax2.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax2.set_ylabel("Number of Samples", fontsize=12)
    ax2.set_title("Confidence Distribution", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration_comparison(
    all_probs: dict[str, np.ndarray],
    all_labels: dict[str, np.ndarray],
    output_path: Path,
):
    """Plot reliability diagrams for multiple models."""
    n_models = len(all_probs)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, probs) in zip(axes, all_probs.items()):
        labels = all_labels[name]
        data = get_reliability_data(probs, labels)

        bin_boundaries = data["bin_boundaries"]
        bin_confidences = data["bin_confidences"]
        bin_accuracies = data["bin_accuracies"]
        bin_counts = data["bin_counts"]
        bar_width = 0.8 * (bin_boundaries[1] - bin_boundaries[0])

        for i in range(len(bin_confidences)):
            if bin_counts[i] > 0:
                x = bin_confidences[i]
                ax.bar(x, bin_accuracies[i], width=bar_width, alpha=0.7, color="steelblue", edgecolor="black")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{name}\nECE={data['ece']:.4f}")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


class TemperatureScaler(nn.Module):
    """Learnable temperature scaling for post-hoc calibration."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Fit temperature on validation set."""
        model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                outputs = model(images)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            scaled = logits / self.temperature
            loss = nll_criterion(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return float(self.temperature.item())


def evaluate_calibration(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_name: str = "model",
    output_dir: Optional[Path] = None,
) -> CalibrationResult:
    """Full calibration analysis for a model."""
    model.eval()
    all_probs = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Calibration {model_name}", leave=False):
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
                probs = outputs.get("probabilities", F.softmax(logits, dim=1))
            else:
                logits = outputs
                probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_logits.append(logits.cpu())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    ece = compute_ece(probs, labels)
    mce = compute_mce(probs, labels)
    brier = compute_brier(probs, labels)
    nll_val = compute_nll(probs, labels)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    result = CalibrationResult(
        ece=ece,
        mce=mce,
        brier=brier,
        nll=nll_val,
        temperature=1.0,
        mean_confidence=float(np.mean(confidences)),
        mean_accuracy=float(np.mean(accuracies)),
        num_samples=len(labels),
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_reliability_diagram(probs, labels, model_name, output_dir / f"reliability_{model_name}.png")

        with open(output_dir / f"calibration_{model_name}.json", "w") as f:
            json.dump({
                "model": model_name,
                "ece": ece,
                "mce": mce,
                "brier": brier,
                "nll": nll_val,
                "mean_confidence": result.mean_confidence,
                "mean_accuracy": result.mean_accuracy,
                "num_samples": result.num_samples,
            }, f, indent=2)

    return result
