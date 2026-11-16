"""
Cross-dataset analysis for domain shift investigation.

Compares internal vs external (BRISC 2025) performance.
Measures domain shift performance drop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class CrossDatasetResult:
    model_name: str
    internal_accuracy: float
    internal_f1: float
    internal_mcc: float
    internal_roc_auc: float
    internal_confidence: float
    external_accuracy: float
    external_f1: float
    external_mcc: float
    external_roc_auc: float
    external_confidence: float
    accuracy_drop: float
    f1_drop: float
    mcc_drop: float
    confidence_drop: float
    internal_confusion_matrix: list = field(default_factory=list)
    external_confusion_matrix: list = field(default_factory=list)


def evaluate_on_loader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, dict):
                probs = outputs.get("probabilities", F.softmax(outputs["logits"], dim=1))
                preds = outputs["logits"].argmax(dim=1)
            else:
                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_confidences.extend(probs.max(dim=1).values.cpu().numpy())

    preds = np.array(all_preds)
    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)
    confidences = np.array(all_confidences)

    accuracy = accuracy_score(labels_arr, preds)
    f1 = f1_score(labels_arr, preds, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(labels_arr, preds)
    kappa = cohen_kappa_score(labels_arr, preds)

    try:
        roc_auc = roc_auc_score(labels_arr, probs_arr, multi_class="ovr", average="weighted")
    except ValueError:
        roc_auc = 0.0

    try:
        ll = log_loss(labels_arr, probs_arr)
    except ValueError:
        ll = float("inf")

    cm = confusion_matrix(labels_arr, preds).tolist()

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "roc_auc": float(roc_auc),
        "logloss": float(ll),
        "mean_confidence": float(np.mean(confidences)),
        "confusion_matrix": cm,
    }


def analyze_domain_shift(
    model: nn.Module,
    model_name: str,
    internal_loader: DataLoader,
    external_loader: DataLoader,
    device: torch.device,
) -> CrossDatasetResult:
    """Analyze domain shift between internal and external datasets."""
    internal_metrics = evaluate_on_loader(model, internal_loader, device)
    external_metrics = evaluate_on_loader(model, external_loader, device)

    accuracy_drop = internal_metrics["accuracy"] - external_metrics["accuracy"]
    f1_drop = internal_metrics["f1"] - external_metrics["f1"]
    mcc_drop = internal_metrics["mcc"] - external_metrics["mcc"]
    confidence_drop = internal_metrics["mean_confidence"] - external_metrics["mean_confidence"]

    return CrossDatasetResult(
        model_name=model_name,
        internal_accuracy=internal_metrics["accuracy"],
        internal_f1=internal_metrics["f1"],
        internal_mcc=internal_metrics["mcc"],
        internal_roc_auc=internal_metrics["roc_auc"],
        internal_confidence=internal_metrics["mean_confidence"],
        external_accuracy=external_metrics["accuracy"],
        external_f1=external_metrics["f1"],
        external_mcc=external_metrics["mcc"],
        external_roc_auc=external_metrics["roc_auc"],
        external_confidence=external_metrics["mean_confidence"],
        accuracy_drop=accuracy_drop,
        f1_drop=f1_drop,
        mcc_drop=mcc_drop,
        confidence_drop=confidence_drop,
        internal_confusion_matrix=internal_metrics["confusion_matrix"],
        external_confusion_matrix=external_metrics["confusion_matrix"],
    )


def generate_cross_dataset_table(
    results: list[CrossDatasetResult],
    output_dir: Path,
):
    """Generate cross-dataset comparison table."""
    output_dir.mkdir(parents=True, exist_ok=True)

    header = "| Model | Internal Acc | External Acc | Acc Drop | Internal F1 | External F1 | F1 Drop |"
    separator = "|-------|-------------|-------------|----------|------------|------------|---------|"
    rows = []
    for r in results:
        rows.append(
            f"| {r.model_name} | {r.internal_accuracy:.4f} | {r.external_accuracy:.4f} | "
            f"{r.accuracy_drop:.4f} | {r.internal_f1:.4f} | {r.external_f1:.4f} | {r.f1_drop:.4f} |"
        )

    table = "\n".join([header, separator] + rows)
    with open(output_dir / "cross_dataset_table.md", "w") as f:
        f.write(table)

    with open(output_dir / "cross_dataset_results.json", "w") as f:
        json.dump([
            {
                "model": r.model_name,
                "internal_accuracy": r.internal_accuracy,
                "external_accuracy": r.external_accuracy,
                "accuracy_drop": r.accuracy_drop,
                "internal_f1": r.internal_f1,
                "external_f1": r.external_f1,
                "f1_drop": r.f1_drop,
                "internal_mcc": r.internal_mcc,
                "external_mcc": r.external_mcc,
                "mcc_drop": r.mcc_drop,
                "internal_confidence": r.internal_confidence,
                "external_confidence": r.external_confidence,
                "confidence_drop": r.confidence_drop,
            }
            for r in results
        ], f, indent=2)
