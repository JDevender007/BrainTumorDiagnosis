"""
Ablation study framework for brain tumor classification.

Implements systematic ablation experiments:
A: Baseline classifier
B: Baseline + SimCLR
C: Baseline + segmentation
D: Baseline + tumor-aware features
E: Baseline + tumor-aware features + SimCLR
F: Full TumorAwareMultiTaskNet
G: Full model without fusion attention
H: Full model without segmentation supervision
I: Full model without tumor features
J: Full model without SimCLR
K: Full TURMA-Net (proposed novel model)
L: TURMA-Net without ATGFM
M: TURMA-Net without consistency regularization
N: TURMA-Net without uncertainty weighting
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    log_loss,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class AblationConfig:
    experiments: list[str] = field(default_factory=lambda: [
        "A_baseline",
        "B_baseline_simclr",
        "C_baseline_seg",
        "D_tumor_aware",
        "E_tumor_aware_simclr",
        "F_full_model",
    ])
    device: str = "cuda"
    output_dir: str = "outputs/research_experiments/ablations"


@dataclass
class AblationResult:
    experiment: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    roc_auc: float
    pr_auc: float
    mcc: float
    kappa: float
    logloss: float
    mean_confidence: float
    num_params: int
    model_size_mb: float
    notes: str = ""


def compute_specificity(y_true, y_pred, num_classes=4):
    """Compute macro specificity."""
    specificities = []
    for i in range(num_classes):
        binary_true = (np.array(y_true) == i).astype(int)
        binary_pred = (np.array(y_pred) == i).astype(int)
        tn = ((binary_true == 0) & (binary_pred == 0)).sum()
        fp = ((binary_true == 0) & (binary_pred == 1)).sum()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    return float(np.mean(specificities))


def evaluate_model_on_loader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model and return metrics."""
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
                probs = outputs.get("probabilities", torch.softmax(outputs["logits"], dim=1))
                preds = outputs["logits"].argmax(dim=1)
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_confidences.extend(probs.max(dim=1).values.cpu().numpy())

    preds = np.array(all_preds)
    labels_arr = np.array(all_labels)
    probs = np.array(all_probs)

    accuracy = accuracy_score(labels_arr, preds)
    precision = precision_score(labels_arr, preds, average="weighted", zero_division=0)
    recall = recall_score(labels_arr, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels_arr, preds, average="weighted", zero_division=0)
    specificity = compute_specificity(labels_arr, preds)
    mcc = matthews_corrcoef(labels_arr, preds)
    kappa = cohen_kappa_score(labels_arr, preds)

    try:
        roc_auc = roc_auc_score(labels_arr, probs, multi_class="ovr", average="weighted")
    except ValueError:
        roc_auc = 0.0

    try:
        ll = log_loss(labels_arr, probs)
    except ValueError:
        ll = float("inf")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc),
        "mcc": float(mcc),
        "kappa": float(kappa),
        "logloss": float(ll),
        "mean_confidence": float(np.mean(all_confidences)),
    }


def count_parameters(model: nn.Module) -> tuple[int, int, float]:
    """Count parameters and model size."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return total, trainable, size_mb


def run_ablation_experiment(
    model: nn.Module,
    experiment_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
    epochs: int = 30,
    lr: float = 0.0001,
    save_dir: Optional[Path] = None,
) -> AblationResult:
    """Run a single ablation experiment with training and evaluation."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                if isinstance(outputs, dict):
                    preds = outputs["logits"].argmax(dim=1)
                else:
                    preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    metrics = evaluate_model_on_loader(model, test_loader, device)
    total_params, trainable_params, size_mb = count_parameters(model)

    result = AblationResult(
        experiment=experiment_name,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        specificity=metrics["specificity"],
        roc_auc=metrics["roc_auc"],
        pr_auc=0.0,
        mcc=metrics["mcc"],
        kappa=metrics["kappa"],
        logloss=metrics["logloss"],
        mean_confidence=metrics["mean_confidence"],
        num_params=total_params,
        model_size_mb=size_mb,
    )

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / f"{experiment_name}.json", "w") as f:
            json.dump({
                "experiment": experiment_name,
                "accuracy": result.accuracy,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "specificity": result.specificity,
                "roc_auc": result.roc_auc,
                "mcc": result.mcc,
                "kappa": result.kappa,
                "logloss": result.logloss,
                "mean_confidence": result.mean_confidence,
                "num_params": result.num_params,
                "model_size_mb": result.model_size_mb,
            }, f, indent=2)

    return result


def generate_ablation_table(results: list[AblationResult], output_dir: Path):
    """Generate ablation comparison table."""
    output_dir.mkdir(parents=True, exist_ok=True)

    header = "| Experiment | Accuracy | F1 | MCC | Kappa | ROC-AUC | Params (M) | Size (MB) |"
    separator = "|------------|----------|-----|-----|-------|---------|------------|-----------|"
    rows = []
    for r in results:
        rows.append(
            f"| {r.experiment} | {r.accuracy:.4f} | {r.f1:.4f} | {r.mcc:.4f} | "
            f"{r.kappa:.4f} | {r.roc_auc:.4f} | {r.num_params / 1e6:.2f} | {r.model_size_mb:.2f} |"
        )

    table = "\n".join([header, separator] + rows)
    with open(output_dir / "ablation_table.md", "w") as f:
        f.write(table)

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump([
            {
                "experiment": r.experiment,
                "accuracy": r.accuracy,
                "f1": r.f1,
                "mcc": r.mcc,
                "kappa": r.kappa,
                "roc_auc": r.roc_auc,
                "num_params": r.num_params,
                "model_size_mb": r.model_size_mb,
            }
            for r in results
        ], f, indent=2)
