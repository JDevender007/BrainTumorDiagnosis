"""
Uncertainty estimation for brain tumor classification.

Implements Monte Carlo Dropout for epistemic uncertainty,
predictive entropy, mutual information, and confidence calibration.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class UncertaintyConfig:
    mc_passes: int = 10
    temperature: float = 1.0
    confidence_threshold_low: float = 0.7
    confidence_threshold_high: float = 0.9
    device: str = "cuda"


@dataclass
class UncertaintyResult:
    prediction: int
    confidence: float
    predictive_entropy: float
    mutual_information: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    uncertainty_level: str
    mean_probabilities: np.ndarray
    all_predictions: list[int] = field(default_factory=list)


def compute_uncertainty_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_passes: int = 10,
) -> dict:
    """Compute uncertainty metrics for all samples in dataloader."""
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []
    all_entropies = []
    all_mutual_info = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Uncertainty", leave=False):
            images = images.to(device)

            batch_probs = []
            for _ in range(num_passes):
                for m in model.modules():
                    if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                        m.train()
                outputs = model(images)
                if isinstance(outputs, dict):
                    probs = outputs.get("probabilities", F.softmax(outputs["logits"], dim=1))
                else:
                    probs = F.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu())

            stacked = torch.stack(batch_probs)
            mean_probs = stacked.mean(dim=0)

            predictive_entropy = -torch.sum(
                mean_probs * torch.log(mean_probs + 1e-10), dim=1
            )
            batch_entropy_val = -torch.sum(
                stacked * torch.log(stacked + 1e-10), dim=2
            )
            expected_entropy = batch_entropy_val.mean(dim=0)
            mutual_info = predictive_entropy - expected_entropy

            all_probs.append(mean_probs.numpy())
            all_labels.append(labels.numpy())
            all_preds.append(mean_probs.argmax(dim=1).numpy())
            all_entropies.append(predictive_entropy.numpy())
            all_mutual_info.append(mutual_info.numpy())
            all_confidences.append(mean_probs.max(dim=1).values.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    entropies = np.concatenate(all_entropies)
    mutual_infos = np.concatenate(all_mutual_info)
    confidences = np.concatenate(all_confidences)

    correct_mask = preds == labels
    incorrect_mask = ~correct_mask

    return {
        "mean_confidence": float(np.mean(confidences)),
        "mean_confidence_correct": float(np.mean(confidences[correct_mask])) if correct_mask.any() else 0.0,
        "mean_confidence_incorrect": float(np.mean(confidences[incorrect_mask])) if incorrect_mask.any() else 0.0,
        "mean_entropy": float(np.mean(entropies)),
        "mean_mutual_info": float(np.mean(mutual_infos)),
        "mean_epistemic": float(np.mean(mutual_infos)),
        "accuracy": float(np.mean(correct_mask)),
        "ece": compute_ece(probs, labels, n_bins=15),
        "brier": compute_brier(probs, labels),
        "num_samples": len(labels),
    }


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


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier Score (lower is better)."""
    num_classes = probs.shape[1]
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def temperature_scale(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Temperature scaling for calibration."""
    return logits / temperature


def find_optimal_temperature(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    lr: float = 0.01,
    max_iter: int = 100,
) -> float:
    """Find optimal temperature via validation NLL."""
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

    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    nll_criterion = nn.CrossEntropyLoss()

    def eval_step():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        loss = nll_criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval_step)
    return float(temperature.item())


def classify_uncertainty_level(
    confidence: float,
    entropy: float,
    config: UncertaintyConfig,
) -> str:
    """Classify prediction uncertainty into LOW/MEDIUM/HIGH."""
    if confidence >= config.confidence_threshold_high and entropy < 0.5:
        return "LOW"
    elif confidence >= config.confidence_threshold_low:
        return "MEDIUM"
    else:
        return "HIGH"


def predict_with_uncertainty(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    config: UncertaintyConfig,
) -> UncertaintyResult:
    """Single-image prediction with uncertainty estimation."""
    model.train()
    image = image.unsqueeze(0).to(device) if image.dim() == 3 else image.to(device)

    all_probs = []
    with torch.no_grad():
        for _ in range(config.mc_passes):
            outputs = model(image)
            if isinstance(outputs, dict):
                probs = outputs.get("probabilities", F.softmax(outputs["logits"], dim=1))
            else:
                probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())

    stacked = torch.stack(all_probs)
    mean_probs = stacked.mean(dim=0).squeeze()

    prediction = int(mean_probs.argmax())
    confidence = float(mean_probs.max())

    entropy = float(-torch.sum(mean_probs * torch.log(mean_probs + 1e-10)))
    batch_entropy = -torch.sum(stacked * torch.log(stacked + 1e-10), dim=2)
    expected_entropy = batch_entropy.mean(dim=0).squeeze()
    mutual_info = float((mean_probs * torch.log(mean_probs + 1e-10)).sum() + expected_entropy.mean())

    uncertainty_level = classify_uncertainty_level(confidence, entropy, config)

    all_preds = stacked.argmax(dim=2).squeeze().numpy().tolist()

    return UncertaintyResult(
        prediction=prediction,
        confidence=confidence,
        predictive_entropy=entropy,
        mutual_information=mutual_info,
        epistemic_uncertainty=mutual_info,
        aleatoric_uncertainty=float(stacked.var(dim=0).mean()),
        uncertainty_level=uncertainty_level,
        mean_probabilities=mean_probs.numpy(),
        all_predictions=all_preds if isinstance(all_preds, list) else [all_preds],
    )
