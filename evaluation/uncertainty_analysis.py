"""
Uncertainty utility analysis for brain tumor classification.

Implements:
1. Uncertainty-error correlation
2. AUROC for error detection
3. AUPRC for error detection
4. Risk-coverage curve
5. Selective accuracy
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from typing import Any


def uncertainty_error_correlation(
    uncertainties: np.ndarray,
    errors: np.ndarray,
) -> dict[str, float]:
    """
    Compute correlation between uncertainty and errors.

    Args:
        uncertainties: uncertainty values for each sample
        errors: binary error indicator (1 if incorrect, 0 if correct)

    Returns:
        dict with correlation metrics
    """
    # Spearman correlation
    from scipy.stats import spearmanr, pearsonr

    spearman_corr, spearman_p = spearmanr(uncertainties, errors)
    pearson_corr, pearson_p = pearsonr(uncertainties, errors)

    return {
        "spearman_correlation": float(spearman_corr),
        "spearman_p_value": float(spearman_p),
        "pearson_correlation": float(pearson_corr),
        "pearson_p_value": float(pearson_p),
    }


def auroc_error_detection(
    uncertainties: np.ndarray,
    errors: np.ndarray,
) -> float:
    """
    Compute AUROC for error detection.

    Higher AUROC means uncertainty better distinguishes correct from incorrect.

    Args:
        uncertainties: uncertainty values
        errors: binary error indicator

    Returns:
        AUROC score
    """
    if len(np.unique(errors)) < 2:
        return 0.5  # No discrimination possible

    return float(roc_auc_score(errors, uncertainties))


def auprc_error_detection(
    uncertainties: np.ndarray,
    errors: np.ndarray,
) -> float:
    """
    Compute AUPRC for error detection.

    Args:
        uncertainties: uncertainty values
        errors: binary error indicator

    Returns:
        AUPRC score
    """
    if len(np.unique(errors)) < 2:
        return 0.0

    return float(average_precision_score(errors, uncertainties))


def risk_coverage_curve(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    n_thresholds: int = 100,
) -> dict[str, list[float]]:
    """
    Compute risk-coverage curve.

    Risk = error rate of selected samples
    Coverage = fraction of samples selected

    Args:
        uncertainties: uncertainty values
        errors: binary error indicator
        n_thresholds: number of thresholds to evaluate

    Returns:
        dict with coverage and risk curves
    """
    thresholds = np.linspace(uncertainties.min(), uncertainties.max(), n_thresholds)

    coverages = []
    risks = []

    for threshold in thresholds:
        # Select samples with uncertainty below threshold
        selected = uncertainties <= threshold
        coverage = selected.mean()

        if selected.sum() > 0:
            risk = errors[selected].mean()
        else:
            risk = 0.0

        coverages.append(float(coverage))
        risks.append(float(risk))

    return {
        "coverages": coverages,
        "risks": risks,
        "thresholds": thresholds.tolist(),
    }


def selective_accuracy(
    uncertainties: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    coverage_levels: list[float] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute accuracy at different coverage levels.

    Args:
        uncertainties: uncertainty values
        predictions: predicted labels
        labels: true labels
        coverage_levels: list of coverage levels to evaluate

    Returns:
        dict mapping coverage level to accuracy and related metrics
    """
    if coverage_levels is None:
        coverage_levels = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    correct = predictions == labels
    results = {}

    for coverage in coverage_levels:
        # Select most confident samples
        threshold = np.percentile(uncertainties, (1 - coverage) * 100)
        selected = uncertainties <= threshold

        if selected.sum() > 0:
            selective_acc = correct[selected].mean()
            num_selected = selected.sum()
        else:
            selective_acc = 0.0
            num_selected = 0

        results[f"{coverage:.0%}"] = {
            "accuracy": float(selective_acc),
            "coverage": float(coverage),
            "num_selected": int(num_selected),
            "num_total": len(labels),
        }

    return results


def compute_comprehensive_uncertainty_analysis(
    uncertainties: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    model_name: str = "model",
) -> dict[str, Any]:
    """
    Compute comprehensive uncertainty utility analysis.

    Args:
        uncertainties: uncertainty values
        predictions: predicted labels
        labels: true labels
        model_name: name of the model

    Returns:
        dict with all uncertainty analysis results
    """
    errors = (predictions != labels).astype(float)

    # Uncertainty-error correlation
    correlation = uncertainty_error_correlation(uncertainties, errors)

    # AUROC and AUPRC for error detection
    auroc = auroc_error_detection(uncertainties, errors)
    auprc = auprc_error_detection(uncertainties, errors)

    # Risk-coverage curve
    rc_curve = risk_coverage_curve(uncertainties, errors)

    # Selective accuracy
    selective = selective_accuracy(uncertainties, predictions, labels)

    return {
        "model": model_name,
        "num_samples": len(labels),
        "error_rate": float(errors.mean()),
        "mean_uncertainty": float(uncertainties.mean()),
        "std_uncertainty": float(uncertainties.std()),
        "correlation": correlation,
        "auroc_error_detection": auroc,
        "auprc_error_detection": auprc,
        "risk_coverage_curve": rc_curve,
        "selective_accuracy": selective,
    }
