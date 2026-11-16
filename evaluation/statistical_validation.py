"""
Statistical validation utilities for publication-ready research.

Implements:
1. Bootstrap confidence intervals
2. McNemar's test for paired classification
3. Paired bootstrap test
4. Effect size (Cohen's d)
5. Multiple comparison correction
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Any


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Any,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        metric_fn: metric function (y_true, y_pred) -> float
        n_bootstrap: number of bootstrap samples
        confidence_level: confidence level (e.g., 0.95 for 95% CI)
        seed: random seed

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    # Point estimate
    point_estimate = metric_fn(y_true, y_pred)

    # Bootstrap
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, n, replace=True)
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)

    # Confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

    return point_estimate, ci_lower, ci_upper


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    exact: bool = True,
) -> dict[str, float]:
    """
    McNemar's test for paired classification comparison.

    Tests whether two classifiers have the same error rate.

    Args:
        y_true: ground truth labels
        y_pred_1: predictions from model 1
        y_pred_2: predictions from model 2
        exact: use exact test (for small samples)

    Returns:
        dict with test statistic and p-value
    """
    # Contingency table
    correct_1 = y_pred_1 == y_true
    correct_2 = y_pred_2 == y_true

    b = np.sum(correct_1 & ~correct_2)  # model 1 correct, model 2 wrong
    c = np.sum(~correct_1 & correct_2)  # model 1 wrong, model 2 correct

    if exact:
        # Exact binomial test
        result = stats.binomtest(b, b + c, 0.5)
        statistic = b
        p_value = result.pvalue
    else:
        # Asymptotic test
        if b + c == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            statistic = (b - c) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "b": int(b),
        "c": int(c),
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    metric_fn: Any,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Paired bootstrap test for comparing two classifiers.

    Args:
        y_true: ground truth labels
        y_pred_1: predictions from model 1
        y_pred_2: predictions from model 2
        metric_fn: metric function
        n_bootstrap: number of bootstrap samples
        seed: random seed

    Returns:
        dict with test statistic, p-value, and confidence interval
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    # Observed difference
    diff_observed = metric_fn(y_true, y_pred_1) - metric_fn(y_true, y_pred_2)

    # Bootstrap differences
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, n, replace=True)
        diff = metric_fn(y_true[indices], y_pred_1[indices]) - metric_fn(y_true[indices], y_pred_2[indices])
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # p-value: proportion of bootstrap samples where difference <= 0
    p_value = np.mean(bootstrap_diffs <= 0)

    # Two-tailed p-value
    p_value_two_tailed = 2 * min(p_value, 1 - p_value)

    # 95% CI for the difference
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)

    return {
        "observed_difference": float(diff_observed),
        "p_value": float(p_value_two_tailed),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "mean_bootstrap_difference": float(np.mean(bootstrap_diffs)),
    }


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    Args:
        group1: first group of measurements
        group2: second group of measurements

    Returns:
        Cohen's d value
    """
    diff = group1 - group2
    n = len(diff)

    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    if std_diff == 0:
        return 0.0

    return mean_diff / std_diff


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cliff's delta non-parametric effect size.

    Args:
        group1: first group of measurements
        group2: second group of measurements

    Returns:
        Cliff's delta value in [-1, 1]
    """
    n1 = len(group1)
    n2 = len(group2)

    dominance = 0
    for x in group1:
        for y in group2:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1

    return dominance / (n1 * n2)


def bonferroni_correction(p_values: list[float]) -> list[float]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: list of p-values

    Returns:
        list of corrected p-values
    """
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def fdr_correction(p_values: list[float]) -> list[float]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: list of p-values

    Returns:
        list of corrected p-values
    """
    n = len(p_values)
    ranked = np.argsort(p_values)
    corrected = np.zeros(n)

    for i, idx in enumerate(ranked):
        corrected[idx] = p_values[idx] * n / (i + 1)

    # Ensure monotonicity
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.minimum(corrected, 1.0)

    return corrected.tolist()


def compute_all_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Any,
    metric_name: str = "metric",
) -> dict[str, Any]:
    """
    Compute comprehensive statistics for a single model.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        metric_fn: metric function
        metric_name: name of the metric

    Returns:
        dict with point estimate, CI, and other statistics
    """
    point_est, ci_lower, ci_upper = bootstrap_confidence_interval(
        y_true, y_pred, metric_fn
    )

    return {
        f"{metric_name}_point": float(point_est),
        f"{metric_name}_ci_lower": float(ci_lower),
        f"{metric_name}_ci_upper": float(ci_upper),
    }


def compare_models(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    metric_fn: Any,
    metric_name: str = "metric",
    model_1_name: str = "model_1",
    model_2_name: str = "model_2",
) -> dict[str, Any]:
    """
    Comprehensive comparison between two models.

    Args:
        y_true: ground truth labels
        y_pred_1: predictions from model 1
        y_pred_2: predictions from model 2
        metric_fn: metric function
        metric_name: name of the metric
        model_1_name: name of model 1
        model_2_name: name of model 2

    Returns:
        dict with comparison results
    """
    # Bootstrap CI for each model
    stats_1 = compute_all_statistics(y_true, y_pred_1, metric_fn, f"{model_1_name}_{metric_name}")
    stats_2 = compute_all_statistics(y_true, y_pred_2, metric_fn, f"{model_2_name}_{metric_name}")

    # Paired bootstrap test
    bootstrap_result = paired_bootstrap_test(y_true, y_pred_1, y_pred_2, metric_fn)

    # McNemar's test
    mcnemar_result = mcnemar_test(y_true, y_pred_1, y_pred_2, exact=False)

    # Effect sizes
    scores_1 = np.array([metric_fn(y_true[np.random.choice(len(y_true), len(y_true), replace=True)],
                                    y_pred_1[np.random.choice(len(y_true), len(y_true), replace=True)])
                         for _ in range(100)])
    scores_2 = np.array([metric_fn(y_true[np.random.choice(len(y_true), len(y_true), replace=True)],
                                    y_pred_2[np.random.choice(len(y_true), len(y_true), replace=True)])
                         for _ in range(100)])

    return {
        **stats_1,
        **stats_2,
        "bootstrap_test": bootstrap_result,
        "mcnemar_test": mcnemar_result,
        "cohens_d": float(cohens_d(scores_1, scores_2)),
        "cliffs_delta": float(cliffs_delta(scores_1, scores_2)),
    }
