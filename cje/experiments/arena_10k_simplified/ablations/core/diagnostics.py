"""Diagnostic functions for ablation experiments."""

import numpy as np
from typing import Optional, Dict, Any, Tuple


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute effective sample size (ESS).

    ESS = (sum w)^2 / sum(w^2)

    This measures how many "effective" samples we have after importance weighting.
    An ESS of 100 means the weights are as variable as if we had 100 equal samples.

    Args:
        weights: Importance weights (should sum to n for Hajek, or be normalized)

    Returns:
        ESS (absolute number)
    """
    if len(weights) == 0:
        return 0.0

    weights = np.asarray(weights)
    weights = weights[np.isfinite(weights)]  # Remove NaN/inf

    if len(weights) == 0:
        return 0.0

    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights**2)

    if sum_w2 == 0:
        return 0.0

    return sum_w**2 / sum_w2


def hill_alpha(weights: np.ndarray, k: Optional[int] = None) -> float:
    """Compute Hill estimator of tail index α.

    The tail index measures how heavy the tail is:
    - α > 2: Light tails, finite variance (good)
    - α ≤ 2: Heavy tails, infinite variance (bad)
    - α ≤ 1: Very heavy tails, infinite mean (very bad)

    Args:
        weights: Importance weights
        k: Number of tail observations to use (default: 5% or 50, whichever is larger)

    Returns:
        Hill estimator α̂
    """
    weights = np.asarray(weights)
    weights = weights[weights > 0]  # Only positive weights
    weights = weights[np.isfinite(weights)]  # Remove NaN/inf

    if len(weights) < 10:
        return np.nan  # Not enough data

    # Default k: 5% of data or 50, whichever is larger
    if k is None:
        k = max(50, int(0.05 * len(weights)))
        k = min(k, len(weights) // 2)  # But at most half the data

    # Sort and get tail
    x_sorted = np.sort(weights)
    x_tail = x_sorted[-k:]  # k largest values
    x_k = x_sorted[-k]  # k-th largest (threshold)

    # Hill estimator: 1 / mean(log(X_i / X_k)) for X_i > X_k
    log_ratios = np.log(x_tail / x_k)

    # Avoid division by zero
    mean_log_ratio = np.mean(log_ratios)
    if mean_log_ratio == 0:
        return np.inf

    return 1.0 / mean_log_ratio


def simcal_distortion(W_original: np.ndarray, W_calibrated: np.ndarray) -> float:
    """Compute relative L2 distortion from SIMCal calibration.

    δ = ||W_cal - W_orig||_2 / ||W_orig||_2

    This measures how much the weights were changed by calibration.
    Lower is better (less distortion).

    Args:
        W_original: Original importance weights
        W_calibrated: Calibrated weights

    Returns:
        Relative distortion δ
    """
    W_original = np.asarray(W_original)
    W_calibrated = np.asarray(W_calibrated)

    # Handle shape mismatch
    if W_original.shape != W_calibrated.shape:
        return np.nan

    # Remove NaN/inf
    mask = np.isfinite(W_original) & np.isfinite(W_calibrated)
    W_original = W_original[mask]
    W_calibrated = W_calibrated[mask]

    if len(W_original) == 0:
        return np.nan

    norm_orig = np.linalg.norm(W_original)
    if norm_orig == 0:
        return np.nan

    return np.linalg.norm(W_calibrated - W_original) / norm_orig


def weight_cv(weights: np.ndarray) -> float:
    """Compute coefficient of variation of weights.

    CV = std(w) / mean(w)

    This is a normalized measure of weight variability.
    Lower is better (more uniform weights).

    Args:
        weights: Importance weights

    Returns:
        Coefficient of variation
    """
    weights = np.asarray(weights)
    weights = weights[np.isfinite(weights)]

    if len(weights) == 0:
        return np.nan

    mean_w = np.mean(weights)
    if mean_w == 0:
        return np.nan

    return np.std(weights) / mean_w


def compute_rmse(estimates: Dict[str, float], truths: Dict[str, float]) -> float:
    """Compute RMSE between estimates and oracle truths.

    Args:
        estimates: Policy -> estimate
        truths: Policy -> oracle truth

    Returns:
        Root mean squared error
    """
    if not estimates or not truths:
        return np.nan

    squared_errors = []
    for policy in estimates:
        if policy in truths:
            est = estimates[policy]
            truth = truths[policy]
            if np.isfinite(est) and np.isfinite(truth):
                squared_errors.append((est - truth) ** 2)

    if not squared_errors:
        return np.nan

    return np.sqrt(np.mean(squared_errors))


def compute_calibration_metrics(
    judge_scores: np.ndarray,
    oracle_labels: np.ndarray,
    calibrated_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute calibration quality metrics.

    Args:
        judge_scores: Original judge scores
        oracle_labels: Oracle ground truth labels
        calibrated_scores: Calibrated predictions (optional)

    Returns:
        Dictionary of calibration metrics
    """
    # Remove missing values
    mask = np.isfinite(judge_scores) & np.isfinite(oracle_labels)
    judge_scores = judge_scores[mask]
    oracle_labels = oracle_labels[mask]

    if len(judge_scores) == 0:
        return {"calibration_rmse": np.nan, "kendall_tau": np.nan}

    metrics = {}

    # RMSE of judge vs oracle
    metrics["judge_oracle_rmse"] = np.sqrt(np.mean((judge_scores - oracle_labels) ** 2))

    # Kendall's tau (rank correlation)
    from scipy import stats

    tau, _ = stats.kendalltau(judge_scores, oracle_labels)
    metrics["kendall_tau"] = tau

    # If calibrated scores provided, compute improvement
    if calibrated_scores is not None:
        calibrated_scores = calibrated_scores[mask]
        metrics["calibrated_rmse"] = np.sqrt(
            np.mean((calibrated_scores - oracle_labels) ** 2)
        )
        metrics["calibration_improvement"] = (
            metrics["judge_oracle_rmse"] - metrics["calibrated_rmse"]
        ) / metrics["judge_oracle_rmse"]

    return metrics


def compute_paired_delta(
    results_baseline: Dict[str, Any],
    results_treatment: Dict[str, Any],
    metric: str = "rmse_vs_oracle",
) -> Tuple[float, float]:
    """Compute paired difference with standard error.

    Used for comparing parameter settings (e.g., ρ=1.0 vs ρ=0.6)
    with the same seeds and oracle slices.

    Args:
        results_baseline: Baseline results
        results_treatment: Treatment results
        metric: Which metric to compare

    Returns:
        (delta, se) where delta = treatment - baseline
    """
    # This would need the raw paired differences, not just aggregates
    # For now, return simple difference
    baseline_val = results_baseline.get(metric, np.nan)
    treatment_val = results_treatment.get(metric, np.nan)

    if not np.isfinite(baseline_val) or not np.isfinite(treatment_val):
        return np.nan, np.nan

    delta = treatment_val - baseline_val
    # SE would require access to paired seed-level differences
    se = np.nan  # Placeholder

    return delta, se
