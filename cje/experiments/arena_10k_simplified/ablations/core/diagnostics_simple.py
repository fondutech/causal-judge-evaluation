"""Simplified diagnostic functions for ablation experiments."""

import numpy as np
from typing import Optional, Dict, Any


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute effective sample size (ESS).

    ESS = (sum w)^2 / sum(w^2)

    Args:
        weights: Importance weights

    Returns:
        ESS value (higher is better)
    """
    weights = np.asarray(weights)
    weights = weights[np.isfinite(weights)]

    if len(weights) == 0:
        return 0.0

    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights**2)

    if sum_w2 == 0:
        return 0.0

    return sum_w**2 / sum_w2


def hill_alpha(weights: np.ndarray, k: Optional[int] = None) -> float:
    """Compute Hill estimator of tail index α.

    α > 2: Light tails (good)
    α ≤ 2: Heavy tails (bad - infinite variance)

    Args:
        weights: Importance weights
        k: Number of tail observations (default: 5% or 50)

    Returns:
        Hill estimator α̂
    """
    weights = np.asarray(weights)
    weights = weights[(weights > 0) & np.isfinite(weights)]

    if len(weights) < 10:
        return np.nan

    if k is None:
        k = max(50, int(0.05 * len(weights)))
        k = min(k, len(weights) // 2)

    # Sort and get tail
    x_sorted = np.sort(weights)
    x_tail = x_sorted[-k:]
    x_k = x_sorted[-k]

    # Hill estimator
    log_ratios = np.log(x_tail / x_k)
    mean_log_ratio = np.mean(log_ratios)

    if mean_log_ratio == 0:
        return np.inf

    return 1.0 / mean_log_ratio


def weight_cv(weights: np.ndarray) -> float:
    """Coefficient of variation of weights.

    CV = std(w) / mean(w)

    Args:
        weights: Importance weights

    Returns:
        CV value (lower is better)
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
