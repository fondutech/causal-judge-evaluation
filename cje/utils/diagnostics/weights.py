"""
Weight diagnostic computations for importance sampling.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

from ...data.diagnostics import Status

logger = logging.getLogger(__name__)


# ========== Core Metrics ==========


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute ESS = (sum(w))^2 / sum(w^2).

    ESS measures how many "effective" samples we have after weighting.
    ESS = n means perfect overlap, ESS << n means poor overlap.
    """
    s = weights.sum()
    s2 = np.sum(weights**2)
    return float((s * s) / np.maximum(s2, 1e-12))


def compute_ess(weights: np.ndarray) -> float:
    """Alias for effective_sample_size() - kept for backward compatibility."""
    return effective_sample_size(weights)


def tail_weight_ratio(
    weights: np.ndarray, q_low: float = 0.05, q_high: float = 0.99
) -> float:
    """Compute ratio of high to low quantiles.

    Args:
        weights: Importance weights
        q_low: Lower quantile (default 0.05 to avoid instability)
        q_high: Upper quantile (default 0.99)

    Returns:
        Ratio of high/low quantiles (inf if low quantile is ~0)
    """
    lo = np.quantile(weights, q_low)
    hi = np.quantile(weights, q_high)
    if lo <= 1e-12:
        return float(np.inf)
    return float(hi / lo)


def mass_concentration(weights: np.ndarray, top_pct: float = 0.01) -> float:
    """Fraction of total weight held by top x% of samples.

    Args:
        weights: Importance weights
        top_pct: Top percentage to consider (0.01 = top 1%)

    Returns:
        Fraction of total weight in top samples
    """
    n = len(weights)
    k = max(1, int(n * top_pct))
    sorted_weights = np.sort(weights)[::-1]  # Descending
    return float(sorted_weights[:k].sum() / weights.sum())


# ========== Diagnostic Computation ==========


def compute_weight_diagnostics(
    weights: np.ndarray, policy: str = "unknown"
) -> Dict[str, Any]:
    """Compute weight diagnostics for a single policy.

    Returns dict with: ess_fraction, max_weight, tail_ratio_99_5, status
    """
    n = len(weights)
    ess = effective_sample_size(weights)
    ess_fraction = ess / n if n > 0 else 0.0

    # Tail ratio (p99/p5)
    tail_ratio = tail_weight_ratio(weights, 0.05, 0.99)

    # Determine status
    if ess_fraction < 0.01 or tail_ratio > 1000:
        status = Status.CRITICAL
    elif ess_fraction < 0.1 or tail_ratio > 100:
        status = Status.WARNING
    else:
        status = Status.GOOD

    return {
        "ess_fraction": ess_fraction,
        "max_weight": float(weights.max()),
        "tail_ratio_99_5": tail_ratio,
        "status": status,
    }
