"""Comprehensive diagnostic utilities for CJE estimators.

This module provides diagnostic functions for weight-based estimators,
focusing on overlap, balance, and reliability metrics.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute ESS = (sum(w))^2 / sum(w^2).

    ESS measures how many "effective" samples we have after weighting.
    ESS = n means perfect overlap, ESS << n means poor overlap.
    """
    s = weights.sum()
    s2 = np.sum(weights**2)
    return float((s * s) / np.maximum(s2, 1e-12))


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


def weight_diagnostics(
    weights: np.ndarray, n_total: Optional[int] = None, n_valid: Optional[int] = None
) -> Dict[str, Any]:
    """Comprehensive weight diagnostics.

    Args:
        weights: Importance weights (should have mean ≈ 1)
        n_total: Total samples before filtering
        n_valid: Valid samples after filtering

    Returns:
        Dictionary with nested diagnostic metrics:
        - weights: ESS, quantiles, tail ratios, concentration
        - data: filtering statistics (if provided)
    """
    n = len(weights)
    ess = effective_sample_size(weights)

    diagnostics = {
        "weights": {
            # Basic statistics
            "n": n,
            "ess": float(ess),
            "ess_fraction": float(ess / n),
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "cv": float(weights.std() / (weights.mean() + 1e-12)),
            # Quantiles
            "min": float(weights.min()),
            "p25": float(np.quantile(weights, 0.25)),
            "p50": float(np.quantile(weights, 0.50)),
            "p75": float(np.quantile(weights, 0.75)),
            "p95": float(np.quantile(weights, 0.95)),
            "p99": float(np.quantile(weights, 0.99)),
            "max": float(weights.max()),
            # Tail behavior (using p5 instead of p1 for stability)
            "tail_ratio_99_5": tail_weight_ratio(weights, 0.05, 0.99),
            "tail_ratio_99_1": tail_weight_ratio(weights, 0.01, 0.99),
            "tail_ratio_95_5": tail_weight_ratio(weights, 0.05, 0.95),
            # Mass concentration
            "top1_share": mass_concentration(weights, 0.01),
            "top5_share": mass_concentration(weights, 0.05),
            # Extreme weights
            "share_above_10x": float((weights > 10).mean()),
            "share_above_100x": float((weights > 100).mean()),
        }
    }

    # Add data filtering info if provided
    if n_total is not None and n_valid is not None:
        diagnostics["data"] = {
            "n_total": n_total,
            "n_valid": n_valid,
            "filter_rate": float(1 - n_valid / n_total) if n_total > 0 else 0.0,
        }

    return diagnostics


def standardized_diffs(features: np.ndarray, weights: np.ndarray) -> Dict[str, Any]:
    """Compute standardized differences for balance assessment.

    Args:
        features: (n, p) array of context-only features
        weights: (n,) array with mean ≈ 1 (Hajek normalized)

    Returns:
        Balance diagnostics including ASAMD (Average Standardized Absolute Mean Difference)
    """
    n, p = features.shape

    # Unweighted moments
    mu0 = features.mean(axis=0)
    sd0 = features.std(axis=0, ddof=1) + 1e-12

    # Weighted moments (Hajek: mean(w)=1 => divide by n)
    muw = (weights[:, None] * features).sum(axis=0) / n

    # Standardized differences using unweighted SD
    sdiff = (muw - mu0) / sd0
    asamd = np.abs(sdiff)

    return {
        "avg_asamd": float(asamd.mean()),
        "max_std_diff": float(np.max(asamd)),
        "per_feature": asamd.tolist() if p <= 20 else None,  # Limit output size
        "n_features": p,
    }


def evaluate_status(diagnostics: Dict[str, Any]) -> str:
    """Evaluate diagnostic status (green/amber/red).

    Advisory thresholds - tune based on domain experience.

    Args:
        diagnostics: Output from weight_diagnostics()

    Returns:
        Status string: 'green', 'amber', or 'red'
    """
    weights = diagnostics.get("weights", {})
    balance = diagnostics.get("balance", {})

    # Extract key metrics
    ess_frac = weights.get("ess_fraction", 0)
    tail_ratio = weights.get("tail_ratio_99_5", np.inf)
    asamd = balance.get("avg_asamd", 0)

    # Green thresholds (conservative)
    if ess_frac >= 0.30 and tail_ratio <= 100 and asamd <= 0.05:
        return "green"

    # Amber thresholds (acceptable)
    if ess_frac >= 0.20 and tail_ratio <= 500 and asamd <= 0.10:
        return "amber"

    return "red"


def format_diagnostic_summary(diagnostics: Dict[str, Any]) -> str:
    """Format diagnostics as a human-readable summary.

    Args:
        diagnostics: Output from weight_diagnostics()

    Returns:
        Formatted string summary
    """
    w = diagnostics.get("weights", {})
    d = diagnostics.get("data", {})

    lines = []

    # Data filtering (if available)
    if d:
        lines.append(
            f"Data: {d['n_valid']}/{d['n_total']} valid ({100*(1-d['filter_rate']):.1f}%)"
        )

    # Key weight metrics
    lines.append(f"ESS: {w['ess']:.1f} ({100*w['ess_fraction']:.1f}% of n={w['n']})")
    lines.append(f"Weight range: [{w['min']:.3f}, {w['max']:.3f}]")
    lines.append(f"Tail ratio (p99/p5): {w['tail_ratio_99_5']:.1f}")
    lines.append(f"Top 1% hold {100*w['top1_share']:.1f}% of weight")
    lines.append(f"Top 5% hold {100*w['top5_share']:.1f}% of weight")

    # Extreme weights
    if w["share_above_10x"] > 0:
        lines.append(f"Weights >10x mean: {100*w['share_above_10x']:.1f}%")
    if w["share_above_100x"] > 0:
        lines.append(f"Weights >100x mean: {100*w['share_above_100x']:.1f}%")

    return "\n".join(lines)
