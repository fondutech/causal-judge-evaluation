"""Comprehensive weight diagnostics for importance sampling.

This module provides:
- Core metric calculations (ESS, tail ratios, etc.)
- Diagnostic computation (both dict and dataclass interfaces)
- Formatting and visualization utilities
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ========== Core Metrics (Pure Functions) ==========


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute ESS = (sum(w))^2 / sum(w^2).

    ESS measures how many "effective" samples we have after weighting.
    ESS = n means perfect overlap, ESS << n means poor overlap.
    """
    s = weights.sum()
    s2 = np.sum(weights**2)
    return float((s * s) / np.maximum(s2, 1e-12))


def compute_ess(weights: np.ndarray) -> float:
    """Compute Effective Sample Size (ESS) from importance weights.

    Alias for effective_sample_size() - kept for backward compatibility.
    """
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


# ========== Data Models ==========


@dataclass
class WeightDiagnostics:
    """Container for weight diagnostic results (public API)."""

    policy_name: str
    min_weight: float
    max_weight: float
    mean_weight: float
    median_weight: float
    ess_fraction: float  # Effective Sample Size as fraction of N
    extreme_weight_count: int  # Weights beyond thresholds
    zero_weight_count: int  # Exactly zero weights
    consistency_flag: str  # "GOOD", "WARNING", "CRITICAL"

    def summary(self) -> str:
        """Format diagnostics as readable summary."""
        flag_emoji = {"GOOD": "✅", "WARNING": "⚠️", "CRITICAL": "❌"}
        emoji = flag_emoji.get(self.consistency_flag, "❓")

        lines = [
            f"{emoji} {self.policy_name} Weight Diagnostics:",
            f"  ESS: {self.ess_fraction:.1%} (Effective Sample Size)",
            f"  Range: {self.min_weight:.2e} to {self.max_weight:.2e}",
            f"  Mean: {self.mean_weight:.4f}, Median: {self.median_weight:.4f}",
        ]

        if self.extreme_weight_count > 0:
            lines.append(f"  Extreme weights: {self.extreme_weight_count}")
        if self.zero_weight_count > 0:
            lines.append(f"  Zero weights: {self.zero_weight_count}")

        if self.consistency_flag != "GOOD":
            lines.append(f"  Status: {self.consistency_flag}")

        return "\n".join(lines)


# ========== Main Diagnostic Functions ==========


def weight_diagnostics(
    weights: np.ndarray, n_total: Optional[int] = None, n_valid: Optional[int] = None
) -> Dict[str, Any]:
    """Comprehensive weight diagnostics (internal use, returns dict).

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


def diagnose_weights(
    weights: np.ndarray,
    policy_name: str = "Unknown",
    expected_weight: Optional[float] = None,
    extreme_threshold_high: float = 100.0,
    extreme_threshold_low: float = 0.01,
) -> WeightDiagnostics:
    """Compute weight diagnostics (public API, returns dataclass).

    Args:
        weights: Array of importance weights
        policy_name: Name of the policy
        expected_weight: Expected mean weight (e.g., 1.0 for identical policies)
        extreme_threshold_high: Weights above this are considered extreme
        extreme_threshold_low: Weights below this are considered extreme

    Returns:
        WeightDiagnostics dataclass with detailed statistics
    """
    if len(weights) == 0:
        return WeightDiagnostics(
            policy_name=policy_name,
            min_weight=0.0,
            max_weight=0.0,
            mean_weight=0.0,
            median_weight=0.0,
            ess_fraction=0.0,
            extreme_weight_count=0,
            zero_weight_count=0,
            consistency_flag="CRITICAL",
        )

    # Get comprehensive diagnostics using internal function
    dict_result = weight_diagnostics(weights)
    w = dict_result["weights"]

    # Count problematic weights
    extreme_count = int(
        np.sum(
            (weights > extreme_threshold_high)
            | (weights < extreme_threshold_low)
            | ~np.isfinite(weights)
        )
    )
    zero_count = int(np.sum(weights == 0.0))

    # Determine consistency flag
    consistency_flag = "GOOD"
    ess_fraction = w["ess_fraction"]

    # Critical issues
    if ess_fraction < 0.01:  # < 1% ESS
        consistency_flag = "CRITICAL"
    elif extreme_count > len(weights) * 0.5:  # > 50% extreme weights
        consistency_flag = "CRITICAL"
    # Check expected weight (e.g., for identical policies)
    elif expected_weight is not None and abs(w["mean"] - expected_weight) > 0.1:
        consistency_flag = "CRITICAL"
    # Warning conditions
    elif ess_fraction < 0.1:  # < 10% ESS
        consistency_flag = "WARNING"
    elif extreme_count > len(weights) * 0.1:  # > 10% extreme weights
        consistency_flag = "WARNING"

    return WeightDiagnostics(
        policy_name=policy_name,
        min_weight=w["min"],
        max_weight=w["max"],
        mean_weight=w["mean"],
        median_weight=w["p50"],
        ess_fraction=ess_fraction,
        extreme_weight_count=extreme_count,
        zero_weight_count=zero_count,
        consistency_flag=consistency_flag,
    )


# ========== Status Evaluation ==========


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

    # Balance may not be populated (requires features)
    # Default to None if not available
    asamd = balance.get("avg_asamd", None) if balance else None

    # Evaluate based on available metrics
    if asamd is not None:
        # Full evaluation with balance
        # Green thresholds (conservative)
        if ess_frac >= 0.30 and tail_ratio <= 100 and asamd <= 0.05:
            return "green"

        # Amber thresholds (acceptable)
        if ess_frac >= 0.20 and tail_ratio <= 500 and asamd <= 0.10:
            return "amber"
    else:
        # Weight-only evaluation (no balance metrics)
        # Green thresholds (conservative, weight-only)
        if ess_frac >= 0.30 and tail_ratio <= 100:
            return "green"

        # Amber thresholds (acceptable, weight-only)
        if ess_frac >= 0.20 and tail_ratio <= 500:
            return "amber"

    return "red"


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


# ========== Formatting Utilities ==========


def create_weight_summary_table(all_diagnostics: Dict[str, WeightDiagnostics]) -> str:
    """Create a summary table of weight diagnostics for multiple policies.

    Args:
        all_diagnostics: Dict mapping policy names to WeightDiagnostics

    Returns:
        Formatted table as string
    """
    if not all_diagnostics:
        return "No weight diagnostics available."

    lines = [
        "Weight Summary",
        "-" * 70,
        f"{'Policy':<20} {'ESS':>10} {'Mean Weight':>15} {'Status':<10}",
        "-" * 70,
    ]

    for policy_name, diag in all_diagnostics.items():
        status_str = diag.consistency_flag
        lines.append(
            f"{policy_name:<20} {diag.ess_fraction:>10.1%} "
            f"{diag.mean_weight:>15.4f} {status_str:<10}"
        )

    return "\n".join(lines)


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
