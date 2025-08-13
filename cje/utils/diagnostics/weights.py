"""
Weight diagnostic computations for importance sampling.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
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


# ========== Legacy Interface (for backward compatibility) ==========


@dataclass
class WeightDiagnostics:
    """Container for weight diagnostic results (legacy interface)."""

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
            f"  Extreme weights: {self.extreme_weight_count}",
            f"  Zero weights: {self.zero_weight_count}",
            f"  Status: {self.consistency_flag}",
        ]
        return "\n".join(lines)


def diagnose_weights(
    weights: np.ndarray,
    policy_name: str = "target",
    extreme_quantile: float = 0.99,
) -> WeightDiagnostics:
    """Compute comprehensive weight diagnostics (legacy interface).

    Args:
        weights: Importance weights
        policy_name: Name of the policy
        extreme_quantile: Quantile for defining extreme weights

    Returns:
        WeightDiagnostics dataclass with all metrics
    """
    n = len(weights)
    if n == 0:
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

    # Compute ESS
    ess = effective_sample_size(weights)
    ess_fraction = ess / n

    # Compute thresholds
    extreme_threshold = np.quantile(weights, extreme_quantile)

    # Count special cases
    extreme_count = int(np.sum(weights > extreme_threshold))
    zero_count = int(np.sum(weights == 0))

    # Determine consistency flag
    if ess_fraction < 0.01 or zero_count > n * 0.1:
        flag = "CRITICAL"
    elif ess_fraction < 0.1 or extreme_count > n * 0.05:
        flag = "WARNING"
    else:
        flag = "GOOD"

    return WeightDiagnostics(
        policy_name=policy_name,
        min_weight=float(weights.min()),
        max_weight=float(weights.max()),
        mean_weight=float(weights.mean()),
        median_weight=float(np.median(weights)),
        ess_fraction=ess_fraction,
        extreme_weight_count=extreme_count,
        zero_weight_count=zero_count,
        consistency_flag=flag,
    )
