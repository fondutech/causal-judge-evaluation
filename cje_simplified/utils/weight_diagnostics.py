"""Weight diagnostics for debugging importance sampling issues."""

import math
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class WeightDiagnostics:
    """Container for weight diagnostic results."""

    policy_name: str
    min_weight: float
    max_weight: float
    mean_weight: float
    median_weight: float
    ess_fraction: float  # Effective Sample Size as fraction of N
    extreme_weight_count: int  # Weights > 100 or < 0.01
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


def compute_ess(weights: np.ndarray) -> float:
    """Compute Effective Sample Size (ESS) from importance weights.

    ESS = (sum w)^2 / sum(w^2)

    Returns value between 0 and n_samples.
    """
    if len(weights) == 0:
        return 0.0

    # Handle infinite or zero weights
    finite_weights = weights[np.isfinite(weights) & (weights > 0)]
    if len(finite_weights) == 0:
        return 0.0

    sum_w = np.sum(finite_weights)
    sum_w2 = np.sum(finite_weights**2)

    if sum_w2 == 0:
        return 0.0

    return (sum_w**2) / sum_w2


def diagnose_weights(
    weights: np.ndarray,
    policy_name: str = "Unknown",
    expected_weight: Optional[float] = None,
) -> WeightDiagnostics:
    """Compute comprehensive weight diagnostics.

    Args:
        weights: Array of importance weights
        policy_name: Name of the policy
        expected_weight: Expected mean weight (e.g., 1.0 for identical policies)

    Returns:
        WeightDiagnostics object with detailed statistics
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

    # Basic statistics
    finite_weights = weights[np.isfinite(weights)]
    if len(finite_weights) == 0:
        finite_weights = np.array([0.0])

    min_weight = float(np.min(finite_weights))
    max_weight = float(np.max(finite_weights))
    mean_weight = float(np.mean(finite_weights))
    median_weight = float(np.median(finite_weights))

    # Count problematic weights
    extreme_count = int(
        np.sum((weights > 100) | (weights < 0.01) | ~np.isfinite(weights))
    )
    zero_count = int(np.sum(weights == 0.0))

    # ESS fraction
    ess = compute_ess(weights)
    ess_fraction = ess / len(weights)

    # Determine consistency flag
    consistency_flag = "GOOD"

    # Critical issues
    if ess_fraction < 0.01:  # < 1% ESS
        consistency_flag = "CRITICAL"
    elif extreme_count > len(weights) * 0.5:  # > 50% extreme weights
        consistency_flag = "CRITICAL"
    # Check expected weight (e.g., for identical policies)
    elif expected_weight is not None and abs(mean_weight - expected_weight) > 0.1:
        consistency_flag = "CRITICAL"
    # Warning conditions
    elif ess_fraction < 0.1:  # < 10% ESS
        consistency_flag = "WARNING"
    elif extreme_count > len(weights) * 0.1:  # > 10% extreme weights
        consistency_flag = "WARNING"

    return WeightDiagnostics(
        policy_name=policy_name,
        min_weight=min_weight,
        max_weight=max_weight,
        mean_weight=mean_weight,
        median_weight=median_weight,
        ess_fraction=ess_fraction,
        extreme_weight_count=extreme_count,
        zero_weight_count=zero_count,
        consistency_flag=consistency_flag,
    )


def create_weight_summary_table(all_diagnostics: Dict[str, WeightDiagnostics]) -> str:
    """Create a summary table of weight diagnostics for multiple policies.

    Args:
        all_diagnostics: Dict mapping policy names to diagnostics

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


def detect_api_nondeterminism(
    sampler, baseline_policy: str = "pi_clone", tolerance: float = 0.05
) -> Dict[str, Any]:
    """Detect if API returns inconsistent log probabilities.

    Checks if a baseline policy (that should be identical to behavior policy)
    has mean weight significantly different from 1.0.

    Args:
        sampler: PrecomputedSampler with data
        baseline_policy: Name of policy that should be identical to behavior
        tolerance: Tolerance for mean weight deviation from 1.0

    Returns:
        Dict with detection results and recommendations
    """
    if baseline_policy not in sampler.target_policies:
        return {
            "detected": False,
            "reason": f"Baseline policy {baseline_policy} not found",
        }

    # Compute weights for baseline
    weights = sampler.compute_importance_weights(baseline_policy)
    mean_weight = np.mean(weights[np.isfinite(weights)])
    deviation = abs(mean_weight - 1.0)

    detected = deviation > tolerance

    return {
        "detected": detected,
        "mean_weight": float(mean_weight),
        "deviation": float(deviation),
        "recommendation": (
            (
                "API non-determinism detected. Consider:\n"
                "1. Using force_continuation=True in teacher forcing\n"
                "2. Averaging multiple API calls\n"
                "3. Using a deterministic evaluation mode"
            )
            if detected
            else "No significant API non-determinism detected"
        ),
    }
