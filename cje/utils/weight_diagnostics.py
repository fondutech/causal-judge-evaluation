"""
Weight Diagnostics Utilities

Compute and display importance weight diagnostics to catch
consistency issues and low ESS problems early.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
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
    extreme_weight_count: int  # Weights > 1000 or < 0.001
    zero_weight_count: int  # Exactly zero weights
    consistency_flag: str  # "GOOD", "WARNING", "CRITICAL"
    log_weight_range: float  # Max - Min log weights
    weight_coefficient_variation: float  # Std/Mean of weights


def compute_importance_weights(
    behavior_logprobs: List[float], target_logprobs: List[float]
) -> List[float]:
    """Compute importance weights from log probabilities."""
    if len(behavior_logprobs) != len(target_logprobs):
        raise ValueError("Behavior and target log-probs must have same length")

    weights = []
    for logp_behavior, logp_target in zip(behavior_logprobs, target_logprobs):
        try:
            weight = math.exp(logp_target - logp_behavior)
            weights.append(weight)
        except OverflowError:
            # Handle extreme log-prob differences
            log_diff = logp_target - logp_behavior
            if log_diff > 50:  # exp(50) ≈ 5e21
                weights.append(float("inf"))
            else:  # log_diff < -50, exp(-50) ≈ 2e-22
                weights.append(0.0)

    return weights


def compute_ess(weights: List[float]) -> float:
    """Compute Effective Sample Size (ESS) from importance weights."""
    if not weights:
        return 0.0

    # Handle infinite or zero weights
    finite_weights = [w for w in weights if math.isfinite(w) and w > 0]
    if not finite_weights:
        return 0.0

    # ESS = (sum w)^2 / sum(w^2)
    sum_w = sum(finite_weights)
    sum_w2 = sum(w * w for w in finite_weights)

    if sum_w2 == 0:
        return 0.0

    return (sum_w * sum_w) / sum_w2


def diagnose_weights(
    weights: List[float],
    policy_name: str = "Unknown",
    expected_weight: Optional[float] = None,
) -> WeightDiagnostics:
    """Compute comprehensive weight diagnostics."""

    if not weights:
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
            log_weight_range=0.0,
            weight_coefficient_variation=0.0,
        )

    # Basic statistics
    finite_weights = [w for w in weights if math.isfinite(w)]
    if not finite_weights:
        finite_weights = [0.0]  # Fallback

    min_weight = min(finite_weights)
    max_weight = max(finite_weights)
    mean_weight = float(np.mean(finite_weights))  # Convert to Python float
    median_weight = float(np.median(finite_weights))  # Convert to Python float

    # Count problematic weights
    extreme_count = sum(
        1 for w in weights if not math.isfinite(w) or w > 1000 or w < 0.001
    )
    zero_count = sum(1 for w in weights if w == 0.0)

    # ESS fraction
    ess = compute_ess(weights)
    ess_fraction = ess / len(weights) if weights else 0.0

    # Log weight range (for detecting extreme differences)
    positive_weights = [w for w in finite_weights if w > 0]
    if positive_weights:
        log_weights = [math.log(w) for w in positive_weights]
        log_weight_range = max(log_weights) - min(log_weights)
    else:
        log_weight_range = 0.0

    # Coefficient of variation
    if mean_weight > 0:
        weight_std = float(np.std(finite_weights))  # Convert to Python float
        cv = weight_std / mean_weight
    else:
        cv = float("inf")

    # Determine consistency flag
    consistency_flag = determine_consistency_flag(
        ess_fraction, extreme_count, len(weights), expected_weight, mean_weight
    )

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
        log_weight_range=log_weight_range,
        weight_coefficient_variation=cv,
    )


def determine_consistency_flag(
    ess_fraction: float,
    extreme_count: int,
    total_count: int,
    expected_weight: Optional[float],
    mean_weight: float,
) -> str:
    """Determine if weights indicate a problem."""

    # Critical issues
    if ess_fraction < 0.01:  # < 1% ESS
        return "CRITICAL"
    if extreme_count > total_count * 0.5:  # > 50% extreme weights
        return "CRITICAL"

    # Check for identical policy consistency
    if expected_weight is not None:
        weight_error = abs(mean_weight - expected_weight)
        if weight_error > 0.1:  # Mean weight should be ~expected for identical policies
            return "CRITICAL"

    # Warning conditions
    if ess_fraction < 0.1:  # < 10% ESS
        return "WARNING"
    if extreme_count > total_count * 0.1:  # > 10% extreme weights
        return "WARNING"

    return "GOOD"


def format_weight_diagnostics(diagnostics: WeightDiagnostics) -> str:
    """Format weight diagnostics for display."""

    flag_emoji = {"GOOD": "✅", "WARNING": "⚠️", "CRITICAL": "❌"}

    emoji = flag_emoji.get(diagnostics.consistency_flag, "❓")

    lines = [
        f"{emoji} **{diagnostics.policy_name}** Weight Diagnostics:",
        f"   ESS: {diagnostics.ess_fraction:.1%} (Effective Sample Size)",
        f"   Range: {diagnostics.min_weight:.2e} to {diagnostics.max_weight:.2e}",
        f"   Mean: {diagnostics.mean_weight:.4f}, Median: {diagnostics.median_weight:.4f}",
        f"   Extreme weights: {diagnostics.extreme_weight_count} (>{1000} or <{0.001})",
        f"   Zero weights: {diagnostics.zero_weight_count}",
    ]

    if diagnostics.consistency_flag != "GOOD":
        lines.append(f"   🚨 Status: {diagnostics.consistency_flag}")

        if diagnostics.ess_fraction < 0.1:
            lines.append(f"      - Low ESS: {diagnostics.ess_fraction:.1%} < 10%")
        if diagnostics.extreme_weight_count > 0:
            lines.append(
                f"      - {diagnostics.extreme_weight_count} extreme weights detected"
            )

    return "\n".join(lines)


def analyze_arena_weights(data: List[Dict[str, Any]]) -> Dict[str, WeightDiagnostics]:
    """Analyze importance weights for all policies in arena data."""

    if not data:
        return {}

    # Extract log probabilities
    behavior_logprobs = [record["logp"] for record in data]

    # Get all target policies
    target_policies = set()
    for record in data:
        if "logp_target_all" in record:
            target_policies.update(record["logp_target_all"].keys())

    diagnostics = {}

    for policy_name in target_policies:
        # Extract target log-probs for this policy
        target_logprobs = []
        for record in data:
            logp_target = record.get("logp_target_all", {}).get(policy_name, 0.0)
            target_logprobs.append(logp_target)

        # Compute weights
        weights = compute_importance_weights(behavior_logprobs, target_logprobs)

        # Determine expected weight (1.0 for scout/identical policies)
        expected_weight = 1.0 if "scout" in policy_name.lower() else None

        # Diagnose
        policy_diagnostics = diagnose_weights(weights, policy_name, expected_weight)
        diagnostics[policy_name] = policy_diagnostics

    return diagnostics


def create_weight_summary_table(diagnostics: Dict[str, WeightDiagnostics]) -> str:
    """Create a summary table of weight diagnostics."""

    if not diagnostics:
        return "No weight diagnostics available."

    lines = [
        "📊 **Importance Weight Summary**",
        "",
        "| Policy | ESS | Mean Weight | Status | Issues |",
        "|--------|-----|-------------|--------|--------|",
    ]

    for policy_name, diag in diagnostics.items():
        status_emoji = {"GOOD": "✅", "WARNING": "⚠️", "CRITICAL": "❌"}[
            diag.consistency_flag
        ]

        issues = []
        if diag.ess_fraction < 0.1:
            issues.append(f"Low ESS ({diag.ess_fraction:.1%})")
        if diag.extreme_weight_count > 0:
            issues.append(f"{diag.extreme_weight_count} extreme")
        if diag.zero_weight_count > 0:
            issues.append(f"{diag.zero_weight_count} zero")

        issues_str = ", ".join(issues) if issues else "None"

        lines.append(
            f"| {policy_name} | {diag.ess_fraction:.1%} | {diag.mean_weight:.4f} | "
            f"{status_emoji} {diag.consistency_flag} | {issues_str} |"
        )

    return "\n".join(lines)
