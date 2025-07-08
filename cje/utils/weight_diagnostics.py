"""
Weight Diagnostics Utilities

Compute and display importance weight diagnostics to catch
consistency issues and low ESS problems early.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats


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
    # Overlap diagnostics
    overlap_score: Optional[float] = None  # 0-1, higher is better
    common_support_fraction: Optional[float] = None  # Fraction with good support
    # API non-determinism indicators
    api_noise_suspected: bool = False  # True if API non-determinism detected
    baseline_deviation: Optional[float] = (
        None  # For identical policies, deviation from 1.0
    )


@dataclass
class OverlapDiagnostics:
    """Container for overlap analysis results."""

    overlap_score: float  # 0-1 overlap measure
    common_support_fraction: float  # Fraction with good propensity support
    log_ratio_percentiles: Dict[int, float]  # 5th, 25th, 50th, 75th, 95th
    extreme_log_ratio_fraction: float  # Fraction with |log_ratio| > threshold
    positivity_violations: int  # Count of near-zero propensities


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

    # Check for API noise (baseline policies deviating from expected weight of 1.0)
    api_noise_suspected = False
    baseline_deviation = None
    if expected_weight is not None:
        baseline_deviation = abs(mean_weight - expected_weight)
        # If a baseline policy has mean weight > 0.05 from expected, suspect API noise
        if baseline_deviation > 0.05:
            api_noise_suspected = True

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
        api_noise_suspected=api_noise_suspected,
        baseline_deviation=baseline_deviation,
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

    # Add overlap diagnostics if available
    if diagnostics.overlap_score is not None:
        lines.append(f"   Overlap score: {diagnostics.overlap_score:.2f} (0-1 scale)")
        lines.append(f"   Common support: {diagnostics.common_support_fraction:.1%}")

    # Add API noise warning
    if diagnostics.api_noise_suspected:
        lines.append(f"   ⚠️  API non-determinism suspected")
        if diagnostics.baseline_deviation is not None:
            lines.append(
                f"      - Baseline weight deviation: {diagnostics.baseline_deviation:.3f} from expected 1.0"
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

        # Determine expected weight (1.0 for scout/clone/identical policies)
        is_baseline = any(
            marker in policy_name.lower() for marker in ["scout", "clone", "p0"]
        )
        expected_weight = 1.0 if is_baseline else None

        # Diagnose with overlap
        policy_diagnostics = diagnose_weights_with_overlap(
            weights, behavior_logprobs, target_logprobs, policy_name, expected_weight
        )
        diagnostics[policy_name] = policy_diagnostics

    return diagnostics


def compute_overlap_diagnostics(
    behavior_logprobs: List[float],
    target_logprobs: List[float],
    min_logprob_threshold: float = -50.0,
) -> OverlapDiagnostics:
    """
    Compute overlap diagnostics between behavior and target policies.

    Args:
        behavior_logprobs: Log probabilities under behavior policy
        target_logprobs: Log probabilities under target policy
        min_logprob_threshold: Threshold for near-zero probabilities

    Returns:
        OverlapDiagnostics object with overlap metrics
    """
    if len(behavior_logprobs) != len(target_logprobs):
        raise ValueError("Behavior and target log-probs must have same length")

    n = len(behavior_logprobs)
    log_ratios = []
    positivity_violations = 0

    for logp_b, logp_t in zip(behavior_logprobs, target_logprobs):
        # Check for positivity violations
        if logp_b < min_logprob_threshold or logp_t < min_logprob_threshold:
            positivity_violations += 1
            continue

        log_ratio = logp_t - logp_b
        log_ratios.append(log_ratio)

    if not log_ratios:
        # No valid samples for overlap computation
        return OverlapDiagnostics(
            overlap_score=0.0,
            common_support_fraction=0.0,
            log_ratio_percentiles={5: 0.0, 25: 0.0, 50: 0.0, 75: 0.0, 95: 0.0},
            extreme_log_ratio_fraction=1.0,
            positivity_violations=positivity_violations,
        )

    # Compute percentiles
    percentiles = np.percentile(log_ratios, [5, 25, 50, 75, 95])
    log_ratio_percentiles = {
        5: float(percentiles[0]),
        25: float(percentiles[1]),
        50: float(percentiles[2]),
        75: float(percentiles[3]),
        95: float(percentiles[4]),
    }

    # Compute overlap score (based on normalized entropy of weights)
    weights = [math.exp(lr) for lr in log_ratios]
    weights_normalized = np.array(weights) / np.sum(weights)
    # Overlap score: 1 - normalized entropy (higher means better overlap)
    entropy = -np.sum(weights_normalized * np.log(weights_normalized + 1e-10))
    max_entropy = math.log(len(weights))
    overlap_score = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

    # Common support fraction
    common_support_fraction = len(log_ratios) / n

    # Extreme log ratio fraction (|log_ratio| > 5)
    extreme_threshold = 5.0
    extreme_count = sum(1 for lr in log_ratios if abs(lr) > extreme_threshold)
    extreme_log_ratio_fraction = extreme_count / len(log_ratios) if log_ratios else 0.0

    return OverlapDiagnostics(
        overlap_score=overlap_score,
        common_support_fraction=common_support_fraction,
        log_ratio_percentiles=log_ratio_percentiles,
        extreme_log_ratio_fraction=extreme_log_ratio_fraction,
        positivity_violations=positivity_violations,
    )


def diagnose_weights_with_overlap(
    weights: List[float],
    behavior_logprobs: List[float],
    target_logprobs: List[float],
    policy_name: str = "Unknown",
    expected_weight: Optional[float] = None,
) -> WeightDiagnostics:
    """
    Compute comprehensive weight diagnostics including overlap analysis.

    This extends the basic diagnose_weights function with overlap metrics.
    """
    # Get basic weight diagnostics
    basic_diagnostics = diagnose_weights(weights, policy_name, expected_weight)

    # Compute overlap diagnostics
    overlap_diag = compute_overlap_diagnostics(behavior_logprobs, target_logprobs)

    # Add overlap metrics to weight diagnostics
    basic_diagnostics.overlap_score = overlap_diag.overlap_score
    basic_diagnostics.common_support_fraction = overlap_diag.common_support_fraction

    # Update consistency flag based on overlap
    if overlap_diag.common_support_fraction < 0.5:  # Less than 50% common support
        if basic_diagnostics.consistency_flag == "GOOD":
            basic_diagnostics.consistency_flag = "WARNING"
        elif basic_diagnostics.consistency_flag == "WARNING":
            basic_diagnostics.consistency_flag = "CRITICAL"

    # Check for API noise in baseline comparisons
    # If this is a baseline policy (e.g., pi_clone) with poor overlap, suspect API noise
    if "clone" in policy_name.lower() or "scout" in policy_name.lower():
        if overlap_diag.overlap_score < 0.7:  # Poor overlap for identical policy
            basic_diagnostics.api_noise_suspected = True

    return basic_diagnostics


def create_weight_summary_table(diagnostics: Dict[str, WeightDiagnostics]) -> str:
    """Create a summary table of weight diagnostics."""

    if not diagnostics:
        return "No weight diagnostics available."

    lines = [
        "📊 **Importance Weight Summary**",
        "",
        "| Policy | ESS | Mean Weight | Overlap | Status | Issues |",
        "|--------|-----|-------------|---------|--------|--------|",
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
        if (
            diag.common_support_fraction is not None
            and diag.common_support_fraction < 0.5
        ):
            issues.append(f"Poor overlap ({diag.common_support_fraction:.1%})")
        if diag.api_noise_suspected:
            issues.append("API noise")

        issues_str = ", ".join(issues) if issues else "None"

        # Format overlap score
        overlap_str = (
            f"{diag.overlap_score:.2f}" if diag.overlap_score is not None else "N/A"
        )

        lines.append(
            f"| {policy_name} | {diag.ess_fraction:.1%} | {diag.mean_weight:.4f} | "
            f"{overlap_str} | {status_emoji} {diag.consistency_flag} | {issues_str} |"
        )

    return "\n".join(lines)


def format_overlap_diagnostics(
    overlap_diag: OverlapDiagnostics,
    policy_name: str = "Unknown",
) -> str:
    """Format detailed overlap diagnostics for display."""

    lines = [
        f"📈 **{policy_name}** Overlap Analysis:",
        f"   Overlap Score: {overlap_diag.overlap_score:.3f} (0=poor, 1=perfect)",
        f"   Common Support: {overlap_diag.common_support_fraction:.1%}",
        f"   Positivity Violations: {overlap_diag.positivity_violations}",
    ]

    # Add log ratio percentiles
    lines.append("   Log Ratio Percentiles:")
    for p, value in overlap_diag.log_ratio_percentiles.items():
        lines.append(f"      P{p}: {value:+.2f}")

    # Add warnings
    if overlap_diag.extreme_log_ratio_fraction > 0.1:
        lines.append(
            f"   ⚠️  {overlap_diag.extreme_log_ratio_fraction:.1%} of samples have extreme log ratios"
        )

    if overlap_diag.common_support_fraction < 0.8:
        lines.append(
            f"   ⚠️  Limited overlap: only {overlap_diag.common_support_fraction:.1%} common support"
        )

    return "\n".join(lines)


def detect_api_nondeterminism(
    data: List[Dict[str, Any]],
    behavior_policy_name: str = "p0",
    baseline_policy_name: str = "pi_clone",
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """
    Detect API non-determinism by analyzing log probability differences
    between supposedly identical policies.

    Args:
        data: List of records with log probabilities
        behavior_policy_name: Name of behavior policy (default: "p0")
        baseline_policy_name: Name of baseline policy that should be identical (default: "pi_clone")
        tolerance: Tolerance for mean weight deviation from 1.0

    Returns:
        Dictionary with detection results and statistics
    """
    if not data:
        return {"detected": False, "reason": "No data provided"}

    # Check if baseline policy exists in data
    sample_record = data[0]
    if "logp_target_all" not in sample_record:
        return {"detected": False, "reason": "No target log probabilities found"}

    if baseline_policy_name not in sample_record["logp_target_all"]:
        return {
            "detected": False,
            "reason": f"Baseline policy {baseline_policy_name} not found",
        }

    # Extract log probabilities
    behavior_logprobs = []
    baseline_logprobs = []
    log_diffs = []
    identical_response_diffs = []

    for record in data:
        behavior_logp = record.get("logp", 0.0)
        baseline_logp = record.get("logp_target_all", {}).get(baseline_policy_name, 0.0)

        behavior_logprobs.append(behavior_logp)
        baseline_logprobs.append(baseline_logp)

        # Calculate log difference
        if behavior_logp != 0.0 or baseline_logp != 0.0:  # Skip empty responses
            log_diff = abs(baseline_logp - behavior_logp)
            log_diffs.append(log_diff)

            # Check if responses are identical
            behavior_response = record.get("response", "")
            baseline_response = record.get("responses_target_all", {}).get(
                baseline_policy_name, ""
            )

            if behavior_response == baseline_response and behavior_response != "":
                identical_response_diffs.append(log_diff)

    if not log_diffs:
        return {
            "detected": False,
            "reason": "No valid log probability differences found",
        }

    # Compute importance weights
    weights = compute_importance_weights(behavior_logprobs, baseline_logprobs)
    finite_weights = [w for w in weights if math.isfinite(w) and w > 0]

    if not finite_weights:
        return {"detected": False, "reason": "No valid importance weights computed"}

    # Analyze results
    mean_weight = float(np.mean(finite_weights))
    weight_deviation = abs(mean_weight - 1.0)

    # Detect non-determinism
    detected = False
    reasons = []

    if weight_deviation > tolerance:
        detected = True
        reasons.append(
            f"Mean weight {mean_weight:.3f} deviates from expected 1.0 by {weight_deviation:.3f}"
        )

    if identical_response_diffs:
        max_identical_diff = max(identical_response_diffs)
        if max_identical_diff > 0.1:
            detected = True
            reasons.append(
                f"Found {len(identical_response_diffs)} identical responses with log prob differences up to {max_identical_diff:.2f}"
            )

    # Compute statistics
    results = {
        "detected": detected,
        "reasons": reasons,
        "statistics": {
            "mean_weight": mean_weight,
            "weight_deviation": weight_deviation,
            "mean_log_diff": float(np.mean(log_diffs)) if log_diffs else 0.0,
            "max_log_diff": max(log_diffs) if log_diffs else 0.0,
            "identical_response_count": len(identical_response_diffs),
            "max_identical_response_diff": (
                max(identical_response_diffs) if identical_response_diffs else 0.0
            ),
            "ess_fraction": (
                compute_ess(finite_weights) / len(weights) if weights else 0.0
            ),
        },
        "recommendation": "Consider averaging multiple API calls or using a deterministic evaluation mode to reduce variance.",
    }

    return results
