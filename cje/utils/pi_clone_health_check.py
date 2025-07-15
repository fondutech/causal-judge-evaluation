#!/usr/bin/env python3
"""
Pi_clone health check utility.

Validates that importance weights for identical policies are within acceptable bounds,
detecting API non-determinism issues early in the pipeline.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class HealthCheckResult:
    """Result of pi_clone health check."""

    passed: bool
    weight_mean: float
    weight_cv: float  # Coefficient of variation
    weight_median: float
    ess_fraction: float
    issues: List[str]
    recommendations: List[str]

    def __str__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"Pi_clone Health Check: {status}",
            f"  Mean weight: {self.weight_mean:.3f} (expected: 1.0)",
            f"  Median weight: {self.weight_median:.3f}",
            f"  Coefficient of variation: {self.weight_cv:.3f}",
            f"  ESS fraction: {self.ess_fraction:.1%}",
        ]

        if self.issues:
            lines.append("\nIssues detected:")
            for issue in self.issues:
                lines.append(f"  - {issue}")

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


def check_pi_clone_health(
    weights: List[float],
    mean_threshold: float = 0.3,
    cv_threshold: float = 0.3,
    ess_threshold: float = 0.1,
) -> HealthCheckResult:
    """
    Check health of pi_clone importance weights.

    Args:
        weights: List of importance weights for pi_clone
        mean_threshold: Maximum acceptable deviation of mean from 1.0
        cv_threshold: Maximum acceptable coefficient of variation
        ess_threshold: Minimum acceptable ESS fraction

    Returns:
        HealthCheckResult with diagnostics and recommendations
    """
    if not weights:
        return HealthCheckResult(
            passed=False,
            weight_mean=0.0,
            weight_cv=0.0,
            weight_median=0.0,
            ess_fraction=0.0,
            issues=["No weights provided"],
            recommendations=["Check data pipeline"],
        )

    # Filter finite weights
    finite_weights = [w for w in weights if np.isfinite(w) and w > 0]
    if not finite_weights:
        return HealthCheckResult(
            passed=False,
            weight_mean=0.0,
            weight_cv=0.0,
            weight_median=0.0,
            ess_fraction=0.0,
            issues=["All weights are non-finite or zero"],
            recommendations=["Check log probability computation"],
        )

    # Compute statistics
    weight_array = np.array(finite_weights)
    weight_mean = float(np.mean(weight_array))
    weight_median = float(np.median(weight_array))
    weight_std = float(np.std(weight_array))
    weight_cv = weight_std / weight_mean if weight_mean > 0 else float("inf")

    # Compute ESS
    ess = compute_ess(finite_weights)
    ess_fraction = ess / len(weights)

    # Check for issues
    issues = []
    recommendations = []

    # Check mean deviation
    mean_deviation = abs(weight_mean - 1.0)
    if mean_deviation > mean_threshold:
        issues.append(f"Mean weight deviates by {mean_deviation:.3f} from expected 1.0")
        recommendations.append("Set temperature=1.0 and top_p=1.0 in log prob calls")

    # Check coefficient of variation
    if weight_cv > cv_threshold:
        issues.append(f"High coefficient of variation: {weight_cv:.3f}")
        recommendations.append(
            "Use seed parameter and skip_cache=true for Fireworks API"
        )
        if weight_cv > 1.0:
            recommendations.append("Consider averaging multiple API calls (K=3)")

    # Check ESS
    if ess_fraction < ess_threshold:
        issues.append(f"Low effective sample size: {ess_fraction:.1%}")
        recommendations.append(
            "Consider local evaluation with llama.cpp for deterministic scoring"
        )

    # Additional checks
    extreme_weights = sum(1 for w in finite_weights if w > 100 or w < 0.01)
    if extreme_weights > len(finite_weights) * 0.05:
        issues.append(f"{extreme_weights} extreme weights (>100x or <0.01x)")
        recommendations.append("Check for system prompt differences or padding issues")

    # Determine if passed
    passed = len(issues) == 0

    # Add success message if passed
    if passed:
        recommendations.append(
            "Importance weights look healthy - proceed with analysis"
        )

    return HealthCheckResult(
        passed=passed,
        weight_mean=weight_mean,
        weight_cv=weight_cv,
        weight_median=weight_median,
        ess_fraction=ess_fraction,
        issues=issues,
        recommendations=recommendations,
    )


def compute_ess(weights: List[float]) -> float:
    """Compute Effective Sample Size from importance weights."""
    if not weights:
        return 0.0

    weight_array = np.array(weights)
    sum_w = np.sum(weight_array)
    sum_w2 = np.sum(weight_array**2)

    if sum_w2 == 0:
        return 0.0

    return (sum_w**2) / sum_w2


def run_health_check_on_data(
    behavior_logprobs: List[Optional[float]],
    clone_logprobs: List[Optional[float]],
    verbose: bool = True,
) -> Tuple[HealthCheckResult, List[float]]:
    """
    Run health check on log probability data.

    Args:
        behavior_logprobs: Log probabilities under behavior policy
        clone_logprobs: Log probabilities under pi_clone
        verbose: Whether to print results

    Returns:
        Tuple of (HealthCheckResult, computed_weights)
    """
    # Compute weights for valid samples
    weights = []
    for b_lp, c_lp in zip(behavior_logprobs, clone_logprobs):
        if b_lp is not None and c_lp is not None:
            try:
                weight = np.exp(c_lp - b_lp)
                weights.append(weight)
            except OverflowError:
                # Handle extreme differences
                log_diff = c_lp - b_lp
                if log_diff > 50:
                    weights.append(float("inf"))
                else:
                    weights.append(0.0)

    # Run health check
    result = check_pi_clone_health(weights)

    if verbose:
        print(result)

    # Issue warning if failed
    if not result.passed:
        warnings.warn(
            f"Pi_clone health check failed! Mean weight: {result.weight_mean:.3f}, "
            f"CV: {result.weight_cv:.3f}, ESS: {result.ess_fraction:.1%}",
            RuntimeWarning,
        )

    return result, weights


def suggest_mitigation_config(result: HealthCheckResult) -> Dict[str, any]:
    """
    Suggest configuration changes based on health check results.

    Args:
        result: Health check result

    Returns:
        Dictionary of suggested configuration parameters
    """
    config = {
        "temperature": 1.0,  # Always use 1.0 for log probs
        "top_p": 1.0,  # No nucleus sampling
        "seed": 42,  # Use consistent seed
    }

    # Add Fireworks-specific parameters if needed
    if result.weight_cv > 0.5:
        config["skip_cache"] = True
        config["num_averaging_calls"] = 3

    # Suggest local evaluation for severe issues
    if result.ess_fraction < 0.05 or result.weight_cv > 1.0:
        config["use_local_evaluation"] = True
        config["local_model_path"] = "path/to/llama-scout.gguf"

    return config


# Example usage in a pipeline
if __name__ == "__main__":
    # Example: simulate some weights
    np.random.seed(42)

    # Good case: weights near 1.0
    good_weights = np.random.lognormal(0, 0.1, 100)
    print("Good case:")
    result_good = check_pi_clone_health(good_weights.tolist())
    print(result_good)
    print("\nSuggested config:", suggest_mitigation_config(result_good))

    print("\n" + "=" * 60 + "\n")

    # Bad case: high variance weights
    bad_weights = np.random.lognormal(1.0, 1.5, 100)
    print("Bad case (high variance):")
    result_bad = check_pi_clone_health(bad_weights.tolist())
    print(result_bad)
    print("\nSuggested config:", suggest_mitigation_config(result_bad))
