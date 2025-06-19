"""Variance-based weight shrinkage for uncertainty-aware evaluation.

This module implements optimal shrinkage to improve effective sample size
when judge uncertainty is high.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import logging

from .schemas import JudgeScore, CalibratedReward

logger = logging.getLogger(__name__)


def compute_optimal_shrinkage(
    weights: np.ndarray,
    rewards: np.ndarray,
    variances: np.ndarray,
    value_estimate: float,
) -> float:
    """Compute optimal shrinkage parameter λ* using covariance formula.

    λ* = Cov[w²v, w(r-μ)²] / E[w²v²]

    Args:
        weights: Importance weights
        rewards: Reward values
        variances: Judge variances
        value_estimate: Current estimate of value (μ)

    Returns:
        Optimal shrinkage parameter (clipped to [0, inf))
    """
    if len(weights) != len(rewards) or len(weights) != len(variances):
        raise ValueError("Input arrays must have same length")

    # Compute squared residuals
    residuals_squared = (rewards - value_estimate) ** 2

    # Terms for covariance
    w2v = weights**2 * variances
    wr2 = weights * residuals_squared

    # Compute covariance and denominator
    if len(weights) > 1:
        cov_term = np.cov(w2v, wr2)[0, 1]
    else:
        cov_term = 0.0

    denom = np.mean(weights**2 * variances**2)

    # Handle edge cases
    if denom < 1e-10:
        logger.debug("Denominator too small for shrinkage computation, using λ=0")
        return 0.0

    # Compute optimal lambda
    lambda_star = max(0.0, cov_term / denom)

    # Log result
    logger.info(f"Optimal shrinkage parameter: λ* = {lambda_star:.4f}")
    if lambda_star > 10:
        logger.warning(
            f"Large shrinkage parameter ({lambda_star:.2f}) may indicate high uncertainty"
        )

    return float(lambda_star)


def apply_shrinkage(
    weights: np.ndarray,
    variances: np.ndarray,
    lambda_param: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply variance-based weight shrinkage: w* = w / (1 + λv).

    Args:
        weights: Original importance weights
        variances: Judge variances
        lambda_param: Shrinkage parameter

    Returns:
        Tuple of (shrunk_weights, diagnostics_dict)
    """
    if lambda_param < 0:
        raise ValueError(
            f"Shrinkage parameter must be non-negative, got {lambda_param}"
        )

    # Apply shrinkage
    shrunk_weights = weights / (1 + lambda_param * variances)

    # Compute diagnostics
    ess_original = compute_ess(weights)
    ess_shrunk = compute_ess(shrunk_weights)
    ess_improvement = (ess_shrunk - ess_original) / ess_original * 100

    # Weight reduction statistics
    weight_reductions = 1 - shrunk_weights / weights
    max_reduction = np.max(weight_reductions) * 100
    mean_reduction = np.mean(weight_reductions) * 100

    diagnostics = {
        "lambda": lambda_param,
        "ess_original": ess_original,
        "ess_shrunk": ess_shrunk,
        "ess_improvement_pct": ess_improvement,
        "max_weight_reduction_pct": max_reduction,
        "mean_weight_reduction_pct": mean_reduction,
        "n_weights_reduced": np.sum(weight_reductions > 0.01),
    }

    # Log key statistics
    logger.info(
        f"Shrinkage applied: ESS {ess_original:.1f} → {ess_shrunk:.1f} "
        f"({ess_improvement:+.1f}%)"
    )

    return shrunk_weights, diagnostics


def compute_ess(weights: np.ndarray) -> float:
    """Compute effective sample size from weights.

    ESS = (Σw)² / Σw²

    Args:
        weights: Importance weights

    Returns:
        Effective sample size
    """
    if len(weights) == 0:
        return 0.0

    sum_w = np.sum(weights)
    sum_w2 = np.sum(weights**2)

    if sum_w2 < 1e-10:
        return 0.0

    return float(sum_w**2 / sum_w2)


def adaptive_shrinkage(
    weights: np.ndarray,
    rewards: np.ndarray,
    variances: np.ndarray,
    value_estimate: float,
    min_ess_ratio: float = 0.1,
    max_lambda: float = 100.0,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Apply adaptive shrinkage with ESS constraint.

    Finds the largest λ that maintains ESS ≥ min_ess_ratio * n.

    Args:
        weights: Importance weights
        rewards: Reward values
        variances: Judge variances
        value_estimate: Current value estimate
        min_ess_ratio: Minimum ESS as fraction of n
        max_lambda: Maximum allowed lambda

    Returns:
        Tuple of (shrunk_weights, lambda_used, diagnostics)
    """
    n = len(weights)
    min_ess = min_ess_ratio * n

    # Compute optimal lambda
    lambda_optimal = compute_optimal_shrinkage(
        weights, rewards, variances, value_estimate
    )

    # If optimal lambda maintains ESS constraint, use it
    test_weights = weights / (1 + lambda_optimal * variances)
    test_ess = compute_ess(test_weights)

    if test_ess >= min_ess:
        logger.info(f"Using optimal λ = {lambda_optimal:.4f} (ESS = {test_ess:.1f})")
        shrunk_weights, diag = apply_shrinkage(weights, variances, lambda_optimal)
        return shrunk_weights, lambda_optimal, diag

    # Otherwise, binary search for largest lambda maintaining constraint
    logger.info(f"Optimal λ violates ESS constraint, searching for constrained λ...")

    lambda_low = 0.0
    lambda_high = min(lambda_optimal, max_lambda)

    for _ in range(20):  # Max iterations
        lambda_mid = (lambda_low + lambda_high) / 2
        test_weights = weights / (1 + lambda_mid * variances)
        test_ess = compute_ess(test_weights)

        if test_ess >= min_ess:
            lambda_low = lambda_mid
        else:
            lambda_high = lambda_mid

        if lambda_high - lambda_low < 0.001:
            break

    # Use the found lambda
    final_lambda = lambda_low
    shrunk_weights, diag = apply_shrinkage(weights, variances, final_lambda)

    logger.info(
        f"Constrained λ = {final_lambda:.4f} (optimal was {lambda_optimal:.4f})"
    )

    return shrunk_weights, final_lambda, diag


def analyze_shrinkage_impact(
    original_weights: np.ndarray,
    shrunk_weights: np.ndarray,
    variances: np.ndarray,
    lambda_param: float,
) -> Dict[str, Any]:
    """Analyze the impact of shrinkage on weights and ESS.

    Args:
        original_weights: Weights before shrinkage
        shrunk_weights: Weights after shrinkage
        variances: Judge variances used for shrinkage
        lambda_param: Shrinkage parameter used

    Returns:
        Dictionary with detailed shrinkage analysis
    """
    # Weight changes
    weight_ratios = shrunk_weights / np.maximum(original_weights, 1e-10)

    # Identify most affected samples
    shrinkage_factors = 1 + lambda_param * variances
    most_shrunk_idx = np.argsort(shrinkage_factors)[-10:]

    # ESS analysis
    ess_orig = compute_ess(original_weights)
    ess_shrunk = compute_ess(shrunk_weights)

    # Variance concentration before/after
    var_contrib_orig = original_weights**2 * variances
    var_contrib_shrunk = shrunk_weights**2 * variances

    sorted_orig = np.sort(var_contrib_orig)[::-1]
    sorted_shrunk = np.sort(var_contrib_shrunk)[::-1]

    n_top = max(1, len(original_weights) // 10)
    conc_orig = np.sum(sorted_orig[:n_top]) / np.sum(var_contrib_orig)
    conc_shrunk = np.sum(sorted_shrunk[:n_top]) / np.sum(var_contrib_shrunk)

    return {
        "lambda": lambda_param,
        "ess": {
            "original": ess_orig,
            "shrunk": ess_shrunk,
            "improvement": ess_shrunk - ess_orig,
            "improvement_pct": (ess_shrunk - ess_orig) / ess_orig * 100,
        },
        "weight_changes": {
            "min_ratio": np.min(weight_ratios),
            "mean_ratio": np.mean(weight_ratios),
            "max_ratio": np.max(weight_ratios),
            "n_reduced_50pct": np.sum(weight_ratios < 0.5),
        },
        "variance_concentration": {
            "top_10pct_original": conc_orig,
            "top_10pct_shrunk": conc_shrunk,
            "reduction": conc_orig - conc_shrunk,
        },
        "most_affected_samples": most_shrunk_idx.tolist(),
    }
