"""
Utility functions for importance weight computation.

This module consolidates common importance weight calculations
to avoid duplication across different sampler implementations.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Default weight clipping value
DEFAULT_MAX_WEIGHT = 50


def compute_importance_weight(
    target_logp: float,
    base_logp: float,
    max_weight: int = DEFAULT_MAX_WEIGHT,
) -> float:
    """
    Compute single importance weight with maximum clipping only.

    Args:
        target_logp: Log probability under target policy
        base_logp: Log probability under base policy
        max_weight: Maximum weight value (default: 50). No minimum clipping.

    Returns:
        Importance weight: clipped to [0, max_weight]
    """
    log_ratio = target_logp - base_logp

    # Compute weight
    weight = float(np.exp(log_ratio))

    # Only clip maximum values
    if weight > max_weight:
        logger.warning(
            f"Large weight {weight:.2f} (log ratio: {log_ratio:.2f}), "
            f"clipping to {max_weight}"
        )
        weight = float(max_weight)

    return weight


def compute_weight_statistics(
    weights: np.ndarray,
    policy_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute common statistics for importance weights.

    Args:
        weights: Weight matrix of shape (n_samples, n_policies)
        policy_names: Optional list of policy names

    Returns:
        Dictionary with weight statistics including ESS
    """
    n_samples, n_policies = weights.shape

    if policy_names is None:
        policy_names = [f"policy_{i}" for i in range(n_policies)]

    stats: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_policies": n_policies,
        "weight_range": [],
        "ess_values": [],
        "ess_percentage": 0.0,
        "n_missing": 0,
        "policy_stats": {},
    }

    # Compute per-policy statistics
    for j, policy_name in enumerate(policy_names):
        w = weights[:, j]
        valid_w = w[~np.isnan(w)]

        if len(valid_w) > 0:
            # Weight range
            stats["weight_range"].append(
                (float(np.min(valid_w)), float(np.max(valid_w)))
            )

            # Effective Sample Size (ESS)
            if valid_w.sum() > 0:
                ess = (valid_w.sum() ** 2) / (valid_w**2).sum()
            else:
                ess = 0.0
            stats["ess_values"].append(float(ess))

            # Detailed statistics
            stats["policy_stats"][policy_name] = {
                "mean": float(np.mean(valid_w)),
                "std": float(np.std(valid_w)),
                "min": float(np.min(valid_w)),
                "max": float(np.max(valid_w)),
                "n_valid": len(valid_w),
                "n_missing": len(w) - len(valid_w),
                "ess": float(ess),
                "ess_percentage": float(100.0 * ess / n_samples),
            }
        else:
            stats["weight_range"].append((np.nan, np.nan))
            stats["ess_values"].append(0.0)
            stats["policy_stats"][policy_name] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "n_valid": 0,
                "n_missing": len(w),
                "ess": 0.0,
                "ess_percentage": 0.0,
            }

    # Overall statistics
    stats["n_missing"] = int(np.isnan(weights).any(axis=1).sum())
    if stats["ess_values"]:
        stats["ess_percentage"] = float(np.mean(stats["ess_values"]) / n_samples * 100)

    return stats


def validate_log_prob(
    logp: float,
    context: str = "",
    response: str = "",
    policy_name: str = "",
) -> Tuple[bool, Optional[str]]:
    """
    Validate a log probability value.

    Args:
        logp: Log probability to validate
        context: Optional context for error messages
        response: Optional response for error messages
        policy_name: Optional policy name for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if finite
    if not np.isfinite(logp):
        return False, f"Non-finite log prob: {logp}"

    # Log probs should be <= 0
    if logp > 0:
        # Small positive values might be floating point errors
        if logp < 1e-6:
            logger.debug(f"Small positive log prob {logp}, treating as 0")
            return True, None
        else:
            return False, f"Positive log prob: {logp}"

    # Check for unreasonably small values
    if logp < -1000:
        logger.warning(f"Very small log prob {logp} for {policy_name}")
        # Still valid, just unusual

    return True, None
