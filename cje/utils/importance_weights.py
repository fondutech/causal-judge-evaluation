"""
Utility functions for importance weight computation.

This module consolidates common importance weight calculations
to avoid duplication across different sampler implementations.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Constants
DEFAULT_LOG_RATIO_CLIP = 20.0  # Clip log ratios to [-20, 20]


def compute_importance_weight(
    target_logp: float,
    base_logp: float,
    clip_min: float = -DEFAULT_LOG_RATIO_CLIP,
    clip_max: float = DEFAULT_LOG_RATIO_CLIP,
) -> float:
    """
    Compute single importance weight with clipping.

    Args:
        target_logp: Log probability under target policy
        base_logp: Log probability under base policy
        clip_min: Minimum log ratio (default: -20)
        clip_max: Maximum log ratio (default: 20)

    Returns:
        Importance weight: exp(target_logp - base_logp)
    """
    log_ratio = target_logp - base_logp

    # Clip for numerical stability
    if log_ratio > clip_max:
        logger.warning(f"Large log ratio {log_ratio:.2f}, clipping to {clip_max}")
        log_ratio = clip_max
    elif log_ratio < clip_min:
        logger.warning(f"Small log ratio {log_ratio:.2f}, clipping to {clip_min}")
        log_ratio = clip_min

    return float(np.exp(log_ratio))


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
