"""
Weight stabilization methods for importance sampling.

Implements various heavy-tail smoothers mentioned in the paper:
- Hard clipping
- SWITCH (Truncated Importance Sampling with Bias Correction)
- Log-exp smoothing
"""

from typing import Literal, Optional, Tuple, Dict, Any
import numpy as np


WeightStabilizationMethod = Literal["clip", "switch", "log_exp", "none"]


def stabilize_weights(
    weights: np.ndarray,
    method: WeightStabilizationMethod = "clip",
    threshold: float = 20.0,
    temperature: float = 0.1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply weight stabilization to reduce variance from heavy-tailed importance weights.

    Args:
        weights: Raw importance weights
        method: Stabilization method to use
        threshold: Threshold parameter (interpretation depends on method)
        temperature: Temperature for smooth methods (only for log_exp)

    Returns:
        Tuple of (stabilized_weights, diagnostics)
    """
    diagnostics: Dict[str, Any] = {
        "method": method,
        "original_max": float(np.max(weights)),
        "original_mean": float(np.mean(weights)),
        "original_std": float(np.std(weights)),
    }

    if method == "clip":
        stabilized = np.minimum(weights, threshold)
        diagnostics["n_clipped"] = int(np.sum(weights > threshold))
        diagnostics["clipped_fraction"] = float(np.mean(weights > threshold))

    elif method == "switch":
        # SWITCH: Self-normalized with threshold
        # w_switch = w * I(w <= τ) / P(w <= τ)
        mask = weights <= threshold
        if np.any(mask):
            p_below = np.mean(mask)
            stabilized = np.where(mask, weights / p_below, 0.0)
        else:
            # All weights above threshold - fall back to clipping
            stabilized = np.minimum(weights, threshold)
        diagnostics["switch_probability"] = float(np.mean(mask))

    elif method == "log_exp":
        # Smooth transformation: w_new = exp(temp * log(w)) / normalizer
        # This compresses large weights more than small ones
        log_weights = np.log(np.maximum(weights, 1e-10))
        scaled_log = temperature * log_weights

        # Prevent numerical overflow
        max_scaled = np.max(scaled_log)
        if max_scaled > 100:  # exp(100) is already huge
            scaled_log = scaled_log - max_scaled + 100

        exp_weights = np.exp(scaled_log)
        # Normalize to maintain mean
        stabilized = exp_weights * (np.mean(weights) / np.mean(exp_weights))
        diagnostics["temperature"] = temperature
        diagnostics["compression_ratio"] = float(np.max(weights) / np.max(stabilized))

    else:  # "none"
        stabilized = weights

    # Common diagnostics
    diagnostics["stabilized_max"] = float(np.max(stabilized))
    diagnostics["stabilized_mean"] = float(np.mean(stabilized))
    diagnostics["stabilized_std"] = float(np.std(stabilized))

    # Effective sample size
    if np.sum(stabilized) > 0:
        ess = np.sum(stabilized) ** 2 / np.sum(stabilized**2)
        diagnostics["ess"] = float(ess)
        diagnostics["ess_ratio"] = float(ess / len(weights))
    else:
        diagnostics["ess"] = 0.0
        diagnostics["ess_ratio"] = 0.0

    return stabilized, diagnostics


def adaptive_stabilization(
    weights: np.ndarray,
    target_ess_ratio: float = 0.5,
    max_iterations: int = 10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Adaptively choose stabilization parameters to achieve target ESS.

    Args:
        weights: Raw importance weights
        target_ess_ratio: Target ESS as fraction of n (default: 0.5)
        max_iterations: Maximum iterations for adaptation

    Returns:
        Tuple of (stabilized_weights, diagnostics)
    """
    n = len(weights)
    target_ess = target_ess_ratio * n

    # Start with mild clipping
    threshold = np.percentile(weights, 99)
    best_weights = weights
    best_ess = n**2 / np.sum(weights**2)
    best_diagnostics = {"method": "none", "ess": best_ess}

    for i in range(max_iterations):
        # Try clipping at current threshold
        clipped, diag = stabilize_weights(weights, "clip", threshold)
        current_ess = diag["ess"]

        if abs(current_ess - target_ess) < 0.1 * target_ess:
            # Close enough
            return clipped, diag

        if current_ess < target_ess:
            # Too much clipping, increase threshold
            threshold *= 1.5
        else:
            # Not enough clipping, decrease threshold
            threshold *= 0.8

        if diag["ess"] > best_diagnostics["ess"]:
            best_weights = clipped
            best_diagnostics = diag

    best_diagnostics["adaptive_iterations"] = max_iterations
    best_diagnostics["target_ess_ratio"] = target_ess_ratio
    return best_weights, best_diagnostics
