"""Stability gates and mitigation strategies for ablations."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional
from .diagnostics import effective_sample_size, hill_alpha


@dataclass
class GateConfig:
    """Configuration for stability gates.

    Uses adaptive thresholds that adjust to sample size.
    """

    # Absolute thresholds (for large samples)
    ess_threshold_absolute: float = 1000.0  # Minimum acceptable ESS

    # Relative thresholds (for small samples)
    ess_threshold_percent: float = 10.0  # Minimum as percentage of n

    # Other gates
    tail_alpha_min: float = 2.0  # Minimum safe tail index
    max_weight_threshold: float = 0.1  # Max fraction on single sample

    def get_ess_threshold(self, n_total: int) -> float:
        """Get adaptive ESS threshold based on sample size.

        For small samples: require 10% of n
        For large samples: require at least 1000
        """
        if n_total is None or n_total <= 0:
            return self.ess_threshold_absolute

        # Adaptive: min of absolute or percentage-based
        relative_threshold = self.ess_threshold_percent * n_total / 100.0
        return min(self.ess_threshold_absolute, relative_threshold)


def check_gates(
    weights: np.ndarray,
    config: Optional[GateConfig] = None,
    n_total: Optional[int] = None,
) -> Dict[str, bool]:
    """Check all stability gates.

    Args:
        weights: Importance weights
        config: Gate configuration (uses defaults if None)
        n_total: Total number of samples (for ESS percentage)

    Returns:
        Dictionary of gate_name -> pass/fail
    """
    if config is None:
        config = GateConfig()

    weights = np.asarray(weights)
    weights = weights[np.isfinite(weights)]  # Remove NaN/inf

    if len(weights) == 0:
        return {"ess": False, "tail": False, "max_weight": False, "all_pass": False}

    # Compute diagnostics
    ess = effective_sample_size(weights)
    alpha = hill_alpha(weights)

    # Normalize weights for max weight check
    weights_norm = weights / np.sum(weights)
    max_w = np.max(weights_norm) if len(weights_norm) > 0 else 1.0

    # Get adaptive ESS threshold
    ess_threshold = config.get_ess_threshold(n_total)

    # Check gates
    gates = {
        "ess": ess >= ess_threshold,
        "tail": alpha > config.tail_alpha_min if np.isfinite(alpha) else False,
        "max_weight": max_w < config.max_weight_threshold,
    }

    # Add ESS details for transparency
    if n_total is not None and n_total > 0:
        ess_percent = 100.0 * ess / n_total
        gates["ess_percent"] = ess_percent
        gates["ess_threshold_used"] = ess_threshold

    # Overall pass/fail
    gates["all_pass"] = all([gates["ess"], gates["tail"], gates["max_weight"]])

    return gates


def temper_weights(weights: np.ndarray, lam: float) -> np.ndarray:
    """Apply trust region tempering W^λ.

    This geometrically interpolates between uniform (λ=0) and target (λ=1).

    Args:
        weights: Original importance weights
        lam: Tempering parameter in [0, 1]

    Returns:
        Tempered weights, renormalized to same sum
    """
    weights = np.asarray(weights)

    # Handle edge cases
    if lam == 1.0:
        return weights
    if lam == 0.0:
        return np.ones_like(weights) * np.mean(weights)

    # Temper: W^λ
    weights_tempered = np.power(np.maximum(weights, 1e-300), lam)

    # Renormalize to preserve mean
    weights_tempered = weights_tempered * (np.sum(weights) / np.sum(weights_tempered))

    return weights_tempered


def apply_variance_cap(weights: np.ndarray, rho: float) -> np.ndarray:
    """Apply variance cap (simplified version).

    In practice, this would call into the full SIMCal machinery.
    Here we do a simple clipping for demonstration.

    Args:
        weights: Original weights
        rho: Variance cap parameter

    Returns:
        Capped weights
    """
    if rho >= 1.0:
        return weights

    # Simple version: clip extreme weights
    # Real version would use full SIMCal projection
    weights = np.asarray(weights)
    mean_w = np.mean(weights)
    std_w = np.std(weights)

    # Clip at mean ± rho * original_range
    max_w = mean_w + rho * 3 * std_w
    min_w = max(0, mean_w - rho * 3 * std_w)

    weights_capped = np.clip(weights, min_w, max_w)

    # Renormalize
    weights_capped = weights_capped * (np.sum(weights) / np.sum(weights_capped))

    return weights_capped


def apply_overlap_weights(weights: np.ndarray) -> np.ndarray:
    """Apply overlap weighting heuristic W/(1+W).

    This shrinks large weights more than small ones.

    Args:
        weights: Original weights

    Returns:
        Overlap-weighted weights
    """
    weights = np.asarray(weights)

    # Normalize to mean 1 first
    weights_norm = weights / np.mean(weights)

    # Apply overlap transformation
    weights_overlap = weights_norm / (1.0 + weights_norm)

    # Renormalize to preserve sum
    weights_overlap = weights_overlap * (np.sum(weights) / np.sum(weights_overlap))

    return weights_overlap


def apply_mitigation_ladder(
    weights: np.ndarray,
    config: Optional[GateConfig] = None,
    n_total: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply mitigation strategies in order until gates pass.

    Tries:
    1. Variance cap (ρ = 0.8, 0.6, 0.5)
    2. Trust region tempering (λ = 0.7, 0.5, 0.3)
    3. Overlap weights as fallback

    Args:
        weights: Original importance weights
        config: Gate configuration
        n_total: Total number of samples
        verbose: Print progress

    Returns:
        (mitigated_weights, mitigation_info)
    """
    if config is None:
        config = GateConfig()

    weights_original = np.array(weights)

    # Check if already passes
    gates = check_gates(weights_original, config, n_total)
    if gates["all_pass"]:
        return weights_original, {"method": "none", "gates_passed": True}

    mitigations_tried = []

    # 1. Try variance cap
    for rho in [0.8, 0.6, 0.5]:
        if verbose:
            print(f"  Trying variance cap ρ={rho}")

        weights_new = apply_variance_cap(weights_original, rho)
        gates = check_gates(weights_new, config, n_total)

        if gates["all_pass"]:
            return weights_new, {
                "method": "variance_cap",
                "rho": rho,
                "gates_passed": True,
                "ess_after": effective_sample_size(weights_new),
                "alpha_after": hill_alpha(weights_new),
            }

        mitigations_tried.append(f"rho={rho}")

    # 2. Try trust region tempering
    for lam in [0.7, 0.5, 0.3]:
        if verbose:
            print(f"  Trying trust region λ={lam}")

        weights_new = temper_weights(weights_original, lam)
        gates = check_gates(weights_new, config, n_total)

        if gates["all_pass"]:
            return weights_new, {
                "method": "trust_region",
                "lambda": lam,
                "gates_passed": True,
                "ess_after": effective_sample_size(weights_new),
                "alpha_after": hill_alpha(weights_new),
            }

        mitigations_tried.append(f"lambda={lam}")

    # 3. Fall back to overlap weights
    if verbose:
        print("  Falling back to overlap weights")

    weights_new = apply_overlap_weights(weights_original)
    gates = check_gates(weights_new, config, n_total)

    return weights_new, {
        "method": "overlap_weights",
        "gates_passed": gates["all_pass"],
        "tried": mitigations_tried,
        "ess_after": effective_sample_size(weights_new),
        "alpha_after": hill_alpha(weights_new),
        "should_skip": not gates["all_pass"],  # Still fails after all mitigations
    }


def create_gate_summary(
    weights_by_policy: Dict[str, np.ndarray],
    config: Optional[GateConfig] = None,
    n_total: Optional[int] = None,
) -> Dict[str, Any]:
    """Create summary of gate status across policies.

    Args:
        weights_by_policy: Policy -> weights
        config: Gate configuration
        n_total: Total samples

    Returns:
        Summary with per-policy and overall statistics
    """
    if config is None:
        config = GateConfig()

    summary = {
        "policies": {},
        "n_pass": 0,
        "n_fail": 0,
        "failures_by_gate": {"ess": [], "tail": [], "max_weight": []},
    }

    for policy, weights in weights_by_policy.items():
        gates = check_gates(weights, config, n_total)
        ess = effective_sample_size(weights)
        alpha = hill_alpha(weights)

        summary["policies"][policy] = {
            "gates": gates,
            "ess": ess,
            "ess_percent": 100.0 * ess / n_total if n_total else None,
            "tail_alpha": alpha,
            "max_weight": (
                np.max(weights / np.sum(weights)) if len(weights) > 0 else None
            ),
            "pass": gates["all_pass"],
        }

        if gates["all_pass"]:
            summary["n_pass"] += 1
        else:
            summary["n_fail"] += 1

            # Track which gates failed
            if not gates["ess"]:
                summary["failures_by_gate"]["ess"].append(policy)
            if not gates["tail"]:
                summary["failures_by_gate"]["tail"].append(policy)
            if not gates["max_weight"]:
                summary["failures_by_gate"]["max_weight"].append(policy)

    summary["all_pass"] = summary["n_fail"] == 0

    return summary
