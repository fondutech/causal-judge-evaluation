"""Identification width computation for CF-bits.

This module computes the identification width (Wid) which represents
structural uncertainty from limited overlap and calibration.

Phase 1 implementation - binned isotonic bands with Hoeffding confidence bounds.
"""

from typing import Optional, Tuple, Dict, Any, List, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    from ..estimators.base_estimator import BaseCJEEstimator

logger = logging.getLogger(__name__)


def compute_identification_width(
    estimator: "BaseCJEEstimator",
    policy: str,
    alpha: float = 0.05,
    n_bins: int = 20,
    min_labels_per_bin: int = 3,
    random_state: Optional[int] = None,
    ci_boot: int = 0,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Compute identification width (Wid) using Phase-1 certificate.

    Implements binned isotonic bands with Hoeffding confidence bounds.
    Uses Hájek weights for target mass computation on σ(S).

    Algorithm:
    1. Bin judge scores S into quantile bins
    2. Compute Hoeffding bands for oracle labels per bin
    3. Apply monotone correction to ensure isotonic feasibility
    4. Compute target mass p' under policy π' using Hájek weights
    5. Return Wid as difference between extremal isotonic solutions

    Args:
        estimator: Fitted CJE estimator
        policy: Target policy name
        alpha: Significance level (default 0.05 for 95% CI)
        n_bins: Number of bins for isotonic bands (default 20)
        min_labels_per_bin: Minimum oracle labels per bin (default 3)
        random_state: Random seed for bootstrap CI (optional)
        ci_boot: Number of bootstrap samples for Wid CI (0 = fast default)

    Returns:
        Tuple of (Wid or None, diagnostics dict)
    """
    try:
        # Get sampler from estimator
        sampler = estimator.sampler

        # 1. Get all judge scores and oracle labels
        # Use the dataset.samples which is the canonical source
        all_samples = sampler.dataset.samples
        if not all_samples:
            logger.warning("No samples available for Wid computation")
            return None, {
                "implemented": True,
                "reason": "no_samples",
                "alpha": alpha,
            }

        # Extract judge scores for all samples
        S_all = []
        for sample in all_samples:
            if sample.metadata and "judge_score" in sample.metadata:
                S_all.append(sample.metadata["judge_score"])
            else:
                S_all.append(None)

        # Check if we have judge scores
        S_all = np.array([s for s in S_all if s is not None])
        if len(S_all) == 0:
            logger.warning("No judge scores available for Wid computation")
            return None, {
                "implemented": True,
                "reason": "no_judge_scores",
                "alpha": alpha,
            }

        # Get oracle slice (samples with ground truth labels)
        oracle_samples = [
            s
            for s in all_samples
            if s.metadata
            and "oracle_label" in s.metadata
            and s.metadata["oracle_label"] is not None
        ]

        if not oracle_samples:
            logger.warning("No oracle labels available for Wid computation")
            return None, {
                "implemented": True,
                "reason": "no_oracle_labels",
                "alpha": alpha,
            }

        # Extract S and Y for oracle slice (filter out None judge scores)
        S_oracle = []
        Y_oracle = []
        for s in oracle_samples:
            if "judge_score" in s.metadata and s.metadata["judge_score"] is not None:
                S_oracle.append(s.metadata["judge_score"])
                Y_oracle.append(s.metadata["oracle_label"])

        S_oracle = np.array(S_oracle) if S_oracle else np.array([])
        Y_oracle = np.array(Y_oracle) if Y_oracle else np.array([])

        if len(S_oracle) == 0 or len(Y_oracle) == 0:
            logger.warning("Oracle samples missing judge scores")
            return None, {
                "implemented": True,
                "reason": "oracle_missing_scores",
                "alpha": alpha,
            }

        # 2. Compute Hájek weights for all samples
        W_raw = sampler.compute_importance_weights(policy, mode="hajek")
        if W_raw is None or len(W_raw) != len(all_samples):
            logger.debug("Could not compute importance weights")
            return None, {
                "implemented": True,
                "reason": "no_weights",
                "alpha": alpha,
            }

        # Filter weights to match S_all
        W_filtered = []
        for i, sample in enumerate(all_samples):
            if (
                sample.metadata
                and "judge_score" in sample.metadata
                and sample.metadata["judge_score"] is not None
            ):
                W_filtered.append(W_raw[i])
        W_filtered = np.array(W_filtered)

        # 3. Create bins based on quantiles of S_all
        # Adapt number of bins based on oracle sample size and min_labels_per_bin
        J = min(n_bins, max(6, len(S_oracle) // max(1, min_labels_per_bin)))

        # Compute quantile edges
        bin_edges = np.quantile(S_all, np.linspace(0, 1, J + 1))

        # Guard against duplicate edges (when S has many ties)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) - 1 < 3:
            # Too few distinct bins, fall back to equal-width
            logger.warning(f"Only {len(bin_edges)-1} distinct bins, using equal-width")
            s_min, s_max = np.min(S_all), np.max(S_all)
            if s_max > s_min:
                bin_edges = np.linspace(s_min - 1e-10, s_max + 1e-10, min(J + 1, 7))
            else:
                # All scores are identical
                bin_edges = np.array([s_min - 1e-10, s_max + 1e-10])

        # Update J to actual number of bins
        J = len(bin_edges) - 1

        # Ensure edges include all data
        bin_edges[0] -= 1e-10  # Ensure leftmost edge includes minimum
        bin_edges[-1] += 1e-10  # Ensure rightmost edge includes maximum

        # 4. Compute statistics per bin
        m_j = np.zeros(J)  # Oracle sample count per bin
        Y_bar_j = np.zeros(J)  # Mean oracle label per bin
        p_prime_j = np.zeros(J)  # Target mass per bin

        # Assign oracle samples to bins
        for j in range(J):
            # Oracle samples in this bin
            if j < J - 1:
                mask_oracle = (S_oracle >= bin_edges[j]) & (S_oracle < bin_edges[j + 1])
            else:
                mask_oracle = (S_oracle >= bin_edges[j]) & (
                    S_oracle <= bin_edges[j + 1]
                )

            m_j[j] = np.sum(mask_oracle)
            if m_j[j] > 0:
                Y_bar_j[j] = np.mean(Y_oracle[mask_oracle])
            else:
                Y_bar_j[j] = 0.5  # No data - use midpoint

            # All samples in this bin (for target mass)
            if j < J - 1:
                mask_all = (S_all >= bin_edges[j]) & (S_all < bin_edges[j + 1])
            else:
                mask_all = (S_all >= bin_edges[j]) & (S_all <= bin_edges[j + 1])

            # Compute Hájek mass
            if np.sum(mask_all) > 0:
                p_prime_j[j] = np.sum(W_filtered[mask_all]) / np.sum(W_filtered)

        # Normalize p_prime to ensure it sums to 1 (numerical safety)
        p_prime_j = p_prime_j / max(p_prime_j.sum(), 1e-12)

        # 5. Compute Hoeffding bands
        epsilon_j = np.sqrt(np.log(2 * J / alpha) / (2 * np.maximum(1, m_j)))
        ell_j = np.maximum(0, Y_bar_j - epsilon_j)  # Lower bounds
        u_j = np.minimum(1, Y_bar_j + epsilon_j)  # Upper bounds

        # Handle bins with no oracle data
        for j in range(J):
            if m_j[j] == 0:
                ell_j[j] = 0
                u_j[j] = 1

        # 6. Monotone correction
        ell_up = np.maximum.accumulate(ell_j)  # Cumulative max (isotonic lower)
        u_down = np.minimum.accumulate(u_j[::-1])[
            ::-1
        ]  # Reverse cumulative min (isotonic upper)

        # Check for violations and pool if needed
        violations = ell_up > u_down
        if np.any(violations):
            # Use while loop to avoid double-incrementing index
            i = 0
            while i < J:
                if not violations[i]:
                    i += 1
                    continue

                # Find contiguous violation block
                start = i
                while i < J and violations[i]:
                    i += 1
                end = i

                # Pool this block
                block_mass = np.sum(p_prime_j[start:end])
                if block_mass > 0:
                    pooled_value = (
                        np.sum(p_prime_j[start:end] * Y_bar_j[start:end]) / block_mass
                    )
                    pooled_value = float(np.clip(pooled_value, 0.0, 1.0))
                    ell_up[start:end] = pooled_value
                    u_down[start:end] = pooled_value

        # 7. Compute Wid
        psi_min = np.sum(p_prime_j * ell_up)
        psi_max = np.sum(p_prime_j * u_down)
        wid = float(psi_max - psi_min)

        # 8. Prepare diagnostics
        # Compute mass on unlabeled bins (support gap diagnostic)
        p_mass_unlabeled = float(np.sum(p_prime_j[m_j == 0]))

        diagnostics = {
            "implemented": True,
            "method": "phase1_certificate",
            "alpha": alpha,
            "n_bins": int(J),
            "n_oracle": int(len(S_oracle)),
            "bin_edges": bin_edges.tolist(),
            "m_j": m_j.tolist(),
            "Y_bar_j": Y_bar_j.tolist(),
            "ell_j": ell_j.tolist(),
            "u_j": u_j.tolist(),
            "ell_up": ell_up.tolist(),
            "u_down": u_down.tolist(),
            "p_prime_j": p_prime_j.tolist(),
            "contrib_j": (p_prime_j * (u_down - ell_up)).tolist(),
            "psi_min": float(psi_min),
            "psi_max": float(psi_max),
            "violations_found": int(np.sum(violations)),
            "p_mass_unlabeled": p_mass_unlabeled,  # Mass on bins with no oracle labels
        }

        # Optional: Bootstrap CI for Wid
        if ci_boot > 0:
            logger.info(f"Computing bootstrap CI for Wid with {ci_boot} samples")
            # This would bootstrap only the oracle slice
            # For now, we'll skip this in Phase-1
            diagnostics["ci_boot"] = None

        logger.info(f"Computed Wid={wid:.3f} for {policy} using {J} bins")
        return wid, diagnostics

    except Exception as e:
        logger.error(f"Error computing identification width: {e}")
        return None, {
            "implemented": True,
            "error": str(e),
            "alpha": alpha,
        }


def compute_isotonic_bands(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute confidence bands for isotonic regression.

    Phase 2 feature - not yet implemented.

    Args:
        X: Covariate values (e.g., judge scores)
        Y: Response values
        alpha: Significance level
        n_boot: Number of bootstrap samples

    Returns:
        Tuple of (lower_band, upper_band)
    """
    # Placeholder
    n = len(X)
    fitted = np.mean(Y) * np.ones(n)
    margin = 0.1

    return fitted - margin, fitted + margin


def compute_overlap_bounds(
    weights: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Compute bounds on overlap contribution to Wid.

    Phase 2 feature - not yet implemented.

    Args:
        weights: Importance weights
        alpha: Significance level

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Placeholder
    return 0.0, 0.2
