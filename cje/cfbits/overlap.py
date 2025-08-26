"""Overlap metrics on σ(S) for structural information bounds.

These metrics measure the structural overlap on the judge score marginal,
providing ceilings on what any S-indexed calibration can achieve.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import logging
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


@dataclass
class OverlapFloors:
    """Overlap metrics that bound achievable efficiency.

    Attributes:
        aessf: Adjusted ESS Fraction on σ(S) ∈ (0,1]
        bc: Bhattacharyya coefficient on σ(S) ∈ (0,1]
        chi2_s: χ² divergence on judge marginal
        ci_aessf: Confidence interval for A-ESSF
        ci_bc: Confidence interval for BC
        omega_profile: Optional profile of ω(s) = E[W|S=s]
    """

    aessf: float
    bc: float
    chi2_s: float
    ci_aessf: Tuple[float, float]
    ci_bc: Tuple[float, float]
    omega_profile: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "aessf": self.aessf,
            "bc": self.bc,
            "chi2_s": self.chi2_s,
            "ci_aessf": list(self.ci_aessf),
            "ci_bc": list(self.ci_bc),
        }
        if self.omega_profile:
            result["omega_profile"] = self.omega_profile
        return result

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Structural overlap: A-ESSF={self.aessf:.1%} "
            f"[{self.ci_aessf[0]:.1%}, {self.ci_aessf[1]:.1%}], "
            f"BC={self.bc:.1%} [Theory: A-ESSF ≤ BC²={self.bc**2:.1%}]"
        )


def estimate_omega_conservative(
    S: np.ndarray,
    W: np.ndarray,
    method: str = "bins",
    n_bins: int = 20,
    use_cross_fit: bool = True,
    n_folds: int = 5,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Conservative estimate of ω(s) = E[W|S=s].

    Uses binning or kernel methods to estimate E[W|S] without assuming monotonicity.
    For A-ESSF computation, we want a conservative (high variance) estimate.

    Args:
        S: Judge scores
        W: Raw importance weights
        method: "bins" for histogram, "isotonic_both" for max of ↑↓
        n_bins: Number of bins for histogram method
        use_cross_fit: Whether to use cross-fitting
        n_folds: Number of folds for cross-fitting
        random_state: Random seed for reproducibility

    Returns:
        Conservative ω(S) estimates
    """
    n = len(S)

    # Set up random number generator
    rng = np.random.default_rng(random_state)

    if method == "bins":
        # Histogram-based estimation
        # Create bins based on quantiles
        bin_edges = np.percentile(S, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] -= 1e-10
        bin_edges[-1] += 1e-10

        omega_hat = np.zeros(n)

        if use_cross_fit and n_folds > 1 and n >= 50:
            # Cross-fitted version
            fold_indices = np.arange(n) % n_folds
            rng.shuffle(fold_indices)

            for fold in range(n_folds):
                train_mask = fold_indices != fold
                test_mask = fold_indices == fold

                if np.sum(test_mask) == 0 or np.sum(train_mask) < 20:
                    continue

                # Compute bin means on training
                bin_means = {}
                for i in range(n_bins):
                    bin_mask = (S[train_mask] >= bin_edges[i]) & (
                        S[train_mask] < bin_edges[i + 1]
                    )
                    if np.sum(bin_mask) > 0:
                        bin_means[i] = np.mean(W[train_mask][bin_mask])
                    else:
                        bin_means[i] = np.mean(W[train_mask])  # Fallback

                # Apply to test
                for i in range(n_bins):
                    test_bin_mask = (
                        test_mask & (S >= bin_edges[i]) & (S < bin_edges[i + 1])
                    )
                    omega_hat[test_bin_mask] = bin_means[i]

            # Fill any zeros with simple estimate
            if np.any(omega_hat == 0):
                for i in range(n_bins):
                    bin_mask = (S >= bin_edges[i]) & (S < bin_edges[i + 1])
                    if np.sum(bin_mask) > 0:
                        omega_hat[bin_mask & (omega_hat == 0)] = np.mean(W[bin_mask])
        else:
            # Simple binning
            for i in range(n_bins):
                bin_mask = (S >= bin_edges[i]) & (S < bin_edges[i + 1])
                if np.sum(bin_mask) > 0:
                    omega_hat[bin_mask] = np.mean(W[bin_mask])
                else:
                    omega_hat[bin_mask] = np.mean(W)

    elif method == "isotonic_both":
        # Try both increasing and decreasing, take max variance
        from sklearn.isotonic import IsotonicRegression

        # Increasing fit
        iso_inc = IsotonicRegression(y_min=0, increasing=True)
        omega_inc = iso_inc.fit_transform(S, W)

        # Decreasing fit
        iso_dec = IsotonicRegression(y_min=0, increasing=False)
        omega_dec = iso_dec.fit_transform(S, W)

        # Choose the one with higher variance (more conservative for A-ESSF)
        var_inc = np.var(omega_inc)
        var_dec = np.var(omega_dec)

        if var_inc >= var_dec:
            omega_hat = omega_inc
            logger.debug(f"Using increasing fit (var={var_inc:.3f} vs {var_dec:.3f})")
        else:
            omega_hat = omega_dec
            logger.debug(f"Using decreasing fit (var={var_dec:.3f} vs {var_inc:.3f})")
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure mean-1 normalization
    if np.mean(omega_hat) > 0:
        omega_hat = omega_hat * (np.mean(W) / np.mean(omega_hat))
    else:
        omega_hat = np.ones_like(W)

    return omega_hat


def compute_chi2_divergence(omega: np.ndarray) -> float:
    """Compute χ² divergence from fitted ω(S).

    χ²_S = E[ω(S)²] - 1

    Args:
        omega: Fitted ω(S) values (should be mean-1)

    Returns:
        χ² divergence
    """
    return float(np.mean(omega**2) - 1)


def compute_aessf(chi2_s: float) -> float:
    """Compute Adjusted ESS Fraction from χ² divergence.

    A-ESSF = 1 / (1 + χ²_S) = exp(-D₂(P'_S || P_S))

    Args:
        chi2_s: χ² divergence on judge marginal

    Returns:
        A-ESSF ∈ (0, 1]
    """
    return 1.0 / (1.0 + chi2_s)


def compute_bhattacharyya_coefficient(
    S: np.ndarray,
    W: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Compute Bhattacharyya coefficient on σ(S).

    BC = ∫ √(dP'_S × dP_S)

    Uses adaptive binning for numerical stability.

    Args:
        S: Judge scores
        W: Importance weights
        n_bins: Number of bins for discretization

    Returns:
        BC ∈ (0, 1]
    """
    # Create bins based on quantiles of S
    bin_edges = np.percentile(S, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] -= 1e-10  # Ensure all points included
    bin_edges[-1] += 1e-10

    # Assign to bins
    bin_indices = np.digitize(S, bin_edges) - 1

    # Compute probabilities under P and P'
    p0 = np.zeros(n_bins)
    p1 = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            p0[i] = np.mean(mask)  # Empirical probability under P
            p1[i] = np.mean(W[mask]) * p0[i]  # Probability under P'

    # Normalize
    p0 = p0 / np.sum(p0)
    p1 = p1 / np.sum(p1)

    # Compute BC
    bc = float(np.sum(np.sqrt(p0 * p1)))

    return min(bc, 1.0)  # Ensure ≤ 1


def bootstrap_overlap_metrics(
    S: np.ndarray,
    W: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap confidence intervals for overlap metrics.

    Uses paired bootstrap, resampling (S, W) pairs together.

    Args:
        S: Judge scores
        W: Importance weights
        n_boot: Number of bootstrap samples
        alpha: Significance level
        seed: Random seed for reproducibility

    Returns:
        Dict with CIs for 'aessf' and 'bc'
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(S)
    aessf_boot = []
    bc_boot = []

    for _ in range(n_boot):
        # Resample with replacement
        idx = np.random.choice(n, n, replace=True)
        S_boot = S[idx]
        W_boot = W[idx]

        # Compute metrics using conservative estimation
        # Use a derived seed for reproducibility within bootstrap
        boot_seed = None if seed is None else seed + _ + 1
        omega_boot = estimate_omega_conservative(
            S_boot, W_boot, method="bins", use_cross_fit=False, random_state=boot_seed
        )
        chi2_boot = compute_chi2_divergence(omega_boot)
        aessf_boot.append(compute_aessf(chi2_boot))

        bc_boot.append(compute_bhattacharyya_coefficient(S_boot, W_boot))

    # Compute percentile CIs
    aessf_ci = (
        np.percentile(aessf_boot, 100 * alpha / 2),
        np.percentile(aessf_boot, 100 * (1 - alpha / 2)),
    )

    bc_ci = (
        np.percentile(bc_boot, 100 * alpha / 2),
        np.percentile(bc_boot, 100 * (1 - alpha / 2)),
    )

    return {
        "aessf": aessf_ci,
        "bc": bc_ci,
    }


def estimate_overlap_floors(
    S: np.ndarray,
    W: np.ndarray,
    method: str = "conservative",
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
    return_omega_profile: bool = False,
) -> OverlapFloors:
    """Estimate structural overlap floors on σ(S).

    These metrics provide upper bounds on achievable efficiency for any
    S-indexed calibration method.

    Args:
        S: Judge scores
        W: Raw importance weights (not necessarily calibrated)
        method: Method for ω(S) estimation ("conservative", "bins", "isotonic_both")
        n_boot: Number of bootstrap samples for CIs
        alpha: Significance level for CIs
        random_state: Random seed for reproducibility
        return_omega_profile: Whether to include ω(S) profile in output

    Returns:
        OverlapFloors with A-ESSF, BC, and confidence intervals
    """
    # Validate inputs
    if len(S) != len(W):
        raise ValueError(f"S and W must have same length, got {len(S)} and {len(W)}")

    if len(S) < 10:
        raise ValueError(f"Need at least 10 samples, got {len(S)}")

    # Remove any invalid weights (including tiny negatives from numerical noise)
    valid_mask = np.isfinite(W) & (W >= -1e-10)
    if not np.all(valid_mask):
        logger.warning(f"Removing {np.sum(~valid_mask)} invalid weights")
        S = S[valid_mask]
        W = W[valid_mask]

    # Clip any tiny negative values to 0
    W = np.maximum(W, 0.0)

    # Normalize weights to mean 1
    if np.mean(W) > 0:
        W = W / np.mean(W)
    else:
        raise ValueError("All weights are zero or negative")

    # Estimate ω(S) = E[W|S] conservatively
    if method == "conservative":
        # Default conservative method: bins
        omega = estimate_omega_conservative(
            S, W, method="bins", use_cross_fit=True, random_state=random_state
        )
    elif method in ["bins", "isotonic_both"]:
        omega = estimate_omega_conservative(
            S, W, method=method, use_cross_fit=True, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute χ² divergence and A-ESSF
    chi2_s = compute_chi2_divergence(omega)
    aessf = compute_aessf(chi2_s)

    # Compute Bhattacharyya coefficient
    bc = compute_bhattacharyya_coefficient(S, W)

    # Bootstrap confidence intervals
    if n_boot > 0:
        cis = bootstrap_overlap_metrics(S, W, n_boot, alpha, random_state)
        ci_aessf = cis["aessf"]
        ci_bc = cis["bc"]
    else:
        ci_aessf = (aessf, aessf)
        ci_bc = (bc, bc)

    # Prepare omega profile if requested
    omega_profile = None
    if return_omega_profile:
        # Create grid of S values
        s_grid = np.percentile(S, np.linspace(0, 100, 100))

        # For profile, we can show the actual omega values
        # Interpolate conservatively
        from scipy.interpolate import interp1d

        # Sort by S for interpolation
        sort_idx = np.argsort(S)
        S_sorted = S[sort_idx]
        omega_sorted = omega[sort_idx]

        # Create interpolator with nearest neighbor for extrapolation
        interp = interp1d(
            S_sorted,
            omega_sorted,
            kind="nearest",
            bounds_error=False,
            fill_value=(omega_sorted[0], omega_sorted[-1]),
        )
        omega_grid = interp(s_grid)

        omega_profile = {
            "s_grid": s_grid.tolist(),
            "omega_grid": omega_grid.tolist(),
            "mean": float(np.mean(omega)),
            "std": float(np.std(omega)),
        }

    # Verify theoretical constraint: A-ESSF ≤ BC²
    if aessf > bc**2 + 0.01:  # Allow small numerical tolerance
        logger.warning(
            f"Theoretical violation: A-ESSF={aessf:.3f} > BC²={bc**2:.3f}. "
            "This suggests numerical issues."
        )

    return OverlapFloors(
        aessf=aessf,
        bc=bc,
        chi2_s=chi2_s,
        ci_aessf=ci_aessf,
        ci_bc=ci_bc,
        omega_profile=omega_profile,
    )
