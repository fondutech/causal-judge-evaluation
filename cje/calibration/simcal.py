"""Stacked Score-Indexed Monotone Calibration (SIMCal) for importance weights.

This module implements stacked SIMCal, which combines {baseline, increasing,
decreasing} candidates via convex optimization to minimize out-of-fold (OOF)
influence function variance.

The stacking approach:
1. Builds candidate weight vectors (raw, isotonic increasing/decreasing)
2. Computes OOF influence functions for each candidate
3. Solves a quadratic program on the simplex to find optimal mixture
4. Applies uniform blending to satisfy ESS/variance constraints
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
import warnings


@dataclass
class SimcalConfig:
    """Configuration for stacked SIMCal calibration.

    Stacked SIMCal combines multiple candidate weight vectors (baseline,
    increasing, decreasing) to minimize OOF influence function variance,
    then applies uniform blending to meet ESS/variance constraints.

    Args:
        ess_floor: Minimum ESS as fraction of n (e.g., 0.2 => ESS >= 0.2 * n)
        var_cap: Maximum allowed variance of calibrated weights
        epsilon: Small constant for numerical stability
        include_baseline: Whether to include raw weights in the stack (default True)
        ridge_lambda: Ridge regularization for covariance matrix (default 1e-8)
        n_folds: Number of folds for OOF if fold_ids not provided (default 5)
        baseline_shrink: Shrinkage toward baseline for stability (default 0.05)
    """

    ess_floor: Optional[float] = 0.2
    var_cap: Optional[float] = None
    epsilon: float = 1e-9
    include_baseline: bool = True
    ridge_lambda: float = 1e-8
    n_folds: int = 5
    baseline_shrink: float = 0.05

    def __post_init__(self) -> None:
        if self.ess_floor is not None and not (0 < self.ess_floor <= 1):
            raise ValueError(f"ess_floor must be in (0, 1], got {self.ess_floor}")
        if self.var_cap is not None and self.var_cap <= 0:
            raise ValueError(f"var_cap must be positive, got {self.var_cap}")
        if self.baseline_shrink < 0 or self.baseline_shrink > 1:
            raise ValueError(
                f"baseline_shrink must be in [0, 1], got {self.baseline_shrink}"
            )

        # Validate consistency between ess_floor and var_cap
        if self.ess_floor is not None and self.var_cap is not None:
            # ESS = n/(1 + Var) implies Var <= 1/ess_floor - 1
            implied_var_cap = (1.0 / self.ess_floor) - 1.0
            if self.var_cap > implied_var_cap:
                warnings.warn(
                    f"var_cap={self.var_cap:.3f} is looser than ESS-implied cap "
                    f"{implied_var_cap:.3f} from ess_floor={self.ess_floor}. "
                    f"The ESS constraint will dominate.",
                    UserWarning,
                )


class SIMCalibrator:
    """Stacked Score-Indexed Monotone Calibrator.

    Combines {baseline, increasing, decreasing} candidates to minimize
    OOF influence function variance, then applies uniform blending to
    meet ESS/variance constraints.
    """

    def __init__(self, config: SimcalConfig):
        """Initialize SIMCalibrator with configuration.

        Args:
            config: SimcalConfig with calibration parameters
        """
        self.cfg = config

    @staticmethod
    def implied_var_cap(ess_floor: float) -> float:
        """Compute the implied variance cap from an ESS floor constraint.

        Since ESS = n/(1 + Var), requiring ESS >= ess_floor * n
        implies Var <= 1/ess_floor - 1.

        Args:
            ess_floor: Minimum ESS as fraction of n (must be in (0, 1])

        Returns:
            Maximum allowed variance to satisfy the ESS constraint
        """
        if not (0 < ess_floor <= 1):
            raise ValueError(f"ess_floor must be in (0, 1], got {ess_floor}")
        return (1.0 / ess_floor) - 1.0

    def transform(
        self,
        w: np.ndarray,
        s: np.ndarray,
        *,
        rewards: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calibrate weights using stacked score-indexed monotone projection.

        Algorithm:
        1. Build candidate weight vectors: {baseline?, increasing, decreasing}
        2. Compute OOF influence functions for each candidate
        3. Solve quadratic program to find optimal mixture on simplex
        4. Apply single γ-blend toward uniform for constraints
        5. Optional: Apply baseline shrinkage for stability

        Args:
            w: Raw importance weights (must be positive, will be normalized to mean 1)
            s: Score index (e.g., judge scores) for ordering
            rewards: Rewards for IPS influence functions (optional, uses weights only if None)
            residuals: DR residuals (R - g_oof(S)) for DR influence functions
            fold_ids: Pre-assigned fold IDs for OOF computation (optional)

        Returns:
            Tuple of (calibrated_weights, info_dict) where info_dict contains:
                - mixture_weights: Optimal convex combination weights
                - candidates: Names of candidate weight vectors
                - gamma: Uniform blending parameter for constraints
                - var_before: Variance of input weights
                - var_after: Final variance after all adjustments
                - ess_before: ESS of input weights
                - ess_after: Final ESS after all adjustments
                - oof_variance_reduction: Ratio of stacked to best single candidate

        Raises:
            ValueError: If weights contain non-positive, NaN, or infinite values
        """
        # Input validation
        w = np.asarray(w, dtype=float)
        s = np.asarray(s, dtype=float)

        if len(w) != len(s):
            raise ValueError(f"Length mismatch: weights={len(w)}, scores={len(s)}")

        if not np.all(np.isfinite(w)) or not np.all(np.isfinite(s)):
            raise ValueError("SIMCal: NaNs or infinities in inputs")

        if np.any(w <= 0):
            raise ValueError("SIMCal: weights must be positive")

        # Ensure mean-one normalization
        w = w / w.mean()
        n = len(w)

        # Build candidate weight vectors
        candidates = []
        candidate_names = []

        # 1. Baseline (raw weights)
        if self.cfg.include_baseline:
            candidates.append(w.copy())
            candidate_names.append("baseline")

        # 2. Isotonic increasing
        iso_inc = IsotonicRegression(increasing=True, out_of_bounds="clip")
        w_inc = iso_inc.fit(s, w).predict(s)
        w_inc = np.maximum(w_inc, self.cfg.epsilon)
        w_inc = w_inc / w_inc.mean()
        candidates.append(w_inc)
        candidate_names.append("increasing")

        # 3. Isotonic decreasing
        iso_dec = IsotonicRegression(increasing=False, out_of_bounds="clip")
        w_dec = iso_dec.fit(s, w).predict(s)
        w_dec = np.maximum(w_dec, self.cfg.epsilon)
        w_dec = w_dec / w_dec.mean()
        candidates.append(w_dec)
        candidate_names.append("decreasing")

        K = len(candidates)

        # Determine what to use for influence functions
        if residuals is not None:
            # DR influence functions: w * (R - g_oof(S))
            if_targets = residuals
            if_type = "dr"
        elif rewards is not None:
            # IPS influence functions: w * R
            if_targets = rewards
            if_type = "ips"
        else:
            # Weight-only influence functions: w itself
            if_targets = np.ones(n)  # Makes IF = w - 1
            if_type = "weight"

        # Compute OOF influence functions
        if fold_ids is not None:
            # Use provided fold assignments
            unique_folds = np.unique(fold_ids)
            n_folds = len(unique_folds)
        else:
            # Generate fold assignments
            kf = KFold(n_splits=self.cfg.n_folds, shuffle=True, random_state=42)
            fold_ids = np.zeros(n, dtype=int)
            for fold_idx, (_, test_idx) in enumerate(kf.split(np.arange(n))):
                fold_ids[test_idx] = fold_idx
            n_folds = self.cfg.n_folds

        # Compute OOF influence matrix (n x K)
        IF_matrix = np.zeros((n, K))

        for k, w_cand in enumerate(candidates):
            # For each fold, compute influence relative to training mean
            for fold_id in range(n_folds):
                test_mask = fold_ids == fold_id
                train_mask = ~test_mask

                if np.sum(train_mask) == 0:
                    continue

                # Compute mean on training folds
                train_mean = np.mean(w_cand[train_mask] * if_targets[train_mask])

                # OOF influence for test fold
                IF_matrix[test_mask, k] = (
                    w_cand[test_mask] * if_targets[test_mask] - train_mean
                )

        # Compute empirical covariance matrix
        Sigma = np.cov(IF_matrix.T)  # K x K

        # Add ridge regularization for stability
        if self.cfg.ridge_lambda > 0:
            reg_amount = self.cfg.ridge_lambda * np.trace(Sigma) / K
            Sigma = Sigma + reg_amount * np.eye(K)

        # Solve quadratic program on simplex: min_π π^T Σ π s.t. π ≥ 0, 1^T π = 1
        mixture_weights = self._solve_simplex_qp(Sigma)

        # Compute stacked weights
        w_stacked = np.zeros(n)
        for k, pi_k in enumerate(mixture_weights):
            w_stacked += pi_k * candidates[k]

        # Apply constraints via uniform blending
        w_final, gamma = self._apply_constraints(w_stacked)

        # Optional: Apply baseline shrinkage for stability
        if self.cfg.baseline_shrink > 0:
            w_final = (
                1 - self.cfg.baseline_shrink
            ) * w_final + self.cfg.baseline_shrink * w
            w_final = w_final / w_final.mean()

        # Compute diagnostics
        var_before = float(np.var(w))
        var_after = float(np.var(w_final))
        ess_before = n / (1 + var_before)
        ess_after = n / (1 + var_after)

        # Compute variance reduction vs best single candidate
        single_variances = [np.var(IF_matrix[:, k]) for k in range(K)]
        best_single_var = min(single_variances)
        stacked_var = np.var(IF_matrix @ mixture_weights)
        variance_reduction = (
            stacked_var / best_single_var if best_single_var > 0 else 1.0
        )

        info = {
            "mixture_weights": mixture_weights.tolist(),
            "candidates": candidate_names,
            "gamma": gamma,
            "var_before": var_before,
            "var_after": var_after,
            "ess_before": ess_before,
            "ess_after": ess_after,
            "oof_variance_reduction": float(variance_reduction),
            "if_type": if_type,
            "n_folds": n_folds,
            "baseline_shrink": self.cfg.baseline_shrink,
        }

        return w_final, info

    def _solve_simplex_qp(self, Sigma: np.ndarray) -> np.ndarray:
        """Solve quadratic program on simplex using active set method.

        Minimize π^T Σ π subject to π ≥ 0, 1^T π = 1

        Args:
            Sigma: K x K positive semi-definite covariance matrix

        Returns:
            Optimal mixture weights on simplex
        """
        K = Sigma.shape[0]

        # Start with uniform weights
        active_set = set(range(K))

        max_iterations = 20
        for _ in range(max_iterations):
            # Solve on current active set
            if len(active_set) == 0:
                # Degenerate case - return uniform
                return np.ones(K) / K

            # Build reduced system
            active_idx = sorted(active_set)
            Sigma_active = Sigma[np.ix_(active_idx, active_idx)]

            # Solve equality-constrained QP: min π^T Σ π s.t. 1^T π = 1
            # Solution: π = Σ^{-1} 1 / (1^T Σ^{-1} 1)
            try:
                ones = np.ones(len(active_idx))
                Sigma_inv_ones = np.linalg.solve(Sigma_active, ones)
                denom = np.dot(ones, Sigma_inv_ones)

                if abs(denom) < 1e-10:
                    # Near-singular - use uniform on active set
                    pi_active = ones / len(active_idx)
                else:
                    pi_active = Sigma_inv_ones / denom
            except np.linalg.LinAlgError:
                # Singular - use uniform on active set
                pi_active = np.ones(len(active_idx)) / len(active_idx)

            # Check for negative weights
            if np.all(pi_active >= -1e-10):
                # Feasible - construct full solution
                pi_full = np.zeros(K)
                for i, idx in enumerate(active_idx):
                    pi_full[idx] = max(0, pi_active[i])

                # Renormalize to ensure exact sum to 1
                pi_full = pi_full / pi_full.sum()
                return pi_full

            # Remove most negative from active set
            min_idx = np.argmin(pi_active)
            active_set.remove(active_idx[min_idx])

        # Fallback to uniform if no convergence
        return np.ones(K) / K

    def _apply_constraints(self, w: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply ESS/variance constraints via uniform blending.

        Args:
            w: Mean-one weight vector

        Returns:
            Tuple of (constrained_weights, gamma) where gamma is the blending parameter
        """
        n = len(w)
        var_w = np.var(w)
        gamma = 0.0

        if var_w > 0:
            # Check ESS constraint
            if self.cfg.ess_floor is not None:
                var_max_ess = (1.0 / self.cfg.ess_floor) - 1.0
                if var_w > var_max_ess:
                    gamma = max(gamma, 1.0 - np.sqrt(var_max_ess / var_w))

            # Check variance cap
            if self.cfg.var_cap is not None and var_w > self.cfg.var_cap:
                gamma = max(gamma, 1.0 - np.sqrt(self.cfg.var_cap / var_w))

        gamma = float(np.clip(gamma, 0.0, 1.0))

        # Apply blending: w ← 1 + (1-γ)(w-1)
        w_constrained = 1.0 + (1.0 - gamma) * (w - 1.0)
        w_constrained = np.maximum(w_constrained, self.cfg.epsilon)
        w_constrained = w_constrained / w_constrained.mean()

        return w_constrained, gamma
