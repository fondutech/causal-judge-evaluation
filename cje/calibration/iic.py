"""
Isotonic Influence Control (IIC) for variance reduction.

IIC residualizes influence functions against judge scores to reduce variance
without changing the target estimand. This is a pure variance reduction
technique that operates on the influence function layer.

Theory:
    For any asymptotically linear estimator with IF φ:
    - Fit isotonic regression: ĥ(s) ≈ E[φ|S=s]
    - Residualize: φ̃ = φ - ĥ(S)
    - Properties:
        * E[φ̃] = E[φ] = 0 (unbiased)
        * Var(φ̃) ≤ Var(φ) (variance reduction)
        * Same asymptotic distribution, tighter CIs

This is a key contribution from the CJE paper that provides "free" variance
reduction by exploiting the judge score structure.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)


@dataclass
class IICConfig:
    """Configuration for Isotonic Influence Control.

    Args:
        enable: Whether to apply IIC (default True - it's free variance reduction!)
        use_cross_fit: Whether to use fold-honest fitting (recommended)
        min_samples_for_iic: Minimum samples to attempt IIC
        compute_diagnostics: Whether to compute R² and other diagnostics
        store_components: Whether to store E[φ|S] for visualization
    """

    enable: bool = True  # On by default - no reason not to use it
    use_cross_fit: bool = True
    min_samples_for_iic: int = 50  # Need enough data for isotonic regression
    compute_diagnostics: bool = True
    store_components: bool = False  # For debugging/visualization


class IsotonicInfluenceControl:
    """Reduce influence function variance via isotonic residualization.

    This implements the IIC component from the CJE paper, providing
    variance reduction by exploiting the relationship between influence
    functions and judge scores.

    The key insight: influence functions often correlate with judge scores
    (since both relate to outcome quality). By removing the predictable
    component E[φ|S], we keep the same estimand but reduce variance.
    """

    def __init__(self, config: Optional[IICConfig] = None):
        """Initialize IIC.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or IICConfig()
        self._diagnostics: Dict[str, Dict[str, Any]] = {}
        self._fitted_components: Dict[str, np.ndarray] = {}

    def residualize(
        self,
        influence: np.ndarray,
        judge_scores: np.ndarray,
        policy: str,
        fold_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Residualize influence function against judge scores.

        This is the main entry point for IIC. It fits E[φ|S] using isotonic
        regression and returns the residuals φ - Ê[φ|S].

        The direction (increasing/decreasing) is chosen automatically based on
        which gives better fit, measured by Spearman correlation.

        Args:
            influence: Raw influence function values (per-sample contributions)
            judge_scores: Judge scores S (ordering index)
            policy: Policy name (for diagnostics)
            fold_ids: Optional fold assignments for cross-fitting
                     (use same folds as reward calibration for consistency)

        Returns:
            residualized_if: φ̃ = φ - Ê[φ|S] (same mean, less variance)
            diagnostics: Dict with R², variance reduction, etc.
        """
        if not self.config.enable:
            return influence, {"applied": False, "reason": "disabled"}

        n = len(influence)
        if n < self.config.min_samples_for_iic:
            logger.debug(f"Too few samples ({n}) for IIC")
            return influence, {"applied": False, "reason": "insufficient_samples"}

        # Validate inputs
        if len(judge_scores) != n:
            logger.warning(
                f"Length mismatch: {n} influences, {len(judge_scores)} scores"
            )
            return influence, {"applied": False, "reason": "length_mismatch"}

        # Check for valid values
        if not np.all(np.isfinite(influence)):
            logger.warning(f"Non-finite influence values for {policy}")
            return influence, {"applied": False, "reason": "non_finite_influence"}

        valid_scores = np.isfinite(judge_scores)
        if not np.all(valid_scores):
            # Handle missing judge scores gracefully
            if valid_scores.sum() < self.config.min_samples_for_iic:
                return influence, {
                    "applied": False,
                    "reason": "insufficient_valid_scores",
                }
            # We'll only fit on valid scores

        # Fit E[φ|S] using isotonic regression
        if self.config.use_cross_fit and fold_ids is not None:
            fitted_values = self._fit_cross_fitted(
                influence, judge_scores, fold_ids, valid_scores
            )
        else:
            fitted_values = self._fit_global(influence, judge_scores, valid_scores)

        # Compute residuals: φ̃ = φ - Ê[φ|S]
        residuals = influence - fitted_values

        # Store fitted component if requested (for visualization)
        if self.config.store_components:
            self._fitted_components[policy] = fitted_values

        # Compute diagnostics
        diagnostics = {"applied": True}
        if self.config.compute_diagnostics:
            diagnostics.update(
                self._compute_diagnostics(influence, fitted_values, residuals, policy)
            )

        self._diagnostics[policy] = diagnostics

        logger.debug(
            f"IIC for {policy}: R²={diagnostics.get('r_squared', 0):.3f}, "
            f"variance reduction={diagnostics.get('var_reduction', 0):.1%}"
        )

        return residuals, diagnostics

    def _fit_global(
        self, influence: np.ndarray, judge_scores: np.ndarray, valid_mask: np.ndarray
    ) -> np.ndarray:
        """Global isotonic fit (simpler but may overfit).

        This fits a single isotonic regression on all data.
        Direction (increasing/decreasing) is chosen based on Spearman correlation.
        """
        fitted = np.zeros_like(influence)

        if valid_mask.sum() < 2:
            # Not enough valid data
            return fitted

        # Determine direction based on Spearman correlation
        from scipy.stats import spearmanr

        corr, _ = spearmanr(judge_scores[valid_mask], influence[valid_mask])

        # Choose direction: increasing if positive correlation, decreasing if negative
        increasing = corr >= 0

        # Fit isotonic regression on valid data with chosen direction
        iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        iso.fit(judge_scores[valid_mask], influence[valid_mask])

        # Predict for all data
        fitted[valid_mask] = iso.predict(judge_scores[valid_mask])

        # For invalid scores, use mean of influence (no reduction)
        fitted[~valid_mask] = influence[valid_mask].mean()

        # Store direction in diagnostics
        self._last_direction = "increasing" if increasing else "decreasing"
        self._last_correlation = float(corr)

        return fitted

    def _fit_cross_fitted(
        self,
        influence: np.ndarray,
        judge_scores: np.ndarray,
        fold_ids: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Fold-honest isotonic fit (recommended).

        This prevents overfitting by using out-of-fold predictions.
        Each sample's E[φ|S] is predicted using a model trained on other folds.
        Direction is chosen per fold based on training data correlation.
        """
        fitted = np.zeros_like(influence)
        unique_folds = np.unique(fold_ids[fold_ids >= 0])

        if len(unique_folds) < 2:
            # Not enough folds for cross-fitting
            logger.debug("Insufficient folds for cross-fitting, using global fit")
            return self._fit_global(influence, judge_scores, valid_mask)

        # Track directions used across folds for diagnostics
        directions_used = []
        correlations = []

        from scipy.stats import spearmanr

        for fold in unique_folds:
            # Define train and test sets
            train_mask = (fold_ids >= 0) & (fold_ids != fold) & valid_mask
            test_mask = (fold_ids == fold) & valid_mask

            if train_mask.sum() < 10 or test_mask.sum() == 0:
                # Not enough data in this fold
                if test_mask.sum() > 0:
                    fitted[test_mask] = (
                        influence[train_mask].mean() if train_mask.sum() > 0 else 0
                    )
                continue

            # Determine direction based on training data correlation
            corr, _ = spearmanr(judge_scores[train_mask], influence[train_mask])
            increasing = corr >= 0
            directions_used.append("increasing" if increasing else "decreasing")
            correlations.append(float(corr))

            # Fit isotonic regression on training folds with chosen direction
            iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
            iso.fit(judge_scores[train_mask], influence[train_mask])

            # Predict on test fold
            fitted[test_mask] = iso.predict(judge_scores[test_mask])

        # Handle samples not in any fold (shouldn't happen with our fold system)
        unfitted = (fold_ids < 0) | ~valid_mask
        if unfitted.sum() > 0:
            fitted[unfitted] = (
                influence[valid_mask].mean() if valid_mask.sum() > 0 else 0
            )

        # Store aggregated direction info for diagnostics
        if directions_used:
            # Most common direction across folds
            from collections import Counter

            direction_counts = Counter(directions_used)
            self._last_direction = direction_counts.most_common(1)[0][0]
            self._last_correlation = (
                float(np.mean(correlations)) if correlations else 0.0
            )
            self._fold_directions = directions_used  # Store for detailed diagnostics

        return fitted

    def _compute_diagnostics(
        self,
        original: np.ndarray,
        fitted: np.ndarray,
        residual: np.ndarray,
        policy: str,
    ) -> Dict[str, Any]:
        """Compute IIC diagnostics for reporting and visualization.

        Key metrics:
        - R²: How much of the IF variance is explained by judge scores
        - Variance reduction: How much we reduced the IF variance
        - SE reduction: How much we reduced the standard error
        """
        var_original = np.var(original, ddof=1)
        var_residual = np.var(residual, ddof=1)
        var_fitted = np.var(fitted, ddof=1)

        # R² of the isotonic fit (how well S predicts φ)
        ss_tot = np.sum((original - original.mean()) ** 2)
        ss_res = np.sum((original - fitted) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Variance reduction (main benefit)
        var_reduction = 1 - (var_residual / var_original) if var_original > 0 else 0

        # Standard error reduction (what users care about)
        se_reduction = (
            1 - np.sqrt(max(0, var_residual) / var_original) if var_original > 0 else 0
        )

        # Effective sample size gain (interpretable)
        # ESS ∝ 1/Var, so ESS gain = var_original/var_residual
        ess_gain = var_original / var_residual if var_residual > 0 else 1.0

        diagnostics = {
            "policy": policy,
            "r_squared": float(max(0, r_squared)),  # Can be negative if fit is terrible
            "var_original": float(var_original),
            "var_residual": float(var_residual),
            "var_fitted": float(var_fitted),
            "var_reduction": float(max(0, var_reduction)),
            "se_reduction": float(max(0, se_reduction)),
            "ess_gain": float(ess_gain),
            "n_samples": len(original),
            "mean_preserved": abs(residual.mean() - original.mean()) < 1e-10,
        }

        # Add direction information if available
        if hasattr(self, "_last_direction"):
            diagnostics["direction"] = self._last_direction
            diagnostics["correlation"] = self._last_correlation

        # Add per-fold directions if cross-fitting was used
        if hasattr(self, "_fold_directions"):
            from collections import Counter

            direction_counts = Counter(self._fold_directions)
            diagnostics["fold_directions"] = dict(direction_counts)

        return diagnostics

    def get_diagnostics(self, policy: Optional[str] = None) -> Dict[str, Any]:
        """Get IIC diagnostics for reporting.

        Args:
            policy: Specific policy (None for all)

        Returns:
            Dictionary of diagnostics
        """
        if policy is not None:
            return self._diagnostics.get(policy, {})
        return self._diagnostics.copy()

    def get_fitted_component(self, policy: str) -> Optional[np.ndarray]:
        """Get fitted E[φ|S] component for visualization.

        Only available if store_components=True in config.
        """
        return self._fitted_components.get(policy)

    def summary(self) -> str:
        """Generate summary of IIC performance across policies."""
        if not self._diagnostics:
            return "No IIC diagnostics available"

        lines = ["IIC Performance Summary:"]
        for policy, diag in self._diagnostics.items():
            if diag.get("applied", False):
                lines.append(
                    f"  {policy}: R²={diag.get('r_squared', 0):.3f}, "
                    f"SE reduction={diag.get('se_reduction', 0):.1%}, "
                    f"ESS gain={diag.get('ess_gain', 1):.2f}x"
                )
            else:
                lines.append(
                    f"  {policy}: Not applied ({diag.get('reason', 'unknown')})"
                )

        return "\n".join(lines)
