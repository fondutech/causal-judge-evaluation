"""Calibrated Inverse Propensity Scoring (IPS) estimator.

This is the core CJE estimator that uses isotonic calibration of importance
weights to achieve unbiased, efficient estimation.
"""

import numpy as np
from typing import Dict, Optional, Any
import logging

from .base_estimator import BaseCJEEstimator
from ..data.models import EstimationResult, WeightCalibrationConfig
from ..data.precomputed_sampler import PrecomputedSampler
from ..calibration.isotonic import calibrate_to_target_mean
from ..utils.diagnostics import weight_diagnostics, evaluate_status

logger = logging.getLogger(__name__)


class CalibratedIPS(BaseCJEEstimator):
    """Calibrated IPS estimator with cross-fitting.

    Uses isotonic regression to calibrate importance weights, ensuring E[w]=1
    while preserving ranking. By default, prevents variance explosion through
    variance-safe blending when calibration would increase variance.

    Args:
        sampler: PrecomputedSampler with data
        k_folds: Number of cross-fitting folds (default 5, minimum 2)
        clip_weight: Maximum weight value before calibration (default 1e10, i.e., no clipping)
        random_seed: Random seed for reproducibility
        enforce_variance_nonincrease: Prevent variance explosion (default True).
                                     Set to False for pure isotonic without variance control.
        compute_diagnostics: Compute detailed weight diagnostics (default True)
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        k_folds: int = 5,
        clip_weight: float = 1e10,  # Default: no clipping, let calibration handle extremes
        random_seed: int = 42,
        enforce_variance_nonincrease: bool = True,  # Default: prevent variance explosion
        compute_diagnostics: bool = True,  # compute detailed diagnostics
    ):
        # Create config
        config = WeightCalibrationConfig(
            k_folds=k_folds,
            clip_weight=clip_weight,
            target_mean=1.0,
            random_seed=random_seed,
        )
        super().__init__(sampler, config)
        self.enforce_variance_nonincrease = enforce_variance_nonincrease
        self.compute_diagnostics = compute_diagnostics

    def fit(self) -> None:
        """Fit weight calibration for all target policies with comprehensive diagnostics."""

        for policy in self.sampler.target_policies:
            # Get raw weights (with optional pre-clipping)
            raw_weights = self.sampler.compute_importance_weights(
                policy, clip_weight=self.config.clip_weight
            )
            if raw_weights is None:
                continue

            # Store raw weight statistics for comparison
            raw_var = raw_weights.var()
            raw_mean = raw_weights.mean()

            logger.debug(
                f"Calibrating weights for policy '{policy}': "
                f"n_samples={len(raw_weights)}, raw_mean={raw_mean:.3f}, "
                f"raw_std={raw_weights.std():.3f}, raw_var={raw_var:.6f}, "
                f"raw_range=[{raw_weights.min():.3f}, {raw_weights.max():.3f}]"
            )

            # Calibrate weights with optional variance constraint
            calibrated = calibrate_to_target_mean(
                raw_weights,
                target_mean=self.config.target_mean,
                k_folds=self.config.k_folds,
                random_seed=self.config.random_seed,
                enforce_variance_nonincrease=self.enforce_variance_nonincrease,
            )

            # Cache calibrated weights in base class storage
            self._weights_cache[policy] = calibrated

            # Compute comprehensive diagnostics if requested
            if self.compute_diagnostics:
                # Get data info for filter transparency
                data = self.sampler.get_data_for_policy(policy)
                n_total = (
                    len(self.sampler.dataset.samples)
                    if hasattr(self.sampler.dataset, "samples")
                    else None
                )
                n_valid = len(data) if data else 0

                # Weight diagnostics (including data filtering info)
                diag = weight_diagnostics(calibrated, n_total=n_total, n_valid=n_valid)

                # Calibration-specific diagnostics (compare on normalized scale)
                raw_mean = float(raw_weights.mean())
                raw_norm = (
                    raw_weights * (self.config.target_mean / raw_mean)
                    if raw_mean
                    else raw_weights
                )
                raw_norm_var = float(raw_norm.var())
                cal_var = float(calibrated.var())

                diag["calibration"] = {
                    "variance_reduction": (
                        float(1.0 - cal_var / raw_norm_var) if raw_norm_var > 0 else 0.0
                    ),
                    "mean_preserved": bool(
                        abs(calibrated.mean() - self.config.target_mean) < 1e-10
                    ),
                    "variance_safe": bool(
                        cal_var <= raw_norm_var * 1.001
                    ),  # VAR_TOL from isotonic.py
                    "mode": "isotonic",
                    "params": {
                        "k_folds": self.config.k_folds,
                        "enforce_variance": self.enforce_variance_nonincrease,
                        "clip_weight": self.config.clip_weight,
                        "random_seed": self.config.random_seed,
                    },
                }

                # Evaluate overall status
                diag["status"] = evaluate_status(diag)

                # Store diagnostics
                self._diagnostics[policy] = diag

                # Log summary
                logger.info(
                    f"Calibrated weights for '{policy}': "
                    f"ESS={diag['weights']['ess']:.1f} ({diag['weights']['ess_fraction']:.1%}), "
                    f"tail_ratio_99_5={diag['weights']['tail_ratio_99_5']:.1f}, "
                    f"var_reduction={diag['calibration']['variance_reduction']:.1%}, "
                    f"status={diag['status']}"
                )

        self._fitted = True

    def estimate(self) -> EstimationResult:
        """Compute estimates for all target policies."""
        self._validate_fitted()

        estimates = []
        standard_errors = []
        n_samples_used = {}

        for policy in self.sampler.target_policies:
            if policy not in self._weights_cache:
                # No valid data for this policy
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            # Get data and weights for this policy
            data = self.sampler.get_data_for_policy(policy)
            weights = self._weights_cache[policy]

            if len(data) != len(weights):
                raise ValueError(f"Data/weight mismatch for {policy}")

            # Extract rewards
            rewards = np.array([d["reward"] for d in data])

            # Compute weighted estimate
            wy = weights * rewards
            estimate = wy.mean()

            # Compute standard error using correct formula
            n = len(wy)
            var_hat = float(np.var(wy, ddof=1)) if n > 1 else 0.0
            se = np.sqrt(var_hat / n)

            estimates.append(estimate)
            standard_errors.append(se)
            n_samples_used[policy] = len(data)

        return EstimationResult(
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="calibrated_ips",
            metadata={
                "k_folds": self.config.k_folds,
                "clip_weight": self.config.clip_weight,
                "diagnostics": self._diagnostics,  # Include diagnostics
            },
        )

    def get_diagnostics(self, target_policy: Optional[str] = None) -> Dict[str, Any]:
        """Get diagnostics for a specific policy or all policies.

        Args:
            target_policy: Policy name or None for all diagnostics

        Returns:
            Dictionary of diagnostics
        """
        if target_policy is None:
            return dict(self._diagnostics)
        return dict(self._diagnostics.get(target_policy, {}))
