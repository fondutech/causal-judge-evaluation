"""Calibrated Inverse Propensity Scoring (IPS) estimator.

This is the core CJE estimator that uses isotonic calibration with variance control
to stabilize IPS in heavy-tail regimes. It trades a small amount of bias for
substantially reduced variance (i.e., not strictly unbiased).
"""

import numpy as np
from typing import Dict, Optional, Any, Union
import logging

from .base_estimator import BaseCJEEstimator
from ..data.models import EstimationResult, WeightCalibrationConfig
from ..data.precomputed_sampler import PrecomputedSampler
from ..calibration.isotonic import calibrate_to_target_mean
from ..utils.diagnostics import weight_diagnostics, evaluate_status

logger = logging.getLogger(__name__)


class CalibratedIPS(BaseCJEEstimator):
    """Variance-controlled IPS estimator using isotonic calibration.

    Uses SNIPS-style PAV calibration to reduce variance and heavy-tail
    pathologies in importance weights. This is a deliberate variance-bias
    tradeoff for more stable estimation (not strictly unbiased).

    Features:
    - Single-pass PAV (no cross-fitting overhead)
    - Closed-form variance-safe blending with feasibility handling
    - Comprehensive diagnostics on weight quality and calibration

    Args:
        sampler: PrecomputedSampler with data
        clip_weight: Maximum weight value before calibration (default None = no clipping)
        enforce_variance_nonincrease: Prevent variance explosion (default True)
        max_variance_ratio: Maximum allowed variance ratio (default 1.0 = no increase)
        compute_diagnostics: Compute detailed weight diagnostics (default True)
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        clip_weight: Optional[
            float
        ] = None,  # Default: no clipping, let calibration handle extremes
        enforce_variance_nonincrease: bool = True,  # Default: prevent variance explosion
        max_variance_ratio: float = 1.0,  # ≤1.0 = no increase, <1.0 = force reduction
        compute_diagnostics: bool = True,  # compute detailed diagnostics
    ):
        # Create config (k_folds not used but required by config)
        config = WeightCalibrationConfig(
            k_folds=2,  # Not used in optimized version
            clip_weight=clip_weight,
            target_mean=1.0,
            random_seed=42,  # Not used in optimized version
        )
        super().__init__(sampler, config)
        self.enforce_variance_nonincrease = enforce_variance_nonincrease
        self.max_variance_ratio = max_variance_ratio
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
            raw_mean = raw_weights.mean()

            logger.debug(
                f"Calibrating weights for policy '{policy}': "
                f"n_samples={len(raw_weights)}, raw_mean={raw_mean:.3f}, "
                f"raw_std={raw_weights.std():.3f}, "
                f"raw_range=[{raw_weights.min():.3f}, {raw_weights.max():.3f}]"
            )

            # Calibrate weights with optimized single-pass algorithm
            calibrated, calib_info = calibrate_to_target_mean(
                raw_weights,
                target_mean=self.config.target_mean,
                enforce_variance_nonincrease=self.enforce_variance_nonincrease,
                max_variance_ratio=self.max_variance_ratio,
                return_diagnostics=True,
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

                # Respect the chosen variance cap in variance_safe flag
                target_var = raw_norm_var * self.max_variance_ratio

                # Compute ESS uplift vs raw weights
                diag_raw = weight_diagnostics(
                    raw_norm, n_total=n_total, n_valid=n_valid
                )
                ess_raw = diag_raw["weights"]["ess"]
                ess_cal = diag["weights"]["ess"]
                ess_uplift_ratio = (ess_cal / ess_raw) if ess_raw > 0 else np.nan

                diag["calibration"] = {
                    "variance_reduction": (
                        float(1.0 - cal_var / raw_norm_var) if raw_norm_var > 0 else 0.0
                    ),
                    "mean_preserved": bool(
                        abs(calibrated.mean() - self.config.target_mean) < 1e-10
                    ),
                    "variance_safe": bool(
                        cal_var <= target_var * 1.001
                    ),  # Respect the chosen cap
                    "ess_uplift_ratio": ess_uplift_ratio,
                    "ess_raw": ess_raw,
                    "ess_calibrated": ess_cal,
                    "mode": "calibrated_ips",
                    "alpha_blend": calib_info.get("alpha_blend"),
                    "variance_cap_feasible": calib_info.get("feasible"),
                    "achieved_var_ratio": calib_info.get("achieved_var_ratio"),
                    "target_var_ratio": self.max_variance_ratio,
                    "params": {
                        "enforce_variance": self.enforce_variance_nonincrease,
                        "max_variance_ratio": self.max_variance_ratio,
                        "clip_weight": self.config.clip_weight,
                    },
                }
                if "note" in calib_info:
                    diag["calibration"]["note"] = calib_info["note"]
                if "n_negative_clipped" in calib_info:
                    diag["calibration"]["n_negative_clipped"] = calib_info[
                        "n_negative_clipped"
                    ]

                # Evaluate overall status
                diag["status"] = evaluate_status(diag)

                # Store diagnostics
                self._diagnostics[policy] = diag

                # Log summary
                logger.info(
                    f"Calibrated weights for '{policy}': "
                    f"ESS={diag['weights']['ess']:.1f} ({diag['weights']['ess_fraction']:.1%}), "
                    f"ESS_uplift={ess_uplift_ratio:.1f}x, "
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
            if data is None:
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            weights = self._weights_cache[policy]

            if len(data) != len(weights):
                raise ValueError(f"Data/weight mismatch for {policy}")

            # Extract rewards
            rewards = np.array([d["reward"] for d in data])

            # Compute weighted estimate
            wy = weights * rewards
            estimate = wy.mean()

            # Compute standard error using delta-method for SNIPS (ratio estimator)
            n = len(wy)
            if n > 1:
                centered = weights * (
                    rewards - estimate
                )  # Ratio (SNIPS) delta-method residuals
                var_hat = float(np.var(centered, ddof=1))
                se = np.sqrt(var_hat / n)
            else:
                se = 0.0

            estimates.append(estimate)
            standard_errors.append(se)
            n_samples_used[policy] = len(data) if data else 0

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
