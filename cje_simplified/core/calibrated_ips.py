"""Calibrated Inverse Propensity Scoring (IPS) estimator.

This is the core CJE estimator that uses isotonic calibration of importance
weights to achieve unbiased, efficient estimation.
"""

import numpy as np
from typing import Dict, Optional
import logging

from .base_estimator import BaseCJEEstimator
from ..data.models import EstimationResult, WeightCalibrationConfig
from ..data.precomputed_sampler import PrecomputedSampler
from ..calibration.isotonic import calibrate_to_target_mean

logger = logging.getLogger(__name__)


class CalibratedIPS(BaseCJEEstimator):
    """Calibrated IPS estimator with cross-fitting.

    Uses isotonic regression to calibrate importance weights, ensuring E[w]=1
    while preserving ranking. This achieves optimal efficiency even when
    the propensity model has errors.

    Args:
        sampler: PrecomputedSampler with data
        k_folds: Number of cross-fitting folds (default 5, minimum 2)
        clip_weight: Maximum weight value for variance control (default 100)
        random_seed: Random seed for reproducibility
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        k_folds: int = 5,
        clip_weight: float = 100.0,
        random_seed: int = 42,
    ):
        # Create config
        config = WeightCalibrationConfig(
            k_folds=k_folds,
            clip_weight=clip_weight,
            target_mean=1.0,
            random_seed=random_seed,
        )
        super().__init__(sampler, config)

    def fit(self) -> None:
        """Fit weight calibration for all target policies."""

        for policy in self.sampler.target_policies:
            # Get raw weights
            raw_weights = self.sampler.compute_importance_weights(policy)
            if raw_weights is None:
                continue

            logger.debug(
                f"Calibrating weights for policy '{policy}': "
                f"n_samples={len(raw_weights)}, raw_mean={raw_weights.mean():.3f}, "
                f"raw_std={raw_weights.std():.3f}, raw_var={raw_weights.var():.6f}, "
                f"raw_range=[{raw_weights.min():.3f}, {raw_weights.max():.3f}]"
            )

            # Calibrate weights
            calibrated = calibrate_to_target_mean(
                raw_weights,
                target_mean=self.config.target_mean,
                k_folds=self.config.k_folds,
                random_seed=self.config.random_seed,
            )

            # Cache calibrated weights in base class storage
            self._weights_cache[policy] = calibrated

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
            estimate = np.sum(weights * rewards) / len(rewards)

            # Compute standard error
            residuals = rewards - estimate
            var_estimate = np.sum(weights**2 * residuals**2) / len(rewards)
            se = np.sqrt(var_estimate / len(rewards))

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
            },
        )
