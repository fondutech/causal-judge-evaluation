"""Raw IPS estimator without weight calibration."""

import numpy as np
import logging

from cje_simplified.data.precomputed_sampler import PrecomputedSampler

from .base_estimator import BaseCJEEstimator
from ..data.models import EstimationResult

logger = logging.getLogger(__name__)


class RawIPS(BaseCJEEstimator):
    """Standard importance sampling estimator without weight calibration.

    This estimator uses raw importance weights directly without any
    calibration or adjustment beyond clipping.
    """

    def __init__(self, sampler: PrecomputedSampler, clip_weight: float = 100.0):
        """Initialize raw IPS estimator.

        Args:
            sampler: PrecomputedSampler with data
            clip_weight: Maximum weight value for variance control
        """
        super().__init__(sampler)
        self.clip_weight = clip_weight

    def fit(self) -> None:
        """Compute raw importance weights for all policies."""
        for policy in self.sampler.target_policies:
            # Get raw weights with clipping
            weights = self.sampler.compute_importance_weights(
                policy, clip_weight=self.clip_weight
            )

            if weights is None:
                continue

            logger.debug(
                f"Raw IPS weights for policy '{policy}': "
                f"mean={weights.mean():.3f}, std={weights.std():.3f}, "
                f"min={weights.min():.3f}, max={weights.max():.3f}"
            )

            # Cache weights
            self._weights_cache[policy] = weights

        self._fitted = True

    def estimate(self) -> EstimationResult:
        """Compute IPS estimates using raw weights."""
        self._validate_fitted()

        rewards = self.sampler.get_rewards()
        estimates = []
        standard_errors = []
        n_samples_used = {}

        for policy in self.sampler.target_policies:
            weights = self._weights_cache.get(policy)

            if weights is None:
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            # Standard IPS estimate
            weighted_rewards = weights * rewards
            estimate = np.mean(weighted_rewards)

            # Standard error using delta method
            n = len(rewards)
            var_term = np.mean((weighted_rewards - estimate) ** 2)
            se = np.sqrt(var_term / n)

            estimates.append(estimate)
            standard_errors.append(se)
            n_samples_used[policy] = n

        return EstimationResult(
            target_policies=self.sampler.target_policies,
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="RawIPS",
            metadata={
                "estimator": "RawIPS",
                "clip_weight": self.clip_weight,
            },
        )
