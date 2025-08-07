"""Raw IPS estimator without weight calibration."""

import numpy as np
import logging
from typing import Dict, Optional, Any

from cje_simplified.data.precomputed_sampler import PrecomputedSampler

from .base_estimator import BaseCJEEstimator
from ..data.models import EstimationResult
from ..utils.diagnostics import weight_diagnostics, evaluate_status

logger = logging.getLogger(__name__)


class RawIPS(BaseCJEEstimator):
    """Standard importance sampling estimator without weight calibration.

    This estimator uses raw importance weights directly without any
    calibration or adjustment beyond clipping.
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        clip_weight: float = 100.0,
        compute_diagnostics: bool = True,  # compute detailed diagnostics
    ):
        """Initialize raw IPS estimator.

        Args:
            sampler: PrecomputedSampler with data
            clip_weight: Maximum weight value for variance control
            compute_diagnostics: Whether to compute detailed diagnostics
        """
        super().__init__(sampler)
        self.clip_weight = clip_weight
        self.compute_diagnostics = compute_diagnostics

    def fit(self) -> None:
        """Compute raw importance weights for all policies with diagnostics."""
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

            # Compute diagnostics if requested
            if self.compute_diagnostics:
                # Get data info for filter transparency
                data = self.sampler.get_data_for_policy(policy)
                n_total = (
                    len(self.sampler.dataset.samples)
                    if hasattr(self.sampler.dataset, "samples")
                    else None
                )
                n_valid = len(data) if data else 0

                # Weight diagnostics
                diag = weight_diagnostics(weights, n_total=n_total, n_valid=n_valid)

                # Evaluate overall status
                diag["status"] = evaluate_status(diag)

                # Store diagnostics
                self._diagnostics[policy] = diag

                # Log summary
                logger.info(
                    f"Raw IPS weights for '{policy}': "
                    f"ESS={diag['weights']['ess']:.1f} ({diag['weights']['ess_fraction']:.1%}), "
                    f"tail_ratio_99_5={diag['weights']['tail_ratio_99_5']:.1f}, "
                    f"status={diag['status']}"
                )

        self._fitted = True

    def estimate(self) -> EstimationResult:
        """Compute IPS estimates using raw weights."""
        self._validate_fitted()

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

            # Get data for this policy (ensures consistency with weights)
            data = self.sampler.get_data_for_policy(policy)
            if data is None:
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            # Extract rewards from the filtered data
            rewards = np.array([d["reward"] for d in data])

            if len(rewards) != len(weights):
                raise ValueError(
                    f"Data/weight mismatch for {policy}: {len(rewards)} vs {len(weights)}"
                )

            # Standard IPS estimate
            weighted_rewards = weights * rewards
            estimate = weighted_rewards.mean()

            # Compute standard error using correct formula
            n = len(weighted_rewards)
            var_hat = float(np.var(weighted_rewards, ddof=1)) if n > 1 else 0.0
            se = np.sqrt(var_hat / n)

            estimates.append(estimate)
            standard_errors.append(se)
            n_samples_used[policy] = n

        return EstimationResult(
            target_policies=self.sampler.target_policies,
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="raw_ips",  # Consistent lowercase snake case
            metadata={
                "estimator": "RawIPS",
                "clip_weight": self.clip_weight,
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
