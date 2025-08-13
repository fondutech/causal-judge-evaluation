"""Calibrated Inverse Propensity Scoring (IPS) estimator.

This is the core CJE estimator that uses isotonic calibration with variance control
to stabilize IPS in heavy-tail regimes. It trades a small amount of bias for
substantially reduced variance (i.e., not strictly unbiased).
"""

import numpy as np
from typing import Dict, Optional, Set
import logging

from .base_estimator import BaseCJEEstimator
from ..data.models import EstimationResult
from ..data.precomputed_sampler import PrecomputedSampler
from ..calibration.isotonic import calibrate_to_target_mean

logger = logging.getLogger(__name__)


class CalibratedIPS(BaseCJEEstimator):
    """Variance-controlled IPS estimator using isotonic calibration.

    Uses SNIPS-style PAV calibration to reduce variance and heavy-tail
    pathologies in importance weights. This is a deliberate variance-bias
    tradeoff for more stable estimation (not strictly unbiased).

    Features:
    - Single-pass PAV (no cross-fitting overhead)
    - Closed-form variance-safe blending with feasibility handling
    - Comprehensive diagnostics via DiagnosticSuite

    Args:
        sampler: PrecomputedSampler with data
        clip_weight: Maximum weight value before calibration (default None = no clipping)
        enforce_variance_nonincrease: Prevent variance explosion (default True)
        max_variance_ratio: Maximum allowed variance ratio (default 1.0 = no increase)
        store_influence: Store per-sample influence functions (default False)
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        clip_weight: Optional[float] = None,
        enforce_variance_nonincrease: bool = True,
        max_variance_ratio: float = 1.0,
        store_influence: bool = False,
    ):
        super().__init__(sampler)
        self.clip_weight = clip_weight
        self.enforce_variance_nonincrease = enforce_variance_nonincrease
        self.max_variance_ratio = max_variance_ratio
        self.store_influence = store_influence
        self.target_mean = 1.0  # Always use SNIPS/Hajek normalization
        self._influence_functions: Dict[str, np.ndarray] = {}
        self._no_overlap_policies: Set[str] = set()
        self._calibration_info: Dict[str, Dict] = {}  # Store calibration details

    def fit(self) -> None:
        """Fit weight calibration for all target policies."""
        for policy in self.sampler.target_policies:
            # Get raw weights (with optional pre-clipping)
            raw_weights = self.sampler.compute_importance_weights(
                policy, clip_weight=self.clip_weight
            )
            if raw_weights is None:
                continue

            # Check for no overlap (all weights are zero)
            if np.all(raw_weights == 0):
                logger.warning(
                    f"Policy '{policy}' has no overlap with base policy (all weights zero)."
                )
                self._no_overlap_policies.add(policy)
                continue

            logger.debug(
                f"Calibrating weights for policy '{policy}': "
                f"n_samples={len(raw_weights)}, raw_mean={raw_weights.mean():.3f}"
            )

            # Calibrate weights
            calibrated, calib_info = calibrate_to_target_mean(
                raw_weights,
                target_mean=self.target_mean,
                enforce_variance_nonincrease=self.enforce_variance_nonincrease,
                max_variance_ratio=self.max_variance_ratio,
                return_diagnostics=True,
            )

            # Cache results
            self._weights_cache[policy] = calibrated
            self._calibration_info[policy] = calib_info

        self._fitted = True

    def estimate(self) -> EstimationResult:
        """Compute estimates for all target policies with diagnostics."""
        self._validate_fitted()

        estimates = []
        standard_errors = []
        n_samples_used = {}
        influence_functions = {}

        # Compute estimates for each policy
        for policy in self.sampler.target_policies:
            if policy in self._no_overlap_policies:
                # No overlap - return NaN
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                logger.warning(f"Policy '{policy}' has no overlap - returning NaN")
                continue

            # Get calibrated weights
            weights = self._weights_cache.get(policy)
            if weights is None:
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            # Get rewards
            data = self.sampler.get_data_for_policy(policy)
            if data is None or len(data) == 0:
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            # Extract rewards from the list of dictionaries
            rewards = np.array([d["reward"] for d in data])
            n = len(rewards)
            n_samples_used[policy] = n

            # Compute weighted estimate
            estimate = float(np.sum(weights * rewards) / n)
            estimates.append(estimate)

            # Compute standard error using influence functions
            influence = weights * rewards - estimate
            se = float(np.std(influence, ddof=1) / np.sqrt(n))
            standard_errors.append(se)

            # Store influence functions if requested
            if self.store_influence:
                influence_functions[policy] = influence

        # Store influence functions for later use
        if self.store_influence:
            self._influence_functions = influence_functions

        # Create result WITHOUT legacy metadata
        result = EstimationResult(
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="calibrated_ips",
            metadata={
                "target_policies": list(self.sampler.target_policies),
                "ips_influence": influence_functions if self.store_influence else None,
            },
        )

        # Build complete diagnostic suite using the new system
        self._build_diagnostics(result)

        # Store for later access
        self._results = result

        return result

    def get_raw_weights(self, target_policy: str) -> Optional[np.ndarray]:
        """Get raw (uncalibrated) importance weights.

        Args:
            target_policy: Name of target policy

        Returns:
            Array of raw weights or None if not available
        """
        return self.sampler.compute_importance_weights(
            target_policy, clip_weight=None, mode="raw"
        )

    def get_calibration_info(self, target_policy: str) -> Optional[Dict]:
        """Get calibration information for a policy.

        Args:
            target_policy: Name of target policy

        Returns:
            Dictionary with calibration details or None
        """
        return self._calibration_info.get(target_policy)
