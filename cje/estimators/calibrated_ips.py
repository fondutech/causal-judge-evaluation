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
from ..data.diagnostics import IPSDiagnostics, Status
from ..calibration.isotonic import calibrate_to_target_mean
from ..utils.diagnostics import compute_weight_diagnostics

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
        self._diagnostics: Optional[IPSDiagnostics] = None

    def fit(self) -> None:
        """Fit weight calibration for all target policies."""
        # Get judge scores once (same for all policies)
        judge_scores = self.sampler.get_judge_scores()

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

            # Calibrate weights - use judge scores as ordering index if available
            calibrated, calib_info = calibrate_to_target_mean(
                raw_weights,
                target_mean=self.target_mean,
                enforce_variance_nonincrease=self.enforce_variance_nonincrease,
                max_variance_ratio=self.max_variance_ratio,
                return_diagnostics=True,
                ordering_index=judge_scores,  # Pass judge scores as the ordering index
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

        # Create result
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

        # Build simplified diagnostics
        diagnostics = self._build_diagnostics(result)
        result.diagnostics = diagnostics

        # Store diagnostics for later access
        self._diagnostics = diagnostics

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

    def _build_diagnostics(self, result: EstimationResult) -> IPSDiagnostics:
        """Build simplified diagnostics for this estimation.

        Args:
            result: The estimation result

        Returns:
            IPSDiagnostics object
        """
        # Get dataset info
        dataset = getattr(self.sampler, "dataset", None) or getattr(
            self.sampler, "_dataset", None
        )
        n_total = 0
        if dataset:
            n_total = (
                dataset.n_samples
                if hasattr(dataset, "n_samples")
                else len(dataset.samples)
            )

        # Build estimates dict
        estimates_dict = {}
        se_dict = {}
        policies = list(self.sampler.target_policies)
        for i, policy in enumerate(policies):
            if i < len(result.estimates):
                estimates_dict[policy] = float(result.estimates[i])
                se_dict[policy] = float(result.standard_errors[i])

        # Compute weight diagnostics
        ess_per_policy = {}
        max_weight_per_policy = {}
        tail_ratio_per_policy = {}
        overall_ess = 0.0
        total_n = 0

        for policy in policies:
            weights = self.get_weights(policy)
            if weights is not None and len(weights) > 0:
                w_diag = compute_weight_diagnostics(weights, policy)
                ess_per_policy[policy] = w_diag["ess_fraction"]
                max_weight_per_policy[policy] = w_diag["max_weight"]
                tail_ratio_per_policy[policy] = w_diag["tail_ratio_99_5"]

                # Track overall
                n = len(weights)
                overall_ess += w_diag["ess_fraction"] * n
                total_n += n

        # Compute overall weight ESS
        weight_ess = overall_ess / total_n if total_n > 0 else 0.0

        # Determine status based on ESS and tail ratios
        worst_tail = max(tail_ratio_per_policy.values()) if tail_ratio_per_policy else 0
        if weight_ess < 0.01 or worst_tail > 1000:
            weight_status = Status.CRITICAL
        elif weight_ess < 0.1 or worst_tail > 100:
            weight_status = Status.WARNING
        else:
            weight_status = Status.GOOD

        # Get calibration info if available
        calibration_rmse = None
        calibration_r2 = None
        n_oracle_labels = None

        # If dataset has calibration info in metadata
        if dataset and hasattr(dataset, "metadata"):
            cal_info = dataset.metadata.get("calibration_info", {})
            calibration_rmse = cal_info.get("rmse")
            calibration_r2 = cal_info.get("r2")
            n_oracle_labels = cal_info.get("n_oracle")

        # Create IPSDiagnostics
        diagnostics = IPSDiagnostics(
            estimator_type="CalibratedIPS",
            method="calibrated_ips",
            n_samples_total=n_total,
            n_samples_valid=self.sampler.n_valid_samples,
            n_policies=len(policies),
            policies=policies,
            estimates=estimates_dict,
            standard_errors=se_dict,
            n_samples_used=result.n_samples_used,
            weight_ess=weight_ess,
            weight_status=weight_status,
            ess_per_policy=ess_per_policy,
            max_weight_per_policy=max_weight_per_policy,
            weight_tail_ratio_per_policy=tail_ratio_per_policy,
            # Calibration fields
            calibration_rmse=calibration_rmse,
            calibration_r2=calibration_r2,
            n_oracle_labels=n_oracle_labels,
        )

        return diagnostics

    def get_diagnostics(self) -> Optional[IPSDiagnostics]:
        """Get the diagnostics object."""
        return self._diagnostics
