"""Raw IPS estimator without weight calibration."""

import numpy as np
import logging
from typing import Dict, Optional

from .base_estimator import BaseCJEEstimator
from ..data.models import EstimationResult
from ..data.diagnostics import IPSDiagnostics, Status
from ..data.precomputed_sampler import PrecomputedSampler
from ..utils.diagnostics import compute_weight_diagnostics

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
    ):
        """Initialize raw IPS estimator.

        Args:
            sampler: PrecomputedSampler with data
            clip_weight: Maximum weight value for variance control
        """
        super().__init__(sampler)
        self.clip_weight = clip_weight
        self._diagnostics: Optional[IPSDiagnostics] = None

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
        """Compute IPS estimates using raw weights with diagnostics."""
        self._validate_fitted()

        estimates = []
        standard_errors = []
        n_samples_used = {}
        influence_functions = {}

        for policy in self.sampler.target_policies:
            weights = self._weights_cache.get(policy)

            if weights is None:
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            # Get data for this policy
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

            # SAFETY CHECK: Refuse to provide unreliable estimates
            # Following CLAUDE.md: "Fail Fast and Clearly"
            n = len(weights)
            ess = (
                np.sum(weights) ** 2 / np.sum(weights**2) / n
            )  # Effective sample size fraction
            near_zero = np.sum(weights < 1e-10) / len(
                weights
            )  # Fraction with near-zero weight

            if ess < 0.01 or near_zero > 0.95:
                logger.error(
                    f"Policy '{policy}' has extreme weight concentration: "
                    f"ESS={ess:.1%}, {near_zero:.1%} near-zero weights. "
                    f"Refusing to provide unreliable estimate."
                )
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                influence_functions[policy] = np.full(n, np.nan)
                continue

            # Standard IPS estimate
            weighted_rewards = weights * rewards
            estimate = weighted_rewards.mean()

            # Compute standard error using influence functions
            n = len(weighted_rewards)
            influence = weighted_rewards - estimate
            se = float(np.std(influence, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

            estimates.append(estimate)
            standard_errors.append(se)
            n_samples_used[policy] = n

            # Store influence functions (always needed for proper inference)
            influence_functions[policy] = influence

        # Store influence functions for later use
        self._influence_functions = influence_functions

        # Create result with clean structure
        result = EstimationResult(
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="raw_ips",
            influence_functions=influence_functions,
            metadata={
                "target_policies": list(self.sampler.target_policies),
                "clip_weight": self.clip_weight,
            },
        )

        # Build diagnostics using the new system
        diagnostics = self._build_diagnostics(result)
        result.diagnostics = diagnostics

        # Store diagnostics for later access
        self._diagnostics = diagnostics

        # Store for later access
        self._results = result

        return result

    def _build_diagnostics(self, result: EstimationResult) -> IPSDiagnostics:
        """Build diagnostics for raw IPS estimation.

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
        tail_index_per_policy = {}
        status_per_policy = {}
        overall_ess = 0.0
        total_n = 0

        for policy in policies:
            weights = self.get_weights(policy)
            if weights is not None and len(weights) > 0:
                w_diag = compute_weight_diagnostics(weights, policy)
                ess_per_policy[policy] = w_diag["ess_fraction"]
                max_weight_per_policy[policy] = w_diag["max_weight"]
                status_per_policy[policy] = w_diag["status"]  # Store per-policy status
                # Use tail_index if available (Hill estimator)
                if "tail_index" in w_diag:
                    tail_index_per_policy[policy] = w_diag["tail_index"]

                # Track overall
                n = len(weights)
                overall_ess += w_diag["ess_fraction"] * n
                total_n += n

        # Compute overall weight ESS
        weight_ess = overall_ess / total_n if total_n > 0 else 0.0

        # Determine status based on ESS and tail index
        # For tail index: < 1 is critical (infinite mean), < 2 is warning (infinite variance)
        worst_tail_index = (
            min(tail_index_per_policy.values())
            if tail_index_per_policy
            else float("inf")
        )
        if weight_ess < 0.01 or worst_tail_index < 1:
            weight_status = Status.CRITICAL
        elif weight_ess < 0.1 or worst_tail_index < 2:
            weight_status = Status.WARNING
        else:
            weight_status = Status.GOOD

        # Create IPSDiagnostics
        diagnostics = IPSDiagnostics(
            estimator_type="RawIPS",
            method="raw_ips",
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
            status_per_policy=status_per_policy,
            tail_indices=tail_index_per_policy if tail_index_per_policy else None,
            # No calibration fields for RawIPS
        )

        return diagnostics

    def get_diagnostics(self) -> Optional[IPSDiagnostics]:
        """Get the diagnostics object."""
        return self._diagnostics
