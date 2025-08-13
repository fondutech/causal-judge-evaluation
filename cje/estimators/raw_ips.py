"""Raw IPS estimator without weight calibration."""

import numpy as np
import logging
from typing import Dict, Optional

from .base_estimator import BaseCJEEstimator
from ..data.models import EstimationResult
from ..data.diagnostics import Diagnostics, Status
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
        store_influence: bool = False,
    ):
        """Initialize raw IPS estimator.

        Args:
            sampler: PrecomputedSampler with data
            clip_weight: Maximum weight value for variance control
            store_influence: Store per-sample influence functions
        """
        super().__init__(sampler)
        self.clip_weight = clip_weight
        self.store_influence = store_influence
        self._influence_functions: Dict[str, np.ndarray] = {}

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
            method="raw_ips",
            metadata={
                "target_policies": list(self.sampler.target_policies),
                "clip_weight": self.clip_weight,
                "ips_influence": influence_functions if self.store_influence else None,
            },
        )

        # Build diagnostics using the new system
        diagnostics = self._build_diagnostics(result)
        result.diagnostics = diagnostics

        # Store for later access
        self._results = result

        return result
    
    def _build_diagnostics(self, result: EstimationResult) -> Diagnostics:
        """Build diagnostics for raw IPS estimation.
        
        Args:
            result: The estimation result
            
        Returns:
            Diagnostics object
        """
        # Get dataset info
        dataset = getattr(self.sampler, 'dataset', None) or getattr(self.sampler, '_dataset', None)
        n_total = 0
        if dataset:
            n_total = dataset.n_samples if hasattr(dataset, 'n_samples') else len(dataset.samples)
        
        # Build estimates dict
        estimates_dict = {}
        se_dict = {}
        policies = list(self.sampler.target_policies)
        for i, policy in enumerate(policies):
            if i < len(result.estimates):
                estimates_dict[policy] = float(result.estimates[i])
                se_dict[policy] = float(result.standard_errors[i])
        
        # Create base diagnostics
        diag = Diagnostics(
            estimator_type="RawIPS",
            method="raw_ips",
            n_samples_total=n_total,
            n_samples_valid=self.sampler.n_valid_samples,
            n_policies=len(policies),
            policies=policies,
            estimates=estimates_dict,
            standard_errors=se_dict,
            n_samples_used=result.n_samples_used
        )
        
        # Add weight diagnostics (similar to CalibratedIPS but without calibration)
        ess_per_policy = {}
        max_weight_per_policy = {}
        tail_ratio_per_policy = {}
        overall_ess = 0.0
        total_n = 0
        
        for policy in policies:
            weights = self.get_weights(policy)
            if weights is not None and len(weights) > 0:
                w_diag = compute_weight_diagnostics(weights, policy)
                ess_per_policy[policy] = w_diag['ess_fraction']
                max_weight_per_policy[policy] = w_diag['max_weight']
                tail_ratio_per_policy[policy] = w_diag['tail_ratio_99_5']
                
                # Track overall
                n = len(weights)
                overall_ess += w_diag['ess_fraction'] * n
                total_n += n
        
        if total_n > 0:
            diag.weight_ess = overall_ess / total_n
            diag.ess_per_policy = ess_per_policy
            diag.max_weight_per_policy = max_weight_per_policy
            diag.weight_tail_ratio_per_policy = tail_ratio_per_policy
            
            # Determine status based on ESS and tail ratios
            worst_tail = max(tail_ratio_per_policy.values()) if tail_ratio_per_policy else 0
            if diag.weight_ess < 0.01 or worst_tail > 1000:
                diag.weight_status = Status.CRITICAL
            elif diag.weight_ess < 0.1 or worst_tail > 100:
                diag.weight_status = Status.WARNING
            else:
                diag.weight_status = Status.GOOD
        
        return diag