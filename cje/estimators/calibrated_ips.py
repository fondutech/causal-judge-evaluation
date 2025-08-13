"""Calibrated Inverse Propensity Scoring (IPS) estimator with stacked SIMCal.

This is the core CJE estimator that uses stacked Score-Indexed Monotone Calibration
(SIMCal) to stabilize IPS in heavy-tail regimes. It combines {baseline, increasing,
decreasing} candidates via convex optimization to minimize OOF influence function
variance, then blends toward uniform to meet variance/ESS constraints.
"""

import numpy as np
from typing import Dict, Optional, Set, Any
import logging

from .base_estimator import BaseCJEEstimator
from ..data.models import EstimationResult
from ..data.precomputed_sampler import PrecomputedSampler
from ..data.diagnostics import IPSDiagnostics, Status
from ..utils.diagnostics import compute_weight_diagnostics
from ..calibration.simcal import SIMCalibrator, SimcalConfig

logger = logging.getLogger(__name__)


class CalibratedIPS(BaseCJEEstimator):
    """Stacked SIMCal-based IPS estimator with optimal weight calibration.

    Uses stacked Score-Indexed Monotone Calibration (SIMCal) to reduce variance and
    heavy-tail pathologies in importance weights. Combines {baseline, increasing,
    decreasing} candidates via convex optimization to minimize OOF influence function
    variance, then blends toward uniform to meet constraints.

    Features:
    - Stacked calibration combining multiple candidates optimally
    - OOF influence function variance minimization
    - ESS floor and variance cap constraints
    - Judge score-indexed calibration for better alignment
    - Automatic DR-aware calibration when calibrator available
    - Comprehensive diagnostics

    Args:
        sampler: PrecomputedSampler with data
        clip_weight: Maximum weight value before calibration (default None = no clipping)
        ess_floor: Minimum ESS as fraction of n (default 0.2 = 20% ESS)
        var_cap: Maximum allowed variance of calibrated weights (default None = no cap)
        calibrator: Optional JudgeCalibrator for DR influence functions
        include_baseline: Whether to include raw weights in the stack (default True)
        baseline_shrink: Shrinkage toward baseline for stability (default 0.05)
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        clip_weight: Optional[float] = None,
        ess_floor: Optional[float] = 0.2,
        var_cap: Optional[float] = None,
        calibrator: Optional[Any] = None,
        include_baseline: bool = True,
        baseline_shrink: float = 0.05,
    ):
        super().__init__(sampler)
        self.clip_weight = clip_weight
        self.ess_floor = ess_floor
        self.var_cap = var_cap
        self.calibrator = calibrator
        self.include_baseline = include_baseline
        self.baseline_shrink = baseline_shrink
        self._no_overlap_policies: Set[str] = set()
        self._calibration_info: Dict[str, Dict] = {}  # Store calibration details
        self._diagnostics: Optional[IPSDiagnostics] = None

    def fit(self) -> None:
        """Fit stacked weight calibration for all target policies."""
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

            # Use SIMCal calibration with judge scores as the index
            if judge_scores is None:
                raise ValueError(
                    "Judge scores are required for SIMCal calibration. "
                    "Ensure samples have 'judge_score' in metadata."
                )

            # Get rewards for this policy (always needed for influence functions)
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                logger.warning(f"No data for policy '{policy}'. Skipping.")
                continue
            rewards = np.array([d["reward"] for d in data], dtype=float)

            # Try to get DR residuals if calibrator available
            residuals = None
            fold_ids = None
            if self.calibrator is not None and hasattr(self.calibrator, "predict_oof"):
                try:
                    # Extract fold IDs from data
                    fold_list = [d.get("cv_fold") for d in data]
                    if all(v is not None for v in fold_list) and len(fold_list) == len(
                        judge_scores
                    ):
                        fold_ids = np.asarray(fold_list, dtype=int)
                        # Compute cross-fitted residuals
                        g_oof = self.calibrator.predict_oof(judge_scores, fold_ids)
                        residuals = rewards - g_oof
                        logger.debug(f"Using DR residuals for policy '{policy}'")
                except Exception as e:
                    logger.debug(f"Could not compute DR residuals: {e}")

            # Run stacked SIMCal calibration
            cfg = SimcalConfig(
                ess_floor=self.ess_floor,
                var_cap=self.var_cap,
                include_baseline=self.include_baseline,
                baseline_shrink=self.baseline_shrink,
            )
            sim = SIMCalibrator(cfg)
            calibrated, calib_info = sim.transform(
                raw_weights,
                judge_scores,
                rewards=rewards,  # Always provide rewards
                residuals=residuals,  # Provide if available for DR
                fold_ids=fold_ids,  # Provide if available for consistent OOF
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

            # Store influence functions (always needed for proper inference)
            influence_functions[policy] = influence

        # Store influence functions for later use
        self._influence_functions = influence_functions

        # Create result with clean separation of concerns
        result = EstimationResult(
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="calibrated_ips",
            influence_functions=influence_functions,
            metadata={
                "target_policies": list(self.sampler.target_policies),
                "calibration_method": "simcal",
                "ess_floor": self.ess_floor,
                "var_cap": self.var_cap,
                "calibration_info": self._calibration_info,  # TODO: Move to diagnostics
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
            calibration_r2 = cal_info.get("r2")  # May be None if not computed
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
