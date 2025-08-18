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
from ..diagnostics import IPSDiagnostics, Status
from ..diagnostics import compute_weight_diagnostics
from ..calibration.simcal import SIMCalibrator, SimcalConfig

logger = logging.getLogger(__name__)


class CalibratedIPS(BaseCJEEstimator):
    """IPS estimator with optional SIMCal weight calibration.

    Can operate in two modes:
    1. calibrate=True (default): Uses stacked Score-Indexed Monotone Calibration (SIMCal)
       to reduce variance and heavy-tail pathologies in importance weights
    2. calibrate=False: Uses raw importance weights directly (equivalent to traditional IPS)

    Features when calibrated:
    - Stacked calibration combining multiple candidates optimally
    - OOF influence function variance minimization
    - ESS floor and variance cap constraints
    - Judge score-indexed calibration for better alignment
    - Automatic DR-aware calibration when calibrator available

    Features in both modes:
    - Oracle slice augmentation for honest confidence intervals
    - Comprehensive diagnostics
    - Optional weight clipping

    Args:
        sampler: PrecomputedSampler with data
        calibrate: Whether to apply SIMCal calibration (default True)
        clip_weight: Maximum weight value before calibration (default None = no clipping)
        ess_floor: Minimum ESS as fraction of n (default 0.2 = 20% ESS) [only used if calibrate=True]
        var_cap: Maximum allowed variance of calibrated weights (default None = no cap) [only used if calibrate=True]
        calibrator: Optional JudgeCalibrator for DR influence functions [only used if calibrate=True]
        include_baseline: Whether to include raw weights in the stack (default True) [only used if calibrate=True]
        baseline_shrink: Shrinkage toward baseline for stability (default 0.05) [only used if calibrate=True]
        **kwargs: Additional arguments passed to BaseCJEEstimator (e.g., oracle_slice_config)
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        calibrate: bool = True,
        clip_weight: Optional[float] = None,
        ess_floor: Optional[float] = 0.2,
        var_cap: Optional[float] = None,
        calibrator: Optional[Any] = None,
        include_baseline: bool = True,
        baseline_shrink: float = 0.05,
        run_diagnostics: bool = True,
        **kwargs: Any,
    ):
        # Pass oracle_slice_config to base if provided, otherwise use default "auto"
        super().__init__(
            sampler=sampler,
            run_diagnostics=run_diagnostics,
            diagnostic_config=None,  # Will use defaults
            **kwargs,  # Passes oracle_slice_config if provided
        )
        self.calibrate = calibrate
        self.clip_weight = clip_weight
        self.ess_floor = ess_floor if calibrate else None
        self.var_cap = var_cap if calibrate else None
        self.calibrator = calibrator if calibrate else None
        self.include_baseline = include_baseline if calibrate else True
        self.baseline_shrink = baseline_shrink if calibrate else 0.0
        self._no_overlap_policies: Set[str] = set()
        self._calibration_info: Dict[str, Dict] = {}  # Store calibration details
        self._diagnostics: Optional[IPSDiagnostics] = None

    def fit(self) -> None:
        """Fit weights for all target policies (with or without calibration)."""
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

            # If not calibrating, just use raw weights
            if not self.calibrate:
                logger.debug(
                    f"Raw IPS weights for policy '{policy}': "
                    f"mean={raw_weights.mean():.3f}, std={raw_weights.std():.3f}, "
                    f"min={raw_weights.min():.3f}, max={raw_weights.max():.3f}"
                )

                # Cache raw weights
                self._weights_cache[policy] = raw_weights

                # Fit m̂(S) for oracle slice augmentation
                if judge_scores is not None:
                    self.oracle_augmentation.fit_m_hat(
                        raw_weights, judge_scores, policy, cv_folds=None
                    )

                continue  # Skip calibration for this policy

            # ========== Calibration path (original code) ==========
            logger.debug(
                f"Calibrating weights for policy '{policy}': "
                f"n_samples={len(raw_weights)}, raw_mean={raw_weights.mean():.3f}"
            )

            # Use SIMCal calibration with appropriate ordering index
            # When calibrator available: use cross-fitted g(s) for better alignment with DR
            # Otherwise: fall back to raw judge scores
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

            # Try to get DR residuals and cross-fitted rewards if calibrator available
            residuals = None
            fold_ids = None
            g_oof = None
            if self.calibrator is not None and hasattr(self.calibrator, "predict_oof"):
                try:
                    # Extract fold IDs from data
                    fold_list = [d.get("cv_fold") for d in data]
                    if all(v is not None for v in fold_list) and len(fold_list) == len(
                        judge_scores
                    ):
                        fold_ids = np.asarray(fold_list, dtype=int)
                        # Compute cross-fitted predictions
                        g_oof = self.calibrator.predict_oof(judge_scores, fold_ids)
                        residuals = rewards - g_oof
                        logger.debug(f"Using DR residuals for policy '{policy}'")
                        logger.debug(
                            f"Using cross-fitted rewards as SIMCal ordering index"
                        )
                except Exception as e:
                    logger.debug(f"Could not compute DR residuals: {e}")

            # Determine the ordering index for SIMCal
            # Use cross-fitted calibrated rewards if available, otherwise raw judge scores
            # This aligns the monotone projection with the actual nuisance function used in DR
            ordering_index = g_oof if g_oof is not None else judge_scores

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
                ordering_index,  # Now uses g_oof when available, judge_scores otherwise
                rewards=rewards,  # Always provide rewards
                residuals=residuals,  # Provide if available for DR
                fold_ids=fold_ids,  # Provide if available for consistent OOF
            )

            # Cache results
            self._weights_cache[policy] = calibrated
            self._calibration_info[policy] = calib_info

            # Fit m̂(S) for oracle slice augmentation
            # Use the calibrated weights we'll actually use in estimation
            self.oracle_augmentation.fit_m_hat(
                calibrated, judge_scores, policy, cv_folds=fold_ids
            )

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

            # SAFETY CHECK: Refuse to provide unreliable estimates
            # Following CLAUDE.md: "Fail Fast and Clearly"

            # Check effective sample size
            ess = np.sum(weights) ** 2 / np.sum(weights**2) / n

            # Check weight concentration: What fraction of total weight is on top 5% of samples?
            sorted_weights = np.sort(weights)[::-1]
            top_5pct_count = max(1, int(0.05 * n))
            top_5pct_weight = np.sum(sorted_weights[:top_5pct_count]) / np.sum(weights)

            # Check raw weights for hidden problems (calibration can mask issues)
            raw_weights = self.get_raw_weights(policy)
            raw_near_zero = 0.0
            if raw_weights is not None:
                raw_near_zero = np.sum(raw_weights < 1e-10) / len(raw_weights)

            # Coefficient of variation as additional check
            cv_weights = (
                np.std(weights) / np.mean(weights)
                if np.mean(weights) > 0
                else float("inf")
            )

            # Refuse if multiple indicators suggest unreliability
            # We use percentage-based gates because they measure distribution overlap quality,
            # not just statistical power. Poor overlap means the estimate is dominated by
            # a small subset of data, making it practically unreliable even if statistically valid.
            refuse = False
            reasons = []

            if ess < 0.30:  # Less than 30% effective sample size
                refuse = True
                reasons.append(f"ESS={ess:.1%}")

            if raw_near_zero > 0.85:  # More than 85% of raw weights near zero
                refuse = True
                reasons.append(f"raw_near_zero={raw_near_zero:.1%}")

            if (
                top_5pct_weight > 0.30 and cv_weights > 2.0
            ):  # High concentration AND high variability
                refuse = True
                reasons.append(f"top_5%={top_5pct_weight:.1%} with CV={cv_weights:.1f}")

            if refuse:
                # Provide detailed explanation of what low ESS means practically
                logger.error(
                    f"Cannot reliably estimate policy '{policy}': only {ess:.1%} effective overlap. "
                    f"This means {(1-ess)*100:.0f}% of your data is essentially ignored. "
                    f"Reasons for refusal: {', '.join(reasons)}. "
                    f"Solutions: (1) Use policies with better overlap, "
                    f"(2) Try DR methods with fresh draws, "
                    f"(3) Collect data from more diverse base policies."
                )
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                influence_functions[policy] = np.full(n, np.nan)
                continue

            # Base IPS contribution
            base_contrib = weights * rewards

            # Add oracle slice augmentation for honest CIs
            aug, aug_diagnostics = self.oracle_augmentation.compute_augmentation(
                policy, rewards, data, self.sampler.dataset.samples
            )
            self._aug_diagnostics[policy] = aug_diagnostics

            # Total contribution with augmentation
            total_contrib = base_contrib + aug
            estimate = float(total_contrib.mean())
            estimates.append(estimate)

            # Compute standard error using augmented influence functions
            influence = total_contrib - estimate
            se = float(np.std(influence, ddof=1) / np.sqrt(n))
            standard_errors.append(se)

            # Add slice variance share to diagnostics
            if aug_diagnostics:
                var_base = (
                    np.var(base_contrib - base_contrib.mean(), ddof=1) if n > 1 else 0.0
                )
                var_total = (
                    np.var(total_contrib - total_contrib.mean(), ddof=1)
                    if n > 1
                    else 0.0
                )
                aug_diagnostics["slice_variance_share"] = (
                    float(aug_diagnostics.get("aug_var", 0.0) / var_total)
                    if var_total > 0
                    else 0.0
                )

            # Store influence functions (always needed for proper inference)
            influence_functions[policy] = influence

        # Store influence functions for later use
        self._influence_functions = influence_functions

        # Create result with clean separation of concerns
        result = EstimationResult(
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="calibrated_ips" if self.calibrate else "raw_ips",
            influence_functions=influence_functions,
            metadata={
                "target_policies": list(self.sampler.target_policies),
                "calibration_method": "simcal" if self.calibrate else None,
                "ess_floor": self.ess_floor,
                "var_cap": self.var_cap,
                "calibration_info": self._calibration_info,  # TODO: Move to diagnostics
                "slice_augmentation": self._aug_diagnostics,  # Oracle slice augmentation info
            },
        )

        # Build and attach diagnostics directly
        diagnostics = self._build_diagnostics(result)
        result.diagnostics = diagnostics
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
        tail_indices = {}
        status_per_policy = {}
        overall_ess = 0.0
        total_n = 0

        for policy in policies:
            weights = self.get_weights(policy)
            if weights is not None and len(weights) > 0:
                w_diag = compute_weight_diagnostics(weights, policy, compute_hill=True)
                ess_per_policy[policy] = w_diag["ess_fraction"]
                max_weight_per_policy[policy] = w_diag["max_weight"]
                status_per_policy[policy] = w_diag["status"]  # Store per-policy status

                # Hill tail index is now computed in compute_weight_diagnostics
                if "tail_index" in w_diag:
                    tail_indices[policy] = w_diag["tail_index"]
                else:
                    tail_indices[policy] = None

                # Track overall
                n = len(weights)
                overall_ess += w_diag["ess_fraction"] * n
                total_n += n

        # Compute overall weight ESS
        weight_ess = overall_ess / total_n if total_n > 0 else 0.0

        # Determine status based on ESS and tail indices
        worst_tail_idx = min(
            (idx for idx in tail_indices.values() if idx is not None),
            default=float("inf"),
        )
        if weight_ess < 0.01:
            weight_status = Status.CRITICAL
        elif worst_tail_idx < 1.5:  # Very heavy tails
            weight_status = Status.CRITICAL
        elif weight_ess < 0.1:
            weight_status = Status.WARNING
        elif worst_tail_idx < 2.0:  # Heavy tails (infinite variance)
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

        # Store tail indices in result metadata
        if tail_indices:
            result.metadata["tail_indices"] = tail_indices

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
            status_per_policy=status_per_policy,
            tail_indices=tail_indices,  # Use Hill indices instead of tail ratios
            # Calibration fields
            calibration_rmse=calibration_rmse,
            calibration_r2=calibration_r2,
            n_oracle_labels=n_oracle_labels,
        )

        return diagnostics

    def get_diagnostics(self) -> Optional[IPSDiagnostics]:
        """Get the diagnostics object."""
        return self._diagnostics
