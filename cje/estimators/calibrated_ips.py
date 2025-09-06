"""Calibrated Inverse Propensity Scoring (IPS) estimator with stacked SIMCal.

This is the core CJE estimator that uses stacked Score-Indexed Monotone Calibration
(SIMCal) to stabilize IPS in heavy-tail regimes. It combines {baseline, increasing,
decreasing} candidates via convex optimization to minimize OOF influence function
variance, then blends toward uniform to meet variance/ESS constraints.
"""

import numpy as np
from typing import Dict, Optional, Set, Any, List, cast
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
        weight_mode: "hajek" for mean-one normalized weights, "raw" for unnormalized (default "hajek")
        clip_weight: Maximum weight value before calibration (default None = no clipping)
        ess_floor: Minimum ESS as fraction of n (default 0.2 = 20% ESS) [only used if calibrate=True]
        var_cap: Maximum allowed variance of calibrated weights (default None = no cap) [only used if calibrate=True]
        calibrator: Optional JudgeCalibrator for DR influence functions [only used if calibrate=True]
        include_baseline: Whether to include raw weights in the stack (default False) [only used if calibrate=True]
        baseline_shrink: Shrinkage toward baseline for stability (default 0.0) [only used if calibrate=True]
        refuse_unreliable: Whether to refuse (return NaN) for unreliable estimates (default False)
        **kwargs: Additional arguments passed to BaseCJEEstimator (e.g., oracle_slice_config)
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        calibrate: bool = True,
        weight_mode: str = "hajek",
        clip_weight: Optional[float] = None,
        ess_floor: Optional[float] = 0.2,
        var_cap: Optional[float] = None,
        calibrator: Optional[Any] = None,
        include_baseline: bool = False,
        baseline_shrink: float = 0.0,
        run_diagnostics: bool = True,
        refuse_unreliable: bool = False,
        oua_jackknife: bool = False,
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
        self.weight_mode = weight_mode
        self.clip_weight = clip_weight
        self.ess_floor = ess_floor if calibrate else None
        self.var_cap = var_cap if calibrate else None
        self.calibrator = calibrator if calibrate else None
        self.include_baseline = include_baseline if calibrate else True
        self.baseline_shrink = baseline_shrink if calibrate else 0.0
        self.refuse_unreliable = refuse_unreliable
        # Optional oracle-uncertainty jackknife (disabled by default)
        self.oua_jackknife = bool(oua_jackknife)
        self._no_overlap_policies: Set[str] = set()
        self._calibration_info: Dict[str, Dict] = {}  # Store calibration details
        self._diagnostics: Optional[IPSDiagnostics] = None

    def fit(self) -> None:
        """Fit weights for all target policies (with or without calibration)."""
        for policy in self.sampler.target_policies:
            # Get raw weights (with optional pre-clipping and weight mode)
            raw_weights = self.sampler.compute_importance_weights(
                policy, clip_weight=self.clip_weight, mode=self.weight_mode
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

                # Fit m̂(S) for oracle slice augmentation (use policy subset)
                data = self.sampler.get_data_for_policy(policy)
                if data:
                    S_policy = np.asarray(
                        [d.get("judge_score", np.nan) for d in data], dtype=float
                    )
                    if not np.all(np.isnan(S_policy)):
                        self.oracle_augmentation.fit_m_hat(
                            raw_weights, S_policy, policy, cv_folds=None
                        )

                continue  # Skip calibration for this policy

            # ========== Calibration path (original code) ==========
            logger.debug(
                f"Calibrating weights for policy '{policy}': "
                f"n_samples={len(raw_weights)}, raw_mean={raw_weights.mean():.3f}"
            )

            # Get data and judge scores for this policy
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                logger.warning(f"No data for policy '{policy}'. Skipping.")
                continue

            # Policy-subset judge scores (aligned with raw_weights)
            S_policy = np.asarray(
                [d.get("judge_score", np.nan) for d in data], dtype=float
            )
            if np.all(np.isnan(S_policy)):
                raise ValueError(
                    f"Judge scores are required for SIMCal calibration of policy '{policy}'. "
                    "Ensure samples have 'judge_score' in metadata."
                )

            # Get rewards for this policy (always needed for influence functions)
            rewards = np.array([d["reward"] for d in data], dtype=float)

            # Try to get cross-fitted rewards if calibrator available (for SIMCal ordering)
            # Get OOF predictions for this policy subset
            g_oof = None
            fold_ids: Optional[np.ndarray] = (
                None  # Initialize before conditional blocks
            )

            if self.calibrator is not None:
                try:
                    # Option 1: Use index-based OOF for policy subset
                    if hasattr(self.calibrator, "predict_oof_by_index"):
                        ds_index_by_pid = {
                            str(s.prompt_id): i
                            for i, s in enumerate(self.sampler.dataset.samples)
                        }
                        pids = [str(d.get("prompt_id")) for d in data]
                        ds_idx = np.asarray(
                            [ds_index_by_pid.get(pid, -1) for pid in pids], dtype=int
                        )
                        if np.all(ds_idx >= 0):
                            g_oof = self.calibrator.predict_oof_by_index(ds_idx)
                            if g_oof is not None:
                                logger.debug(
                                    f"Using index-based cross-fitted rewards (g^OOF) as SIMCal ordering for policy '{policy}'"
                                )

                    # Option 2: Use fold-based OOF with policy subset
                    if g_oof is None and hasattr(self.calibrator, "predict_oof"):
                        from ..data.folds import get_fold

                        n_folds = self.sampler.dataset.metadata.get("n_folds", 5)
                        seed = self.sampler.dataset.metadata.get("fold_seed", 42)
                        fold_ids = np.asarray(
                            [
                                get_fold(
                                    str(d.get("prompt_id", f"sample_{i}")),
                                    n_folds,
                                    seed,
                                )
                                for i, d in enumerate(data)
                            ],
                            dtype=int,
                        )
                        g_oof = self.calibrator.predict_oof(S_policy, fold_ids)
                        if g_oof is not None:
                            logger.debug(
                                f"Using fold-based cross-fitted rewards (g^OOF) as SIMCal ordering for policy '{policy}'"
                            )
                except Exception as e:
                    logger.debug(f"SIMCal ordering OOF failed for '{policy}': {e}")
                    g_oof = None

            # Determine the ordering index for SIMCal
            # Use cross-fitted calibrated rewards if available, otherwise raw judge scores
            # Ensure alignment with raw_weights length
            ordering_index = (
                g_oof
                if (g_oof is not None and len(g_oof) == len(raw_weights))
                else S_policy
            )

            # For residuals (used by DR methods), we need policy-specific computations
            # This is handled separately by DR estimators that have access to the subset mapping
            residuals = None

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
                residuals=residuals,  # None for IPS (DR estimators handle this separately)
                fold_ids=fold_ids,  # None for IPS (DR estimators handle this separately)
            )

            # Cache results
            self._weights_cache[policy] = calibrated
            self._calibration_info[policy] = calib_info

            # Fit m̂(S) for oracle slice augmentation
            # Use the calibrated weights we'll actually use in estimation
            self.oracle_augmentation.fit_m_hat(
                calibrated, S_policy, policy, cv_folds=fold_ids
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
                raw_near_zero = float(np.sum(raw_weights < 1e-10) / len(raw_weights))

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
                warning_msg = (
                    f"Policy '{policy}' has poor overlap: ESS fraction = {ess:.1%} (heavy-tailed weighting). "
                    f"Estimates may be dominated by a small subset of samples. "
                    f"Reasons: {', '.join(reasons)}. "
                    f"Solutions: (1) Use policies with better overlap, "
                    f"(2) Try DR methods with fresh draws, "
                    f"(3) Collect data from more diverse base policies."
                )

                if self.refuse_unreliable:
                    logger.error(f"Cannot reliably estimate {warning_msg}")
                    estimates.append(np.nan)
                    standard_errors.append(np.nan)
                    influence_functions[policy] = np.full(n, np.nan)
                    continue
                else:
                    # Provide estimate with strong warning
                    logger.warning(f"⚠️ UNRELIABLE ESTIMATE: {warning_msg}")

            # Base IPS contribution
            base_contrib = weights * rewards

            # Add oracle slice augmentation for honest CIs
            aug, aug_diagnostics = self.oracle_augmentation.compute_augmentation(
                policy,
                rewards,
                cast(List[Dict[str, Any]], data),
                self.sampler.dataset.samples,
            )
            self._aug_diagnostics[policy] = aug_diagnostics

            # Total contribution with augmentation
            total_contrib = base_contrib + aug
            estimate = float(total_contrib.mean())
            estimates.append(estimate)

            # Compute influence functions
            influence = total_contrib - estimate

            # Get fold assignments if using IIC
            fold_ids = None
            if self.use_iic:
                # Compute fold assignments for cross-fitting
                from ..data.folds import get_fold

                prompt_ids = [
                    d.get("prompt_id", f"sample_{i}") for i, d in enumerate(data)
                ]
                # Use n_folds from dataset metadata (default to 5 if not set)
                n_folds = self.sampler.dataset.metadata.get("n_folds", 5)
                seed = self.sampler.dataset.metadata.get("fold_seed", 42)
                fold_ids = np.array(
                    [get_fold(pid, n_folds, seed) for pid in prompt_ids]
                )

            # Apply IIC for variance reduction (if enabled)
            influence, iic_adjustment = self._apply_iic(
                influence, policy, fold_ids=fold_ids
            )

            # Store IIC adjustment for transparency
            if not hasattr(self, "_iic_adjustments"):
                self._iic_adjustments = {}
            self._iic_adjustments[policy] = iic_adjustment

            # Adjust the point estimate to maintain consistency with the influence function
            estimate += iic_adjustment
            estimates[-1] = estimate  # Update the stored estimate

            # Compute standard error from the (possibly residualized) influence functions
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
            diagnostics=None,  # Will be set below
            robust_standard_errors=None,
            robust_confidence_intervals=None,
            metadata={
                "target_policies": list(self.sampler.target_policies),
                "iic_diagnostics": self._iic_diagnostics if self.use_iic else None,
                "iic_adjustments": getattr(
                    self, "_iic_adjustments", {}
                ),  # IIC adjustments applied
                "iic_estimate_adjusted": self.use_iic,  # Flag: estimates already adjusted
                "calibration_method": "simcal" if self.calibrate else None,
                "ess_floor": self.ess_floor,
                "var_cap": self.var_cap,
                "calibration_info": self._calibration_info,  # TODO: Move to diagnostics
                "slice_augmentation": self._aug_diagnostics,  # Oracle slice augmentation info
            },
        )

        # Add calibration-floor metrics (logged only) per policy
        try:
            cal_info = getattr(self.sampler.dataset, "metadata", {}).get(
                "calibration_info", {}
            )
            f_min = float(cal_info.get("f_min", float("nan")))
            eps = 1e-6
            floor_meta: Dict[str, Dict[str, float]] = {}
            for policy in self.sampler.target_policies:
                data = self.sampler.get_data_for_policy(policy)
                if not data:
                    continue
                rewards = np.array([d["reward"] for d in data], dtype=float)
                if np.isfinite(f_min):
                    floor_mass_logged = float(np.mean(np.abs(rewards - f_min) <= eps))
                else:
                    floor_mass_logged = float("nan")
                floor_meta[policy] = {
                    "f_min": f_min,
                    "floor_mass_logged": floor_mass_logged,
                }
            # Attach to metadata
            if isinstance(result.metadata, dict):
                result.metadata["calibration_floor"] = floor_meta
        except Exception:
            pass

        # Optionally add oracle-uncertainty jackknife variance
        if self.oua_jackknife and self.calibrator is not None:
            oua_ses: List[float] = []
            var_oracle_map: Dict[str, float] = {}
            jk_counts: Dict[str, int] = {}
            base_se = result.standard_errors
            for i, policy in enumerate(self.sampler.target_policies):
                var_orc = 0.0
                K = 0
                jack = self.get_oracle_jackknife(policy)
                if jack is not None and len(jack) >= 2 and i < len(base_se):
                    K = len(jack)
                    psi_bar = float(np.mean(jack))
                    var_orc = (K - 1) / K * float(np.mean((jack - psi_bar) ** 2))
                var_oracle_map[policy] = var_orc
                jk_counts[policy] = K
                se_main = float(base_se[i]) if i < len(base_se) else float("nan")
                oua_ses.append(float(np.sqrt(se_main**2 + var_orc)))

            result.robust_standard_errors = np.array(oua_ses)
            # Attach OUA metadata
            if isinstance(result.metadata, dict):
                result.metadata.setdefault("oua", {})
                result.metadata["oua"].update(
                    {
                        "var_oracle_per_policy": var_oracle_map,
                        "jackknife_counts": jk_counts,
                    }
                )

        # Build and attach diagnostics directly
        diagnostics = self._build_diagnostics(result)
        result.diagnostics = diagnostics
        self._diagnostics = diagnostics

        # Attach compact core summary for empirical analysis (no UX change)
        try:
            core_summary: Dict[str, Dict[str, Any]] = {}
            ess = diagnostics.ess_per_policy if diagnostics else {}
            tails = getattr(diagnostics, "tail_indices", None) or {}
            hell_all = getattr(diagnostics, "hellinger_affinity", None)
            hell_per = getattr(diagnostics, "hellinger_per_policy", None) or {}
            cal_floor = (
                result.metadata.get("calibration_floor", {})
                if isinstance(result.metadata, dict)
                else {}
            )
            for policy in self.sampler.target_policies:
                core_summary[policy] = {
                    "ess_fraction": float(ess.get(policy, 0.0)) if ess else None,
                    "tail_index": (
                        float(tails[policy])
                        if policy in tails and tails[policy] is not None
                        else None
                    ),
                    "hellinger_affinity": (
                        float(hell_per[policy])
                        if policy in hell_per and hell_per[policy] is not None
                        else (float(hell_all) if hell_all is not None else None)
                    ),
                }
                if policy in cal_floor:
                    core_summary[policy].update(cal_floor[policy])
            if isinstance(result.metadata, dict):
                result.metadata["core_summary"] = core_summary
        except Exception:
            pass

        # Store for later access
        self._results = result

        return result

    def get_oracle_jackknife(self, policy: str) -> Optional[np.ndarray]:
        """Leave-one-oracle-fold jackknife estimates for IPS.

        For each calibrator fold model f^(−k), recompute the IPS estimate using
        rewards R^(−k) = f^(−k)(S) and the same calibrated weights, and include
        oracle augmentation with the updated residuals.

        Returns an array of K estimates, or None if not applicable.
        """
        try:
            if self.calibrator is None or not hasattr(self.calibrator, "_fold_models"):
                return None
            fold_models = getattr(self.calibrator, "_fold_models", {})
            if not fold_models:
                return None

            # Get required data
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                return None
            weights = self.get_weights(policy)
            if weights is None:
                return None

            judge_scores = np.array([d.get("judge_score") for d in data], dtype=float)

            # Sanity check alignment
            if len(judge_scores) != len(weights):
                return None

            jack: List[float] = []
            for fold_id, fold_model in fold_models.items():
                # Recompute rewards under leave-one-fold calibrator
                rewards_loo = np.clip(fold_model.predict(judge_scores), 0.0, 1.0)

                # Recompute augmentation with the updated rewards
                aug_vec, _ = self.oracle_augmentation.compute_augmentation(
                    policy,
                    rewards_loo,
                    cast(List[Dict[str, Any]], data),
                    self.sampler.dataset.samples,
                )
                contrib = weights * rewards_loo + aug_vec
                jack.append(float(np.mean(contrib)))

            return np.asarray(jack, dtype=float) if jack else None
        except Exception as e:
            logger.debug(f"get_oracle_jackknife failed for {policy}: {e}")
            return None

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
        hellinger_per_policy = {}  # New: Hellinger affinity per policy
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

                # Compute Hellinger affinity for this policy (use raw weights)
                raw_weights = self.get_raw_weights(policy)
                if raw_weights is not None and len(raw_weights) > 0:
                    from ..diagnostics.overlap import hellinger_affinity

                    hellinger_per_policy[policy] = hellinger_affinity(raw_weights)

                # Track overall
                n = len(weights)
                overall_ess += w_diag["ess_fraction"] * n
                total_n += n

        # Compute overall weight ESS
        weight_ess = overall_ess / total_n if total_n > 0 else 0.0

        # Compute overall Hellinger affinity (average across policies)
        overall_hellinger = None
        overlap_quality = None
        if hellinger_per_policy:
            overall_hellinger = float(np.mean(list(hellinger_per_policy.values())))
            # Determine overlap quality based on Hellinger
            if overall_hellinger < 0.20:
                overlap_quality = "catastrophic"
            elif overall_hellinger < 0.35:
                overlap_quality = "poor"
            elif overall_hellinger < 0.50:
                overlap_quality = "marginal"
            else:
                overlap_quality = "good"

        # Determine status based on ESS, Hellinger, and tail indices
        worst_tail_idx = min(
            (idx for idx in tail_indices.values() if idx is not None),
            default=float("inf"),
        )

        # Include Hellinger in status determination
        if overlap_quality == "catastrophic" or weight_ess < 0.01:
            weight_status = Status.CRITICAL
        elif worst_tail_idx < 1.5:  # Very heavy tails
            weight_status = Status.CRITICAL
        elif overlap_quality == "poor" or weight_ess < 0.1:
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

        # Create IPSDiagnostics with new overlap metrics
        diagnostics = IPSDiagnostics(
            estimator_type="CalibratedIPS",
            method="calibrated_ips" if self.calibrate else "raw_ips",
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
            # New overlap metrics
            hellinger_affinity=overall_hellinger,
            hellinger_per_policy=hellinger_per_policy if hellinger_per_policy else None,
            overlap_quality=overlap_quality,
            # Calibration fields
            calibration_rmse=calibration_rmse,
            calibration_r2=calibration_r2,
            n_oracle_labels=n_oracle_labels,
        )

        return diagnostics

    def get_diagnostics(self) -> Optional[IPSDiagnostics]:
        """Get the diagnostics object."""
        return self._diagnostics
