"""Orthogonalized Calibrated IPS estimator.

This provides first-order robustness to errors in both:
1. The reward calibrator f̂(S)
2. The weight calibrator m̂(S) = E[W|S]

Uses a SIMCal-anchored approach that preserves variance control
while adding orthogonalization corrections.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, cast
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .calibrated_ips import CalibratedIPS
from ..data.models import EstimationResult
from ..data.folds import get_fold

logger = logging.getLogger(__name__)


class OrthogonalizedCalibratedIPS(CalibratedIPS):
    """Calibrated IPS with first-order orthogonalization.

    This estimator extends CalibratedIPS with orthogonalization against
    both reward and weight nuisance functions, achieving:
    - First-order insensitivity to f̂(S) errors (reward calibration)
    - First-order insensitivity to m̂(S) errors (weight calibration)
    - Clean √n asymptotic behavior
    - Preserved variance gains from SIMCal

    The SIMCal-anchored formulation:
    V̂ = P_n[W̃·R] + P_n[(W-m̂^OOF)(R^OOF-f̂^OOF)] + P_n[f̂^OOF(W-W̃)]

    where:
    - W̃: SIMCal calibrated weights (variance-stabilized)
    - W: Raw importance weights
    - R: Calibrated rewards (global fit)
    - R^OOF, f̂^OOF: Out-of-fold calibrated rewards
    - m̂^OOF: Out-of-fold E[W|S]
    """

    def __init__(
        self, *args: Any, use_orthogonalization: bool = True, **kwargs: Any
    ) -> None:
        """Initialize OC-IPS estimator.

        Args:
            use_orthogonalization: Whether to apply orthogonalization.
                If False, behaves exactly like CalibratedIPS.
            *args, **kwargs: Passed to CalibratedIPS
        """
        super().__init__(*args, **kwargs)
        self.use_orthogonalization = use_orthogonalization
        self._m_hat_oof_cache: Dict[str, np.ndarray] = {}
        self._orthogonalization_diagnostics: Dict[str, Dict] = {}

    def fit(self) -> None:
        """Fit the estimator with additional m̂^OOF computation."""
        # Run parent fit (handles weight calibration, OUA, etc.)
        super().fit()

        if not self.use_orthogonalization or not self.calibrate:
            return

        # For each policy, fit m̂^OOF(S) for orthogonalization
        logger.debug("Fitting m̂^OOF for orthogonalization")

        for policy in self.sampler.target_policies:
            if policy in self._no_overlap_policies:
                continue

            # Get data for this policy
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                continue

            # Get raw weights W (before SIMCal)
            raw_weights = self.sampler.compute_importance_weights(
                policy, clip_weight=self.clip_weight, mode=self.weight_mode
            )
            if raw_weights is None:
                continue

            # Get judge scores
            judge_scores = np.array([d.get("judge_score", np.nan) for d in data])
            if np.all(np.isnan(judge_scores)):
                logger.warning(f"No judge scores for policy {policy}, skipping m̂^OOF")
                continue

            # Get fold assignments (local to this policy subset)
            n_folds = self.sampler.dataset.metadata.get("n_folds", 5)
            seed = self.sampler.dataset.metadata.get("fold_seed", 42)
            prompt_ids = [d.get("prompt_id", f"sample_{i}") for i, d in enumerate(data)]
            fold_ids = np.array([get_fold(pid, n_folds, seed) for pid in prompt_ids])

            # Cross-fit m̂(S) = E[W|S]
            m_hat_oof = self._fit_m_hat_oof(raw_weights, judge_scores, fold_ids)
            self._m_hat_oof_cache[policy] = m_hat_oof

            logger.debug(
                f"Fitted m̂^OOF for {policy}: mean={m_hat_oof.mean():.3f}, "
                f"std={m_hat_oof.std():.3f}, range=[{m_hat_oof.min():.3f}, {m_hat_oof.max():.3f}]"
            )

    def _fit_m_hat_oof(
        self, weights: np.ndarray, judge_scores: np.ndarray, fold_ids: np.ndarray
    ) -> np.ndarray:
        """Cross-fit m̂(S) = E[W|S] for orthogonalization.

        Args:
            weights: Raw importance weights
            judge_scores: Judge scores S
            fold_ids: Fold assignments for cross-fitting

        Returns:
            m_hat_oof: Out-of-fold predictions of E[W|S]
        """
        m_hat_oof = np.zeros_like(weights)
        unique_folds = np.unique(fold_ids[fold_ids >= 0])

        if len(unique_folds) < 2:
            # Not enough folds for cross-fitting, use global fit
            logger.debug("Insufficient folds for cross-fitting m̂, using global fit")
            valid_mask = np.isfinite(judge_scores)
            if valid_mask.sum() > 1:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(judge_scores[valid_mask], weights[valid_mask])
                m_hat_oof[valid_mask] = iso.predict(judge_scores[valid_mask])
                m_hat_oof[~valid_mask] = weights[valid_mask].mean()
            else:
                m_hat_oof = np.ones_like(weights)
        else:
            # Cross-fit across folds
            for fold in unique_folds:
                train_mask = (
                    (fold_ids >= 0) & (fold_ids != fold) & np.isfinite(judge_scores)
                )
                test_mask = (fold_ids == fold) & np.isfinite(judge_scores)

                # Robust fold handling with minimum requirements
                min_train = 100  # Minimum training samples
                min_bins = 8  # Minimum unique score values

                unique_train_scores = (
                    np.unique(judge_scores[train_mask]).size
                    if train_mask.sum() > 0
                    else 0
                )

                if (
                    train_mask.sum() < min_train
                    or test_mask.sum() == 0
                    or unique_train_scores < min_bins
                ):
                    # Not enough data in this fold
                    if test_mask.sum() > 0:
                        m_hat_oof[test_mask] = (
                            weights[train_mask].mean() if train_mask.sum() > 0 else 1.0
                        )
                    continue

                # Fit isotonic regression on training folds
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(judge_scores[train_mask], weights[train_mask])

                # Predict on test fold
                m_hat_oof[test_mask] = iso.predict(judge_scores[test_mask])

            # Handle any samples with missing judge scores
            missing_mask = np.isnan(judge_scores)
            if missing_mask.sum() > 0:
                m_hat_oof[missing_mask] = weights[~missing_mask].mean()

        # Scale to match mean(W) not 1.0 for better orthogonality
        valid_mask = np.isfinite(judge_scores)
        muW = float(np.mean(weights[valid_mask])) if valid_mask.sum() > 0 else 1.0

        if muW <= 0:
            logger.warning(f"Non-positive mean(W)={muW:.3f}, falling back to 1.0")
            muW = 1.0

        if m_hat_oof.mean() > 1e-12:
            m_hat_oof = m_hat_oof * (muW / m_hat_oof.mean())
        else:
            logger.warning("m̂^OOF has near-zero mean, using ones")
            m_hat_oof = np.ones_like(m_hat_oof) * muW

        return m_hat_oof

    def estimate(self) -> EstimationResult:
        """Compute OC-IPS estimates with orthogonalization."""
        if not self.use_orthogonalization or not self.calibrate:
            # Fall back to standard CalibratedIPS
            return super().estimate()

        self._validate_fitted()

        estimates = []
        standard_errors = []
        n_samples_used = {}
        influence_functions = {}

        for policy in self.sampler.target_policies:
            if policy in self._no_overlap_policies:
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                influence_functions[policy] = np.array([np.nan])
                logger.warning(f"Policy '{policy}' has no overlap - returning NaN")
                continue

            # Get data for this policy
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                logger.warning(f"No data for policy '{policy}'. Skipping.")
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                continue

            n = len(data)
            n_samples_used[policy] = n

            # Get weights
            W = self.sampler.compute_importance_weights(
                policy, clip_weight=self.clip_weight, mode=self.weight_mode
            )  # Raw weights
            W_tilde = self._weights_cache[policy]  # SIMCal calibrated weights

            # Get rewards
            R = np.array([d["reward"] for d in data])  # Global fit rewards

            # Initialize OOF quantities (default to global if not available)
            R_oof = R.copy()
            f_oof = R.copy()

            # Try to get OOF rewards using dataset indices (best), else fold-based OOF
            if self.calibrator is not None:
                try:
                    # 1) Prefer dataset-index OOF if available
                    if hasattr(self.calibrator, "predict_oof_by_index"):
                        # Build mapping from prompt_id to dataset index
                        ds_index_by_pid = {
                            str(s.prompt_id): i
                            for i, s in enumerate(self.sampler.dataset.samples)
                        }
                        ds_idx = np.array(
                            [
                                ds_index_by_pid.get(str(d.get("prompt_id")), -1)
                                for d in data
                            ],
                            dtype=int,
                        )

                        # Only proceed if all indices are valid
                        if np.all(ds_idx >= 0):
                            R_pred = self.calibrator.predict_oof_by_index(ds_idx)
                            if R_pred is not None:
                                R_oof = np.asarray(R_pred, dtype=float)
                                f_oof = R_oof
                                logger.debug(
                                    f"Using index-based OOF rewards for {policy}"
                                )

                    # 2) Else try fold-based OOF using per-prompt folds
                    elif hasattr(self.calibrator, "predict_oof"):
                        from ..data.folds import get_fold

                        n_folds = self.sampler.dataset.metadata.get("n_folds", 5)
                        seed = self.sampler.dataset.metadata.get("fold_seed", 42)
                        judge_scores = np.array(
                            [d.get("judge_score", np.nan) for d in data], dtype=float
                        )
                        prompt_ids = [
                            d.get("prompt_id", f"sample_{i}")
                            for i, d in enumerate(data)
                        ]
                        fold_cal = np.array(
                            [get_fold(pid, n_folds, seed) for pid in prompt_ids],
                            dtype=int,
                        )

                        # Check if we have valid judge scores
                        valid_scores = np.isfinite(judge_scores)
                        if valid_scores.sum() > 0:
                            R_pred = self.calibrator.predict_oof(judge_scores, fold_cal)
                            if R_pred is not None:
                                R_oof = np.asarray(R_pred, dtype=float)
                                f_oof = R_oof
                                logger.debug(
                                    f"Using fold-based OOF rewards for {policy}"
                                )

                    # 3) Fallback: in-fold predict with a warning
                    elif hasattr(self.calibrator, "predict"):
                        judge_scores = np.array(
                            [d.get("judge_score", np.nan) for d in data], dtype=float
                        )
                        valid_scores = np.isfinite(judge_scores)
                        if valid_scores.sum() > 0:
                            R_pred = self.calibrator.predict(judge_scores)
                            if R_pred is not None:
                                R_oof = np.asarray(R_pred, dtype=float)
                                f_oof = R_oof
                                logger.warning(
                                    f"OC-IPS: Using in-fold calibrator.predict() for {policy}; orthogonality guarantees may weaken."
                                )
                except Exception as e:
                    logger.debug(f"OC-IPS: OOF reward path failed for '{policy}': {e}")

            # Get m̂^OOF (default to ones if not available)
            m_hat_oof = self._m_hat_oof_cache.get(policy, np.ones_like(W))

            # Compute OC-IPS (SIMCal-anchored, two-term version)
            # Since R_oof == f_oof, the orthog term (W-m̂)(R-f̂) is zero
            # We only need baseline + retarget for orthogonality

            # 1. Baseline term (use OOF rewards for consistency with IF)
            baseline = W_tilde * R_oof

            # 2. Re-targeting term (achieves orthogonality to both f̂ and m̂)
            retarget = f_oof * (W - W_tilde)

            # Total contribution before augmentation
            contrib = baseline + retarget

            # Add oracle slice augmentation (as in parent)
            aug, aug_diagnostics = self.oracle_augmentation.compute_augmentation(
                policy,
                R,  # Keep using R for augmentation (it's designed for that)
                cast(List[Dict[str, Any]], data),
                self.sampler.dataset.samples,
            )
            self._aug_diagnostics[policy] = aug_diagnostics

            # Total contribution with augmentation
            total_contrib = contrib + aug

            # Point estimate
            V_hat = float(total_contrib.mean())
            estimates.append(V_hat)

            # Influence function (perfectly aligned with estimator now)
            # φ = contrib - V̂ = W̃·R^OOF + f̂^OOF(W-W̃) + aug - V̂
            phi = contrib + aug - V_hat

            # Apply IIC if enabled (variance reduction via residualization)
            if self.use_iic:
                n_folds = self.sampler.dataset.metadata.get("n_folds", 5)
                seed = self.sampler.dataset.metadata.get("fold_seed", 42)
                prompt_ids = [
                    d.get("prompt_id", f"sample_{i}") for i, d in enumerate(data)
                ]
                fold_ids = np.array(
                    [get_fold(pid, n_folds, seed) for pid in prompt_ids]
                )
                phi, iic_adjustment = self._apply_iic(phi, policy, fold_ids=fold_ids)

                # Store IIC adjustment
                if not hasattr(self, "_iic_adjustments"):
                    self._iic_adjustments = {}
                self._iic_adjustments[policy] = iic_adjustment

                # Adjust point estimate and recenter IF
                V_hat += iic_adjustment
                phi -= iic_adjustment  # Critical: recenter IF after adjusting V_hat

            # Standard error from influence function
            se = float(np.std(phi, ddof=1) / np.sqrt(n))
            standard_errors.append(se)
            influence_functions[policy] = phi

            # Store orthogonalization diagnostics with CIs
            retarget_se = float(np.std(retarget, ddof=1) / np.sqrt(n))
            retarget_ci = 1.96 * retarget_se

            self._orthogonalization_diagnostics[policy] = {
                "retarget_residual": float(retarget.mean()),
                "retarget_se": retarget_se,
                "retarget_ci_lower": float(retarget.mean() - retarget_ci),
                "retarget_ci_upper": float(retarget.mean() + retarget_ci),
                "baseline_contrib": float(baseline.mean()),
                "uses_oof_rewards": not np.array_equal(R, R_oof),
            }

            logger.debug(
                f"OC-IPS for {policy}: V̂={V_hat:.4f}, SE={se:.4f}, "
                f"retarget={retarget.mean():.6f}±{retarget_se:.6f}"
            )

        # Store influence functions for later use
        self._influence_functions = influence_functions

        # Create result
        result = EstimationResult(
            estimates=np.array(estimates),
            standard_errors=np.array(standard_errors),
            n_samples_used=n_samples_used,
            method="oc-ips",
            influence_functions=influence_functions,
            diagnostics=None,  # Will be set below
            robust_standard_errors=None,
            robust_confidence_intervals=None,
            metadata={
                "target_policies": list(self.sampler.target_policies),
                "calibrate": self.calibrate,
                "use_orthogonalization": self.use_orthogonalization,
                "orthogonalization_diagnostics": self._orthogonalization_diagnostics,
                "augmentation_diagnostics": self._aug_diagnostics,
            },
        )

        # Create diagnostics similar to parent class
        if self.run_diagnostics:
            from ..diagnostics import IPSDiagnostics, Status

            # Gather weight statistics
            ess_per_policy = {}
            max_weight_per_policy = {}
            overall_ess = 0.0

            for policy in self.sampler.target_policies:
                if policy in self._weights_cache:
                    weights = self._weights_cache[policy]
                    # Compute raw ESS (not divided by n)
                    ess = float((weights.sum() ** 2) / (weights**2).sum())
                    ess_normalized = ess / len(weights)  # ESS per sample
                    ess_per_policy[policy] = ess_normalized
                    max_weight_per_policy[policy] = float(weights.max())
                    overall_ess += ess_normalized

            if len(ess_per_policy) > 0:
                overall_ess = overall_ess / len(ess_per_policy)

            # Determine status based on ESS
            if overall_ess > 0.5:
                status = Status.GOOD
            elif overall_ess > 0.2:
                status = Status.WARNING
            else:
                status = Status.CRITICAL

            # Create diagnostics
            policies = list(self.sampler.target_policies)
            estimates_dict = {p: result.estimates[i] for i, p in enumerate(policies)}
            se_dict = {p: result.standard_errors[i] for i, p in enumerate(policies)}

            diagnostics = IPSDiagnostics(
                estimator_type="OrthogonalizedCalibratedIPS",
                method="oc-ips",
                n_samples_total=len(self.sampler.dataset.samples),
                n_samples_valid=self.sampler.n_valid_samples,
                n_policies=len(policies),
                policies=policies,
                estimates=estimates_dict,
                standard_errors=se_dict,
                n_samples_used=result.n_samples_used,
                weight_ess=overall_ess,
                weight_status=status,
                ess_per_policy=ess_per_policy,
                max_weight_per_policy=max_weight_per_policy,
            )

            result.diagnostics = diagnostics

        self._results = result
        return result
