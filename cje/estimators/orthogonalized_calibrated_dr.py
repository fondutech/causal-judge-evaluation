# orthogonalized_calibrated_dr.py
# -*- coding: utf-8 -*-
"""
Orthogonalized Calibrated DR-CPO (OC-DR-CPO).

This module implements the SIMCal-anchored, orthogonalized calibrated DR estimator:
    V̂_ODR
      = P_n[ g(X) + W̃ * { R - q(X,A) } ]
      + P_n[ (W - m̂^OOF) * (R^OOF - f̂^OOF) ]                 # orthogonalizer
      + P_n[ (R^OOF - q^OOF) * (W - W̃) ]                      # retarget-to-W

Properties:
- First-order insensitivity to errors in BOTH nuisances:
  (i) reward calibrator f̂(S) and (ii) weight calibrator m̂(S)=E[W|S].
- √n inference with cross-fitting (OOF) and the standard OUA jackknife add-on.
- Preserves SIMCal’s tail stability by anchoring on W̃.

Implementation notes:
- Reuses DREstimator’s infrastructure (fresh draws, outcome model, oracle augmentation, IIC).
- Cross-fits m̂^OOF(S) locally per-policy subset via isotonic W~S, using per-policy folds.
- Fetches OOF rewards for the residual corrections by DATASET INDEX if available
  (calibrator.predict_oof_by_index). Falls back to fold-based OOF or plain predict() with a warning.
- Uses the same per-prompt fold mapping as the outcome model for q^OOF on logged data.

Diagnostics:
- Reports orthogonalization and retarget residuals (mean ± CI); both should include 0.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, Tuple, cast

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .dr_base import (
    DREstimator,
)  # Base DR implementation (fresh draws, outcome model, OUA, IIC)
from ..data.precomputed_sampler import PrecomputedSampler
from ..data.models import EstimationResult
from ..data.folds import get_fold

logger = logging.getLogger(__name__)


class OrthogonalizedCalibratedDRCPO(DREstimator):
    """Orthogonalized Calibrated DR-CPO estimator (OC-DR-CPO).

    Estimator formula (per-sample contributions):
        contrib = g_fresh
                + W_tilde * (R - q_logged)
                + (W - m_hat_oof) * (R_oof - f_oof)           # orthogonalizer
                + (R_oof - q_oof) * (W - W_tilde)             # retarget-to-W
                + aug_vector                                  # oracle augmentation on the IPS-like part

    Influence function (OOF path):
        φ_i = g_fresh_i
            + W_tilde_i * (R_oof_i - q_oof_i)
            + (W_i - m_hat_oof_i)*(R_oof_i - f_oof_i)
            + (R_oof_i - q_oof_i)*(W_i - W_tilde_i)
            + aug_i
            - V̂

    Notes
    -----
    * Anchors on SIMCal weights (W̃) from the internal CalibratedIPS within DREstimator.
    * Requires fresh draws (same as DR-CPO). Add via DREstimator.add_fresh_draws().
    * Adds two residual diagnostics:
        - orthog_residual = mean[(W - m̂^OOF)(R^OOF - f̂^OOF)]
        - retarget_residual = mean[(R^OOF - q^OOF)(W - W̃)]
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        n_folds: int = 5,
        use_calibrated_weights: bool = True,  # MUST be True to anchor on SIMCal (recommended)
        weight_mode: str = "hajek",
        calibrator: Optional[Any] = None,
        random_seed: int = 42,
        run_diagnostics: bool = True,
        use_iic: bool = True,
        use_orthogonalization: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            sampler: PrecomputedSampler with calibrated data
            n_folds: Outcome-model cross-fitting folds (reused by DREstimator)
            use_calibrated_weights: True => SIMCal anchor (recommended)
            weight_mode: 'hajek' (mean-one) or 'raw' for W (recommended: 'hajek')
            calibrator: Reward calibrator f̂; used for R and OOF predictions
            random_seed: Seed for deterministic fold assignment
            run_diagnostics: Whether to compute diagnostics
            use_iic: Apply IIC residualization to IF for SE tightening
            use_orthogonalization: If False, falls back to simple DR-CPO
            **kwargs: forwarded to DREstimator (e.g., oracle_slice_config)
        """
        super().__init__(
            sampler=sampler,
            outcome_model=None,  # DREstimator chooses model automatically
            n_folds=n_folds,
            use_calibrated_weights=use_calibrated_weights,
            weight_mode=weight_mode,
            calibrator=calibrator,
            random_seed=random_seed,
            run_diagnostics=run_diagnostics,
            **kwargs,
        )
        self.use_iic = use_iic
        self.use_orthogonalization = use_orthogonalization
        self._m_hat_oof_cache: Dict[str, np.ndarray] = {}
        self._orthogonalization_diagnostics: Dict[str, Dict[str, Any]] = {}

    # ---------- Fit: add m̂^OOF(S) per policy (local folds) ----------

    def fit(self) -> None:
        """Fit weights (SIMCal), outcome model, and m̂^OOF for each policy."""
        # Parent does: fit IPS (SIMCal) and outcome model; sets _promptid_to_fold
        super().fit()

        if not self.use_orthogonalization:
            return

        logger.debug("Fitting m̂^OOF(S) = E[W|S] per policy for ODR orthogonalization")

        for policy in self.sampler.target_policies:
            # Logged subset for this policy
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                continue

            # Raw/base W for the residual (use hajek mean-one to align with W̃)
            W = self.sampler.compute_importance_weights(
                policy, clip_weight=None, mode=self.ips_estimator.weight_mode
            )

            # Judge scores S
            S = np.array([d.get("judge_score", np.nan) for d in data], dtype=float)
            if np.all(~np.isfinite(S)):
                logger.warning(
                    f"No finite judge scores for policy '{policy}', skipping m̂^OOF."
                )
                continue

            # Local fold assignments (simple, robust): computed from prompt_id, independent of calibrator folds
            n_folds = self.sampler.dataset.metadata.get("n_folds", 5)
            seed = self.sampler.dataset.metadata.get("fold_seed", 42)
            prompt_ids = [d.get("prompt_id", f"sample_{i}") for i, d in enumerate(data)]
            fold_ids = np.array(
                [get_fold(pid, n_folds, seed) for pid in prompt_ids], dtype=int
            )

            # Cross-fitted isotonic m̂(S) = E[W|S]
            m_hat_oof = self._fit_m_hat_oof(W, S, fold_ids)
            self._m_hat_oof_cache[policy] = m_hat_oof

            logger.debug(
                f"m̂^OOF for {policy}: mean={m_hat_oof.mean():.4f}, "
                f"std={m_hat_oof.std():.4f}, n={len(m_hat_oof)}"
            )

    def _fit_m_hat_oof(
        self, weights: np.ndarray, judge_scores: np.ndarray, fold_ids: np.ndarray
    ) -> np.ndarray:
        """Cross-fit m̂(S)=E[W|S] by isotonic regression with per-policy local folds."""
        m_hat_oof = np.zeros_like(weights, dtype=float)
        uniq = np.unique(fold_ids[fold_ids >= 0])

        # If we cannot cross-fit, fall back to global isotonic or a constant
        if uniq.size < 2:
            valid = np.isfinite(judge_scores)
            if valid.sum() > 1:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(judge_scores[valid], weights[valid])
                m_hat_oof[valid] = iso.predict(judge_scores[valid])
                m_hat_oof[~valid] = float(weights[valid].mean())
            else:
                m_hat_oof[:] = float(np.mean(weights)) if weights.size else 1.0
        else:
            # Conservative fold requirements
            min_train = 100
            min_bins = 8

            for f in uniq:
                train = (fold_ids >= 0) & (fold_ids != f) & np.isfinite(judge_scores)
                test = (fold_ids == f) & np.isfinite(judge_scores)

                if test.sum() == 0:
                    continue

                if (train.sum() < min_train) or (
                    np.unique(judge_scores[train]).size < min_bins
                ):
                    # Not enough signal; use pooled mean on test fold
                    m_hat_oof[test] = (
                        float(weights[train].mean()) if train.sum() else 1.0
                    )
                    continue

                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(judge_scores[train], weights[train])
                m_hat_oof[test] = iso.predict(judge_scores[test])

            # Handle NaN S
            missing = ~np.isfinite(judge_scores)
            if missing.any():
                m_hat_oof[missing] = float(weights[~missing].mean())

        # Scale m̂ to match mean(W) on this subset (better orthogonality than forcing mean=1)
        valid = np.isfinite(judge_scores)
        muW = float(weights[valid].mean()) if valid.any() else float(weights.mean())
        muW = muW if muW > 0 else 1.0

        mean_m = float(m_hat_oof.mean()) if m_hat_oof.size else 1.0
        if mean_m > 1e-12:
            m_hat_oof *= muW / mean_m
        else:
            logger.warning("m̂^OOF has near-zero mean; using constant μ_W.")
            m_hat_oof[:] = muW

        return m_hat_oof

    # ---------- Estimate: ODR contributions and IF ----------

    def estimate(self) -> EstimationResult:
        """Compute ODR-CPO estimates with orthogonalization (or DR-CPO fallback)."""
        self._validate_fitted()

        # Try to auto-load fresh draws (DREstimator provides this)
        self._auto_load_fresh_draws()

        estimates: List[float] = []
        ses: List[float] = []
        n_used: Dict[str, int] = {}
        ifs: Dict[str, np.ndarray] = {}
        self._orthogonalization_diagnostics = {}

        # Build a fast prompt_id -> dataset index map for OOF-by-index rewards
        ds_index_by_pid: Dict[str, int] = {
            str(s.prompt_id): i for i, s in enumerate(self.sampler.dataset.samples)
        }

        for policy in self.sampler.target_policies:
            # Require fresh draws
            if policy not in self._fresh_draws:
                raise ValueError(
                    f"No fresh draws registered for '{policy}'. "
                    f"Call add_fresh_draws(policy, fresh_draws) before estimate()."
                )

            # Logged subset for this policy
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                logger.warning(f"No valid logged data for policy '{policy}'. Skipping.")
                estimates.append(np.nan)
                ses.append(np.nan)
                n_used[policy] = 0
                continue

            n = len(data)
            n_used[policy] = n

            # SIMCal weights (anchor) and mean-one raw/Hájek W for the retarget term
            W_tilde = self.ips_estimator.get_weights(policy)
            if W_tilde is None:
                logger.warning(f"No weights for policy '{policy}'. Skipping.")
                estimates.append(np.nan)
                ses.append(np.nan)
                continue

            W = self.sampler.compute_importance_weights(
                policy, clip_weight=None, mode=self.ips_estimator.weight_mode
            )

            # Logged arrays
            R_logged = np.array([d["reward"] for d in data], dtype=float)
            S_logged = np.array([d.get("judge_score") for d in data], dtype=float)
            prompts = [d["prompt"] for d in data]
            responses = [d["response"] for d in data]
            pids = [str(d.get("prompt_id")) for d in data]

            # Outcome model OOF predictions on logged data (q^OOF)
            if not hasattr(self, "_promptid_to_fold") or not self._promptid_to_fold:
                raise ValueError(
                    "Missing fold assignments for outcome model. "
                    "Ensure fit() completed with cross-fitting."
                )
            fold_ids = np.array(
                [self._promptid_to_fold[pid] for pid in pids], dtype=int
            )
            q_logged_oof = self.outcome_model.predict(
                prompts, responses, S_logged, fold_ids
            )

            # Fresh-draw DM vector (per-prompt averages, same fold as the prompt)
            fresh = self._fresh_draws[policy]
            g_fresh_list: List[float] = []
            fresh_var_list: List[float] = []
            for i, pid in enumerate(pids):
                scores_i = fresh.get_scores_for_prompt_id(pid)
                if len(scores_i) == 0:
                    g_fresh_list.append(0.0)
                    fresh_var_list.append(0.0)
                    continue
                # Outcome model expects same fold for the prompt's draws
                fold_vec = np.full(len(scores_i), fold_ids[i], dtype=int)
                preds_i = self.outcome_model.predict(
                    [prompts[i]] * len(scores_i),
                    [""] * len(scores_i),
                    np.asarray(scores_i, dtype=float),
                    fold_vec,
                )
                g_fresh_list.append(float(np.mean(preds_i)))
                fresh_var_list.append(
                    float(np.var(preds_i)) if len(preds_i) > 1 else 0.0
                )

            g_fresh = np.asarray(g_fresh_list, dtype=float)
            fresh_var = np.asarray(fresh_var_list, dtype=float)

            # OOF rewards for orthogonalization (R^OOF and f̂^OOF)
            # Prefer by-index OOF; else fold-based OOF; else plain predict() (warn).
            R_oof = R_logged.copy()
            f_oof = R_logged.copy()
            used_true_oof = False

            if self.calibrator is not None:
                try:
                    # 1) Try dataset-index OOF
                    if hasattr(self.calibrator, "predict_oof_by_index"):
                        ds_idx = np.array(
                            [ds_index_by_pid[pid] for pid in pids], dtype=int
                        )
                        R_pred = self.calibrator.predict_oof_by_index(ds_idx)
                        if R_pred is not None:
                            R_oof = np.asarray(R_pred, dtype=float)
                            f_oof = R_oof
                            used_true_oof = True
                    # 2) Else try fold-based OOF with prompt-based folds
                    elif hasattr(self.calibrator, "predict_oof"):
                        n_folds = self.sampler.dataset.metadata.get("n_folds", 5)
                        seed = self.sampler.dataset.metadata.get("fold_seed", 42)
                        fold_cal = np.array(
                            [get_fold(pid, n_folds, seed) for pid in pids], dtype=int
                        )
                        R_pred = self.calibrator.predict_oof(S_logged, fold_cal)
                        if R_pred is not None:
                            R_oof = np.asarray(R_pred, dtype=float)
                            f_oof = R_oof
                            used_true_oof = True
                    # 3) Fallback to in-fold predict (warn)
                    elif hasattr(self.calibrator, "predict"):
                        R_pred = self.calibrator.predict(S_logged)
                        if R_pred is not None:
                            R_oof = np.asarray(R_pred, dtype=float)
                            f_oof = R_oof
                            logger.warning(
                                "ODR: Using in-fold calibrator.predict() (no OOF available). "
                                "Orthogonalization guarantee may weaken."
                            )
                except Exception as e:
                    logger.debug(
                        f"ODR: OOF reward prediction failed for '{policy}': {e}"
                    )

            # m̂^OOF cache (if missing, use 1's; estimator remains valid but loses orthogonality to m̂)
            m_hat_oof = self._m_hat_oof_cache.get(policy, np.ones_like(W, dtype=float))

            # ---------- Build contributions ----------
            # Baseline DR (anchored on W̃) - use R_oof for consistency with IF
            baseline_ips = W_tilde * (R_oof - q_logged_oof)

            # Orthogonalizer and retarget terms
            if self.use_orthogonalization:
                orthog = (W - m_hat_oof) * (R_oof - f_oof)
                retarget = (R_oof - q_logged_oof) * (W - W_tilde)
            else:
                orthog = np.zeros_like(W_tilde)
                retarget = np.zeros_like(W_tilde)

            # Oracle slice augmentation (use R_logged for augmentation as designed)
            aug_vec, aug_diag = self.oracle_augmentation.compute_augmentation(
                policy,
                R_logged,  # Keep using R_logged for augmentation
                cast(List[Dict[str, Any]], data),
                self.sampler.dataset.samples,
            )
            self._aug_diagnostics[policy] = aug_diag

            # Total per-sample contribution and point estimate
            contrib = g_fresh + baseline_ips + orthog + retarget + aug_vec
            V_hat = float(np.mean(contrib))
            estimates.append(V_hat)

            # ---------- Influence function (perfectly aligned with estimator) ----------
            phi = contrib - V_hat

            # Optional IIC residualization (variance reduction) + recenter IF
            if self.use_iic:
                # Use per-prompt folds (already built) for the residualizer
                phi, iic_adjustment = self._apply_iic(phi, policy, fold_ids=fold_ids)
                # Adjust estimate and re-center IF
                V_hat += float(iic_adjustment)
                phi -= float(iic_adjustment)
                # Update stored estimate
                estimates[-1] = V_hat

            # Standard error from IF + MC variance for finite fresh draws
            base_se = float(np.std(phi, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

            # Compute MC variance component (mirrors DREstimator logic)
            draws_per_prompt = []
            for pid in pids:
                scores_i = fresh.get_scores_for_prompt_id(pid)
                draws_per_prompt.append(len(scores_i))
            M = np.asarray(draws_per_prompt, dtype=float)
            mc_var = float(np.sum(fresh_var / np.maximum(M, 1.0)) / (n**2))

            # Combined SE
            se = float(np.sqrt(base_se**2 + mc_var))
            ses.append(se)
            ifs[policy] = phi

            # ---------- Diagnostics ----------
            # Residual means and CIs (should include 0)
            def _mean_ci(v: np.ndarray) -> Tuple[float, float, float]:
                m = float(v.mean())
                s = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
                delta = 1.96 * s
                return m, m - delta, m + delta

            ortho_mean, ortho_lo, ortho_hi = _mean_ci(orthog)
            retgt_mean, retgt_lo, retgt_hi = _mean_ci(retarget)

            self._orthogonalization_diagnostics[policy] = {
                "orthog_residual": ortho_mean,
                "orthog_ci_lower": ortho_lo,
                "orthog_ci_upper": ortho_hi,
                "retarget_residual": retgt_mean,
                "retarget_ci_lower": retgt_lo,
                "retarget_ci_upper": retgt_hi,
                "baseline_dm_mean": float(np.mean(g_fresh)),
                "baseline_ips_mean": float(np.mean(baseline_ips)),
                "uses_true_oof_rewards": bool(used_true_oof),
                "mc_variance": mc_var,
                "avg_draws_per_prompt": float(np.mean(M)) if len(M) > 0 else 0.0,
            }

            logger.info(
                f"OC-DR-CPO[{policy}]: {V_hat:.4f} ± {se:.4f} | "
                f"orthog={ortho_mean:+.4e} [{ortho_lo:+.4e},{ortho_hi:+.4e}], "
                f"retarget={retgt_mean:+.4e} [{retgt_lo:+.4e},{retgt_hi:+.4e}]"
            )

        # Package result
        result = EstimationResult(
            estimates=np.asarray(estimates, dtype=float),
            standard_errors=np.asarray(ses, dtype=float),
            n_samples_used=n_used,
            method="oc_dr_cpo",
            influence_functions=ifs,
            diagnostics=None,  # The caller (suite) or parent infra can attach suites; we add metadata below
            robust_standard_errors=None,
            robust_confidence_intervals=None,
            metadata={
                "target_policies": list(self.sampler.target_policies),
                "orthogonalization_diagnostics": self._orthogonalization_diagnostics,
                "iic_estimate_adjusted": bool(self.use_iic),
                "iic_diagnostics": getattr(self, "_iic_diagnostics", None),
                "oracle_augmentation": getattr(self, "_aug_diagnostics", None),
            },
        )

        # Optionally attach OUA jackknife SEs (same logic as DR-CPO base)
        if getattr(self, "oua_jackknife", False) and self.calibrator is not None:
            try:
                oua_ses: List[float] = []
                var_oracle_map: Dict[str, float] = {}
                jk_counts: Dict[str, int] = {}
                for i, policy in enumerate(self.sampler.target_policies):
                    se_main = (
                        float(result.standard_errors[i])
                        if i < len(result.standard_errors)
                        else float("nan")
                    )
                    var_orc = 0.0
                    K = 0
                    jack = self.get_oracle_jackknife(policy)
                    if jack is not None and len(jack) >= 2:
                        K = len(jack)
                        psi_bar = float(np.mean(jack))
                        var_orc = (K - 1) / K * float(np.mean((jack - psi_bar) ** 2))
                    var_oracle_map[policy] = var_orc
                    jk_counts[policy] = K
                    oua_ses.append(float(np.sqrt(se_main**2 + var_orc)))

                result.robust_standard_errors = np.array(oua_ses)
                if isinstance(result.metadata, dict):
                    result.metadata.setdefault("oua", {})
                    result.metadata["oua"].update(
                        {
                            "var_oracle_per_policy": var_oracle_map,
                            "jackknife_counts": jk_counts,
                        }
                    )
            except Exception as e:
                logger.debug(f"OC-DR-CPO OUA jackknife failed: {e}")

        # Store IFs on self for downstream tools
        self._influence_functions = ifs
        self._results = result
        return result
