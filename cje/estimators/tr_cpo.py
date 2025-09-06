# cje/estimators/tr_cpo.py
# -*- coding: utf-8 -*-
"""
Triply-Robust Calibrated DR-CPO (TR-CPO) with efficient label correction.

Estimator:
  V̂_TR
    = P_n[ g(X) + w^{m1} * { R - q(X,A) } ]
    + P_n[ (L / π̂_L(X,S)) * m̂(S) * { Y - R } ]

  where m̂(S) = E[W|S] is cross-fitted for variance reduction

Influence function (OOF path):
  φ_i
    = g_fresh_i
    + w^{m1}_i * { R^{OOF}_i - q^{OOF}_i }
    + (L_i / π̂_L^{OOF}(Z_i)) * m̂^{OOF}(S_i) * { Y_i - R^{OOF}_i }
    - V̂_TR

Notes:
  * Uses raw/Hájek weights (NOT SIMCal) to preserve triply-robust guarantees.
  * Replaces W with m̂(S)=E[W|S] in label term for local efficiency under MAR.
  * Cross-fits both π̂_L and m̂ against judge score S via isotonic regression.
  * Falls back gracefully if no oracle labels are present (reduces to DR-CPO).
  * Reuses outcome-model & fresh-draw scaffolding from DREstimator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast
import logging
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .dr_base import DREstimator  # outcome model, fresh draws, OUA, IIC scaffolding
from ..data.precomputed_sampler import PrecomputedSampler
from ..data.models import EstimationResult
from ..data.folds import get_fold

logger = logging.getLogger(__name__)


class TRCPOEstimator(DREstimator):
    """Triply-Robust DR-CPO.

    Args:
        sampler: PrecomputedSampler
        n_folds: Cross-fitting folds (for outcome and π̂_L)
        weight_mode: 'hajek' (recommended) or 'raw' for w^{m1}
        calibrator: Reward calibrator f̂ for R and OOF predictions
        random_seed: Seed for fold hashing
        min_pi: Lower clip for π̂_L (default 1e-3)
        max_pi: Upper clip for π̂_L (default 1 - 1e-3)
        use_iic: Residualize IF against S to reduce variance
        run_diagnostics: Build DR-like diagnostics
        **kwargs: passed to base class (e.g., oracle_slice_config)
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        n_folds: int = 5,
        weight_mode: str = "hajek",
        calibrator: Optional[Any] = None,
        random_seed: int = 42,
        min_pi: float = 1e-3,
        max_pi: float = 1 - 1e-3,
        use_iic: bool = True,
        run_diagnostics: bool = True,
        **kwargs: Any,
    ):
        # TR uses raw/Hájek weights; disable SIMCal in parent (but reuse all DR infra)
        super().__init__(
            sampler=sampler,
            outcome_model=None,  # parent will choose (isotonic or calibrator-backed)
            n_folds=n_folds,
            use_calibrated_weights=False,  # IMPORTANT: w^{m1} only for TR
            weight_mode=weight_mode,
            calibrator=calibrator,
            random_seed=random_seed,
            run_diagnostics=run_diagnostics,
            **kwargs,
        )
        self.weight_mode = weight_mode  # Store weight_mode as instance attribute
        self.min_pi = float(min_pi)
        self.max_pi = float(max_pi)
        self.use_iic = bool(use_iic)
        self._piL_oof_cache: Dict[str, np.ndarray] = {}
        self._m_hat_oof_cache: Dict[str, np.ndarray] = {}  # Cache for m̂(S) = E[W|S]
        self._tr_diagnostics: Dict[str, Dict[str, Any]] = {}

    # ---------- Label-propensity (π̂_L) ----------

    def _fit_piL_oof(
        self, L: np.ndarray, S: np.ndarray, fold_ids: np.ndarray
    ) -> np.ndarray:
        """Cross-fitted isotonic π̂_L(S) with robust fallbacks."""
        n = len(L)
        pi_oof = np.empty(n, dtype=float)
        unique_folds = np.unique(fold_ids[fold_ids >= 0])

        # Fallback if no folds (shouldn't happen in normal flow)
        if unique_folds.size < 2:
            p = float(np.mean(L)) if n > 0 else 0.5
            pi_oof.fill(p)
            return np.clip(pi_oof, self.min_pi, self.max_pi)

        # Conservative fold requirements
        min_train = 100
        min_bins = 6

        for f in unique_folds:
            tr = (fold_ids != f) & np.isfinite(S)
            te = (fold_ids == f) & np.isfinite(S)

            if te.sum() == 0:
                continue

            if (tr.sum() < min_train) or (np.unique(S[tr]).size < min_bins):
                p = float(np.mean(L[tr])) if tr.sum() else float(np.mean(L))
                pi_oof[te] = p
                continue

            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(S[tr], L[tr].astype(float))
            pi_oof[te] = iso.predict(S[te])

        # Missing S → global propensity
        miss = ~np.isfinite(S)
        if miss.any():
            p = (
                float(np.mean(L[np.isfinite(S)]))
                if np.isfinite(S).any()
                else float(np.mean(L))
            )
            pi_oof[miss] = p

        return np.clip(pi_oof, self.min_pi, self.max_pi)

    def _fit_m_hat_oof(
        self, weights: np.ndarray, judge_scores: np.ndarray, fold_ids: np.ndarray
    ) -> np.ndarray:
        """Cross-fit m̂(S)=E[W|S] by isotonic regression for variance reduction."""
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

        # Scale m̂ to match mean(W) on this subset (preserves unbiasedness)
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

    # ---------- Fit ----------

    def fit(self) -> None:
        """Fit outcome model; fit π̂_L OOF per policy."""
        # Clear caches to prevent stale data
        self._piL_oof_cache.clear()
        self._m_hat_oof_cache.clear()

        # Parent fit: IPS (raw/Hájek), outcome model; builds _promptid_to_fold map
        super().fit()

        # Build π̂_L OOF for each policy (based on policy subset)
        for policy in self.sampler.target_policies:
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                continue

            # Oracle labels and S
            Y_list: List[float] = []
            L_list: List[float] = []
            S_list: List[float] = []

            pids: List[str] = []
            for d in data:
                y = d.get("oracle_label", None)
                Y_list.append(float(y) if y is not None else 0.0)
                L_list.append(1.0 if y is not None else 0.0)
                s = d.get("judge_score", np.nan)
                S_list.append(float(s) if s is not None else np.nan)
                pids.append(str(d.get("prompt_id")))

            L = np.asarray(L_list, dtype=float)
            S = np.asarray(S_list, dtype=float)

            # Per-prompt folds (same hashing rule as elsewhere)
            n_folds = self.n_folds
            seed = self.random_seed
            fold_ids = np.array(
                [get_fold(pid, n_folds, seed) for pid in pids], dtype=int
            )

            # Always fit m̂(S) = E[W|S] for efficient label correction (label-independent)
            w = self.sampler.compute_importance_weights(
                policy, clip_weight=None, mode=self.weight_mode
            )
            if w is not None and len(w) == len(data):
                m_hat_oof = self._fit_m_hat_oof(w, S, fold_ids)
                self._m_hat_oof_cache[policy] = m_hat_oof

                # Compute R²(W~S) diagnostic
                valid = np.isfinite(S)
                if valid.sum() > 1:
                    ss_tot = np.var(w[valid], ddof=1)
                    ss_res = np.var(w[valid] - m_hat_oof[valid], ddof=1)
                    r2_w_s = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                    logger.info(
                        f"TR: R²(W~S) = {r2_w_s:.3f} for policy '{policy}' (higher → more variance reduction)"
                    )

            if np.sum(L) == 0:
                # No oracle labels at all; TR term will vanish at estimate() time
                logger.info(
                    f"TR: no oracle labels for policy '{policy}', skipping π̂_L fit."
                )
                # Store a flat propensity ~ 0.5 so that L/π̂_L stays bounded when L=0
                self._piL_oof_cache[policy] = np.clip(
                    np.full_like(L, fill_value=0.5), self.min_pi, self.max_pi
                )
                continue

            pi_oof = self._fit_piL_oof(L, S, fold_ids)
            self._piL_oof_cache[policy] = pi_oof

        self._fitted = True

    # ---------- Estimate ----------

    def estimate(self) -> EstimationResult:
        """Compute TR-CPO estimates + IF-based SEs (IIC optional)."""
        self._validate_fitted()
        self._auto_load_fresh_draws()  # ensure DM fresh draws are available

        estimates: List[float] = []
        ses: List[float] = []
        n_used: Dict[str, int] = {}
        if_map: Dict[str, np.ndarray] = {}
        self._tr_diagnostics = {}

        for policy in self.sampler.target_policies:
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                logger.warning(f"No data for policy '{policy}', skipping.")
                estimates.append(np.nan)
                ses.append(np.nan)
                n_used[policy] = 0
                continue

            n = len(data)
            n_used[policy] = n

            # Raw/Hájek weights (critical: no SIMCal here)
            w = self.sampler.compute_importance_weights(
                policy, clip_weight=None, mode=self.weight_mode
            )
            if w is None or len(w) != n:
                raise ValueError(f"TR: weight retrieval failed for policy '{policy}'.")

            # Logged arrays
            R = np.asarray([d["reward"] for d in data], dtype=float)
            S = np.asarray([d.get("judge_score", np.nan) for d in data], dtype=float)
            pids = [str(d.get("prompt_id")) for d in data]
            Y_list = []
            for d in data:
                label = d.get("oracle_label")
                Y_list.append(float(label) if label is not None else 0.0)
            Y = np.asarray(Y_list, dtype=float)
            L = np.asarray(
                [1.0 if d.get("oracle_label") is not None else 0.0 for d in data],
                dtype=float,
            )

            # Folds
            if not hasattr(self, "_promptid_to_fold") or not self._promptid_to_fold:
                raise ValueError("TR: missing fold map; ensure fit() completed.")
            fold_ids = np.asarray(
                [self._promptid_to_fold[pid] for pid in pids], dtype=int
            )

            # Outcome model predictions (OOF) on logged data
            prompts = [d["prompt"] for d in data]
            responses = [d["response"] for d in data]
            q_oof = self.outcome_model.predict(prompts, responses, S, fold_ids)

            # Fresh-draw DM term g_fresh_i per prompt (average over draws)
            if policy not in self._fresh_draws:
                raise ValueError(
                    f"TR: no fresh draws for '{policy}'. Add via add_fresh_draws() or place in auto path."
                )
            fresh = self._fresh_draws[policy]
            g_fresh_vals = []
            fresh_var_list = []
            draws_per_prompt = []

            for i, pid in enumerate(pids):
                scores_i = fresh.get_scores_for_prompt_id(pid)
                draws_per_prompt.append(len(scores_i))
                if len(scores_i) == 0:
                    g_fresh_vals.append(0.0)
                    fresh_var_list.append(0.0)
                    logger.warning(f"TR: No fresh draws for prompt {pid} - using 0")
                    continue
                fold_vec = np.full(len(scores_i), fold_ids[i], dtype=int)
                preds_i = self.outcome_model.predict(
                    [prompts[i]] * len(scores_i),
                    [""] * len(scores_i),
                    np.asarray(scores_i, dtype=float),
                    fold_vec,
                )
                g_fresh_vals.append(float(np.mean(preds_i)))
                fresh_var_list.append(
                    float(np.var(preds_i, ddof=1)) if len(preds_i) > 1 else 0.0
                )

            g_fresh = np.asarray(g_fresh_vals, dtype=float)
            fresh_var = np.asarray(fresh_var_list, dtype=float)
            M = np.asarray(draws_per_prompt, dtype=float)

            # R^{OOF} for IF (out-of-fold calibrated rewards)
            # Prefer index-based OOF for best alignment
            R_oof = R.copy()
            if self.calibrator is not None:
                try:
                    # First try index-based OOF (most reliable)
                    if hasattr(self.calibrator, "predict_oof_by_index"):
                        ds_index_by_pid = {
                            str(s.prompt_id): i
                            for i, s in enumerate(self.sampler.dataset.samples)
                        }
                        ds_idx = np.array(
                            [ds_index_by_pid.get(pid, -1) for pid in pids], dtype=int
                        )
                        if np.all(ds_idx >= 0):
                            ro = self.calibrator.predict_oof_by_index(ds_idx)
                            if ro is not None:
                                R_oof = np.asarray(ro, dtype=float)
                    # Fallback to fold-based OOF
                    elif hasattr(self.calibrator, "predict_oof"):
                        R_oof = np.asarray(
                            self.calibrator.predict_oof(S, fold_ids), dtype=float
                        )
                    # Last resort: in-fold prediction
                    elif hasattr(self.calibrator, "predict"):
                        logger.warning(
                            "TR: using in-fold calibrator.predict(); IF may be slightly optimistic."
                        )
                        R_oof = np.asarray(self.calibrator.predict(S), dtype=float)
                except Exception as e:
                    logger.debug(f"TR: OOF reward prediction failed: {e}")

            # π̂_L^{OOF}
            if policy in self._piL_oof_cache:
                pi_oof = np.asarray(self._piL_oof_cache[policy], dtype=float)
                if pi_oof.shape[0] != n:
                    # As a guard, recompute for this subset
                    pi_oof = self._fit_piL_oof(L, S, fold_ids)
            else:
                pi_oof = self._fit_piL_oof(L, S, fold_ids)

            # ----- Point estimate -----
            # DM mean
            dm_term = float(np.mean(g_fresh))

            # DR correction with raw/Hájek weights
            dr_corr = float(np.mean(w * (R - q_oof)))

            # Get m̂(S) = E[W|S] for efficient label correction
            if (
                policy in self._m_hat_oof_cache
                and self._m_hat_oof_cache[policy].shape[0] == n
            ):
                m_hat = self._m_hat_oof_cache[policy]
            else:
                # This should not happen in normal operation - log warning and fallback
                logger.warning(
                    f"TR-CPO: m̂(S) not found in cache for policy '{policy}'. "
                    f"Falling back to raw weights W (less efficient). "
                    f"This may indicate a bug - m̂ should have been fitted during fit()."
                )
                m_hat = w

            # Define clipped pi_oof once
            pi_clipped = np.clip(pi_oof, self.min_pi, self.max_pi)

            # Two-phase correction using m̂(S) instead of W for efficiency
            tr_vec = (L / pi_clipped) * m_hat * (Y - R)
            tr_corr = float(np.mean(tr_vec))

            V_hat = dm_term + dr_corr + tr_corr
            estimates.append(V_hat)

            # ----- Influence function (OOF path) -----
            # φ = g_fresh + w*(R_oof - q_oof) + (L/pi_oof)*m̂*(Y - R_oof) - V_hat
            # Use m̂(S) in label term for consistency with point estimate
            phi = (
                g_fresh
                + w * (R_oof - q_oof)
                + (L / pi_clipped) * m_hat * (Y - R_oof)
                - V_hat
            )

            # Optional IIC (variance-only tightening)
            if self.use_iic:
                phi, iic_adj = self._apply_iic(phi, policy, fold_ids=fold_ids)
                V_hat += float(iic_adj)
                phi -= float(iic_adj)  # recenter IF
                estimates[-1] = V_hat

            # Compute SE with Monte Carlo variance adjustment
            base_se = float(np.std(phi, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            mc_var = float(np.sum(fresh_var / np.maximum(M, 1.0)) / (n**2))
            se = float(np.sqrt(base_se**2 + mc_var))
            ses.append(se)
            if_map[policy] = phi

            # Store MC diagnostics
            if not hasattr(self, "_mc_diagnostics"):
                self._mc_diagnostics = {}
            self._mc_diagnostics[policy] = {
                "base_se": base_se,
                "mc_var": mc_var,
                "mc_share": (
                    mc_var / (base_se**2 + mc_var) if (base_se**2 + mc_var) > 0 else 0.0
                ),
                "avg_draws_per_prompt": float(M.mean()) if M.size else 0.0,
                "min_draws_per_prompt": int(M.min()) if M.size else 0,
                "max_draws_per_prompt": int(M.max()) if M.size else 0,
            }

            # ----- Diagnostics -----
            label_frac = float(np.mean(L)) if n > 0 else 0.0

            # Compute R²(W~S) to show variance reduction from using m̂(S)
            r2_w_s = 0.0
            uses_efficient_correction = False
            if policy in self._m_hat_oof_cache and not np.allclose(m_hat, w):
                uses_efficient_correction = True
                valid = np.isfinite(S)
                if valid.sum() > 1:
                    ss_tot = np.var(w[valid], ddof=1)
                    ss_res = np.var(w[valid] - m_hat[valid], ddof=1)
                    r2_w_s = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            self._tr_diagnostics[policy] = {
                "dm_mean": dm_term,
                "dr_correction_mean": dr_corr,
                "tr_correction_mean": tr_corr,
                "label_fraction": label_frac,
                "min_pi_hat": float(np.min(pi_oof)) if pi_oof.size else None,
                "median_pi_hat": float(np.median(pi_oof)) if pi_oof.size else None,
                "max_pi_hat": float(np.max(pi_oof)) if pi_oof.size else None,
                "weight_mode": self.weight_mode,
                "uses_oof_rewards": bool(not np.allclose(R, R_oof)),
                "iic_applied": bool(self.use_iic),
                "uses_efficient_correction": uses_efficient_correction,
                "r2_w_given_s": r2_w_s,  # R²(W|S) - higher means more variance reduction
            }

            # Gentle warnings for extreme regimes
            if label_frac < 0.005:
                logger.warning(
                    f"TR: very low label fraction for '{policy}' ({100*label_frac:.2f}%). "
                    "TR remains valid but may be high-variance; stacking should down-weight it."
                )
            if np.any(pi_oof < 5e-4):
                logger.warning(
                    f"TR: extremely small π̂_L detected for '{policy}'. Values are clipped to [{self.min_pi},{self.max_pi}]."
                )

        # Store IFs
        self._influence_functions = if_map

        # Build diagnostics if requested
        diagnostics = None
        if self.run_diagnostics:
            # Get IPS diagnostics from the internal IPS estimator (or None)
            ips_diagnostics = self.get_weight_diagnostics()

            # Build DR diagnostics (reuse parent's method if available)
            if hasattr(self, "_build_dr_diagnostics"):
                diagnostics = self._build_dr_diagnostics(
                    estimates=estimates,
                    standard_errors=ses,
                    n_samples_used=n_used,
                    dr_diagnostics_per_policy=self._tr_diagnostics,
                    ips_diagnostics=ips_diagnostics,
                )
            else:
                # Fallback: create minimal DRDiagnostics
                from ..diagnostics import DRDiagnostics
                from ..diagnostics.core import Status

                policies = list(self.sampler.target_policies)
                estimates_dict = {
                    p: float(e) for p, e in zip(policies, estimates) if not np.isnan(e)
                }
                se_dict = {
                    p: float(se) for p, se in zip(policies, ses) if not np.isnan(se)
                }

                diagnostics = DRDiagnostics(
                    estimator_type="TR_CPO",
                    method="tr_cpo",
                    n_samples_total=len(self.sampler.dataset.samples),
                    n_samples_valid=self.sampler.n_valid_samples,
                    n_policies=len(policies),
                    policies=policies,
                    estimates=estimates_dict,
                    standard_errors=se_dict,
                    n_samples_used=n_used,
                    # Minimal weight fields
                    weight_ess=0.0,
                    weight_status=Status.OK,
                    ess_per_policy={},
                    max_weight_per_policy={},
                    weight_tail_ratio_per_policy={},
                    # DR fields
                    dr_cross_fitted=True,
                    dr_n_folds=self.n_folds,
                    outcome_r2_range=(0.0, 0.0),
                    outcome_rmse_mean=0.0,
                    worst_if_tail_ratio=0.0,
                    dr_diagnostics_per_policy=self._tr_diagnostics,
                    dm_ips_decompositions={},
                    orthogonality_scores={},
                    influence_functions=if_map,
                )

        # Result (we keep metadata compact; stacked DR will consume IFs)
        result = EstimationResult(
            estimates=np.asarray(estimates, dtype=float),
            standard_errors=np.asarray(ses, dtype=float),
            n_samples_used=n_used,
            method="tr_cpo",
            influence_functions=if_map,
            diagnostics=diagnostics,
            robust_standard_errors=None,
            # robust_standard_errors_per_policy removed in latest version
            robust_confidence_intervals=None,
            metadata={
                "target_policies": list(self.sampler.target_policies),
                "tr_diagnostics": self._tr_diagnostics,
                "iic_estimate_adjusted": bool(self.use_iic),
                "weight_mode": self.weight_mode,
                "mc_variance_diagnostics": getattr(self, "_mc_diagnostics", None),
                "iic_diagnostics": getattr(self, "_iic_diagnostics", None),
            },
        )

        # Optional OUA (usually tiny for TR; still supported)
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
                logger.debug(f"TR-CPO OUA jackknife failed: {e}")

        # Keep local copy for stackers
        self._results = result
        return result

    # ---------- OUA jackknife (leave-one-calibrator-fold) ----------

    def get_oracle_jackknife(self, policy: str) -> Optional[np.ndarray]:
        """Compute delete-one-oracle-fold TR estimates (optional OUA)."""
        try:
            if self.calibrator is None or not hasattr(self.calibrator, "_fold_models"):
                return None
            fold_models = getattr(self.calibrator, "_fold_models", {})
            if not fold_models:
                return None

            data = self.sampler.get_data_for_policy(policy)
            if not data:
                return None

            # Raw/Hájek weights
            w = self.sampler.compute_importance_weights(
                policy, clip_weight=None, mode=self.weight_mode
            )
            if w is None:
                return None

            # Logged arrays
            R = np.asarray([d["reward"] for d in data], dtype=float)
            S = np.asarray([d.get("judge_score", np.nan) for d in data], dtype=float)
            pids = [str(d.get("prompt_id")) for d in data]
            Y_list = []
            for d in data:
                label = d.get("oracle_label")
                Y_list.append(float(label) if label is not None else 0.0)
            Y = np.asarray(Y_list, dtype=float)
            L = np.asarray(
                [1.0 if d.get("oracle_label") is not None else 0.0 for d in data],
                dtype=float,
            )

            # Folds
            n_folds = self.n_folds
            seed = self.random_seed
            fold_ids = np.asarray(
                [get_fold(pid, n_folds, seed) for pid in pids], dtype=int
            )

            # Outcome model OOF on logged data
            prompts = [d["prompt"] for d in data]
            responses = [d["response"] for d in data]
            q_oof = self.outcome_model.predict(prompts, responses, S, fold_ids)

            # Fresh draws DM per prompt, per fold model (we average per prompt, using the same fold for the prompt)
            fresh = self._fresh_draws.get(policy)
            if fresh is None:
                return None

            # π̂_L OOF (fixed across oracle jackknife; OK because π̂_L uses logged labels, not oracle calibrator)
            if policy in self._piL_oof_cache and self._piL_oof_cache[policy].shape[
                0
            ] == len(data):
                pi_oof = np.asarray(self._piL_oof_cache[policy], dtype=float)
            else:
                pi_oof = self._fit_piL_oof(L, S, fold_ids)

            # Get m̂(S) for efficient label correction
            if policy in self._m_hat_oof_cache and self._m_hat_oof_cache[policy].shape[
                0
            ] == len(data):
                m_hat = np.asarray(self._m_hat_oof_cache[policy], dtype=float)
            else:
                # This should not happen in normal operation - log warning and fallback
                logger.warning(
                    f"TR-CPO jackknife: m̂(S) not found for policy '{policy}'. "
                    f"Using raw weights W (less efficient). Check if fit() completed properly."
                )
                m_hat = w

            # Compute one estimate per delete-one fold model
            jack: List[float] = []
            for fold_id, f_model in fold_models.items():
                # Logged R_loo
                R_loo = np.clip(f_model.predict(S), 0.0, 1.0)

                # Fresh g_loo per prompt
                g_loo = []
                for i, pid in enumerate(pids):
                    scores_i = fresh.get_scores_for_prompt_id(pid)
                    if len(scores_i) == 0:
                        g_loo.append(0.0)
                        continue
                    f_pred = np.clip(
                        f_model.predict(np.asarray(scores_i, dtype=float)), 0.0, 1.0
                    )
                    g_loo.append(float(np.mean(f_pred)))
                g_loo_array = np.asarray(g_loo, dtype=float)

                # TR estimate with R_loo (use m̂ for efficiency)
                pi_clipped = np.clip(pi_oof, self.min_pi, self.max_pi)
                dm = float(np.mean(g_loo_array))
                dr = float(
                    np.mean(w * (R - q_oof))
                )  # keep R (logged) for the DR term (small diff)
                tr = float(np.mean((L / pi_clipped) * m_hat * (Y - R_loo)))
                jack.append(dm + dr + tr)

            return np.asarray(jack, dtype=float) if jack else None

        except Exception as e:
            logger.debug(f"get_oracle_jackknife(TR) failed for {policy}: {e}")
            return None
