# cje/estimators/mrdr_tmle.py
"""
MRDR+TMLE estimator: policy-specific weighted isotonic outcome models (MRDR)
with a TMLE targeting step applied to the logged component.

Design goals:
- Reuse MRDR's policy-specific, cross-fitted, weighted isotonic outcome models
  (one per target policy) to get g_0.
- Apply a TMLE targeting step with clever covariate W (the Hájek / mean‑one
  importance weights) on the logged term only:
      logit(g*) = logit(g_0) + ε * W                   [logit link]
      g*        = g_0      + ε * W                     [identity link]
- Keep the DM term based on fresh draws unchanged (same fold as its logged prompt).
- Return standard DR influence‑function SEs and DR diagnostics.

Requirements/assumptions:
- Importance weights returned by the internal IPS estimator are mean‑one
  (CalibratedIPS is the default in DREstimator, which ensures this).
- Cross‑fitting folds are available via the calibration pipeline and reused
  for outcome models and fresh draws (strict checks to avoid in‑fold leakage).
"""

from __future__ import annotations
from typing import Dict, Optional, Any, List, Tuple
import logging
import numpy as np

from .mrdr import MRDREstimator  # for fit() and policy-specific weighted models
from ..data.precomputed_sampler import PrecomputedSampler
from ..data.models import EstimationResult
from ..data.fresh_draws import FreshDrawDataset

logger = logging.getLogger(__name__)

_EPS = 1e-7  # numerical guard for logits/probabilities


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p) - np.log(1.0 - p)


class MRDRTMLEEstimator(MRDREstimator):
    """MRDR+TMLE: TMLE targeting on top of MRDR outcome models.

    Inherits MRDR's fitting (policy-specific weighted isotonic outcome models)
    and overrides estimate() to add the TMLE targeting step.

    Args:
        sampler: PrecomputedSampler with logged data
        n_folds: Cross-fitting folds (default 5)
        omega_mode: MRDR regression weighting for outcome models:
            - "w"     -> |W|        (default; most stable)
            - "w2"    -> W^2
            - "snips" -> (W-1)^2
        min_sample_weight: Floor on MRDR regression weights to avoid 0s
        use_calibrated_weights: Use CalibratedIPS for IPS (default True)
        use_policy_specific_models: If False, falls back to simplified MRDR
                                    (shared outcome model) then applies TMLE.
        link: 'logit' (default, for rewards in [0,1]) or 'identity'
        max_iter: Max Newton steps for logistic fluctuation
        tol: Convergence tolerance on normalized score (|score| / sqrt(Fisher))
        calibrator: Optional JudgeCalibrator passed to DREstimator/CalibratedIPS
        **kwargs: Forwarded to DREstimator (e.g., IPS config)
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        n_folds: int = 5,
        omega_mode: str = "w",
        min_sample_weight: float = 1e-8,
        use_calibrated_weights: bool = True,
        use_policy_specific_models: bool = True,
        link: str = "logit",
        max_iter: int = 50,
        tol: float = 1e-8,
        calibrator: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(
            sampler=sampler,
            n_folds=n_folds,
            omega_mode=omega_mode,
            min_sample_weight=min_sample_weight,
            use_calibrated_weights=use_calibrated_weights,
            use_policy_specific_models=use_policy_specific_models,
            calibrator=calibrator,
            **kwargs,
        )

        if link not in {"logit", "identity"}:
            raise ValueError(f"link must be one of ['logit','identity'], got {link}")

        self.link = link
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        # Per-policy targeting diagnostics
        self._tmle_info: Dict[str, Dict[str, Any]] = {}

        # Ensure DR storage members exist (kept consistent with other estimators)
        if not hasattr(self, "_dm_component"):
            self._dm_component: Dict[str, np.ndarray] = {}
        if not hasattr(self, "_ips_correction"):
            self._ips_correction: Dict[str, np.ndarray] = {}
        if not hasattr(self, "_fresh_rewards"):
            self._fresh_rewards: Dict[str, np.ndarray] = {}
        if not hasattr(self, "_outcome_predictions"):
            self._outcome_predictions: Dict[str, np.ndarray] = {}

    # ---------- Targeting solvers (copied and minimally adapted from TMLE) ----------

    def _solve_logistic_fluctuation(
        self, q0_logged: np.ndarray, rewards: np.ndarray, weights: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """Solve for ε in logit(Q*) = logit(Q0) + ε·W using weighted logistic MLE.

        The clever covariate is the mean‑one weight W. Uses a scale‑aware
        convergence criterion: |score|/sqrt(Fisher) ≤ tol.
        """
        q0 = np.clip(q0_logged, _EPS, 1.0 - _EPS)
        eta0 = _logit(q0)

        eps = 0.0
        converged = False
        score_val = 0.0
        fisher_val = 0.0
        normalized_score = 0.0

        if float(np.sum(weights)) <= 0:
            return 0.0, dict(
                epsilon=0.0,
                converged=True,
                iters=0,
                score=0.0,
                fisher=0.0,
                normalized_score=0.0,
            )

        for t in range(self.max_iter):
            mu = _expit(eta0 + eps * weights)
            # score = ∑ W (Y - μ)
            score = float(np.sum(weights * (rewards - mu)))
            # Fisher(ε) = ∑ W² μ(1-μ)
            fisher = float(np.sum((weights**2) * mu * (1.0 - mu)))

            score_val = score
            fisher_val = fisher

            if fisher > 1e-12:
                normalized_score = abs(score) / np.sqrt(fisher)
                if normalized_score <= self.tol:
                    converged = True
                    break
            else:
                # Near-singular Fisher: accept if raw score small
                if abs(score) <= self.tol:
                    converged = True
                    normalized_score = abs(score)
                    break
                logger.warning(
                    "MRDR-TMLE logistic fluctuation: near-singular Fisher; stopping early."
                )
                break

            # Newton step; cap size to avoid huge updates
            step = score / fisher
            if abs(step) > 5.0:
                step = np.sign(step) * 5.0
            eps += step

        if not converged:
            logger.info(
                f"MRDR-TMLE logistic fluctuation did not fully converge: "
                f"iters={self.max_iter}, normalized_score={normalized_score:.3e}, "
                f"|score|={abs(score_val):.3e}, fisher={fisher_val:.3e}"
            )

        return float(eps), dict(
            epsilon=float(eps),
            converged=bool(converged),
            iters=int(t + 1),
            score=score_val,
            fisher=fisher_val,
            normalized_score=normalized_score,
        )

    def _solve_identity_fluctuation(
        self, q0_logged: np.ndarray, rewards: np.ndarray, weights: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """Solve for ε in Q* = Q0 + ε·W via the EIF score equation.

        Clever covariate is W; solution ε = ∑ W (Y - Q0) / ∑ W².
        """
        num = float(np.sum(weights * (rewards - q0_logged)))
        den = float(np.sum(weights**2))
        if den <= 1e-12:
            return 0.0, dict(
                epsilon=0.0, converged=True, iters=1, score=num, fisher=den
            )
        eps = num / den
        return float(eps), dict(
            epsilon=float(eps), converged=True, iters=1, score=num, fisher=den
        )

    # --------------------------------- Estimation ----------------------------------

    def estimate(self) -> EstimationResult:
        """Compute MRDR+TMLE estimates for all target policies.

        Steps per policy:
          1) g_logged0 := cross‑fitted, policy‑specific weighted isotonic predictions
          2) g_fresh0  := cross‑fitted predictions on fresh draws (per prompt mean)
          3) Solve TMLE fluctuation for ε with clever covariate W
          4) Update logged predictions: g_logged* (fresh DM term remains g_fresh0)
          5) ψ = mean(g_fresh0) + mean(W * (R - g_logged*))
          6) SE via empirical IF
        """
        # Ensure MRDR fit ran (policy-specific models + fold mapping)
        self._validate_fitted()

        estimates: List[float] = []
        standard_errors: List[float] = []
        n_samples_used: Dict[str, int] = {}
        self._tmle_info = {}

        for policy in self.sampler.target_policies:
            # Require policy-specific model unless user explicitly opted out
            if self.use_policy_specific_models and policy not in self._policy_models:
                logger.warning(f"No outcome model for policy '{policy}'. Using NaN.")
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            # Require fresh draws (TMLE/DR)
            if policy not in self._fresh_draws:
                raise ValueError(
                    f"No fresh draws registered for '{policy}'. "
                    f"Call add_fresh_draws(policy, fresh_draws) before estimate()."
                )
            fresh_dataset: FreshDrawDataset = self._fresh_draws[policy]

            # Logged data and weights
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                logger.warning(f"No valid data for policy '{policy}'. Skipping.")
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            weights = self.get_weights(policy)
            if weights is None or len(weights) != len(data):
                raise ValueError(
                    f"Weight/data mismatch for policy '{policy}': "
                    f"weights={None if weights is None else len(weights)}, data={len(data)}"
                )

            # Extract arrays
            rewards = np.array([d["reward"] for d in data], dtype=float)
            scores = np.array([d.get("judge_score") for d in data], dtype=float)
            prompt_ids = [str(d.get("prompt_id")) for d in data]
            prompts = [d["prompt"] for d in data]
            responses = [d["response"] for d in data]

            # Strict fold mapping (avoid in-fold leakage)
            if not self._promptid_to_fold:
                raise ValueError(
                    "MRDR+TMLE requires fold assignments from calibration. "
                    "Ensure calibration was run with enable_cross_fit=True."
                )
            unknown_pids = [
                pid for pid in prompt_ids if pid not in self._promptid_to_fold
            ]
            if unknown_pids:
                raise ValueError(
                    f"Missing fold assignments for {len(unknown_pids)} samples in policy '{policy}'. "
                    f"Example prompt_ids: {unknown_pids[:3]}."
                )
            fold_ids = np.array(
                [self._promptid_to_fold[pid] for pid in prompt_ids], dtype=int
            )

            # Initial predictions on logged data (policy-specific weighted model if enabled)
            if self.use_policy_specific_models:
                outcome_model = self._policy_models[policy]
                g_logged0 = outcome_model.predict(prompts, responses, scores, fold_ids)
            else:
                # Fallback to the shared model from the MRDR base (simplified pathway)
                g_logged0 = self.outcome_model.predict(
                    prompts, responses, scores, fold_ids
                )

            # Predictions on fresh draws (per prompt, same fold as logged)
            g_fresh0_all = []
            for i, pid in enumerate(prompt_ids):
                fresh_scores = fresh_dataset.get_scores_for_prompt_id(pid)
                fresh_prompts = [prompts[i]] * len(fresh_scores)
                fresh_responses = [""] * len(fresh_scores)
                fresh_fold_ids = np.full(len(fresh_scores), fold_ids[i])

                if self.use_policy_specific_models:
                    g_fresh_prompt = outcome_model.predict(
                        fresh_prompts, fresh_responses, fresh_scores, fresh_fold_ids
                    )
                else:
                    g_fresh_prompt = self.outcome_model.predict(
                        fresh_prompts, fresh_responses, fresh_scores, fresh_fold_ids
                    )

                g_fresh0_all.append(g_fresh_prompt.mean())

            g_fresh0 = np.array(g_fresh0_all, dtype=float)

            # Sanity check: weights should be ~mean-1
            w_mean = float(weights.mean())
            if not (0.99 <= w_mean <= 1.01):
                logger.warning(
                    f"Weights for policy '{policy}' deviate from expected mean≈1.0: "
                    f"mean={w_mean:.3f}, std={weights.std():.3f}, "
                    f"min={weights.min():.3e}, max={weights.max():.3e}"
                )

            # --- TMLE targeting step on logged term ---
            if self.link == "logit":
                eps, info = self._solve_logistic_fluctuation(
                    g_logged0, rewards, weights
                )
                g_logged_star = _expit(_logit(g_logged0) + eps * weights)
            else:  # identity
                eps, info = self._solve_identity_fluctuation(
                    g_logged0, rewards, weights
                )
                g_logged_star = np.clip(g_logged0 + eps * weights, 0.0, 1.0)

            # DM term stays as g_fresh0 (do NOT shift fresh draws)
            dm_term = float(g_fresh0.mean())
            ips_corr = float(np.mean(weights * (rewards - g_logged_star)))
            psi = dm_term + ips_corr

            # Influence functions and SE
            if_contrib = g_fresh0 + weights * (rewards - g_logged_star) - psi
            se = (
                float(np.std(if_contrib, ddof=1) / np.sqrt(len(if_contrib)))
                if len(if_contrib) > 1
                else 0.0
            )

            # Store per-policy components for diagnostics
            # For TMLE, we want to store the actual contributions to the estimate
            # DM component is the fresh draws predictions, IPS is the weighted correction
            self._dm_component[policy] = g_fresh0  # Fresh outcome predictions
            self._ips_correction[policy] = weights * (
                rewards - g_logged_star
            )  # IPS correction term
            self._fresh_rewards[policy] = (
                rewards  # logged rewards; name kept for compatibility
            )
            # For honest R² diagnostics, store ORIGINAL (pre‑targeting) predictions:
            self._outcome_predictions[policy] = g_logged0
            self._influence_functions[policy] = if_contrib

            # Targeting diagnostics
            info.update(
                dict(
                    link=self.link,
                    epsilon=float(info.get("epsilon", 0.0)),
                    dm=float(dm_term),
                    ips_correction=float(ips_corr),
                    n=int(len(rewards)),
                    mean_weight=w_mean,
                )
            )
            self._tmle_info[policy] = info

            logger.info(
                f"MRDR-TMLE[{policy}]: {psi:.4f} ± {se:.4f} "
                f"(ε={info.get('epsilon', 0.0):+.4f}, DM={dm_term:.4f}, IPS_corr={ips_corr:.4f})"
            )

            estimates.append(psi)
            standard_errors.append(se)
            n_samples_used[policy] = len(rewards)

        # Build per-policy DR diagnostics using the stored components
        dr_diagnostics_per_policy: Dict[str, Dict[str, Any]] = {}
        for idx, policy in enumerate(self.sampler.target_policies):
            if policy not in self._dm_component or np.isnan(estimates[idx]):
                continue
            # Leverage DR base helper for consistent metrics
            policy_diag = self._compute_policy_diagnostics(policy, estimates[idx])
            # Store the diagnostics for this policy
            dr_diagnostics_per_policy[policy] = policy_diag

        # IPS diagnostics from internal IPS estimator, if available
        ips_diag = (
            self.ips_estimator.get_diagnostics()
            if hasattr(self.ips_estimator, "get_diagnostics")
            else None
        )

        # Compose DRDiagnostics with the base helper
        diagnostics = self._build_dr_diagnostics(
            estimates=estimates,
            standard_errors=standard_errors,
            n_samples_used=n_samples_used,
            dr_diagnostics_per_policy=dr_diagnostics_per_policy,
            ips_diagnostics=ips_diag,
        )

        # Metadata (no influence functions here; they are stored top-level)
        metadata = {
            "method": "mrdr_tmle",
            "link": self.link,
            "targeting": self._tmle_info,
            "omega_mode": self.omega_mode,
            "min_sample_weight": self.min_sample_weight,
            "use_policy_specific_models": self.use_policy_specific_models,
            "n_policy_models": len(getattr(self, "_policy_models", {})),
            "cross_fitted": True,
            "n_folds": self.n_folds,
            "fresh_draws_policies": list(self._fresh_draws.keys()),
        }

        return EstimationResult(
            estimates=np.array(estimates, dtype=float),
            standard_errors=np.array(standard_errors, dtype=float),
            n_samples_used=n_samples_used,
            method="mrdr_tmle",
            influence_functions=self._influence_functions,
            diagnostics=diagnostics,
            metadata=metadata,
        )
