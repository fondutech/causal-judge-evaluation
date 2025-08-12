# cje/core/tmle.py
"""
TMLE estimator for policy evaluation with cross-fitted monotone outcome models.

Overview
--------
We estimate V(π') via a targeted minimum loss estimator (TMLE):

  ψ_TMLE = mean_i[ Q*_fresh,i ] + mean_i[ W_i * (R_i - Q*_logged,i) ]

where:
  • Q0(s) is an initial outcome model fit on logged data (monotone in judge score s)
  • The *targeting step* updates Q0 along a 1-d fluctuation submodel to Q* so that
      the empirical EIF estimating equation is solved:
        sum_i W_i * (R_i - Q*_logged,i) ≈ 0
  • For rewards in [0,1] (the default here), we use a *logistic fluctuation*:
        logit Q*(·) = logit Q0(·) + ε
    with ε chosen by (weighted) MLE using clever covariate H=W and offset logit Q0.

Design
------
- We reuse the calibrated, mean-one IPS weights (Hájek/SNIPS) from CalibratedIPS.
- Initial Q0 is learned with *cross-fitted* isotonic regression g(s)=E[R|S=s]
  (unweighted by default; policy-agnostic). We train K fold-specific models and
  always predict out-of-fold on logged data to preserve orthogonality.
- Fresh draws for each prompt/policy are evaluated with the model of the same fold
  as the logged prompt (same fold-id), then averaged per-prompt to form Q0_fresh.
- A single scalar ε per policy is fit via a 1-d Newton solver on the weighted
  logistic likelihood with offset logit(Q0_logged) and weights W.
- We then apply the same fluctuation to both logged and fresh predictions to get Q*.

Requirements
------------
- Dataset must have calibrated rewards (e.g., from JudgeCalibrator).
- For best practice, call judge calibration with enable_cross_fit=True so each record
  has metadata["cv_fold"]. If absent, we assign folds deterministically.
- Fresh draws must be registered via add_fresh_draws(policy, FreshDrawDataset).
"""

from __future__ import annotations
from typing import Dict, Optional, Any, List, Tuple
import logging
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .calibrated_ips import CalibratedIPS
from ..data.precomputed_sampler import PrecomputedSampler
from ..data.models import EstimationResult
from ..data.fresh_draws import FreshDrawDataset
from ..utils.fresh_draws import validate_fresh_draws

logger = logging.getLogger(__name__)


_EPS = 1e-7  # numerical guard for logits/probabilities


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p) - np.log(1.0 - p)


class TMLEEstimator(CalibratedIPS):
    """TMLE with cross-fitted isotonic outcome models.

    Args:
        sampler: PrecomputedSampler with calibrated rewards
        n_folds: Number of cross-fitting folds (default 5)
        link: 'logit' (default, for rewards in [0,1]) or 'identity'
        max_iter: Max Newton steps for logistic targeting
        tol: Convergence tolerance on the (weighted) score
        **kwargs: Passed through to CalibratedIPS (e.g., variance controls)

    Usage:
        # 1) Calibrate dataset with enable_cross_fit=True to populate cv_fold
        calibrated_ds, cal_res = calibrate_dataset(dataset, enable_cross_fit=True, n_folds=5)

        # 2) Build sampler & estimator
        sampler = PrecomputedSampler(calibrated_ds)
        tmle = TMLEEstimator(sampler, n_folds=5, link="logit")

        # 3) Register fresh draws for each evaluation policy
        tmle.add_fresh_draws("policy_a", fresh_draws_a)
        tmle.add_fresh_draws("policy_b", fresh_draws_b)

        # 4) Fit & estimate
        result = tmle.fit_and_estimate()
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        n_folds: int = 5,
        link: str = "logit",
        max_iter: int = 50,
        tol: float = 1e-8,
        **kwargs: Any,
    ):
        super().__init__(sampler, **kwargs)

        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        if link not in {"logit", "identity"}:
            raise ValueError(f"link must be one of ['logit','identity'], got {link}")

        self.n_folds = n_folds
        self.link = link
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        # Outcome models: policy-agnostic (same fits reused for all policies)
        # We still store per-fold models to ensure cross-fitted predictions.
        self._fold_models: Dict[int, IsotonicRegression] = {}

        # Fresh draws per policy
        self._fresh_draws: Dict[str, FreshDrawDataset] = {}

        # prompt_id -> fold mapping
        self._promptid_to_fold: Dict[str, int] = {}

        # Cached arrays shared across policies for initial model fitting
        self._logged_scores: Optional[np.ndarray] = None
        self._logged_rewards: Optional[np.ndarray] = None
        self._logged_prompt_ids: Optional[List[str]] = None
        self._logged_fold_ids: Optional[np.ndarray] = None

        # Per-policy epsilon/diagnostics
        self._tmle_info: Dict[str, Dict[str, Any]] = {}

    # -------------------------- Public API --------------------------

    def add_fresh_draws(self, policy: str, fresh_draws: FreshDrawDataset) -> None:
        """Register fresh draws for an evaluation policy."""
        validate_fresh_draws(fresh_draws, self.sampler.dataset, policy)
        self._fresh_draws[policy] = fresh_draws
        logger.info(
            f"Added fresh draws for policy '{policy}': "
            f"{len(fresh_draws.samples)} samples, draws/prompt={fresh_draws.draws_per_prompt}"
        )

    def fit(self) -> None:
        """Calibrate IPS weights and fit the initial cross-fitted isotonic outcome models."""
        # 1) Calibrate weights via CalibratedIPS
        super().fit()

        # 2) Build prompt_id → fold mapping (prefer cv_fold written by calibration)
        self._promptid_to_fold = self._build_prompt_fold_map(self.n_folds)

        # 3) Prepare logged arrays (scores, rewards, prompt_ids, folds) once
        scores, rewards, prompt_ids, fold_ids = self._collect_logged_arrays()
        self._logged_scores = scores
        self._logged_rewards = rewards
        self._logged_prompt_ids = prompt_ids
        self._logged_fold_ids = fold_ids

        # 4) Fit cross-fitted isotonic models (unweighted) on (S, R)
        self._fit_cross_fitted_isotonic(scores, rewards, fold_ids)

        self._fitted = True
        logger.info(
            f"TMLE fitted initial outcome models with {len(self._fold_models)} folds."
        )

    def estimate(self) -> EstimationResult:
        """Compute TMLE estimates for all target policies."""
        self._validate_fitted()

        estimates: List[float] = []
        standard_errors: List[float] = []
        n_samples_used: Dict[str, int] = {}
        self._tmle_info = {}

        # Must have prepared arrays
        assert self._logged_scores is not None
        assert self._logged_rewards is not None
        assert self._logged_prompt_ids is not None
        assert self._logged_fold_ids is not None

        # Precompute initial out-of-fold predictions on logged data
        g_logged0 = self._predict_logged_oof(self._logged_scores, self._logged_fold_ids)

        for policy in self.sampler.target_policies:
            # Ensure fresh draws available
            if policy not in self._fresh_draws:
                raise ValueError(
                    f"No fresh draws registered for '{policy}'. "
                    f"Call add_fresh_draws(policy, fresh_draws) before estimate()."
                )

            # Align to estimator's valid data for this policy
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                logger.warning(f"No valid data for policy '{policy}'. Skipping.")
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue

            # Calibrated (mean-one) weights for this policy
            weights = self.get_weights(policy)
            if weights is None or len(weights) != len(data):
                raise ValueError(
                    f"Weight/data mismatch for policy '{policy}': "
                    f"weights={None if weights is None else len(weights)}, data={len(data)}"
                )

            # Extract arrays aligned with 'data'
            rewards = np.array([d["reward"] for d in data], dtype=float)
            scores = np.array([d.get("judge_score") for d in data], dtype=float)
            prompt_ids = [str(d.get("prompt_id")) for d in data]

            # Fold ids by prompt
            fold_ids = np.array(
                [self._promptid_to_fold.get(pid, 0) for pid in prompt_ids], dtype=int
            )

            # 1) Initial cross-fitted predictions on logged data (policy-agnostic Q0)
            g_logged0_subset = self._predict_logged_subset_oof(
                scores=scores, prompt_ids=prompt_ids, fold_ids=fold_ids
            )

            # 2) Initial predictions on fresh draws per prompt (same fold)
            g_fresh0 = self._predict_fresh_means(policy, prompt_ids, fold_ids)

            # 3) Targeting step: solve for ε and update Q0 → Q*
            if self.link == "logit":
                eps, info = self._solve_logistic_fluctuation(
                    g_logged0_subset, rewards, weights
                )
                g_logged_star = _expit(_logit(g_logged0_subset) + eps)
                g_fresh_star = _expit(_logit(g_fresh0) + eps)
            else:  # identity link (bounded clip)
                eps, info = self._solve_identity_fluctuation(
                    g_logged0_subset, rewards, weights
                )
                g_logged_star = np.clip(g_logged0_subset + eps, 0.0, 1.0)
                g_fresh_star = np.clip(g_fresh0 + eps, 0.0, 1.0)

            # 4) TMLE estimate = DM + IPS correction
            dm_term = float(g_fresh_star.mean())
            ips_corr = float(np.mean(weights * (rewards - g_logged_star)))
            psi = dm_term + ips_corr

            # 5) Standard error via empirical IF
            if_contrib = g_fresh_star + weights * (rewards - g_logged_star) - psi
            se = (
                float(np.std(if_contrib, ddof=1) / np.sqrt(len(if_contrib)))
                if len(if_contrib) > 1
                else 0.0
            )

            estimates.append(psi)
            standard_errors.append(se)
            n_samples_used[policy] = len(rewards)

            # Keep diagnostics
            info.update(
                dict(
                    link=self.link,
                    epsilon=float(info.get("epsilon", 0.0)),
                    dm=float(dm_term),
                    ips_correction=float(ips_corr),
                    n=len(rewards),
                    mean_weight=float(weights.mean()),
                )
            )
            self._tmle_info[policy] = info

            logger.info(
                f"TMLE[{policy}]: {psi:.4f} ± {se:.4f} "
                f"(ε={info.get('epsilon', 0.0):+.4f}, DM={dm_term:.4f}, IPS_corr={ips_corr:.4f})"
            )

        # Merge IPS diagnostics with TMLE metadata
        meta = {
            "method": "tmle",
            "cross_fitted": True,
            "n_folds": self.n_folds,
            "link": self.link,
            "targeting": self._tmle_info,
            "fresh_draws_policies": list(self._fresh_draws.keys()),
            "ips_diagnostics": getattr(self, "_diagnostics", {}),
        }

        return EstimationResult(
            estimates=np.array(estimates, dtype=float),
            standard_errors=np.array(standard_errors, dtype=float),
            n_samples_used=n_samples_used,
            method="tmle",
            metadata=meta,
        )

    # -------------------------- Fitting helpers --------------------------

    def _collect_logged_arrays(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Collect (S, R, prompt_id, fold_id) for all records usable by any policy."""
        # Use union of valid indices across all policies (as in DR base)
        valid_indices = set()
        for policy in self.sampler.target_policies:
            idx = self.sampler._get_valid_indices(policy)
            valid_indices.update(idx)
        valid_list = sorted(valid_indices)

        scores: List[float] = []
        rewards: List[float] = []
        prompt_ids: List[str] = []
        fold_ids: List[int] = []

        for i in valid_list:
            s = self.sampler.dataset.samples[i]
            if "judge_score" not in s.metadata:
                raise ValueError(
                    "All samples must have 'judge_score' in metadata for TMLE."
                )
            if s.reward is None:
                raise ValueError("All samples must have calibrated rewards for TMLE.")

            scores.append(float(s.metadata["judge_score"]))
            rewards.append(float(s.reward))
            pid = str(s.prompt_id)
            prompt_ids.append(pid)
            fold_ids.append(int(self._promptid_to_fold.get(pid, 0)))

        S = np.asarray(scores, dtype=float)
        R = np.asarray(rewards, dtype=float)
        F = np.asarray(fold_ids, dtype=int)

        if np.isnan(S).any():
            raise ValueError("Found NaNs in judge scores for TMLE fit.")
        return S, R, prompt_ids, F

    def _fit_cross_fitted_isotonic(
        self, scores: np.ndarray, rewards: np.ndarray, fold_ids: np.ndarray
    ) -> None:
        """Fit per-fold isotonic models on other folds (cross-fitting)."""
        unique_folds = sorted(np.unique(fold_ids))
        if len(unique_folds) < 2:
            logger.warning(
                f"Only {len(unique_folds)} unique folds present; "
                "orthogonality may be weak. Prefer enable_cross_fit=True in calibration."
            )

        self._fold_models = {}
        for fold in unique_folds:
            train_mask = fold_ids != fold
            if not np.any(train_mask):
                raise ValueError(f"No training samples for fold {fold} in TMLE fit.")

            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(scores[train_mask], rewards[train_mask])
            self._fold_models[fold] = model

    def _predict_logged_oof(
        self, scores: np.ndarray, fold_ids: np.ndarray
    ) -> np.ndarray:
        """Out-of-fold predictions for the entire logged set."""
        preds = np.zeros_like(scores, dtype=float)
        for fold, model in self._fold_models.items():
            mask = fold_ids == fold
            if np.any(mask):
                preds[mask] = model.predict(scores[mask])
        # Numerical guards for logistic link
        if self.link == "logit":
            preds = np.clip(preds, _EPS, 1.0 - _EPS)
        else:
            preds = np.clip(preds, 0.0, 1.0)
        return preds

    def _predict_logged_subset_oof(
        self, scores: np.ndarray, prompt_ids: List[str], fold_ids: np.ndarray
    ) -> np.ndarray:
        """OOF predictions for a policy-specific subset aligned to `scores`/`prompt_ids`."""
        # Predict using the fold model for each record's fold
        preds = np.zeros_like(scores, dtype=float)
        for f in np.unique(fold_ids):
            model = self._fold_models.get(int(f))
            if model is None:
                raise ValueError(f"Missing fold model {f} in TMLE.")
            m = fold_ids == f
            if np.any(m):
                preds[m] = model.predict(scores[m])
        # Guard
        if self.link == "logit":
            preds = np.clip(preds, _EPS, 1.0 - _EPS)
        else:
            preds = np.clip(preds, 0.0, 1.0)
        return preds

    def _predict_fresh_means(
        self, policy: str, prompt_ids: List[str], fold_ids: np.ndarray
    ) -> np.ndarray:
        """Predict on fresh draws per prompt (same fold), return per-prompt means."""
        fresh = self._fresh_draws[policy]
        means: List[float] = []

        for i, pid in enumerate(prompt_ids):
            fold = int(fold_ids[i])
            model = self._fold_models.get(fold)
            if model is None:
                raise ValueError(f"Missing fold model {fold} for fresh predictions.")

            s_prime = fresh.get_scores_for_prompt_id(pid)
            g_draws = model.predict(np.asarray(s_prime, dtype=float))
            if self.link == "logit":
                g_draws = np.clip(g_draws, _EPS, 1.0 - _EPS)
            else:
                g_draws = np.clip(g_draws, 0.0, 1.0)
            means.append(float(np.mean(g_draws)))

        return np.asarray(means, dtype=float)

    # -------------------------- Targeting (fluctuation) --------------------------

    def _solve_logistic_fluctuation(
        self, q0_logged: np.ndarray, rewards: np.ndarray, weights: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """Solve for ε in logit(Q*) = logit(Q0) + ε using weighted logistic MLE.

        Score(ε) = Σ w_i [y_i - μ_i(ε)], with μ_i(ε)=expit(logit(q0_i)+ε).
        Newton step: ε <- ε + Score / Fisher, where Fisher = Σ w_i μ_i(ε)(1-μ_i(ε)).
        """
        # Guards
        q0 = np.clip(q0_logged, _EPS, 1.0 - _EPS)
        eta0 = _logit(q0)

        eps = 0.0
        converged = False
        score_val = None
        fisher_val = None

        # If all weights ~0, skip targeting
        if float(np.sum(weights)) <= 0:
            return 0.0, dict(
                epsilon=0.0, converged=True, iters=0, score=0.0, fisher=0.0
            )

        for t in range(self.max_iter):
            mu = _expit(eta0 + eps)
            # weighted score (sum w*(y-mu))
            score = float(np.sum(weights * (rewards - mu)))
            fisher = float(np.sum(weights * mu * (1.0 - mu)))

            score_val = score
            fisher_val = fisher

            if abs(score) <= self.tol * len(q0):
                converged = True
                break
            if fisher <= 1e-12:
                logger.warning(
                    "TMLE logistic fluctuation: near-singular Fisher; stopping early."
                )
                break

            # Newton update; cap step to avoid giant jumps
            step = score / fisher
            if abs(step) > 5.0:
                step = np.sign(step) * 5.0
            eps += step

        if not converged:
            logger.info(
                f"TMLE logistic fluctuation did not fully converge: "
                f"iters={self.max_iter}, |score|={abs(score_val):.3e}, fisher={fisher_val:.3e}"
            )

        return float(eps), dict(
            epsilon=float(eps),
            converged=bool(converged),
            iters=int(t + 1),
            score=float(score_val if score_val is not None else 0.0),
            fisher=float(fisher_val if fisher_val is not None else 0.0),
        )

    def _solve_identity_fluctuation(
        self, q0_logged: np.ndarray, rewards: np.ndarray, weights: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """Solve for ε in Q* = Q0 + ε via the EIF score equation: sum w*(y - (q0+ε)) = 0."""
        denom = float(np.sum(weights))
        if denom <= 0:
            return 0.0, dict(
                epsilon=0.0, converged=True, iters=1, score=0.0, fisher=denom
            )
        resid = float(np.sum(weights * (rewards - q0_logged)))
        eps = resid / denom
        return float(eps), dict(
            epsilon=float(eps),
            converged=True,
            iters=1,
            score=float(resid),
            fisher=denom,
        )

    # -------------------------- Misc helpers --------------------------

    def _build_prompt_fold_map(self, n_folds: int) -> Dict[str, int]:
        """Create prompt_id → fold mapping (prefer metadata['cv_fold'])."""
        mapping: Dict[str, int] = {}
        rng = np.random.RandomState(42)

        for s in self.sampler.dataset.samples:
            pid = s.prompt_id
            if pid is None:
                # TMLE requires prompt_id to align logged ↔ fresh; we'll fail later if missing in data paths
                continue
            if "cv_fold" in s.metadata:
                fold = int(s.metadata["cv_fold"])
            else:
                # Fallback: deterministic assignment if no cv_fold present
                fold = int(rng.randint(0, n_folds))
            mapping[str(pid)] = fold

        if not mapping:
            logger.warning(
                "No cv_fold found in dataset metadata; using random fold assignment. "
                "Prefer calibrate_dataset(..., enable_cross_fit=True) for strict orthogonality."
            )
        return mapping
