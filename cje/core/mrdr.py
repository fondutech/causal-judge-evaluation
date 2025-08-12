# cje/core/mrdr.py
"""
MRDR estimator: policy-specific, cross-fitted weighted isotonic outcome models.

Idea (MRDR):
  - Choose g_p to minimize the variance of DR by training the outcome model with
    importance-aware weights ω(W). In practice, we do a weighted regression of
    R on S with ω derived from the (calibrated, mean-one) importance weights W
    for the evaluation policy p, and we cross-fit to preserve orthogonality.
  - This module implements MRDR with isotonic regression (monotone in S).

Key choices:
  • Outcome model per policy p: g_p(s) = isotonic(s; sample_weight = ω_p)
  • Cross-fitting: K-fold; logged predictions use out-of-fold models f_p^{(-k)}
  • Fresh draws: use the same fold as their corresponding logged prompt
  • Default ω mode: "snips"  ->  ω = (W - 1)^2  (good with mean-one/SNIPS weights)
    Other options: "w2" -> ω = W^2, "w" -> ω = |W|

Requirements:
  - Dataset calibrated with enable_cross_fit=True (so each sample has cv_fold)
  - Rewards present (calibrated via JudgeCalibrator)
  - Fresh draws added per target policy via add_fresh_draws()

"""

from __future__ import annotations
from typing import Dict, Optional, Any, List
import logging
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .calibrated_ips import CalibratedIPS
from ..data.precomputed_sampler import PrecomputedSampler
from ..data.models import EstimationResult
from ..data.fresh_draws import FreshDrawDataset
from ..utils.fresh_draws import validate_fresh_draws

logger = logging.getLogger(__name__)


class MRDREstimator(CalibratedIPS):
    """MRDR with cross-fitted, policy-specific weighted isotonic outcome models.

    Args:
        sampler: PrecomputedSampler with calibrated rewards
        n_folds: Cross-fitting folds (default 5)
        omega_mode: Weighting for the MRDR regression. One of:
            - "snips": (W - 1)^2   [default; matches mean-one/SNIPS IF structure]
            - "w2":    W^2
            - "w":     |W|
        min_sample_weight: Floor applied to ω to avoid degenerate 0-weight fits
        **kwargs: Passed through to CalibratedIPS (weight calibration config)

    Usage:
        calibrated_ds, cal_res = calibrate_dataset(
            dataset, enable_cross_fit=True, n_folds=5
        )
        sampler = PrecomputedSampler(calibrated_ds)
        mrdr = MRDREstimator(sampler, n_folds=5, omega_mode="snips")
        mrdr.add_fresh_draws("policy_a", fresh_draws_a)
        ...
        result = mrdr.fit_and_estimate()
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        n_folds: int = 5,
        omega_mode: str = "snips",
        min_sample_weight: float = 1e-8,
        **kwargs: Any,
    ):
        super().__init__(sampler, **kwargs)
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        if omega_mode not in {"snips", "w2", "w"}:
            raise ValueError(
                f"omega_mode must be one of ['snips','w2','w'], got {omega_mode}"
            )

        self.n_folds = n_folds
        self.omega_mode = omega_mode
        self.min_sample_weight = float(min_sample_weight)

        # Per-policy outcome models: dict[policy][fold] -> IsotonicRegression
        self._models: Dict[str, Dict[int, IsotonicRegression]] = {}

        # Fresh draws per policy
        self._fresh_draws: Dict[str, FreshDrawDataset] = {}

        # prompt_id -> fold mapping (from dataset metadata cv_fold, when available)
        self._promptid_to_fold: Dict[str, int] = {}

        # Per-policy cached fold ids & training arrays (aligned to get_data_for_policy(policy))
        self._policy_arrays: Dict[str, Dict[str, Any]] = {}

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
        """Calibrate IPS weights and fit MRDR outcome models (per policy, cross-fitted)."""
        # 1) Calibrate weights via CalibratedIPS
        super().fit()

        # 2) Build prompt_id → fold map (prefer cv_fold written by calibration)
        self._promptid_to_fold = self._build_prompt_fold_map(self.n_folds)

        # 3) For each policy, prepare arrays and fit cross-fitted weighted isotonic models
        self._models = {}
        self._policy_arrays = {}

        for policy in self.sampler.target_policies:
            data = self.sampler.get_data_for_policy(policy)
            if not data:
                logger.warning(
                    f"No valid data for policy '{policy}'; skipping MRDR fit."
                )
                continue

            weights = self.get_weights(policy)
            if weights is None or len(weights) != len(data):
                raise ValueError(
                    f"Weight/data mismatch for policy '{policy}': "
                    f"weights={None if weights is None else len(weights)}, data={len(data)}"
                )

            # Extract arrays aligned with weights/data order
            rewards = np.array([d["reward"] for d in data], dtype=float)
            scores = np.array([d.get("judge_score") for d in data], dtype=float)
            prompt_ids: List[str] = [str(d.get("prompt_id")) for d in data]

            # Fold ids per record (from prompt_id mapping)
            fold_ids = np.array(
                [self._promptid_to_fold.get(pid, 0) for pid in prompt_ids], dtype=int
            )

            # MRDR sample weights ω(W)
            omega = self._omega_from_weights(weights, mode=self.omega_mode)
            # Guard against all-zeros / degeneracy
            omega = np.maximum(omega, self.min_sample_weight)

            # Save arrays for estimation stage
            self._policy_arrays[policy] = dict(
                rewards=rewards,
                scores=scores,
                weights=weights,
                omega=omega,
                fold_ids=fold_ids,
                prompt_ids=prompt_ids,
            )

            # Cross-fitted training
            models_for_policy: Dict[int, IsotonicRegression] = {}
            unique_folds = sorted(np.unique(fold_ids))
            if len(unique_folds) < self.n_folds:
                logger.info(
                    f"Policy '{policy}': only {len(unique_folds)} unique folds present; "
                    f"adjusting from requested {self.n_folds}."
                )

            for fold in unique_folds:
                train_mask = fold_ids != fold
                if not np.any(train_mask):
                    raise ValueError(
                        f"No training samples for fold {fold} in policy '{policy}'"
                    )

                model = IsotonicRegression(out_of_bounds="clip")
                model.fit(
                    X=scores[train_mask],
                    y=rewards[train_mask],
                    sample_weight=omega[train_mask],
                )
                models_for_policy[fold] = model

            self._models[policy] = models_for_policy
            logger.info(
                f"Fitted MRDR outcome models for '{policy}' with {len(models_for_policy)} folds."
            )

        self._fitted = True

    def estimate(self) -> EstimationResult:
        """Compute MRDR estimates for all target policies."""
        self._validate_fitted()

        estimates: List[float] = []
        standard_errors: List[float] = []
        n_samples_used: Dict[str, int] = {}

        for policy in self.sampler.target_policies:
            # Ensure models and fresh draws exist
            if policy not in self._models:
                logger.warning(f"No MRDR outcome models for '{policy}'. Skipping.")
                estimates.append(np.nan)
                standard_errors.append(np.nan)
                n_samples_used[policy] = 0
                continue
            if policy not in self._fresh_draws:
                raise ValueError(
                    f"No fresh draws registered for '{policy}'. "
                    f"Call add_fresh_draws(policy, fresh_draws) before estimate()."
                )

            arrs = self._policy_arrays[policy]
            rewards = arrs["rewards"]
            scores = arrs["scores"]
            weights = arrs["weights"]
            fold_ids = arrs["fold_ids"]
            prompt_ids = arrs["prompt_ids"]

            # 1) Out-of-fold predictions on logged data
            g_logged = np.zeros_like(rewards, dtype=float)
            for fold, model in self._models[policy].items():
                mask = fold_ids == fold
                if np.any(mask):
                    g_logged[mask] = model.predict(scores[mask])

            # 2) Predictions on fresh draws (use the same fold as the logged prompt)
            fresh = self._fresh_draws[policy]
            g_fresh_prompt_means: List[float] = []
            for i, pid in enumerate(prompt_ids):
                fold = fold_ids[i]
                model = self._models[policy].get(fold)
                if model is None:
                    raise ValueError(
                        f"Missing model for fold {fold} in policy '{policy}'. "
                        f"Available: {sorted(self._models[policy].keys())}"
                    )

                s_prime = fresh.get_scores_for_prompt_id(
                    pid
                )  # array of judge scores for draws
                g_draws = model.predict(np.asarray(s_prime, dtype=float))
                g_fresh_prompt_means.append(float(np.mean(g_draws)))

            g_fresh = np.array(g_fresh_prompt_means, dtype=float)

            # 3) MRDR estimate = DM + IPS correction
            dm_term = float(g_fresh.mean())
            ips_correction = float(np.mean(weights * (rewards - g_logged)))
            mrdr_est = dm_term + ips_correction

            # 4) Standard error via empirical IF
            if_contrib = g_fresh + weights * (rewards - g_logged) - mrdr_est
            se = (
                float(np.std(if_contrib, ddof=1) / np.sqrt(len(if_contrib)))
                if len(if_contrib) > 1
                else 0.0
            )

            estimates.append(mrdr_est)
            standard_errors.append(se)
            n_samples_used[policy] = len(rewards)

            logger.info(
                f"MRDR[{policy}]: {mrdr_est:.4f} ± {se:.4f} (DM={dm_term:.4f}, IPS_corr={ips_correction:.4f})"
            )

        # Merge IPS diagnostics with MRDR metadata
        mrdr_meta = {
            "method": "mrdr",
            "cross_fitted": True,
            "n_folds": self.n_folds,
            "omega_mode": self.omega_mode,
            "min_sample_weight": self.min_sample_weight,
            "fresh_draws_policies": list(self._fresh_draws.keys()),
        }
        all_meta = {
            **mrdr_meta,
            **{"ips_diagnostics": getattr(self, "_diagnostics", {})},
        }

        return EstimationResult(
            estimates=np.array(estimates, dtype=float),
            standard_errors=np.array(standard_errors, dtype=float),
            n_samples_used=n_samples_used,
            method="mrdr",
            metadata=all_meta,
        )

    # -------------------------- Helpers --------------------------

    def _omega_from_weights(self, w: np.ndarray, mode: str) -> np.ndarray:
        """Compute MRDR regression weights ω from mean-one IPS weights W."""
        if mode == "snips":
            # Recommended with Hájek (mean-one) weights
            return (w - 1.0) ** 2
        if mode == "w2":
            return w**2
        if mode == "w":
            return np.abs(w)
        raise ValueError(f"Unknown omega_mode: {mode}")

    def _build_prompt_fold_map(self, n_folds: int) -> Dict[str, int]:
        """Create prompt_id → fold mapping (prefer metadata['cv_fold'])."""
        mapping: Dict[str, int] = {}
        rng = np.random.RandomState(42)

        for i, s in enumerate(self.sampler.dataset.samples):
            pid = s.metadata.get("prompt_id")
            if pid is None:
                # DR/MRDR requires prompt_id to align logged ↔ fresh; fail later if missing in data paths
                continue
            if "cv_fold" in s.metadata:
                fold = int(s.metadata["cv_fold"])
            else:
                # Fallback: deterministic assignment if no cv_fold present (discouraged)
                fold = int(rng.randint(0, n_folds))
                # NOTE: We intentionally do NOT warn repeatedly here; final fit will
                #       still be cross-fitted but users should prefer cv_fold from calibration.
            mapping[str(pid)] = fold

        if not mapping:
            logger.warning(
                "No cv_fold found in dataset metadata; using random fold assignment. "
                "Prefer calibrate_dataset(..., enable_cross_fit=True) for strict orthogonality."
            )
        return mapping
