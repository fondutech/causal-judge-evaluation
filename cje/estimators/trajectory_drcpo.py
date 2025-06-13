"""Doubly-Robust CPO estimator for agent trajectories (MDP form).

This is a first-pass implementation that mirrors `MultiDRCPOEstimator` but
operates on `CJETrajectory` objects and uses the new
`MultiTargetTrajectorySampler` for importance weights.

For an initial MVP we treat each trajectory as a single unit with total
reward ``traj.y_true`` (if present) or the sum of per-step rewards.  A future
revision can expose per-step EIFs for even tighter variance.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

# Optional import; may be unavailable on some systems
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover

    class XGBRegressor:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("xgboost could not be imported")


from .base import Estimator
from .results import EstimationResult
from ..data.schema import CJETrajectory
from ..loggers.trajectory_sampler import MultiTargetTrajectorySampler
from .featurizer import Featurizer, BasicFeaturizer
from .auto_outcome import auto_select

ModelType = Any  # scikit-learn compatible model

# Lightweight default model
DEFAULT_OUTCOME_MODEL = Ridge


class MultiDRCPOMDPEstimator(Estimator[Dict[str, Any]]):
    """Cross-fitted DR-CPO estimator for K target *agent* policies.

    Notes
    -----
    *Rewards* – we use ``traj_reward`` defined as::

        traj_reward = traj.y_true if traj.y_true is not None else sum(s.reward or 0 for s in traj.steps)

    *Features* – until a specialised trajectory featurizer arrives, we fall back
    to `BasicFeaturizer` run on a pseudo-sample consisting of
    ``{"context": traj.steps[0].state, "response": traj.steps[-1].action}``.
    This allows the outcome model to run without touching the agent internals.
    """

    def __init__(
        self,
        sampler: MultiTargetTrajectorySampler,
        k: int = 5,
        clip: float = 20.0,
        seed: int = 0,
        outcome_model_cls: Type[ModelType] = DEFAULT_OUTCOME_MODEL,
        outcome_model_kwargs: Optional[Dict[str, Any]] = None,
        featurizer: Optional[Featurizer] = None,
        n_jobs: int | None = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sampler = sampler
        self.k = k
        self.clip = clip
        self.seed = seed
        self.outcome_model_cls = outcome_model_cls
        self.outcome_model_kwargs = outcome_model_kwargs or {}
        self.featurizer = featurizer or BasicFeaturizer()
        self.n_jobs = n_jobs

        self._auto_select_models = (
            outcome_model_cls is XGBRegressor
            and not outcome_model_kwargs
            and featurizer is None
        )

        # Placeholders
        self.n: int = 0
        self.K: int = sampler.K
        self._folds: Optional[List[np.ndarray]] = None
        self._traj_data: List[CJETrajectory] = []
        self._rewards: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None  # (n, K)
        self._features: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _trajectory_reward(traj: CJETrajectory) -> float:
        if traj.y_true is not None:
            return float(traj.y_true)
        return float(sum((s.reward or 0.0) for s in traj.steps))

    def _pseudo_sample(self, traj: CJETrajectory) -> Dict[str, Any]:
        first_state = str(traj.steps[0].state)
        last_action = str(traj.steps[-1].action)
        return {"context": first_state, "response": last_action}

    # ------------------------------------------------------------------
    def fit(self, trajectories: Sequence[CJETrajectory], **kwargs: Any) -> None:  # type: ignore[override]
        self._traj_data = list(trajectories)
        self.n = len(self._traj_data)
        if self.n == 0:
            raise ValueError("No trajectories provided")

        # Auto-select models if applicable
        if self._auto_select_models:
            self.outcome_model_cls, self.outcome_model_kwargs, self.featurizer = (
                auto_select(self.n)
            )

        # Compute rewards vector
        self._rewards = np.array(
            [self._trajectory_reward(t) for t in self._traj_data], dtype=float
        )

        # Importance weights matrix
        self._weights = np.vstack(
            [self.sampler.importance_weights(t) for t in self._traj_data]
        )  # (n, K)

        # Features for outcome model
        pseudo_samples = [self._pseudo_sample(t) for t in self._traj_data]
        self.featurizer.fit(pseudo_samples)
        self._features = self.featurizer.transform(pseudo_samples)

        # Cross-validation folds
        indices = np.arange(self.n)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)
        self._folds = np.array_split(indices, self.k)

    # ------------------------------------------------------------------
    def _compute_mu_pi_matrix(self, idxs: np.ndarray, outcome_model: Any) -> np.ndarray:
        """Monte-Carlo estimate μ_πᵏ(x) for trajectories in idxs."""
        mu_pi = np.zeros((len(idxs), self.K))
        for j, idx in enumerate(idxs):
            traj = self._traj_data[idx]
            context_state = str(traj.steps[0].state)
            samples = self.sampler.sample_many(context_state, n=3)  # 3 MC samples
            # Build pseudo samples per policy
            for k in range(self.K):
                if samples[k]:
                    feats = self.featurizer.transform(
                        [{"context": context_state, "response": s} for s in samples[k]]
                    )
                    mu_pi[j, k] = float(outcome_model.predict(feats).mean())
                else:
                    mu_pi[j, k] = 0.0
        return mu_pi

    def _process_fold(self, test_idx: np.ndarray) -> np.ndarray:
        # Train idx are all others
        train_idx = np.setdiff1d(np.arange(self.n), test_idx)

        # Type assertions for mypy
        assert self._features is not None
        assert self._rewards is not None
        assert self._weights is not None

        # Train outcome model
        model = self.outcome_model_cls(**self.outcome_model_kwargs)
        model.fit(self._features[train_idx], self._rewards[train_idx])

        # Predictions
        mu_hat = model.predict(self._features[test_idx])  # (n_test,)
        mu_pi = self._compute_mu_pi_matrix(test_idx, model)  # (n_test, K)
        W_test = self._weights[test_idx]
        r_test = self._rewards[test_idx]

        eif = mu_pi + W_test * (r_test[:, None] - mu_hat[:, None])
        return np.asarray(eif)

    # ------------------------------------------------------------------
    def estimate(self) -> EstimationResult:  # type: ignore[override]
        if self._folds is None or self._weights is None or self._features is None:
            raise RuntimeError("fit() must be called first")

        eif_blocks = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_fold)(idx) for idx in self._folds
        )
        eif_all = np.vstack(eif_blocks)
        v_hat = eif_all.mean(axis=0)
        cov = (
            np.cov(eif_all, rowvar=False) / self.n
            if self.K > 1
            else np.array([[eif_all.var(ddof=1) / self.n]])
        )
        se = np.sqrt(np.diag(cov))

        return EstimationResult(
            v_hat=v_hat,
            se=se,
            n=self.n,
            covariance_matrix=cov,
            estimator_type="DR_CPO_MDP",
            n_policies=self.K,
            metadata={
                "propensity_clip": self.clip,
                "k_folds": self.k,
            },
        )
