# Reference implementation from the CJE paper
# This mirrors Algorithm 1 (DR-CPO) and Section 4 of the paper.
# It intentionally depends only on numpy and scikit-learn so it's easy to read.
# This file provides a more complete, runnable implementation of Algorithm 1
# from the CJE paper, expanding on the conceptual 30-line snippet presented
# in Section 6.3. It makes specific choices for the outcome model (Ridge Regression)
# and target policy sampler (FixedSampler) for demonstrative purposes.

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from sklearn.linear_model import Ridge


class FixedSampler:
    """Minimal target sampler.

    This corresponds to the sampling step described in Section 3 of the paper.
    ``sample`` returns M candidate responses for a single context.
    """

    def __init__(
        self, responses: List[str], logps: Optional[List[float]] = None
    ) -> None:
        self.responses = responses
        self.logps = [0.0] * len(responses) if logps is None else logps

    def sample(self, context: str) -> List[Tuple[str, float]]:
        return list(zip(self.responses, self.logps))


class ReferenceDRCPO:
    """Minimal DR-CPO estimator with variance.

    Implements the algorithm from Section 4 of the paper.  It uses
    k-fold cross-fitting for the outcome model and computes the EIF based
    variance formula.
    """

    def __init__(
        self, sampler: FixedSampler, k: int = 2, clip: float = 20.0, seed: int = 0
    ) -> None:
        self.sampler = sampler
        self.k = k
        self.clip = clip
        self.seed = seed
        self._logs: List[Dict[str, Any]] = []
        self._folds: List[np.ndarray] = []
        self._weights: Optional[np.ndarray] = None
        self.v_hat: Optional[float] = None
        self.var: Optional[float] = None

    def _feat(self, ctx: str, resp: str) -> np.ndarray:
        return np.array([len(ctx), len(resp)], dtype=float)

    def fit(self, logs: List[Dict[str, Any]]) -> None:
        self._logs = logs
        n = len(logs)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)
        self._folds = list(np.array_split(idx, self.k))
        weights = np.exp(
            np.array([l["logp_target"] for l in logs])
            - np.array([l["logp"] for l in logs])
        )
        self._weights = np.clip(weights, 0, self.clip)

    def estimate(self) -> Dict[str, float]:
        if not self._folds or self._weights is None:
            raise RuntimeError("fit() must be called before estimate()")

        phi_all: List[float] = []
        for fold in self._folds:
            train = np.concatenate([f for f in self._folds if f is not fold])
            model = Ridge()
            X_train = np.vstack(
                [
                    self._feat(self._logs[i]["context"], self._logs[i]["response"])
                    for i in train
                ]
            )
            y_train = np.array([self._logs[i]["reward"] for i in train], dtype=float)
            model.fit(X_train, y_train)

            for i in fold:
                log = self._logs[i]
                w = self._weights[i]
                mu_x = model.predict(
                    self._feat(log["context"], log["response"]).reshape(1, -1)
                )[0]
                samples = self.sampler.sample(log["context"])
                feats = np.vstack([self._feat(log["context"], r) for r, _ in samples])
                mu_pi = model.predict(feats).mean()
                phi_all.append(mu_pi + w * (log["reward"] - mu_x))

        phi = np.array(phi_all, dtype=float)
        self.v_hat = float(phi.mean())
        self.var = float(np.var(phi - self.v_hat, ddof=1) / len(phi))
        return {"v_hat": self.v_hat, "var": self.var}
