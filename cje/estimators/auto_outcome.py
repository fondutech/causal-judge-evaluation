from __future__ import annotations
from typing import Any, Dict, List, Tuple, Type
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from .featurizer import (
    Featurizer,
    BasicFeaturizer,
    SentenceEmbeddingFeaturizer,
)
import logging

logger = logging.getLogger(__name__)


class ScoreAugmentFeaturizer(Featurizer):
    """Wrapper that appends the raw judge score as a feature.

    The wrapped *base* featurizer is responsible for producing an array of
    base features (e.g., lengths or embeddings).  We then concatenate a single
    scalar feature:
        score := log_item["score_raw"] if present else log_item.get("score_cal", 0).

    Note: This prioritizes raw judge scores over calibrated scores to avoid
    circular dependencies in the outcome model.
    """

    def __init__(self, base: Featurizer):
        self.base = base

    # Passthrough fit
    def fit(self, logs: List[Dict[str, Any]]) -> "ScoreAugmentFeaturizer":
        self.base.fit(logs)
        return self

    def _extract_score(self, log_item: Dict[str, Any]) -> float:
        # Get score_raw if present, otherwise score_cal, otherwise default to 0.0
        # This prioritizes raw scores to avoid circular dependencies
        score_val = log_item.get("score_raw")
        if score_val is not None:
            return float(score_val)

        score_val = log_item.get("score_cal")
        if score_val is not None:
            return float(score_val)

        return 0.0

    def transform_single(self, log_item: Dict[str, Any]) -> np.ndarray:
        base_feats = self.base.transform_single(log_item)
        score = np.array([self._extract_score(log_item)], dtype=np.float64)
        return np.concatenate([base_feats, score])

    def transform(self, logs: List[Dict[str, Any]]) -> np.ndarray:
        base_matrix = self.base.transform(logs)
        scores = np.array(
            [self._extract_score(l) for l in logs], dtype=np.float64
        ).reshape(-1, 1)
        return np.hstack([base_matrix, scores])


def _choose_base_featurizer(n_samples: int) -> Featurizer:
    # Small datasets: BasicFeaturizer is fine; larger ones -> sentence embeddings
    if n_samples < 1000:
        return BasicFeaturizer()
    elif n_samples < 20000:
        return SentenceEmbeddingFeaturizer(model_name="all-MiniLM-L6-v2")
    else:
        return SentenceEmbeddingFeaturizer(model_name="all-mpnet-base-v2")


def auto_select(n: int) -> Tuple[Type[Any], Dict[str, Any], Featurizer]:
    """Return (model_cls, model_kwargs, featurizer) based on *n* labels.

    Heuristic buckets:
        • n < 20     → DummyRegressor(mean)
        • 20 ≤ n < 1k → Ridge
        • 1k ≤ n < 20k → XGBRegressor shallow
        • n ≥ 20k    → XGBRegressor deeper

    The returned featurizer uses only context/response features by default.
    Judge scores can be explicitly added via ScoreAugmentFeaturizer if needed.
    """

    if n < 20:
        model_cls = DummyRegressor
        model_kwargs: Dict[str, Any] = {"strategy": "mean"}
    elif n < 1000:
        model_cls = Ridge
        model_kwargs = {"alpha": 1.0}
    elif n < 20000:
        model_cls = XGBRegressor
        model_kwargs = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "objective": "reg:squarederror",
        }
    else:
        model_cls = XGBRegressor
        model_kwargs = {
            "n_estimators": 800,
            "max_depth": 10,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
        }

    featurizer = _choose_base_featurizer(n)

    logger.info(
        "[AutoSelect] n=%d → model=%s, featurizer=%s",
        n,
        model_cls.__name__,
        featurizer.__class__.__name__,
    )

    return model_cls, model_kwargs, featurizer
