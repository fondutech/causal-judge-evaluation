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
    """Wrapper that appends judge score and uncertainty as features.

    The wrapped *base* featurizer is responsible for producing an array of
    base features (e.g., lengths or embeddings). We then concatenate score features:
        - score_mean := extracted from unified score format
        - score_variance := extracted from unified score format (if available)

    This leverages the unified judge system that provides both mean and variance.
    """

    def __init__(self, base: Featurizer, include_variance: bool = True):
        self.base = base
        self.include_variance = include_variance

    # Passthrough fit
    def fit(self, logs: List[Dict[str, Any]]) -> "ScoreAugmentFeaturizer":
        self.base.fit(logs)
        return self

    def _extract_score_features(self, log_item: Dict[str, Any]) -> Tuple[float, float]:
        """Extract score mean and variance from unified format.

        Returns:
            Tuple of (mean, variance)
        """
        from ..utils.score_storage import ScoreCompatibilityLayer

        # Use the compatibility layer to handle both formats
        mean = ScoreCompatibilityLayer.get_score_value(log_item, "score_raw")
        variance = ScoreCompatibilityLayer.get_score_variance(log_item, "score_raw")

        return float(mean), float(variance)

    def _extract_score(self, log_item: Dict[str, Any]) -> float:
        """Legacy method for backward compatibility."""
        mean, _ = self._extract_score_features(log_item)
        return mean

    def transform_single(self, log_item: Dict[str, Any]) -> np.ndarray:
        base_feats = self.base.transform_single(log_item)
        mean, variance = self._extract_score_features(log_item)

        if self.include_variance:
            score_feats = np.array([mean, variance], dtype=np.float64)
        else:
            score_feats = np.array([mean], dtype=np.float64)

        return np.concatenate([base_feats, score_feats])

    def transform(self, logs: List[Dict[str, Any]]) -> np.ndarray:
        base_matrix = self.base.transform(logs)

        # Extract both mean and variance
        score_features = [self._extract_score_features(log) for log in logs]
        means = [sf[0] for sf in score_features]
        variances = [sf[1] for sf in score_features]

        if self.include_variance:
            score_matrix = np.column_stack([means, variances])
        else:
            score_matrix = np.array(means, dtype=np.float64).reshape(-1, 1)

        return np.hstack([base_matrix, score_matrix])


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
