from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class Featurizer(ABC):
    """
    Abstract Base Class for featurizers.

    A featurizer is responsible for converting a log item (or parts of it,
    like context and response) into a numpy array of features suitable
    for machine learning models.
    """

    @abstractmethod
    def fit(self, logs: List[Dict[str, Any]]) -> "Featurizer":
        """
        Fit the featurizer on a list of log items.
        This is optional and can be a no-op if the featurizer is stateless.

        Args:
            logs: A list of log dictionaries.
        """
        pass

    @abstractmethod
    def transform_single(self, log_item: Dict[str, Any]) -> NDArray[np.float64]:
        """
        Transform a single log item into a feature vector.

        Args:
            log_item: A single log dictionary.

        Returns:
            A numpy array representing the features.
        """
        pass

    def transform(self, logs: List[Dict[str, Any]]) -> NDArray[np.float64]:
        """
        Transform a list of log items into a batch of feature vectors.
        Default implementation calls transform_single for each item.
        Subclasses can override for efficiency if batch processing is beneficial.

        Args:
            logs: A list of log dictionaries.

        Returns:
            A 2D numpy array where each row is a feature vector.
        """
        return np.vstack([self.transform_single(log) for log in logs])


class BasicFeaturizer(Featurizer):
    """
    A basic featurizer that uses context length and response length.
    This was the default behavior in DRCPOEstimator.
    """

    def fit(self, logs: List[Dict[str, Any]]) -> "BasicFeaturizer":
        # This featurizer is stateless
        return self

    def transform_single(self, log_item: Dict[str, Any]) -> NDArray[np.float64]:
        return np.array(
            [
                len(str(log_item.get("context", ""))),
                len(str(log_item.get("response", ""))),
            ],
            dtype=np.float64,
        )


class SentenceEmbeddingFeaturizer(Featurizer):
    """
    A featurizer that uses a pre-trained SentenceTransformer model to create
    embeddings for context and response, then concatenates them.

    Args:
        model_name (str): The name of the SentenceTransformer model to use.
                          Defaults to 'all-MiniLM-L6-v2'.
        combination_method (str): How to combine context and response embeddings.
                                  Currently supports 'concat'. Defaults to 'concat'.
    """

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", combination_method: str = "concat"
    ):
        super().__init__()
        self.model_name = model_name
        self.combination_method = combination_method
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
        except ImportError:
            logger.error(
                f"Error loading SentenceTransformer model {self.model_name}: sentence-transformers not installed"
            )
            logger.info("Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(
                f"Error loading SentenceTransformer model {self.model_name}: {e}"
            )
            raise
        self._fitted = False  # Though stateless for pre-trained, fit marks it as ready.

    def fit(self, logs: List[Dict[str, Any]]) -> "SentenceEmbeddingFeaturizer":
        """
        Fit the featurizer to the logs.

        This method primarily marks the featurizer as 'ready'.
        """
        if self.model is None:
            raise RuntimeError(
                f"SentenceTransformer model '{self.model_name}' could not be loaded during init. "
                f"Please check the model name and ensure sentence-transformers is installed."
            )
        self._fitted = True
        return self

    def transform_single(self, log_item: Dict[str, Any]) -> NDArray[np.float64]:
        if not self._fitted or self.model is None:
            raise RuntimeError(
                "Featurizer has not been fitted or the model was not loaded. Call fit() first."
            )

        context_str = str(log_item.get("context", ""))
        response_str = str(log_item.get("response", ""))

        context_emb = self.model.encode(context_str)
        response_emb = self.model.encode(response_str)

        if self.combination_method == "concat":
            features = np.concatenate([context_emb, response_emb])
        # Add other potential combination methods later if needed (e.g., mean, diff)
        else:
            raise ValueError(
                f"Unknown combination_method: {self.combination_method}. Supported: ['concat']"
            )

        return np.array(features, dtype=np.float64)

    def transform(self, logs: List[Dict[str, Any]]) -> NDArray[np.float64]:
        """
        Batch transform a list of log items.
        """
        if not self._fitted or self.model is None:
            raise RuntimeError(
                "Featurizer has not been fitted or the model was not loaded. Call fit() first."
            )

        contexts = [str(log.get("context", "")) for log in logs]
        responses = [str(log.get("response", "")) for log in logs]

        # Batch encode for efficiency
        context_embs = self.model.encode(contexts)
        response_embs = self.model.encode(responses)

        feature_list = []
        for i in range(len(logs)):
            if self.combination_method == "concat":
                feature_list.append(np.concatenate([context_embs[i], response_embs[i]]))
            else:
                # Should not be reached if constructor validates, but good practice
                raise ValueError(
                    f"Unknown combination_method: {self.combination_method}."
                )

        return np.array(feature_list, dtype=np.float64)


# ---------------------------------------------------------------------------
# RichFeaturizer – quick, lightweight feature set tailored for DR-CPO
# ---------------------------------------------------------------------------


class RichFeaturizer(Featurizer):
    """Light-weight featurizer that adds proxy-judge signals and an *action id*.

    Features per log item (in this order):

    0. context length (characters)
    1. response length (characters)
    2. raw judge score   – ``score_raw`` mean (0-1 scaled or None → 0)
    3. raw judge variance – ``score_raw`` variance (if include_variance=True)
    4. calibrated reward – ``score_cal`` *or* ``reward`` fallback (0 if missing)
    5. action id index   – normalised in [0,1] based on mapping learned in *fit*
    """

    def __init__(self, include_variance: bool = True) -> None:
        self._action_to_idx: Dict[str, int] = {}
        self._n_actions: int = 0
        self._fitted: bool = False
        self.include_variance = include_variance

    # ------------------------------------------------------------------
    # Featurizer API
    # ------------------------------------------------------------------

    def fit(self, logs: List[Dict[str, Any]]) -> "RichFeaturizer":
        """Learn mapping *action string* → *index* from the data."""

        actions = {str(row.get("action", "")) for row in logs}
        self._action_to_idx = {act: i for i, act in enumerate(sorted(actions))}
        self._n_actions = max(len(self._action_to_idx), 1)
        self._fitted = True
        return self

    def _action_norm(self, action: str) -> float:
        if action in self._action_to_idx:
            return self._action_to_idx[action] / max(self._n_actions - 1, 1)
        else:
            # unseen action → treat as 0
            return 0.0

    def transform_single(self, log_item: Dict[str, Any]) -> NDArray[np.float64]:
        if not self._fitted:
            raise RuntimeError("RichFeaturizer.fit() must be called before transform()")

        ctx_len = len(str(log_item.get("context", "")))
        resp_len = len(str(log_item.get("response", "")))

        # Extract score mean and variance using unified format
        from ..utils.score_storage import ScoreCompatibilityLayer

        raw_score = ScoreCompatibilityLayer.get_score_value(log_item, "score_raw")
        raw_variance = ScoreCompatibilityLayer.get_score_variance(log_item, "score_raw")

        # Extract calibrated score using compatibility layer
        try:
            cal_score = ScoreCompatibilityLayer.get_score_value(log_item, "score_cal")
        except KeyError:
            # Fall back to reward field
            try:
                cal_score = ScoreCompatibilityLayer.get_score_value(log_item, "reward")
            except KeyError:
                cal_score = 0.0
        action_val = self._action_norm(str(log_item.get("action", "")))

        if self.include_variance:
            feats = np.array(
                [ctx_len, resp_len, raw_score, raw_variance, cal_score, action_val],
                dtype=np.float64,
            )
        else:
            feats = np.array(
                [ctx_len, resp_len, raw_score, cal_score, action_val], dtype=np.float64
            )
        return feats

    # Batch transform inherits default implementation from base class


# Update public API
__all__ = [
    "Featurizer",
    "BasicFeaturizer",
    "SentenceEmbeddingFeaturizer",
    "RichFeaturizer",
]
