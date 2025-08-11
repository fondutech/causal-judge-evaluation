"""Outcome models for Doubly Robust estimation.

Outcome models predict E[R|X,A,S] and are used in the direct method
component of DR estimators. They must be cross-fitted to maintain orthogonality.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseOutcomeModel(ABC):
    """Abstract base class for cross-fitted outcome models in DR estimation.

    All outcome models must support cross-fitted prediction where each
    sample is predicted using a model trained on other folds.
    Subclasses only need to implement the single-model training and prediction.
    """

    def __init__(self, n_folds: int = 5):
        """Initialize the outcome model.

        Args:
            n_folds: Number of folds for cross-fitting
        """
        if n_folds < 2:
            raise ValueError(f"n_folds must be at least 2, got {n_folds}")

        self.n_folds = n_folds
        self.fold_models: Dict[int, Any] = {}
        self.fold_assignments: Optional[np.ndarray] = None
        self._fitted = False

    @abstractmethod
    def _fit_single_model(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: np.ndarray,
    ) -> Any:
        """Fit a single model on training data.

        Args:
            prompts: Training prompts
            responses: Training responses
            rewards: Training rewards (calibrated)
            judge_scores: Training judge scores

        Returns:
            A fitted model object
        """
        pass

    @abstractmethod
    def _predict_single_model(
        self,
        model: Any,
        prompts: List[str],
        responses: List[str],
        judge_scores: np.ndarray,
    ) -> np.ndarray:
        """Make predictions using a fitted model.

        Args:
            model: A model returned by _fit_single_model
            prompts: Prompts to predict on
            responses: Responses to predict on
            judge_scores: Judge scores to predict on

        Returns:
            Predicted rewards
        """
        pass

    def fit(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
    ) -> None:
        """Fit cross-fitted models on logged data."""
        if fold_ids is None:
            raise ValueError("fold_ids is required for cross-fitted outcome models")

        if judge_scores is None:
            raise ValueError("judge_scores is required for outcome models")

        # Validate inputs
        n = len(prompts)
        if (
            len(responses) != n
            or len(rewards) != n
            or len(judge_scores) != n
            or len(fold_ids) != n
        ):
            raise ValueError(
                f"Input length mismatch: prompts={len(prompts)}, responses={len(responses)}, "
                f"rewards={len(rewards)}, judge_scores={len(judge_scores)}, fold_ids={len(fold_ids)}"
            )

        self.fold_assignments = fold_ids.astype(int)
        unique_folds = sorted(set(self.fold_assignments))

        # Validate fold IDs are sequential from 0
        if unique_folds != list(range(len(unique_folds))):
            raise ValueError(
                f"Fold IDs must be sequential integers starting from 0. Got: {unique_folds}"
            )

        if len(unique_folds) != self.n_folds:
            logger.warning(
                f"Expected {self.n_folds} folds but got {len(unique_folds)}. "
                f"Adjusting n_folds."
            )
            self.n_folds = len(unique_folds)

        # Train a model for each fold on the other folds
        for fold in unique_folds:
            train_mask = self.fold_assignments != fold

            if not train_mask.any():
                raise ValueError(f"No training data for fold {fold}")

            train_prompts = [p for i, p in enumerate(prompts) if train_mask[i]]
            train_responses = [r for i, r in enumerate(responses) if train_mask[i]]
            train_rewards = rewards[train_mask]
            train_scores = judge_scores[train_mask]

            model = self._fit_single_model(
                train_prompts, train_responses, train_rewards, train_scores
            )
            self.fold_models[fold] = model

            logger.debug(
                f"Fitted model for fold {fold} on {len(train_prompts)} samples"
            )

        self._fitted = True
        logger.info(
            f"{self.__class__.__name__} fitted with {self.n_folds} cross-fitted models"
        )

    def predict(
        self,
        prompts: List[str],
        responses: List[str],
        judge_scores: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict using cross-fitted models."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if judge_scores is None:
            raise ValueError("judge_scores required for prediction")

        # Use provided fold_ids or fall back to stored ones
        if fold_ids is None:
            if self.fold_assignments is None:
                raise ValueError("fold_ids must be provided or set during fit()")
            if len(prompts) != len(self.fold_assignments):
                raise ValueError(
                    f"Using stored fold_assignments but length mismatch: "
                    f"prompts={len(prompts)}, fold_assignments={len(self.fold_assignments)}"
                )
            fold_ids = self.fold_assignments

        # Validate inputs
        n = len(prompts)
        if len(responses) != n or len(judge_scores) != n or len(fold_ids) != n:
            raise ValueError(
                f"Input length mismatch: prompts={len(prompts)}, responses={len(responses)}, "
                f"judge_scores={len(judge_scores)}, fold_ids={len(fold_ids)}"
            )

        fold_ids = fold_ids.astype(int)
        predictions = np.zeros(n)

        # Predict each fold using its out-of-fold model
        for fold in self.fold_models:
            fold_mask = fold_ids == fold
            if not fold_mask.any():
                continue

            fold_prompts = [p for i, p in enumerate(prompts) if fold_mask[i]]
            fold_responses = [r for i, r in enumerate(responses) if fold_mask[i]]
            fold_scores = judge_scores[fold_mask]

            fold_predictions = self._predict_single_model(
                self.fold_models[fold],
                fold_prompts,
                fold_responses,
                fold_scores,
            )

            # Validate prediction shape
            if len(fold_predictions) != fold_mask.sum():
                raise ValueError(
                    f"Model returned {len(fold_predictions)} predictions but expected {fold_mask.sum()}"
                )

            predictions[fold_mask] = fold_predictions

        return predictions


class IsotonicOutcomeModel(BaseOutcomeModel):
    """Cross-fitted isotonic outcome model for DR estimation.

    This model uses g(x,a,s) = f^(-k)(s) where f^(-k) is the isotonic
    calibration function learned from judge scores to rewards, with the
    k-th fold held out for cross-fitting.

    The isotonic models are trained fresh during the DR fit process,
    ensuring proper cross-fitting for orthogonality.
    """

    def __init__(self, n_folds: int = 5):
        """Initialize isotonic outcome model.

        Args:
            n_folds: Number of cross-fitting folds (default 5)
        """
        super().__init__(n_folds)

    def _fit_single_model(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: np.ndarray,
    ) -> Any:
        """Fit an isotonic regression model on training data."""
        from sklearn.isotonic import IsotonicRegression

        # Fit isotonic model from judge scores to rewards
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(judge_scores, rewards)
        return model

    def _predict_single_model(
        self,
        model: Any,
        prompts: List[str],
        responses: List[str],
        judge_scores: np.ndarray,
    ) -> np.ndarray:
        """Predict using the fitted isotonic model."""
        predictions: np.ndarray = model.predict(judge_scores)
        return predictions


class LinearOutcomeModel(BaseOutcomeModel):
    """Example custom outcome model using linear regression.

    This demonstrates how users can implement their own outcome models
    by extending BaseOutcomeModel.
    """

    def __init__(self, n_folds: int = 5, alpha: float = 1.0):
        """Initialize linear outcome model.

        Args:
            n_folds: Number of cross-fitting folds
            alpha: Regularization strength for Ridge regression
        """
        super().__init__(n_folds)
        self.alpha = alpha

    def _fit_single_model(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: np.ndarray,
    ) -> Any:
        """Fit a Ridge regression model on features."""
        from sklearn.linear_model import Ridge

        features = self._extract_features(prompts, responses, judge_scores)
        model = Ridge(alpha=self.alpha)
        model.fit(features, rewards)
        return model

    def _predict_single_model(
        self,
        model: Any,
        prompts: List[str],
        responses: List[str],
        judge_scores: np.ndarray,
    ) -> np.ndarray:
        """Predict using the fitted Ridge model."""
        features = self._extract_features(prompts, responses, judge_scores)
        predictions = model.predict(features)
        clipped: np.ndarray = np.clip(predictions, 0, 1)
        return clipped

    def _extract_features(
        self,
        prompts: List[str],
        responses: List[str],
        judge_scores: np.ndarray,
    ) -> np.ndarray:
        """Extract simple features from inputs."""
        # Length features
        prompt_lengths = np.array([len(p.split()) for p in prompts]).reshape(-1, 1)
        response_lengths = np.array([len(r.split()) for r in responses]).reshape(-1, 1)

        # Judge scores
        scores = judge_scores.reshape(-1, 1)

        # Combine features
        feature_matrix = np.hstack([prompt_lengths, response_lengths, scores])

        # Add bias term
        bias = np.ones((len(prompts), 1))
        final_features: np.ndarray = np.hstack([feature_matrix, bias])

        return final_features
