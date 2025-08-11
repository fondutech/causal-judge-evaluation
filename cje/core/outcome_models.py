"""Outcome models for Doubly Robust estimation.

Outcome models predict E[R|X,A,S] and are used in the direct method
component of DR estimators. They must be cross-fitted to maintain orthogonality.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    from ..calibration.judge import JudgeCalibrator

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

        # Remap fold IDs to be sequential 0..K-1 for the subset
        original_fold_ids = fold_ids.astype(int)
        unique_folds = sorted(np.unique(original_fold_ids))

        # Create mapping from original to sequential
        fold_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_folds)}
        self.fold_assignments = np.vectorize(fold_id_map.__getitem__)(original_fold_ids)

        # Store reverse mapping for potential debugging
        self._fold_id_map = fold_id_map
        self._reverse_fold_map = {v: k for k, v in fold_id_map.items()}

        # Adjust n_folds to actual number of unique folds
        if len(unique_folds) != self.n_folds:
            logger.info(
                f"Adjusting n_folds from {self.n_folds} to {len(unique_folds)} based on data"
            )
            self.n_folds = len(unique_folds)

        # Train a model for each fold on the other folds (using remapped IDs)
        for fold in range(self.n_folds):
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


class CalibratorBackedOutcomeModel(BaseOutcomeModel):
    """Outcome model that reuses cross-fitted reward calibrators.

    Instead of refitting a model on (S, R=f_all(S)), this model reuses
    the cross-fitted calibrators f^(-k) that were already trained during
    reward calibration. This preserves orthogonality and avoids redundant
    computation.

    This is the recommended default for DR estimation when using isotonic
    calibration for rewards.
    """

    def __init__(self, reward_calibrator: "JudgeCalibrator", n_folds: int = 5):
        """Initialize with a fitted reward calibrator.

        Args:
            reward_calibrator: A fitted JudgeCalibrator with cross-fitted models
            n_folds: Number of folds (should match calibrator's n_folds)
        """
        super().__init__(n_folds)
        self.calibrator = reward_calibrator

        # Verify calibrator has cross-fitted models
        if (
            not hasattr(reward_calibrator, "_fold_models")
            or not reward_calibrator._fold_models
        ):
            raise ValueError(
                "CalibratorBackedOutcomeModel requires a calibrator fitted with "
                "fit_cv(). Use enable_cross_fit=True in calibrate_dataset()."
            )

        if reward_calibrator._n_folds != n_folds:
            logger.warning(
                f"Calibrator has {reward_calibrator._n_folds} folds but outcome model "
                f"requested {n_folds}. Using calibrator's fold count."
            )
            self.n_folds = reward_calibrator._n_folds

    def _fit_single_model(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: np.ndarray,
    ) -> Any:
        """No training needed - reuse calibrator's models."""
        # Just return a reference to the calibrator
        return self.calibrator

    def _predict_single_model(
        self,
        model: Any,
        prompts: List[str],
        responses: List[str],
        judge_scores: np.ndarray,
    ) -> np.ndarray:
        """Predict using the calibrator's cross-fitted models.

        This should never be called directly since we override fit() and predict()
        to use the calibrator's predict_oof() method directly.
        """
        # This is a fallback that shouldn't be reached
        return judge_scores

    def fit(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
    ) -> None:
        """Fit by storing fold assignments (no model training needed).

        Args:
            prompts: Training prompts
            responses: Training responses
            rewards: Training rewards (not used)
            judge_scores: Training judge scores
            fold_ids: Pre-assigned fold IDs from calibration
        """
        n_samples = len(prompts)

        if fold_ids is not None:
            # Use provided fold assignments
            self.fold_assignments = np.asarray(fold_ids)
        else:
            # Try to get from calibrator
            if (
                hasattr(self.calibrator, "_fold_ids")
                and self.calibrator._fold_ids is not None
            ):
                if len(self.calibrator._fold_ids) == n_samples:
                    self.fold_assignments = self.calibrator._fold_ids
                else:
                    # Need to subset or map somehow
                    logger.warning(
                        f"Calibrator has {len(self.calibrator._fold_ids)} fold IDs "
                        f"but we have {n_samples} samples. Using random assignment."
                    )
                    np.random.seed(42)
                    self.fold_assignments = np.random.randint(
                        0, self.n_folds, n_samples
                    )
            else:
                # Fall back to random assignment
                np.random.seed(42)
                self.fold_assignments = np.random.randint(0, self.n_folds, n_samples)

        self._fitted = True
        logger.info(
            f"CalibratorBackedOutcomeModel ready: {n_samples} samples, "
            f"{self.n_folds} folds (reusing calibrator models)"
        )

    def predict(
        self,
        prompts: List[str],
        responses: List[str],
        judge_scores: Optional[np.ndarray] = None,
        fold_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict using cross-fitted calibration models.

        Args:
            prompts: Prompts to predict on
            responses: Responses to predict on
            judge_scores: Judge scores to calibrate
            fold_ids: Fold assignments for each sample

        Returns:
            Cross-fitted predictions using f^(-k)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        if fold_ids is None:
            # Use stored fold assignments if they match
            if self.fold_assignments is not None and len(self.fold_assignments) == len(
                prompts
            ):
                fold_ids = self.fold_assignments
            else:
                # For new data, require explicit fold assignments
                raise ValueError(
                    "fold_ids required for CalibratorBackedOutcomeModel.predict() "
                    "when predicting on new data to avoid accidental in-fold predictions. "
                    "Provide fold assignments from the calibration phase."
                )

        if judge_scores is None:
            raise ValueError("judge_scores required for prediction")

        # Use calibrator's out-of-fold predictions
        predictions = self.calibrator.predict_oof(judge_scores, fold_ids)

        return predictions
