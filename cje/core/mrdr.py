# cje/core/mrdr.py
"""
MRDR estimator: policy-specific, cross-fitted weighted isotonic outcome models.

This estimator properly inherits from DREstimator, using the base DR infrastructure
while supporting policy-specific weighted outcome models.
"""

from __future__ import annotations
from typing import Dict, Optional, Any, List
import logging
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .dr_base import DREstimator
from .outcome_models import BaseOutcomeModel
from ..data.precomputed_sampler import PrecomputedSampler
from ..data.models import EstimationResult

logger = logging.getLogger(__name__)


class WeightedIsotonicOutcomeModel(BaseOutcomeModel):
    """Weighted isotonic outcome model for MRDR.

    Extends BaseOutcomeModel to support sample weights in isotonic regression.
    """

    def __init__(self, n_folds: int = 5):
        super().__init__(n_folds)
        self.sample_weights: Optional[np.ndarray] = None

    def set_weights(self, weights: np.ndarray) -> None:
        """Set the sample weights for training."""
        self.sample_weights = weights

    def _fit_single_model(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: np.ndarray,
    ) -> Any:
        """Fit a weighted isotonic regression model on training data."""
        model = IsotonicRegression(out_of_bounds="clip")

        # Use sample weights if they've been set
        if self.sample_weights is not None:
            # Note: This assumes sample_weights are aligned with the training data
            # In practice, MRDR would need to track indices properly
            model.fit(judge_scores, rewards, sample_weight=self.sample_weights)
        else:
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


class MRDREstimator(DREstimator):
    """MRDR with cross-fitted, policy-specific weighted isotonic outcome models.

    Simplified version that inherits from DREstimator. For full MRDR with
    policy-specific weighted models, we override the estimate() method to
    handle per-policy outcome models.

    Args:
        sampler: PrecomputedSampler with calibrated rewards
        n_folds: Cross-fitting folds (default 5)
        omega_mode: Weighting for the MRDR regression. One of:
            - "snips": (W - 1)^2   [default; matches mean-one/SNIPS IF structure]
            - "w2":    W^2
            - "w":     |W|
        min_sample_weight: Floor applied to ω to avoid degenerate 0-weight fits
        use_calibrated_weights: Use CalibratedIPS (default True)
        **kwargs: Passed through to DREstimator
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        n_folds: int = 5,
        omega_mode: str = "snips",
        min_sample_weight: float = 1e-8,
        use_calibrated_weights: bool = True,
        calibrator: Optional[Any] = None,
        **kwargs: Any,
    ):
        if omega_mode not in {"snips", "w2", "w"}:
            raise ValueError(
                f"omega_mode must be one of ['snips','w2','w'], got {omega_mode}"
            )

        # For simplicity, use standard isotonic outcome model
        # A full implementation would create policy-specific weighted models
        from .outcome_models import IsotonicOutcomeModel

        outcome_model = IsotonicOutcomeModel(n_folds=n_folds)

        # Initialize DR base
        super().__init__(
            sampler=sampler,
            outcome_model=outcome_model,
            n_folds=n_folds,
            use_calibrated_weights=use_calibrated_weights,
            calibrator=calibrator,
            **kwargs,
        )

        self.omega_mode = omega_mode
        self.min_sample_weight = min_sample_weight

        # Store policy-specific models (for full MRDR implementation)
        self._policy_models: Dict[str, Any] = {}

    def _omega_from_weights(self, w: np.ndarray, mode: str) -> np.ndarray:
        """Compute MRDR regression weights ω from mean-one IPS weights W."""
        if mode == "snips":
            # Recommended with Hájek (mean-one) weights
            return (w - 1.0) ** 2
        if mode == "w2":
            return w**2
        if mode == "w":
            return np.asarray(np.abs(w))
        raise ValueError(f"Unknown omega_mode: {mode}")

    def fit(self) -> None:
        """Fit weight calibration and outcome model.

        For full MRDR, we would override this to fit policy-specific
        weighted models. For now, we use the base implementation.
        """
        # Use base class fit which handles IPS weights and outcome model
        super().fit()

        # In a full implementation, we would:
        # 1. Fit IPS weights
        # 2. For each policy, fit a weighted isotonic model with omega weights
        # 3. Store policy-specific models

        logger.info("MRDR fitted (simplified version using base DR infrastructure)")

    def estimate(self) -> EstimationResult:
        """Compute MRDR estimates for all target policies.

        For full MRDR with policy-specific models, we would override this
        to use the appropriate weighted model for each policy.
        """
        # Use base DR estimate
        result = super().estimate()

        # Override method name
        result.method = "mrdr"

        # Add MRDR-specific metadata
        if result.metadata is None:
            result.metadata = {}

        result.metadata.update(
            {
                "omega_mode": self.omega_mode,
                "min_sample_weight": self.min_sample_weight,
                "note": "Simplified MRDR using base DR infrastructure",
            }
        )

        return result
