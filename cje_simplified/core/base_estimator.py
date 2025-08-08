"""Base class for CJE estimators."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from ..data.models import Dataset, EstimationResult, WeightCalibrationConfig
from ..data.precomputed_sampler import PrecomputedSampler


class BaseCJEEstimator(ABC):
    """Abstract base class for CJE estimators."""

    def __init__(
        self,
        sampler: PrecomputedSampler,
        calibration_config: Optional[WeightCalibrationConfig] = None,
    ):
        """Initialize estimator.

        Args:
            sampler: Data sampler with precomputed log probabilities
            calibration_config: Configuration for weight calibration
        """
        self.sampler = sampler
        self.config = calibration_config or WeightCalibrationConfig()
        self._fitted = False
        self._weights_cache: Dict[str, np.ndarray] = {}
        self._diagnostics: Dict[str, Any] = {}  # Initialize diagnostics storage

    @abstractmethod
    def fit(self) -> None:
        """Fit the estimator (e.g., calibrate weights)."""
        pass

    @abstractmethod
    def estimate(self) -> EstimationResult:
        """Compute estimates for all target policies."""
        pass

    def fit_and_estimate(self) -> EstimationResult:
        """Convenience method to fit and estimate in one call."""
        self.fit()
        return self.estimate()

    def get_weights(self, target_policy: str) -> Optional[np.ndarray]:
        """Get importance weights for a target policy.

        Args:
            target_policy: Name of target policy

        Returns:
            Array of weights or None if not available
        """
        return self._weights_cache.get(target_policy)

    def get_raw_weights(self, target_policy: str) -> Optional[np.ndarray]:
        """Get raw (uncalibrated) importance weights for a target policy.

        Computes raw weights directly from the sampler. These are the
        importance weights p_target/p_base without any calibration or clipping.

        Args:
            target_policy: Name of target policy

        Returns:
            Array of raw weights or None if not available.
        """
        # Use compute_importance_weights with clip_weight=None for raw weights
        return self.sampler.compute_importance_weights(target_policy, clip_weight=None)

    @property
    def is_fitted(self) -> bool:
        """Check if estimator has been fitted."""
        return self._fitted

    def _validate_fitted(self) -> None:
        """Ensure estimator is fitted before making predictions."""
        if not self._fitted:
            raise RuntimeError("Estimator must be fitted before calling estimate()")
