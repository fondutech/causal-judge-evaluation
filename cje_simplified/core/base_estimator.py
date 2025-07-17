"""Base class for CJE estimators."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from ..data.models import Dataset, EstimationResult, WeightCalibrationConfig
from ..data import PrecomputedSampler


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

    def _compute_raw_weights(self, target_policy: str) -> Optional[np.ndarray]:
        """Compute raw importance weights for a policy.

        Args:
            target_policy: Name of target policy

        Returns:
            Array of raw weights or None if computation fails
        """
        try:
            # Get formatted data for the specific policy
            data = self.sampler.get_data_for_policy(target_policy)
            if data is None:
                return None

            weights = []
            for record in data:
                # Compute w = pi(y|x) / p0(y|x)
                if (
                    record["base_policy_logprob"] is None
                    or record["policy_logprob"] is None
                ):
                    continue

                log_weight = record["policy_logprob"] - record["base_policy_logprob"]
                weight = np.exp(np.clip(log_weight, -20, 20))  # Clip for stability
                weights.append(weight)

            return np.array(weights) if weights else None

        except Exception as e:
            print(f"Error computing weights for {target_policy}: {e}")
            return None

    @property
    def is_fitted(self) -> bool:
        """Check if estimator has been fitted."""
        return self._fitted

    def _validate_fitted(self) -> None:
        """Ensure estimator is fitted before making predictions."""
        if not self._fitted:
            raise RuntimeError("Estimator must be fitted before calling estimate()")
