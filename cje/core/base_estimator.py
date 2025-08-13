"""Base class for CJE estimators."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from ..data.models import Dataset, EstimationResult
from ..data.precomputed_sampler import PrecomputedSampler
from .diagnostics import DiagnosticSuite, DiagnosticManager


class BaseCJEEstimator(ABC):
    """Abstract base class for CJE estimators.

    All estimators must implement:
    - fit(): Prepare the estimator (e.g., calibrate weights)
    - estimate(): Compute estimates and diagnostics

    The estimate() method must populate EstimationResult.diagnostics
    with a complete DiagnosticSuite.
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
    ):
        """Initialize estimator.

        Args:
            sampler: Data sampler with precomputed log probabilities
        """
        self.sampler = sampler
        self._fitted = False
        self._weights_cache: Dict[str, np.ndarray] = {}
        self._results: Optional[EstimationResult] = None
        self._diagnostic_manager = DiagnosticManager()

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
        # Get truly raw weights (not Hajek normalized)
        return self.sampler.compute_importance_weights(
            target_policy, clip_weight=None, mode="raw"
        )

    @property
    def is_fitted(self) -> bool:
        """Check if estimator has been fitted."""
        return self._fitted

    def _validate_fitted(self) -> None:
        """Ensure estimator is fitted before making predictions."""
        if not self._fitted:
            raise RuntimeError("Estimator must be fitted before calling estimate()")

    def _build_diagnostics(
        self,
        result: EstimationResult,
        calibration_result: Optional[Any] = None,
        include_oracle: bool = False,
    ) -> DiagnosticSuite:
        """Build complete diagnostic suite for the estimation result.

        This method should be called by estimate() to populate diagnostics.

        Args:
            result: The estimation result (without diagnostics)
            calibration_result: Optional calibration result
            include_oracle: Whether to compute oracle diagnostics

        Returns:
            Complete DiagnosticSuite
        """
        # Store results BEFORE computing diagnostics so they're available
        self._results = result

        # Get the dataset from sampler
        dataset = self.sampler.dataset if hasattr(self.sampler, "dataset") else None

        # Use diagnostic manager to compute full suite
        suite = self._diagnostic_manager.compute_suite(
            estimator=self,
            dataset=dataset,
            calibration_result=calibration_result,
            include_oracle=include_oracle,
        )

        # Add diagnostics to result
        result.diagnostics = suite

        return suite

    def get_diagnostics(self) -> Optional[DiagnosticSuite]:
        """Get the diagnostic suite from the last estimation.

        Returns:
            DiagnosticSuite if estimate() has been called, None otherwise
        """
        if self._results and self._results.diagnostics:
            return self._results.diagnostics
        return None
