"""Base class for CJE estimators."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import numpy as np
import logging

from ..data.models import Dataset, EstimationResult
from ..data.precomputed_sampler import PrecomputedSampler
from ..calibration.oracle_slice import OracleSliceAugmentation, OracleSliceConfig

logger = logging.getLogger(__name__)


class BaseCJEEstimator(ABC):
    """Abstract base class for CJE estimators.

    All estimators must implement:
    - fit(): Prepare the estimator (e.g., calibrate weights)
    - estimate(): Compute estimates and diagnostics

    The estimate() method must populate EstimationResult.diagnostics
    with IPSDiagnostics or DRDiagnostics as appropriate.
    """

    def __init__(
        self,
        sampler: PrecomputedSampler,
        run_diagnostics: bool = True,
        diagnostic_config: Optional[Dict[str, Any]] = None,
        oracle_slice_config: Union[str, bool, OracleSliceConfig, None] = "auto",
    ):
        """Initialize estimator.

        Args:
            sampler: Data sampler with precomputed log probabilities
            run_diagnostics: Whether to compute diagnostics (default True)
            diagnostic_config: Optional configuration dict (for future use)
            oracle_slice_config: Oracle slice augmentation configuration:
                - "auto" (default): Automatically detect and enable if oracle coverage < 100%
                - True: Always enable with default configuration
                - False/None: Disable augmentation
                - OracleSliceConfig object: Use provided configuration
        """
        self.sampler = sampler
        self.run_diagnostics = run_diagnostics
        self.diagnostic_config = diagnostic_config
        self._fitted = False
        self._weights_cache: Dict[str, np.ndarray] = {}
        self._influence_functions: Dict[str, np.ndarray] = {}
        self._results: Optional[EstimationResult] = None

        # Configure oracle slice augmentation (canonical for all estimators)
        self.oracle_augmentation = self._configure_oracle_augmentation(
            oracle_slice_config
        )
        self._aug_diagnostics: Dict[str, Dict] = {}  # Store augmentation diagnostics

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
        result = self.estimate()

        # All estimators now create diagnostics directly in estimate()
        # The DiagnosticSuite system has been removed for simplicity
        # per CLAUDE.md principles (YAGNI, Do One Thing Well)

        # Verify diagnostics were created
        if self.run_diagnostics and result is not None:
            if not hasattr(result, "diagnostics") or result.diagnostics is None:
                # This shouldn't happen anymore, but log a warning
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"{self.__class__.__name__} did not create diagnostics. "
                    "All estimators should create IPSDiagnostics or DRDiagnostics directly."
                )

        return result

    def get_influence_functions(self, policy: Optional[str] = None) -> Optional[Any]:
        """Get influence functions for a policy or all policies.

        Influence functions capture the per-sample contribution to the estimate
        and are essential for statistical inference (standard errors, confidence
        intervals, hypothesis tests).

        Args:
            policy: Specific policy name, or None for all policies

        Returns:
            If policy specified: array of influence functions for that policy
            If policy is None: dict of all influence functions by policy
            Returns None if not yet estimated
        """
        if not self._influence_functions:
            return None

        if policy is not None:
            return self._influence_functions.get(policy)

        return self._influence_functions

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

    def get_diagnostics(self) -> Optional[Any]:
        """Get the diagnostics from the last estimation.

        Returns:
            Diagnostics if estimate() has been called, None otherwise
        """
        if self._results and self._results.diagnostics:
            return self._results.diagnostics
        return None

    def _is_dr_estimator(self) -> bool:
        """Check if this is a DR-based estimator.

        Returns:
            True if this is a DR variant, False otherwise
        """
        class_name = self.__class__.__name__
        return any(x in class_name for x in ["DR", "MRDR", "TMLE"])

    def _configure_oracle_augmentation(
        self, config: Union[str, bool, OracleSliceConfig, None]
    ) -> OracleSliceAugmentation:
        """Configure oracle slice augmentation based on settings and detected coverage.

        This is the canonical augmentation for all CJE estimators.

        Args:
            config: Configuration setting:
                - "auto": Auto-detect and enable if coverage < 100%
                - True: Always enable with default config
                - False/None: Disable
                - OracleSliceConfig: Use provided config

        Returns:
            Configured OracleSliceAugmentation instance
        """
        # Handle explicit OracleSliceConfig
        if isinstance(config, OracleSliceConfig):
            logger.info("Oracle slice augmentation configured with custom settings")
            return OracleSliceAugmentation(config)

        # Handle boolean/string config
        if config is False or config is None:
            # Explicitly disabled
            return OracleSliceAugmentation(OracleSliceConfig(enable_augmentation=False))

        # For "auto" or True, check if we should enable
        oracle_coverage = self.sampler.oracle_coverage

        if config == "auto":
            # Auto-detect: enable if we have partial oracle coverage
            if oracle_coverage is not None and 0 < oracle_coverage < 1.0:
                logger.info(
                    f"Oracle slice augmentation auto-enabled (coverage={oracle_coverage:.1%})"
                )
                return OracleSliceAugmentation(
                    OracleSliceConfig(enable_augmentation=True, enable_cross_fit=True)
                )
            else:
                # No oracle info or full coverage - disable
                if oracle_coverage == 1.0:
                    logger.debug("Oracle slice augmentation not needed (100% coverage)")
                elif oracle_coverage == 0.0:
                    logger.debug(
                        "Oracle slice augmentation disabled (no oracle labels)"
                    )
                return OracleSliceAugmentation(
                    OracleSliceConfig(enable_augmentation=False)
                )

        elif config is True:
            # Explicitly enabled
            if oracle_coverage is not None:
                logger.info(
                    f"Oracle slice augmentation enabled (coverage={oracle_coverage:.1%})"
                )
            else:
                logger.info("Oracle slice augmentation enabled (coverage unknown)")
            return OracleSliceAugmentation(
                OracleSliceConfig(enable_augmentation=True, enable_cross_fit=True)
            )

        # Should not reach here
        return OracleSliceAugmentation(OracleSliceConfig(enable_augmentation=False))

    def get_mhat(self, target_policy: str) -> Optional[np.ndarray]:
        """Get cached m̂(S) = E[W|S] for oracle augmentation.

        Args:
            target_policy: Name of the target policy

        Returns:
            m_hat: Estimated E[W|S] normalized to mean 1, or None if not fitted
        """
        return self.oracle_augmentation._m_hat_cache.get(target_policy)
