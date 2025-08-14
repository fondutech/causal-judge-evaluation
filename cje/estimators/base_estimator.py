"""Base class for CJE estimators."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, TYPE_CHECKING
import numpy as np

from ..data.models import Dataset, EstimationResult
from ..data.precomputed_sampler import PrecomputedSampler

if TYPE_CHECKING:
    from ..diagnostics import DiagnosticSuite, DiagnosticConfig


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
        run_diagnostics: bool = True,
        diagnostic_config: Optional["DiagnosticConfig"] = None,
        run_gates: bool = False,
        gate_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize estimator.

        Args:
            sampler: Data sampler with precomputed log probabilities
            run_diagnostics: Whether to compute diagnostics (default True)
            diagnostic_config: Configuration for diagnostics (uses defaults if None)
            run_gates: Whether to run automated gates (default False)
            gate_config: Configuration for diagnostic gates
        """
        self.sampler = sampler
        self.run_diagnostics = run_diagnostics
        self.diagnostic_config = diagnostic_config
        self.run_gates = run_gates
        self.gate_config = gate_config or {}
        self._fitted = False
        self._weights_cache: Dict[str, np.ndarray] = {}
        self._influence_functions: Dict[str, np.ndarray] = {}
        self._results: Optional[EstimationResult] = None
        self._diagnostic_suite: Optional["DiagnosticSuite"] = None

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

        # Run unified diagnostics if enabled
        if self.run_diagnostics and result is not None:
            from ..diagnostics import DiagnosticRunner, DiagnosticConfig

            # Create config if not provided
            if self.diagnostic_config is None:
                self.diagnostic_config = DiagnosticConfig(
                    run_gates=self.run_gates,
                    gate_config=self.gate_config,
                )

            # Run diagnostics
            runner = DiagnosticRunner(self.diagnostic_config)
            self._diagnostic_suite = runner.run(self, result)

            # Store in result for access
            result.diagnostic_suite = self._diagnostic_suite

            # Populate legacy fields for backward compatibility
            if hasattr(result, "diagnostics") and result.diagnostics:
                # Keep existing diagnostics (IPSDiagnostics/DRDiagnostics)
                pass

        # Legacy gate running (if diagnostics disabled but gates enabled)
        elif self.run_gates and result is not None:
            result = self._run_diagnostic_gates(result)

        return result

    def _run_diagnostic_gates(self, result: EstimationResult) -> EstimationResult:
        """Run diagnostic gates and add report to result.

        Args:
            result: Estimation result to check

        Returns:
            Updated result with gate report
        """
        from ..utils.diagnostics import run_diagnostic_gates

        # Collect diagnostics for gates
        diagnostics: Dict[str, Any] = {}

        # Add weight diagnostics if available
        if (
            hasattr(result.diagnostics, "ess_per_policy")
            and result.diagnostics.ess_per_policy
        ):
            weight_diags = {}
            for policy in result.diagnostics.ess_per_policy:
                weight_diags[policy] = {
                    "ess": result.diagnostics.ess_per_policy.get(policy, 0)
                    * result.diagnostics.n_samples_valid,
                    "tail_index": result.metadata.get("tail_indices", {}).get(policy),
                    "max_weight": result.diagnostics.max_weight_per_policy.get(policy),
                }
            diagnostics["weight_diagnostics"] = weight_diags

        # Add orthogonality scores and other DR diagnostics
        if "orthogonality_scores" in result.metadata:
            diagnostics["orthogonality_scores"] = result.metadata[
                "orthogonality_scores"
            ]

        # Add number of policies
        diagnostics["n_policies"] = len(self.sampler.target_policies)

        # Add FDR results if present
        if "fdr_results" in result.metadata:
            diagnostics["fdr_results"] = result.metadata["fdr_results"]

        # Run gates
        gate_report = run_diagnostic_gates(
            diagnostics,
            config=self.gate_config,
            verbose=False,  # Don't print here, let caller decide
        )

        # Store gate report
        result.gate_report = gate_report.to_dict()

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

    def get_diagnostic_suite(self) -> Optional["DiagnosticSuite"]:
        """Get the unified diagnostic suite from the last estimation.

        Returns:
            DiagnosticSuite if computed, None otherwise
        """
        return self._diagnostic_suite
