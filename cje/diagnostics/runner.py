"""Centralized diagnostic computation runner."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, TYPE_CHECKING
import numpy as np

from .suite import (
    DiagnosticSuite,
    WeightMetrics,
    EstimationSummary,
    StabilityMetrics,
    DRMetrics,
    RobustInference,
)
from ..utils.diagnostics import (
    hill_tail_index,
    effective_sample_size,
    compute_stability_diagnostics,
    compute_robust_inference,
)

if TYPE_CHECKING:
    from ..estimators.base_estimator import BaseCJEEstimator
    from ..data.models import EstimationResult


@dataclass
class DiagnosticConfig:
    """Configuration for diagnostic computation."""

    # Core diagnostics (always enabled)
    compute_weights: bool = True

    # Optional diagnostics
    check_stability: bool = True  # Changed to True - drift detection is important!
    check_dr_quality: bool = True  # Only for DR estimators
    compute_robust_se: bool = False  # Keep expensive bootstrap off by default

    # Parameters
    n_bootstrap: int = 4000
    bootstrap_method: str = "stationary"
    fdr_alpha: float = 0.05

    # Performance
    lazy_computation: bool = False  # Compute only when accessed
    cache_results: bool = True


class DiagnosticRunner:
    """Centralized diagnostic computation.

    This replaces scattered diagnostic computations throughout the codebase
    with a single, consistent interface.
    """

    def __init__(self, config: Optional[DiagnosticConfig] = None):
        """Initialize runner with configuration.

        Args:
            config: Diagnostic configuration (uses defaults if None)
        """
        self.config = config or DiagnosticConfig()
        self._cache: Optional[DiagnosticSuite] = None

    def run(
        self, estimator: "BaseCJEEstimator", result: "EstimationResult"
    ) -> DiagnosticSuite:
        """Run all configured diagnostics.

        Args:
            estimator: The estimator that produced the results
            result: Estimation results to diagnose

        Returns:
            Complete diagnostic suite
        """
        # Check cache if enabled
        if self.config.cache_results and self._cache is not None:
            return self._cache

        # Always compute core diagnostics
        weight_diagnostics = self._compute_weight_diagnostics(estimator, result)
        estimation_summary = self._summarize_estimation(estimator, result)

        # Initialize suite with core diagnostics
        suite = DiagnosticSuite(
            weight_diagnostics=weight_diagnostics,
            estimation_summary=estimation_summary,
        )

        # Conditional diagnostics
        if self.config.check_stability:
            try:
                suite.stability = self._compute_stability(estimator)
            except Exception as e:
                # Log but don't fail
                print(f"Warning: Stability diagnostics failed: {e}")

        if self.config.check_dr_quality and self._is_dr_estimator(estimator):
            try:
                suite.dr_quality = self._compute_dr_quality(result)
            except Exception as e:
                print(f"Warning: DR quality diagnostics failed: {e}")

        if self.config.compute_robust_se and self._has_influence_functions(result):
            try:
                suite.robust_inference = self._compute_robust_inference(result)
            except Exception as e:
                print(f"Warning: Robust inference failed: {e}")

        # Cache if enabled
        if self.config.cache_results:
            self._cache = suite

        return suite

    def _compute_weight_diagnostics(
        self, estimator: "BaseCJEEstimator", result: "EstimationResult"
    ) -> Dict[str, WeightMetrics]:
        """Compute weight diagnostics for all policies.

        This consolidates weight diagnostic computation that was previously
        scattered across multiple locations.
        """
        diagnostics = {}

        for policy in estimator.sampler.target_policies:
            # Get calibrated weights
            weights = estimator.get_weights(policy)

            if weights is None or len(weights) == 0:
                continue

            # Compute all weight metrics in one place
            metrics = WeightMetrics(
                ess=effective_sample_size(weights),
                max_weight=float(np.max(weights)),
                cv=(
                    float(np.std(weights) / np.mean(weights))
                    if np.mean(weights) > 0
                    else np.inf
                ),
                n_unique=len(np.unique(weights)),
            )

            # Compute Hill tail index (replaces tail_ratio_99_5)
            try:
                metrics.hill_index = hill_tail_index(weights)
            except Exception:
                metrics.hill_index = None

            diagnostics[policy] = metrics

        return diagnostics

    def _summarize_estimation(
        self, estimator: "BaseCJEEstimator", result: "EstimationResult"
    ) -> EstimationSummary:
        """Create estimation summary."""
        # Extract estimates
        estimates = {}
        for i, policy in enumerate(estimator.sampler.target_policies):
            estimates[policy] = result.estimates[i]

        # Extract standard errors if available
        standard_errors = None
        if result.standard_errors is not None:
            standard_errors = {}
            for i, policy in enumerate(estimator.sampler.target_policies):
                standard_errors[policy] = result.standard_errors[i]

        # Extract confidence intervals if available
        confidence_intervals = None
        if (
            hasattr(result, "confidence_intervals")
            and result.confidence_intervals is not None
        ):
            confidence_intervals = {}
            for i, policy in enumerate(estimator.sampler.target_policies):
                confidence_intervals[policy] = tuple(result.confidence_intervals[i])
        elif (
            hasattr(result, "robust_confidence_intervals")
            and result.robust_confidence_intervals is not None
        ):
            # Try robust CI field
            confidence_intervals = {}
            for i, policy in enumerate(estimator.sampler.target_policies):
                confidence_intervals[policy] = tuple(
                    result.robust_confidence_intervals[i]
                )

        return EstimationSummary(
            estimates=estimates,
            standard_errors=standard_errors,
            confidence_intervals=confidence_intervals,
            n_samples=len(estimator.sampler.dataset.samples),
            n_valid_samples=estimator.sampler.n_valid_samples,
            baseline_policy="clone",  # Default baseline policy
        )

    def _compute_stability(self, estimator: "BaseCJEEstimator") -> StabilityMetrics:
        """Compute stability diagnostics.

        This connects the orphaned stability module to the main pipeline.
        """
        # Call the existing stability diagnostic function
        stability_dict = compute_stability_diagnostics(
            estimator.sampler.dataset,
            judge_field="judge_score",
            oracle_field="oracle_label",
        )

        # Convert to structured metrics
        metrics = StabilityMetrics()

        if "drift_detection" in stability_dict:
            drift = stability_dict["drift_detection"]
            metrics.has_drift = drift.get("has_drift", False)
            metrics.max_tau_change = drift.get("max_tau_change")
            metrics.drift_policies = drift.get("policies_with_drift", [])

        if "calibration" in stability_dict:
            cal = stability_dict["calibration"]
            metrics.ece = cal.get("ece")
            metrics.reliability = cal.get("reliability")
            metrics.resolution = cal.get("resolution")
            metrics.uncertainty = cal.get("uncertainty")

        return metrics

    def _compute_dr_quality(self, result: "EstimationResult") -> DRMetrics:
        """Extract DR quality metrics from result."""
        # Extract from metadata (where DR estimators store them)
        orthogonality = result.metadata.get("orthogonality_scores", {})
        decomposition = result.metadata.get("dm_ips_decompositions", {})

        # Parse decomposition
        dm_contributions = {}
        ips_contributions = {}

        for policy, decomp in decomposition.items():
            if isinstance(decomp, dict):
                dm_contributions[policy] = decomp.get("dm_contribution", 0.0)
                ips_contributions[policy] = decomp.get("ips_augmentation", 0.0)

        # Extract orthogonality scores
        ortho_scores = {}
        for policy, ortho in orthogonality.items():
            if isinstance(ortho, dict):
                ortho_scores[policy] = ortho.get("score", 0.0)
            else:
                ortho_scores[policy] = float(ortho)

        return DRMetrics(
            orthogonality_scores=ortho_scores,
            dm_contributions=dm_contributions,
            ips_contributions=ips_contributions,
        )

    def _compute_robust_inference(self, result: "EstimationResult") -> RobustInference:
        """Compute robust standard errors and inference.

        This connects the orphaned robust_inference module to the main pipeline.
        """
        # Get influence functions - check primary location first
        influence_functions = None

        # Primary location: result.influence_functions (where DR stores them)
        if hasattr(result, "influence_functions") and result.influence_functions:
            influence_functions = result.influence_functions

        # Fallback: check metadata (legacy location)
        if not influence_functions:
            influence_functions = result.metadata.get("dr_influence", {})

        # Compute robust inference
        robust_dict = compute_robust_inference(
            estimates=result.estimates,
            influence_functions=influence_functions,
            policy_labels=(
                list(influence_functions.keys()) if influence_functions else []
            ),
            method=self.config.bootstrap_method,
            n_bootstrap=self.config.n_bootstrap,
            fdr_alpha=self.config.fdr_alpha,
        )

        # Convert to structured metrics
        return RobustInference(
            bootstrap_ses=robust_dict.get("bootstrap_ses", {}),
            bootstrap_cis=robust_dict.get("bootstrap_cis", {}),
            n_bootstrap=self.config.n_bootstrap,
            method=self.config.bootstrap_method,
            fdr_adjusted=robust_dict.get("fdr_adjusted"),
            fdr_alpha=self.config.fdr_alpha,
        )

    def _is_dr_estimator(self, estimator: "BaseCJEEstimator") -> bool:
        """Check if estimator is a DR variant."""
        # Check class name or inheritance
        class_name = estimator.__class__.__name__
        return any(x in class_name for x in ["DR", "MRDR", "TMLE"])

    def _has_influence_functions(self, result: "EstimationResult") -> bool:
        """Check if influence functions are available."""
        # Check in metadata first
        if "dr_influence" in result.metadata and result.metadata["dr_influence"]:
            return True

        # Check in result object
        if hasattr(result, "influence_functions") and result.influence_functions:
            return True

        return False

    def clear_cache(self) -> None:
        """Clear cached diagnostic suite."""
        self._cache = None
