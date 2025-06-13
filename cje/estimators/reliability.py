"""
Reliability assessment for CJE confidence intervals.

This module provides structured metadata and guard-rails to detect
unreliable confidence intervals and suggest corrective actions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .results import EstimationResult


@dataclass
class EstimatorMetadata:
    """Structured metadata for estimator diagnostics."""

    # Core estimator settings
    estimator_type: str
    clip_threshold: Optional[float] = None
    k_folds: Optional[int] = None
    normalize_weights: bool = True
    stabilize_weights: bool = True

    # Weight statistics
    ess_values: Optional[List[float]] = None  # ESS per policy
    ess_percentage: Optional[float] = None  # Average ESS as % of n
    n_clipped: int = 0  # Number of clipped weights
    clip_fraction: float = 0.0  # Fraction of weights clipped
    weight_range: Optional[Tuple[float, float]] = None  # (min, max) final weights

    # Calibration quality (when applicable)
    calibration_range: Optional[float] = None  # Range of calibrated scores
    calibration_collapsed: bool = False  # Whether calibration collapsed

    # Additional diagnostic info
    stabilization_applied: bool = False  # Whether weights were stabilized
    bootstrap_available: bool = False  # Whether bootstrap CIs can be computed
    calibrate_weights: bool = True  # Whether weight calibration was enabled
    calibrate_outcome: bool = True  # Whether outcome calibration was enabled
    runtime_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                elif (
                    isinstance(value, (list, tuple))
                    and value
                    and isinstance(value[0], (np.floating, np.integer))
                ):
                    result[key] = [float(v) for v in value]
                elif isinstance(value, (np.floating, np.integer)):
                    result[key] = float(value)
                elif isinstance(value, (np.bool_, bool)):
                    result[key] = bool(value)
                else:
                    result[key] = value
        return result


@dataclass
class ReliabilityAssessment:
    """Assessment of confidence interval reliability."""

    rating: str  # 'high', 'moderate', 'low', 'unreliable'
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    warning_level: str = "info"  # 'info', 'warning', 'error'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "rating": self.rating,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "warning_level": self.warning_level,
        }


def assess_ci_reliability(
    result: "EstimationResult",
    bootstrap_results: Optional[Dict[str, Any]] = None,
    metadata: Optional[EstimatorMetadata] = None,
) -> ReliabilityAssessment:
    """
    Assess reliability of confidence intervals and provide recommendations.

    Args:
        result: EstimationResult containing estimates and standard errors
        bootstrap_results: Bootstrap CI results if available
        metadata: Structured estimator metadata

    Returns:
        ReliabilityAssessment with rating, issues, and recommendations
    """
    from .results import EstimationResult  # Avoid circular import

    issues = []
    recommendations = []
    warning_level = "info"

    # 1. Zero variance check (most critical)
    min_se = np.min(result.se)
    if min_se == 0 or np.max(result.se) - min_se < 1e-12:
        issues.append("zero_variance")
        recommendations.append(
            "Check for degenerate estimation (all rewards identical)"
        )
        warning_level = "error"

    # 2. ESS/overlap assessment
    if metadata and metadata.ess_percentage is not None:
        ess_pct = metadata.ess_percentage
        if ess_pct < 10:
            issues.append(f"poor_overlap (ESS {ess_pct:.1f}%)")
            recommendations.append(
                "Improve policy overlap: collect more diverse data or use closer target policies"
            )
            warning_level = max(warning_level, "error" if ess_pct < 5 else "warning")
        elif ess_pct < 30:
            issues.append(f"moderate_overlap (ESS {ess_pct:.1f}%)")
            recommendations.append("Consider collecting more data to improve overlap")
            warning_level = max(warning_level, "warning")

    # 3. Heavy clipping detection
    if metadata and metadata.clip_fraction > 0.2:
        issues.append(f"heavy_clipping ({metadata.clip_fraction*100:.1f}% of weights)")
        recommendations.append(
            f"Reduce clip threshold (current: {metadata.clip_threshold}) or investigate weight outliers"
        )
        warning_level = max(warning_level, "warning")

    # 4. Calibration collapse
    if metadata and metadata.calibration_collapsed:
        issues.append("calibration_collapse")
        recommendations.append(
            "Calibration failed - using fallback method. Collect more diverse oracle labels."
        )
        warning_level = max(warning_level, "error")

    # 5. Small sample + poor conditions
    if result.n < 50 and (
        metadata and metadata.ess_percentage and metadata.ess_percentage < 20
    ):
        issues.append(
            f"small_sample_poor_overlap (n={result.n}, ESS {metadata.ess_percentage:.1f}%)"
        )
        recommendations.append(
            "Collect at least 100 samples and improve policy overlap"
        )
        warning_level = max(warning_level, "error")
    elif result.n < 30:
        issues.append(f"very_small_sample (n={result.n})")
        recommendations.append("Collect at least 50-100 samples for reliable inference")
        warning_level = max(warning_level, "warning")

    # 6. Bootstrap vs analytical discrepancy
    if bootstrap_results and bootstrap_results.get("available"):
        analytical_se = result.se

        # Extract bootstrap standard errors carefully
        bootstrap_se = None
        if "standard_errors" in bootstrap_results:
            if isinstance(bootstrap_results["standard_errors"], dict):
                bootstrap_se = np.array(
                    list(bootstrap_results["standard_errors"].values())
                )
            else:
                bootstrap_se = np.array(bootstrap_results["standard_errors"])
        elif "bootstrap_se" in bootstrap_results:
            bootstrap_se = np.array(bootstrap_results["bootstrap_se"])

        if bootstrap_se is not None and len(bootstrap_se) == len(analytical_se):
            # Compute relative differences
            se_ratios = bootstrap_se / np.maximum(analytical_se, 1e-12)
            max_discrepancy = np.max(np.abs(se_ratios - 1.0))

            if max_discrepancy > 0.5:  # 50% difference
                issues.append(
                    f"bootstrap_analytical_discrepancy ({max_discrepancy*100:.0f}% difference)"
                )
                recommendations.append(
                    "Large bootstrap vs analytical discrepancy - rely on bootstrap CIs for small samples"
                )
                warning_level = max(warning_level, "warning")

    # 7. Weight stabilization recommendations
    if metadata and not metadata.stabilize_weights and len(issues) > 0:
        recommendations.append(
            "Enable weight stabilization to improve numerical stability"
        )

    # 8. General recommendations for poor conditions
    if result.n < 100 and result.eif_components is not None:
        recommendations.append(
            "Bootstrap CIs automatically enabled for small sample - use these for robustness"
        )

    # Determine overall rating
    if (
        "zero_variance" in [issue.split()[0] for issue in issues]
        or "calibration_collapse" in issues
    ):
        rating = "unreliable"
    elif any(warning_level == "error" for issue in issues) or "poor_overlap" in str(
        issues
    ):
        rating = "low"
    elif any(warning_level == "warning" for issue in issues) or result.n < 50:
        rating = "moderate"
    else:
        rating = "high"

    return ReliabilityAssessment(
        rating=rating,
        issues=issues,
        recommendations=recommendations,
        warning_level=warning_level,
    )


def create_reliability_warning_message(assessment: ReliabilityAssessment) -> str:
    """Create a Rich-formatted warning message for console display."""

    if assessment.rating == "high":
        return ""  # No warning needed

    # Header based on severity
    if assessment.rating == "unreliable":
        header = "🔴 Unreliable confidence intervals detected!"
        style = "red"
    elif assessment.rating == "low":
        header = "🟡 Low reliability confidence intervals detected!"
        style = "yellow"
    else:  # moderate
        header = "🔵 Moderate reliability confidence intervals"
        style = "blue"

    # Format issues
    issues_text = ""
    if assessment.issues:
        issues_text = "\nIssues detected:\n"
        for issue in assessment.issues:
            issues_text += f"  • {issue.replace('_', ' ').title()}\n"

    # Format recommendations
    recs_text = ""
    if assessment.recommendations:
        recs_text = "\nRecommended actions:\n"
        for rec in assessment.recommendations:
            recs_text += f"  ✓ {rec}\n"

    return f"{header}{issues_text}{recs_text}"
