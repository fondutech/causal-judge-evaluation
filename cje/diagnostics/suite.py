"""Unified diagnostic suite - single source of truth for all diagnostics."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class WeightMetrics:
    """Metrics for importance weight diagnostics."""

    ess: float
    max_weight: float
    hill_index: Optional[float] = None
    cv: Optional[float] = None  # Coefficient of variation
    n_unique: Optional[int] = None

    @property
    def has_heavy_tails(self) -> bool:
        """Check if distribution has heavy tails (α < 2)."""
        return self.hill_index is not None and self.hill_index < 2.0

    @property
    def has_marginal_tails(self) -> bool:
        """Check if distribution has marginal tails (2 ≤ α < 2.5)."""
        return self.hill_index is not None and 2.0 <= self.hill_index < 2.5


@dataclass
class EstimationSummary:
    """Summary of estimation results."""

    estimates: Dict[str, float]
    standard_errors: Optional[Dict[str, float]] = None
    confidence_intervals: Optional[Dict[str, tuple]] = None
    n_samples: int = 0
    n_valid_samples: int = 0
    baseline_policy: str = "clone"

    @property
    def data_efficiency(self) -> float:
        """Proportion of samples used in estimation."""
        if self.n_samples > 0:
            return self.n_valid_samples / self.n_samples
        return 0.0


@dataclass
class StabilityMetrics:
    """Metrics for temporal stability and calibration."""

    has_drift: bool = False
    max_tau_change: Optional[float] = None
    drift_policies: List[str] = field(default_factory=list)
    ece: Optional[float] = None  # Expected calibration error
    reliability: Optional[float] = None
    resolution: Optional[float] = None
    uncertainty: Optional[float] = None

    @property
    def needs_recalibration(self) -> bool:
        """Check if recalibration is recommended."""
        return self.has_drift or (self.ece is not None and self.ece > 0.1)


@dataclass
class DRMetrics:
    """Metrics specific to doubly robust estimation."""

    orthogonality_scores: Dict[str, float]
    dm_contributions: Dict[str, float]
    ips_contributions: Dict[str, float]
    outcome_model_r2: Optional[Dict[str, float]] = None

    @property
    def max_orthogonality_violation(self) -> float:
        """Maximum orthogonality score across policies."""
        if self.orthogonality_scores:
            return max(abs(s) for s in self.orthogonality_scores.values())
        return 0.0

    @property
    def is_orthogonal(self) -> bool:
        """Check if orthogonality is satisfied (< 0.01)."""
        return self.max_orthogonality_violation < 0.01


@dataclass
class RobustInference:
    """Robust inference results."""

    bootstrap_ses: Dict[str, float]
    bootstrap_cis: Dict[str, tuple]
    n_bootstrap: int = 4000
    method: str = "stationary"
    fdr_adjusted: Optional[Dict[str, bool]] = None
    fdr_alpha: float = 0.05

    @property
    def n_significant_fdr(self) -> int:
        """Number of policies significant after FDR correction."""
        if self.fdr_adjusted:
            return sum(self.fdr_adjusted.values())
        return 0


@dataclass
class DiagnosticSuite:
    """Single source of truth for all diagnostics.

    This consolidates all diagnostic information that was previously
    scattered across metadata, diagnostic objects, and internal caches.
    """

    # Always computed
    weight_diagnostics: Dict[str, WeightMetrics]
    estimation_summary: EstimationSummary

    # Conditionally computed
    stability: Optional[StabilityMetrics] = None
    dr_quality: Optional[DRMetrics] = None
    robust_inference: Optional[RobustInference] = None

    # Metadata
    timestamp: Optional[str] = None

    @property
    def has_issues(self) -> bool:
        """Quick check if any issues detected."""
        # Check heuristics
        min_ess = min(w.ess for w in self.weight_diagnostics.values())
        if min_ess < 100:
            return True

        # Check for heavy tails
        if any(w.has_heavy_tails for w in self.weight_diagnostics.values()):
            return True

        # Check stability
        if self.stability and self.stability.has_drift:
            return True

        # Check DR quality
        if self.dr_quality and not self.dr_quality.is_orthogonal:
            return True

        return False

    def get_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on diagnostics."""
        recs = []

        # Weight diagnostics
        for policy, metrics in self.weight_diagnostics.items():
            if metrics.ess < 100:
                recs.append(f"⚠️ {policy}: Critically low ESS ({metrics.ess:.0f})")
                recs.append(f"   → Increase sample size or use less extreme policies")
            elif metrics.ess < 500:
                recs.append(f"⚠️ {policy}: Marginal ESS ({metrics.ess:.0f})")
                recs.append(f"   → Consider increasing sample size")

            if metrics.has_heavy_tails:
                recs.append(
                    f"⚠️ {policy}: Heavy tails detected (α={metrics.hill_index:.2f})"
                )
                recs.append(f"   → Use DR estimation or enable variance regularization")
            elif metrics.has_marginal_tails:
                recs.append(
                    f"⚠️ {policy}: Marginal tail behavior (α={metrics.hill_index:.2f})"
                )
                recs.append(f"   → Monitor variance, consider DR if unstable")

        # Stability issues
        if self.stability:
            if self.stability.has_drift:
                recs.append(
                    f"⚠️ Judge drift detected (Δτ={self.stability.max_tau_change:.3f})"
                )
                recs.append(f"   → Refresh oracle labels or retrain judge")

            if self.stability.ece and self.stability.ece > 0.1:
                recs.append(f"⚠️ Poor calibration (ECE={self.stability.ece:.3f})")
                recs.append(f"   → Recalibrate with fresh oracle labels")

        # DR quality issues
        if self.dr_quality:
            if not self.dr_quality.is_orthogonal:
                max_violation = self.dr_quality.max_orthogonality_violation
                recs.append(f"⚠️ Orthogonality violation ({max_violation:.4f})")
                recs.append(f"   → Check for data leakage or model misspecification")

        # Robust inference
        if self.robust_inference and self.robust_inference.fdr_adjusted:
            n_sig = self.robust_inference.n_significant_fdr
            n_total = len(self.robust_inference.fdr_adjusted)
            if n_sig == 0:
                recs.append(f"ℹ️ No policies significant after FDR correction")
                recs.append(
                    f"   → Consider larger sample size or stronger interventions"
                )
            else:
                recs.append(
                    f"✅ {n_sig}/{n_total} policies significant (FDR α={self.robust_inference.fdr_alpha})"
                )

        return recs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "weight_diagnostics": {
                policy: {
                    "ess": m.ess,
                    "max_weight": m.max_weight,
                    "hill_index": m.hill_index,
                    "cv": m.cv,
                    "n_unique": m.n_unique,
                    "has_heavy_tails": m.has_heavy_tails,
                }
                for policy, m in self.weight_diagnostics.items()
            },
            "estimation_summary": {
                "estimates": self.estimation_summary.estimates,
                "standard_errors": self.estimation_summary.standard_errors,
                "confidence_intervals": self.estimation_summary.confidence_intervals,
                "n_samples": self.estimation_summary.n_samples,
                "n_valid_samples": self.estimation_summary.n_valid_samples,
                "data_efficiency": self.estimation_summary.data_efficiency,
            },
            "has_issues": self.has_issues,
        }

        if self.stability:
            result["stability"] = {
                "has_drift": self.stability.has_drift,
                "max_tau_change": self.stability.max_tau_change,
                "ece": self.stability.ece,
                "needs_recalibration": self.stability.needs_recalibration,
            }

        if self.dr_quality:
            result["dr_quality"] = {
                "max_orthogonality_violation": self.dr_quality.max_orthogonality_violation,
                "is_orthogonal": self.dr_quality.is_orthogonal,
                "dm_contributions": self.dr_quality.dm_contributions,
                "ips_contributions": self.dr_quality.ips_contributions,
            }

        if self.robust_inference:
            result["robust_inference"] = {
                "bootstrap_ses": self.robust_inference.bootstrap_ses,
                "bootstrap_cis": self.robust_inference.bootstrap_cis,
                "n_bootstrap": self.robust_inference.n_bootstrap,
                "method": self.robust_inference.method,
                "n_significant_fdr": self.robust_inference.n_significant_fdr,
            }

        return result

    def to_summary(self) -> str:
        """Generate concise human-readable summary."""
        lines = []
        lines.append("DIAGNOSTIC SUMMARY")
        lines.append("=" * 60)

        # Data overview
        lines.append(
            f"\n📊 Data: {self.estimation_summary.n_valid_samples}/{self.estimation_summary.n_samples} samples used"
        )
        lines.append(f"   Efficiency: {self.estimation_summary.data_efficiency:.1%}")

        # Weight diagnostics
        lines.append("\n📈 Weight Diagnostics:")
        for policy, metrics in self.weight_diagnostics.items():
            status = "✅" if metrics.ess > 500 else "⚠️" if metrics.ess > 100 else "❌"
            tail_warning = " ⚠️ HEAVY TAILS" if metrics.has_heavy_tails else ""
            lines.append(
                f"  {status} {policy}: ESS={metrics.ess:.0f}, α={metrics.hill_index:.2f}{tail_warning}"
            )

        # Stability
        if self.stability:
            status = "❌" if self.stability.has_drift else "✅"
            lines.append(f"\n🔄 Stability: {status}")
            if self.stability.has_drift:
                lines.append(
                    f"   Drift detected: Δτ={self.stability.max_tau_change:.3f}"
                )
            if self.stability.ece:
                lines.append(f"   Calibration ECE: {self.stability.ece:.3f}")

        # DR quality
        if self.dr_quality:
            status = "✅" if self.dr_quality.is_orthogonal else "❌"
            lines.append(f"\n🎯 DR Quality: {status}")
            lines.append(
                f"   Max orthogonality: {self.dr_quality.max_orthogonality_violation:.4f}"
            )

        # Gates

        # Overall assessment
        overall = (
            "✅ No issues detected"
            if not self.has_issues
            else "⚠️ Issues require attention"
        )
        lines.append(f"\n{overall}")

        return "\n".join(lines)
