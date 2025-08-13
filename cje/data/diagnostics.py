"""
Diagnostic data models for CJE.

This module contains only the data structures for diagnostics.
Computation logic is in utils/diagnostics/.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum


class Status(Enum):
    """Health status for diagnostics."""
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Diagnostics:
    """Unified diagnostics container for CJE estimation.
    
    A flat structure containing all diagnostic information.
    Optional fields are None when not applicable to the estimator type.
    """
    
    # ========== Core Info (always present) ==========
    estimator_type: str
    method: str
    n_samples_total: int
    n_samples_valid: int
    n_policies: int
    policies: List[str]
    
    # ========== Estimation Results (always present) ==========
    estimates: Dict[str, float]
    standard_errors: Dict[str, float]
    n_samples_used: Dict[str, int]
    
    # ========== Weight Diagnostics (IPS/CalibratedIPS/DR) ==========
    weight_ess: Optional[float] = None  # Overall effective sample size fraction
    weight_status: Optional[Status] = None
    
    # Per-policy weight metrics
    ess_per_policy: Optional[Dict[str, float]] = None
    max_weight_per_policy: Optional[Dict[str, float]] = None
    weight_tail_ratio_per_policy: Optional[Dict[str, float]] = None  # p99/p5
    
    # ========== Calibration Diagnostics ==========
    calibration_rmse: Optional[float] = None
    calibration_r2: Optional[float] = None
    calibration_coverage_01: Optional[float] = None  # P(|pred - oracle| < 0.1)
    n_oracle_labels: Optional[int] = None
    
    # ========== DR Diagnostics ==========
    dr_cross_fitted: Optional[bool] = None
    dr_n_folds: Optional[int] = None
    
    # Per-policy DR metrics
    outcome_r2_per_policy: Optional[Dict[str, float]] = None
    outcome_rmse_per_policy: Optional[Dict[str, float]] = None
    if_tail_ratio_per_policy: Optional[Dict[str, float]] = None
    score_pvalue_per_policy: Optional[Dict[str, float]] = None  # For TMLE
    
    # ========== Computed Properties ==========
    
    @property
    def filter_rate(self) -> float:
        """Fraction of samples filtered out."""
        if self.n_samples_total > 0:
            return 1.0 - (self.n_samples_valid / self.n_samples_total)
        return 0.0
    
    @property
    def best_policy(self) -> str:
        """Policy with highest estimate."""
        if not self.estimates:
            return "none"
        return max(self.estimates.items(), key=lambda x: x[1])[0]
    
    @property
    def worst_weight_tail_ratio(self) -> float:
        """Worst tail ratio across policies."""
        if self.weight_tail_ratio_per_policy:
            return max(self.weight_tail_ratio_per_policy.values())
        return 0.0
    
    @property
    def overall_status(self) -> Status:
        """Overall health status based on all diagnostics."""
        statuses = []
        
        # Weight status
        if self.weight_status:
            statuses.append(self.weight_status)
        
        # DR status based on outcome R²
        if self.outcome_r2_per_policy:
            min_r2 = min(self.outcome_r2_per_policy.values())
            if min_r2 < 0:
                statuses.append(Status.CRITICAL)
            elif min_r2 < 0.1:
                statuses.append(Status.WARNING)
        
        # IF tail status
        if self.if_tail_ratio_per_policy:
            max_tail = max(self.if_tail_ratio_per_policy.values())
            if max_tail > 1000:
                statuses.append(Status.CRITICAL)
            elif max_tail > 100:
                statuses.append(Status.WARNING)
        
        # Return worst status
        if not statuses:
            return Status.GOOD
        if Status.CRITICAL in statuses:
            return Status.CRITICAL
        if Status.WARNING in statuses:
            return Status.WARNING
        return Status.GOOD
    
    def validate(self) -> List[str]:
        """Run self-consistency checks."""
        issues = []
        
        # Basic sanity checks
        if self.n_samples_valid > self.n_samples_total:
            issues.append(
                f"n_valid ({self.n_samples_valid}) > n_total ({self.n_samples_total})"
            )
        
        # ESS should be <= 1
        if self.weight_ess is not None and self.weight_ess > 1.0:
            issues.append(f"ESS fraction > 1.0: {self.weight_ess}")
        
        if self.ess_per_policy:
            for policy, ess in self.ess_per_policy.items():
                if ess > 1.0:
                    issues.append(f"ESS fraction > 1.0 for {policy}: {ess}")
        
        # R² should be <= 1
        if self.calibration_r2 is not None and self.calibration_r2 > 1.0:
            issues.append(f"Calibration R² > 1.0: {self.calibration_r2}")
        
        if self.outcome_r2_per_policy:
            for policy, r2 in self.outcome_r2_per_policy.items():
                if r2 > 1.0:
                    issues.append(f"Outcome R² > 1.0 for {policy}: {r2}")
        
        # Check estimates match policies
        for policy in self.estimates:
            if policy not in self.policies and policy != "base":
                issues.append(f"Estimate for unknown policy: {policy}")
        
        return issues
    
    def summary(self) -> str:
        """Generate concise summary."""
        lines = [
            f"Estimator: {self.estimator_type}",
            f"Method: {self.method}",
            f"Status: {self.overall_status.value}",
            f"Samples: {self.n_samples_valid}/{self.n_samples_total} valid ({100*(1-self.filter_rate):.1f}%)",
            f"Policies: {', '.join(self.policies)}",
            f"Best policy: {self.best_policy}",
        ]
        
        # Add key metrics if available
        if self.weight_ess is not None:
            lines.append(f"Weight ESS: {self.weight_ess:.1%}")
        
        if self.calibration_rmse is not None:
            lines.append(f"Calibration RMSE: {self.calibration_rmse:.3f}")
        
        if self.outcome_r2_per_policy:
            r2_values = list(self.outcome_r2_per_policy.values())
            lines.append(f"Outcome R²: [{min(r2_values):.3f}, {max(r2_values):.3f}]")
        
        # Add any validation issues
        issues = self.validate()
        if issues:
            lines.append("Issues: " + "; ".join(issues[:2]))  # Show first 2 issues
        
        return " | ".join(lines)
    
    def to_dict(self) -> Dict:
        """Export as dictionary for serialization."""
        from dataclasses import asdict
        d = asdict(self)
        # Convert enums to strings
        if self.weight_status:
            d["weight_status"] = self.weight_status.value
        d["overall_status"] = self.overall_status.value
        return d