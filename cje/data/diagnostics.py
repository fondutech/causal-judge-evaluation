"""
Diagnostic data models for CJE.

This module contains the data structures for diagnostics.
Computation logic is in utils/diagnostics/.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Tuple
from enum import Enum
import numpy as np


class Status(Enum):
    """Health status for diagnostics."""

    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class IPSDiagnostics:
    """Diagnostics for IPS-based estimators (RawIPS and CalibratedIPS)."""

    # ========== Core Info (always present) ==========
    estimator_type: str  # "RawIPS" or "CalibratedIPS"
    method: str
    n_samples_total: int
    n_samples_valid: int
    n_policies: int
    policies: List[str]

    # ========== Estimation Results (always present) ==========
    estimates: Dict[str, float]
    standard_errors: Dict[str, float]
    n_samples_used: Dict[str, int]

    # ========== Weight Diagnostics (always present) ==========
    weight_ess: float  # Overall effective sample size fraction
    weight_status: Status

    # Per-policy weight metrics
    ess_per_policy: Dict[str, float]
    max_weight_per_policy: Dict[str, float]
    status_per_policy: Optional[Dict[str, Status]] = None  # Per-policy status
    weight_tail_ratio_per_policy: Optional[Dict[str, float]] = (
        None  # DEPRECATED: Use tail_indices
    )
    tail_indices: Optional[Dict[str, Optional[float]]] = (
        None  # Hill tail index per policy
    )

    # ========== Calibration Diagnostics (None for RawIPS) ==========
    calibration_rmse: Optional[float] = None
    calibration_r2: Optional[float] = None
    calibration_coverage: Optional[float] = None  # P(|pred - oracle| < 0.1)
    n_oracle_labels: Optional[int] = None

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
        """Worst tail ratio across policies.

        DEPRECATED: Use worst_tail_index instead.
        """
        if self.weight_tail_ratio_per_policy:
            return max(self.weight_tail_ratio_per_policy.values())
        return 0.0

    @property
    def worst_tail_index(self) -> Optional[float]:
        """Lowest (worst) Hill tail index across policies."""
        if self.tail_indices:
            valid_indices = [
                idx for idx in self.tail_indices.values() if idx is not None
            ]
            if valid_indices:
                return min(valid_indices)
        return None

    @property
    def is_calibrated(self) -> bool:
        """Check if this has calibration info."""
        return self.calibration_rmse is not None

    @property
    def overall_status(self) -> Status:
        """Overall health status based on diagnostics."""
        # Start with weight status
        if self.weight_status == Status.CRITICAL:
            return Status.CRITICAL
        elif self.weight_status == Status.WARNING:
            return Status.WARNING

        # Check calibration if present
        if self.is_calibrated:
            if self.calibration_r2 is not None and self.calibration_r2 < 0:
                return Status.CRITICAL
            elif self.calibration_r2 is not None and self.calibration_r2 < 0.5:
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

        # Check for high filter rate
        if self.filter_rate > 0.5:
            issues.append(
                f"High filter rate: {self.filter_rate:.1%} of samples filtered"
            )

        # ESS should be <= 1 and check for low ESS
        if self.weight_ess > 1.0:
            issues.append(f"ESS fraction > 1.0: {self.weight_ess}")
        elif self.weight_ess < 0.1:
            issues.append(f"Very low ESS: {self.weight_ess:.1%}")

        for policy, ess in self.ess_per_policy.items():
            if ess > 1.0:
                issues.append(f"ESS fraction > 1.0 for {policy}: {ess}")
            elif ess < 0.1:
                issues.append(f"Low ESS for {policy}: {ess:.1%}")

        # Check for extreme weights
        for policy, max_w in self.max_weight_per_policy.items():
            if max_w > 100:
                issues.append(f"Extreme max weight for {policy}: {max_w:.1f}")

        # Check for heavy tails using Hill index
        if self.tail_indices:
            for policy, tail_idx in self.tail_indices.items():
                if tail_idx is not None:
                    if tail_idx < 1.5:
                        issues.append(
                            f"Extremely heavy tail for {policy}: α={tail_idx:.2f} (infinite mean risk)"
                        )
                    elif tail_idx < 2.0:
                        issues.append(
                            f"Heavy tail for {policy}: α={tail_idx:.2f} (infinite variance)"
                        )
        # Fallback to deprecated tail ratio if available
        elif self.weight_tail_ratio_per_policy:
            for policy, tail_ratio in self.weight_tail_ratio_per_policy.items():
                if tail_ratio > 100:
                    issues.append(f"Heavy tail for {policy}: ratio={tail_ratio:.1f}")

        # R² should be <= 1
        if self.calibration_r2 is not None and self.calibration_r2 > 1.0:
            issues.append(f"Calibration R² > 1.0: {self.calibration_r2}")

        # Check estimates match policies
        for policy in self.estimates:
            if policy not in self.policies:
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
            f"Weight ESS: {self.weight_ess:.1%}",
        ]

        if self.is_calibrated:
            lines.append(f"Calibration RMSE: {self.calibration_rmse:.3f}")
            lines.append(f"Calibration R²: {self.calibration_r2:.3f}")

        # Add any validation issues
        issues = self.validate()
        if issues:
            lines.append("Issues: " + "; ".join(issues[:2]))

        return " | ".join(lines)

    def to_dict(self) -> Dict:
        """Export as dictionary for serialization."""
        from dataclasses import asdict

        d = asdict(self)
        # Convert enums to strings
        d["weight_status"] = self.weight_status.value
        d["overall_status"] = self.overall_status.value
        return d

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict) -> "IPSDiagnostics":
        """Create from dictionary."""
        # Convert status strings back to enum
        if "weight_status" in data and isinstance(data["weight_status"], str):
            data["weight_status"] = Status(data["weight_status"])
        # Remove computed fields that aren't in the constructor
        data.pop("overall_status", None)
        return cls(**data)

    def to_csv_row(self) -> Dict[str, Any]:
        """Export key metrics as a flat dict for CSV export."""
        row = {
            "estimator": self.estimator_type,
            "method": self.method,
            "n_samples_total": self.n_samples_total,
            "n_samples_valid": self.n_samples_valid,
            "filter_rate": self.filter_rate,
            "weight_ess": self.weight_ess,
            "weight_status": self.weight_status.value,
            "n_policies": self.n_policies,
            "best_policy": self.best_policy if self.policies else None,
            "worst_tail_ratio": self.worst_weight_tail_ratio,
        }
        # Add per-policy metrics
        for policy in self.policies:
            row[f"{policy}_estimate"] = self.estimates.get(policy)
            row[f"{policy}_se"] = self.standard_errors.get(policy)
            row[f"{policy}_ess"] = self.ess_per_policy.get(policy)
        # Add calibration metrics if available
        if self.calibration_rmse is not None:
            row["calibration_rmse"] = self.calibration_rmse
            row["calibration_r2"] = self.calibration_r2
        return row


@dataclass
class DRDiagnostics(IPSDiagnostics):
    """Diagnostics for DR estimators, extending IPS diagnostics."""

    # ========== DR-specific fields ==========
    dr_cross_fitted: bool = True
    dr_n_folds: int = 5

    # Outcome model performance summary
    outcome_r2_range: Tuple[float, float] = (0.0, 0.0)  # (min, max) across policies
    outcome_rmse_mean: float = 0.0  # Average RMSE across policies

    # Influence function summary
    worst_if_tail_ratio: float = 0.0  # Worst p99/p5 ratio across policies

    # Detailed per-policy diagnostics (for visualization and debugging)
    dr_diagnostics_per_policy: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Optional influence functions (can be large)
    influence_functions: Optional[Dict[str, np.ndarray]] = None

    # ========== Computed Properties (override parent) ==========

    @property
    def overall_status(self) -> Status:
        """Overall health status including DR-specific checks."""
        # Start with parent status
        parent_status = super().overall_status
        if parent_status == Status.CRITICAL:
            return Status.CRITICAL

        statuses: List[Status] = [parent_status]

        # Check outcome model R²
        min_r2, max_r2 = self.outcome_r2_range
        if min_r2 < 0:
            statuses.append(Status.CRITICAL)
        elif min_r2 < 0.1:
            statuses.append(Status.WARNING)

        # Check influence function tails
        if self.worst_if_tail_ratio > 1000:
            statuses.append(Status.CRITICAL)
        elif self.worst_if_tail_ratio > 100:
            statuses.append(Status.WARNING)

        # Return worst status
        if Status.CRITICAL in statuses:
            return Status.CRITICAL
        if Status.WARNING in statuses:
            return Status.WARNING
        return Status.GOOD

    def validate(self) -> List[str]:
        """Run self-consistency checks including DR-specific ones."""
        issues = super().validate()

        # Check outcome R² range
        min_r2, max_r2 = self.outcome_r2_range
        if min_r2 > max_r2:
            issues.append(f"Invalid R² range: [{min_r2:.3f}, {max_r2:.3f}]")
        if max_r2 > 1.0:
            issues.append(f"Outcome R² > 1.0: {max_r2}")
        if max_r2 < 0.3:
            issues.append(f"Poor outcome model R²: max={max_r2:.3f}")

        # Check influence function tail ratio
        if self.worst_if_tail_ratio > 100:
            issues.append(
                f"Heavy-tailed influence functions: tail ratio={self.worst_if_tail_ratio:.1f}"
            )

        # Check detailed diagnostics consistency
        if self.dr_diagnostics_per_policy:
            for policy in self.policies:
                if policy not in self.dr_diagnostics_per_policy:
                    issues.append(f"Missing detailed diagnostics for {policy}")

        return issues

    def summary(self) -> str:
        """Generate concise summary including DR info."""
        lines = [
            f"Estimator: {self.estimator_type}",
            f"Method: {self.method}",
            f"Status: {self.overall_status.value}",
            f"Samples: {self.n_samples_valid}/{self.n_samples_total} valid ({100*(1-self.filter_rate):.1f}%)",
            f"Policies: {', '.join(self.policies)}",
            f"Best policy: {self.best_policy}",
            f"Weight ESS: {self.weight_ess:.1%}",
        ]

        if self.is_calibrated:
            lines.append(f"Calibration R²: {self.calibration_r2:.3f}")

        # DR-specific info
        min_r2, max_r2 = self.outcome_r2_range
        lines.append(f"Outcome R²: [{min_r2:.3f}, {max_r2:.3f}]")
        lines.append(f"Cross-fitted: {self.dr_cross_fitted} ({self.dr_n_folds} folds)")

        # Add any validation issues
        issues = self.validate()
        if issues:
            lines.append("Issues: " + "; ".join(issues[:2]))

        return " | ".join(lines)

    def get_policy_diagnostics(self, policy: str) -> Optional[Dict[str, Any]]:
        """Get detailed diagnostics for a specific policy."""
        return self.dr_diagnostics_per_policy.get(policy)

    def has_influence_functions(self) -> bool:
        """Check if influence functions are stored."""
        return (
            self.influence_functions is not None and len(self.influence_functions) > 0
        )

    def to_dict(self) -> Dict:
        """Export as dictionary for serialization, handling numpy arrays."""
        import numpy as np
        from dataclasses import asdict

        d = asdict(self)
        # Convert enums to strings
        d["weight_status"] = self.weight_status.value
        d["overall_status"] = self.overall_status.value

        # Handle influence functions (numpy arrays)
        if self.influence_functions:
            # Convert numpy arrays to lists for JSON serialization
            # Or optionally exclude them to save space
            d["influence_functions"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.influence_functions.items()
            }

        return d

    def to_dict_summary(self) -> Dict:
        """Export summary without large arrays (e.g., influence functions)."""
        d = super().to_dict()
        # Add DR-specific summary fields
        d["dr_cross_fitted"] = self.dr_cross_fitted
        d["dr_n_folds"] = self.dr_n_folds
        d["outcome_r2_range"] = self.outcome_r2_range
        d["outcome_rmse_mean"] = self.outcome_rmse_mean
        d["worst_if_tail_ratio"] = self.worst_if_tail_ratio
        # Exclude influence functions and detailed per-policy diagnostics
        d.pop("influence_functions", None)
        d.pop("dr_diagnostics_per_policy", None)
        return d

    def to_csv_row(self) -> Dict[str, Any]:
        """Export key metrics as a flat dict for CSV export."""
        # Start with parent's CSV row
        row = super().to_csv_row()
        # Add DR-specific metrics
        row["dr_cross_fitted"] = self.dr_cross_fitted
        row["dr_n_folds"] = self.dr_n_folds
        row["outcome_r2_min"] = self.outcome_r2_range[0]
        row["outcome_r2_max"] = self.outcome_r2_range[1]
        row["outcome_rmse_mean"] = self.outcome_rmse_mean
        row["worst_if_tail_ratio"] = self.worst_if_tail_ratio
        row["has_influence_functions"] = self.has_influence_functions()
        return row

    @classmethod
    def from_dict(cls, data: Dict) -> "DRDiagnostics":
        """Create from dictionary, handling numpy arrays."""
        import numpy as np

        # Convert status strings back to enum
        if "weight_status" in data and isinstance(data["weight_status"], str):
            data["weight_status"] = Status(data["weight_status"])

        # Convert influence function lists back to numpy arrays
        if "influence_functions" in data and data["influence_functions"]:
            data["influence_functions"] = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in data["influence_functions"].items()
            }

        # Remove computed fields
        data.pop("overall_status", None)

        # Handle tuple for outcome_r2_range
        if "outcome_r2_range" in data and isinstance(data["outcome_r2_range"], list):
            data["outcome_r2_range"] = tuple(data["outcome_r2_range"])

        return cls(**data)
