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


class GateState(Enum):
    """CF-bits gate states (extends Status with REFUSE)."""

    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    REFUSE = "refuse"


@dataclass
class IPSDiagnostics:
    """Diagnostics for IPS-based estimators (CalibratedIPS in both raw and calibrated modes)."""

    # ========== Core Info (always present) ==========
    estimator_type: str  # "CalibratedIPS"
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

    # ========== Overlap Metrics (new comprehensive diagnostics) ==========
    hellinger_affinity: Optional[float] = None  # Overall Hellinger affinity
    hellinger_per_policy: Optional[Dict[str, float]] = None  # Per-policy Hellinger
    overlap_quality: Optional[str] = None  # "good", "marginal", "poor", "catastrophic"

    # ========== Calibration Diagnostics (None for raw mode) ==========
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
            if self.calibration_r2 is not None:
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

        # Convert status_per_policy if present
        if d.get("status_per_policy"):
            d["status_per_policy"] = {
                policy: status.value if hasattr(status, "value") else status
                for policy, status in d["status_per_policy"].items()
            }

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

    # DR decomposition results
    dm_ips_decompositions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    orthogonality_scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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


@dataclass
class CFBitsDiagnostics:
    """CF-bits diagnostics combining uncertainty decomposition with actionable recommendations.

    CF-bits provides an information-theoretic framework that:
    1. Decomposes total uncertainty into identification (structural) and sampling (statistical) components
    2. Converts uncertainties into bits of information (log₂ reduction from baseline)
    3. Provides reliability gates (GOOD/WARNING/CRITICAL/REFUSE) based on thresholds
    4. Suggests concrete budget allocations to achieve target improvements

    Follows the same patterns as IPSDiagnostics and DRDiagnostics.
    """

    # ========== Core Identification ==========
    policy: str  # Target policy being evaluated
    estimator_type: str  # e.g., "CalibratedIPS", "DRCPOEstimator"
    scenario: str  # "fresh_draws" or "logging_only"

    # ========== Width Decomposition ==========
    wid: Optional[float] = None  # Identification width (structural uncertainty)
    wvar: Optional[float] = None  # Sampling width (statistical uncertainty)
    w_tot: Optional[float] = None  # Total width (wid + wvar)
    w_max: Optional[float] = None  # Maximum of wid and wvar (bottleneck)
    w0: float = 1.0  # Baseline width (1.0 for [0,1] KPIs)

    # ========== CF-bits Information ==========
    bits_tot: Optional[float] = None  # Total bits of information
    bits_id: Optional[float] = None  # Bits from identification channel
    bits_var: Optional[float] = None  # Bits from sampling channel

    # ========== Efficiency Metrics ==========
    ifr_main: Optional[float] = None  # Information Fraction Ratio (standard)
    ifr_oua: Optional[float] = None  # IFR with Oracle Uncertainty Augmentation
    aess_main: Optional[float] = None  # Adjusted ESS (n × IFR_main)
    aess_oua: Optional[float] = None  # Adjusted ESS with OUA

    # ========== σ(S) Structural Floors ==========
    aessf_sigmaS: Optional[float] = None  # A-ESSF on judge marginal
    aessf_sigmaS_lcb: Optional[float] = None  # Lower confidence bound
    bc_sigmaS: Optional[float] = None  # Bhattacharyya coefficient on σ(S)

    # ========== Variance Components ==========
    var_main: Optional[float] = None  # Main IF variance
    var_oracle: Optional[float] = None  # Oracle uncertainty contribution
    var_total: Optional[float] = None  # Total variance (main + oracle)

    # ========== Identification Diagnostics ==========
    wid_diagnostics: Optional[Dict[str, Any]] = None  # Details from Wid computation
    p_mass_unlabeled: Optional[float] = None  # Target mass on unlabeled bins
    n_bins_used: Optional[int] = None  # Number of bins in Phase-1 certificate
    n_oracle_available: Optional[int] = None  # Oracle samples available

    # ========== Reliability Gates ==========
    gate_state: GateState = GateState.GOOD  # Overall reliability assessment
    gate_reasons: List[str] = field(default_factory=list)  # Specific issues
    gate_suggestions: Dict[str, Any] = field(
        default_factory=dict
    )  # Actionable recommendations

    # ========== Budget Recommendations ==========
    logs_factor_for_half_bit: Optional[float] = (
        None  # Sample size multiplier for 0.5 bits
    )
    labels_for_wid_reduction: Optional[int] = None  # Additional labels to reduce Wid
    dominant_channel: Optional[str] = None  # "identification" or "sampling"

    def validate(self) -> List[str]:
        """Validate internal consistency of CF-bits diagnostics."""
        issues = []

        # Check width consistency
        if self.wid is not None and self.wvar is not None:
            if self.w_tot is not None:
                expected_tot = self.wid + self.wvar
                if abs(self.w_tot - expected_tot) > 1e-6:
                    issues.append(
                        f"Width inconsistency: w_tot={self.w_tot:.3f} != wid+wvar={expected_tot:.3f}"
                    )

            if self.w_max is not None:
                expected_max = max(self.wid, self.wvar)
                if abs(self.w_max - expected_max) > 1e-6:
                    issues.append(
                        f"Max width inconsistency: w_max={self.w_max:.3f} != max(wid,wvar)={expected_max:.3f}"
                    )

        # Check efficiency bounds
        if self.ifr_main is not None and not (
            0 <= self.ifr_main <= 1.01
        ):  # Allow slight numerical error
            issues.append(f"IFR_main out of bounds: {self.ifr_main:.3f}")

        if self.ifr_oua is not None and not (0 <= self.ifr_oua <= 1.01):
            issues.append(f"IFR_OUA out of bounds: {self.ifr_oua:.3f}")

        # Check overlap bounds
        if self.aessf_sigmaS is not None and not (0 <= self.aessf_sigmaS <= 1.01):
            issues.append(f"A-ESSF out of bounds: {self.aessf_sigmaS:.3f}")

        if self.bc_sigmaS is not None and not (0 <= self.bc_sigmaS <= 1.01):
            issues.append(f"BC out of bounds: {self.bc_sigmaS:.3f}")

        # Check theoretical relationship: A-ESSF <= BC²
        if self.aessf_sigmaS is not None and self.bc_sigmaS is not None:
            if (
                self.aessf_sigmaS > self.bc_sigmaS**2 * 1.1
            ):  # Allow 10% numerical tolerance
                issues.append(
                    f"Theoretical violation: A-ESSF={self.aessf_sigmaS:.3f} > BC²={self.bc_sigmaS**2:.3f}"
                )

        return issues

    def summary(self) -> str:
        """One-line human-readable summary with key metrics and recommendations."""
        parts = []

        # CF-bits and width
        if self.bits_tot is not None:
            parts.append(f"CF-bits: {self.bits_tot:.1f}")
        if self.w_tot is not None:
            parts.append(f"W={self.w_tot:.2f}")

        # Decomposition
        if self.wid is not None and self.wvar is not None:
            parts.append(f"(Wid={self.wid:.2f}, Wvar={self.wvar:.2f})")

        # Efficiency
        if self.ifr_oua is not None:
            parts.append(f"IFR(OUA)={self.ifr_oua:.0%}")
        elif self.ifr_main is not None:
            parts.append(f"IFR={self.ifr_main:.0%}")

        # Structural floors
        if self.aessf_sigmaS_lcb is not None:
            parts.append(f"A-ESSF(LCB)={self.aessf_sigmaS_lcb:.0%}")

        # Gate
        parts.append(f"Gate: {self.gate_state.value.upper()}")

        # Primary recommendation
        if self.gate_state == GateState.REFUSE:
            parts.append("→ Do not use")
        elif self.gate_state == GateState.CRITICAL:
            parts.append("→ Use with extreme caution")
        elif self.gate_state == GateState.WARNING:
            if self.dominant_channel == "identification":
                if self.p_mass_unlabeled is not None and self.p_mass_unlabeled > 0.1:
                    parts.append(
                        f"→ Add labels ({self.p_mass_unlabeled:.0%} mass unlabeled)"
                    )
                else:
                    parts.append("→ Add more oracle labels")
            else:
                if self.logs_factor_for_half_bit is not None:
                    parts.append(
                        f"→ {self.logs_factor_for_half_bit:.1f}× logs for +0.5 bits"
                    )
                else:
                    parts.append("→ Collect more logs")
        else:
            parts.append("→ Estimate reliable")

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "policy": self.policy,
            "estimator_type": self.estimator_type,
            "scenario": self.scenario,
            # Widths
            "wid": self.wid,
            "wvar": self.wvar,
            "w_tot": self.w_tot,
            "w_max": self.w_max,
            "w0": self.w0,
            # CF-bits
            "bits_tot": self.bits_tot,
            "bits_id": self.bits_id,
            "bits_var": self.bits_var,
            # Efficiency
            "ifr_main": self.ifr_main,
            "ifr_oua": self.ifr_oua,
            "aess_main": self.aess_main,
            "aess_oua": self.aess_oua,
            # Structural floors
            "aessf_sigmaS": self.aessf_sigmaS,
            "aessf_sigmaS_lcb": self.aessf_sigmaS_lcb,
            "bc_sigmaS": self.bc_sigmaS,
            # Variance
            "var_main": self.var_main,
            "var_oracle": self.var_oracle,
            "var_total": self.var_total,
            # Diagnostics
            "wid_diagnostics": self.wid_diagnostics,
            "p_mass_unlabeled": self.p_mass_unlabeled,
            "n_bins_used": self.n_bins_used,
            "n_oracle_available": self.n_oracle_available,
            # Gates
            "gate_state": self.gate_state.value,
            "gate_reasons": self.gate_reasons,
            "gate_suggestions": self.gate_suggestions,
            # Budget
            "logs_factor_for_half_bit": self.logs_factor_for_half_bit,
            "labels_for_wid_reduction": self.labels_for_wid_reduction,
            "dominant_channel": self.dominant_channel,
        }
        return {k: v for k, v in d.items() if v is not None}  # Remove None values

    def to_csv_row(self) -> Dict[str, Any]:
        """Flatten for tabular export (excludes nested dicts)."""
        d = self.to_dict()
        # Remove nested structures
        d.pop("wid_diagnostics", None)
        d.pop("gate_suggestions", None)
        # Flatten gate_reasons to string
        if "gate_reasons" in d:
            d["gate_reasons"] = "; ".join(d["gate_reasons"])
        return d

    @property
    def needs_more_labels(self) -> bool:
        """Whether identification uncertainty dominates."""
        if self.wid is None or self.wvar is None:
            return False
        return self.wid > self.wvar

    @property
    def has_catastrophic_overlap(self) -> bool:
        """Whether structural overlap is catastrophically bad."""
        if self.aessf_sigmaS_lcb is not None:
            return self.aessf_sigmaS_lcb < 0.05
        if self.aessf_sigmaS is not None:
            return self.aessf_sigmaS < 0.02
        return False

    @property
    def is_reliable(self) -> bool:
        """Whether estimate passes all reliability checks."""
        return self.gate_state == GateState.GOOD
