"""
Unified diagnostic system for CJE.

This module provides the central diagnostic types and orchestration for all
estimators in CJE. All diagnostics flow through this unified interface.

Design principles:
- Single source of truth: DiagnosticSuite contains all diagnostics
- Type safety: Dataclasses with clear types
- Composability: Modular diagnostics that combine cleanly
- Validation: Self-consistency checks built in
- Versioning: Schema version tracking for compatibility
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any, List, Tuple
from enum import Enum
from datetime import datetime
import numpy as np


class DiagnosticStatus(Enum):
    """Overall diagnostic health status."""

    GOOD = "good"  # All metrics within acceptable ranges
    WARNING = "warning"  # Some concerning metrics
    CRITICAL = "critical"  # Serious issues detected

    def __str__(self) -> str:
        return self.value


# ============= Data Diagnostics =============


@dataclass
class DataDiagnostics:
    """Diagnostics about the dataset and filtering."""

    n_total: int
    n_valid: int
    n_policies: int
    policies: List[str]
    filter_rate: float
    has_rewards: bool
    has_oracle_labels: bool
    oracle_coverage: float

    def validate(self) -> List[str]:
        """Validate data diagnostics."""
        issues = []
        if self.n_valid > self.n_total:
            issues.append(
                f"Invalid: n_valid ({self.n_valid}) > n_total ({self.n_total})"
            )
        if self.filter_rate < 0 or self.filter_rate > 1:
            issues.append(f"Invalid: filter_rate ({self.filter_rate}) not in [0,1]")
        if self.oracle_coverage < 0 or self.oracle_coverage > 1:
            issues.append(
                f"Invalid: oracle_coverage ({self.oracle_coverage}) not in [0,1]"
            )
        return issues


# ============= Estimation Diagnostics =============


@dataclass
class EstimationDiagnostics:
    """Core estimation diagnostics (all estimators)."""

    method: str
    estimates: Dict[str, float]
    standard_errors: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    n_samples_used: Dict[str, int]
    convergence: bool = True
    iterations: int = 1

    def best_policy(self) -> str:
        """Return policy with highest estimate."""
        if not self.estimates:
            return "none"
        return max(self.estimates.items(), key=lambda x: x[1])[0]

    def validate(self) -> List[str]:
        """Validate estimation diagnostics."""
        issues = []
        # Check consistency between estimates and CIs
        for policy in self.estimates:
            if policy in self.confidence_intervals:
                ci_low, ci_high = self.confidence_intervals[policy]
                est = self.estimates[policy]
                if not (ci_low <= est <= ci_high):
                    issues.append(
                        f"Estimate {est} outside CI [{ci_low}, {ci_high}] for {policy}"
                    )
        # Check for NaN/inf
        for policy, est in self.estimates.items():
            if not np.isfinite(est):
                issues.append(f"Non-finite estimate for {policy}: {est}")
        return issues


# ============= Weight Diagnostics =============


@dataclass
class PolicyWeightDiagnostics:
    """Per-policy weight diagnostics."""

    ess_fraction: float
    mean_weight: float
    median_weight: float
    max_weight: float
    tail_ratio_99_5: float
    top1_mass: float
    n_extreme: int
    n_zero: int
    cv: float  # Coefficient of variation

    def status(self) -> DiagnosticStatus:
        """Determine status for this policy."""
        if self.ess_fraction < 0.01 or self.tail_ratio_99_5 > 1000:
            return DiagnosticStatus.CRITICAL
        elif self.ess_fraction < 0.1 or self.tail_ratio_99_5 > 100:
            return DiagnosticStatus.WARNING
        return DiagnosticStatus.GOOD


@dataclass
class WeightDiagnostics:
    """Weight-based diagnostics (IPS/CalibratedIPS/DR)."""

    per_policy: Dict[str, PolicyWeightDiagnostics]
    overall_ess_fraction: float
    worst_tail_ratio: float
    worst_policy: str
    status: DiagnosticStatus

    @classmethod
    def from_estimator(cls, estimator: Any) -> "WeightDiagnostics":
        """Create from an IPS-based estimator."""
        from ..utils.diagnostics import (
            diagnose_weights,
            weight_diagnostics,
            tail_weight_ratio,
            mass_concentration,
        )

        per_policy = {}
        worst_tail = 0.0
        worst_policy = ""
        total_ess = 0.0
        total_n = 0

        for policy in estimator.sampler.target_policies:
            weights = estimator.get_weights(policy)
            if weights is not None and len(weights) > 0:
                # Get basic diagnostics
                diag = diagnose_weights(weights, policy)

                # Compute additional metrics directly
                full_diag = weight_diagnostics(weights)
                w_stats = full_diag["weights"]

                per_policy[policy] = PolicyWeightDiagnostics(
                    ess_fraction=diag.ess_fraction,
                    mean_weight=diag.mean_weight,
                    median_weight=diag.median_weight,
                    max_weight=diag.max_weight,
                    tail_ratio_99_5=w_stats.get("tail_ratio_99_5", 0.0),
                    top1_mass=w_stats.get("top1_share", 0.0),
                    n_extreme=diag.extreme_weight_count,
                    n_zero=diag.zero_weight_count,
                    cv=float(np.std(weights) / (np.mean(weights) + 1e-12)),
                )

                # Track worst metrics
                tail_ratio = per_policy[policy].tail_ratio_99_5
                if tail_ratio > worst_tail:
                    worst_tail = tail_ratio
                    worst_policy = policy

                # Accumulate for overall ESS
                n = len(weights)
                total_ess += diag.ess_fraction * n
                total_n += n

        overall_ess = total_ess / total_n if total_n > 0 else 0.0

        # Determine overall status
        if overall_ess < 0.01 or worst_tail > 1000:
            status = DiagnosticStatus.CRITICAL
        elif overall_ess < 0.1 or worst_tail > 100:
            status = DiagnosticStatus.WARNING
        else:
            status = DiagnosticStatus.GOOD

        return cls(
            per_policy=per_policy,
            overall_ess_fraction=overall_ess,
            worst_tail_ratio=worst_tail,
            worst_policy=worst_policy,
            status=status,
        )

    def validate(self) -> List[str]:
        """Validate weight diagnostics."""
        issues = []
        for policy, diag in self.per_policy.items():
            if diag.ess_fraction > 1.0:
                issues.append(f"Invalid: ESS fraction > 1.0 for {policy}")
            if diag.mean_weight < 0:
                issues.append(f"Invalid: negative mean weight for {policy}")
        return issues


# ============= DR Diagnostics =============


@dataclass
class PolicyDRDiagnostics:
    """Per-policy DR diagnostics."""

    dm_estimate: float
    ips_correction: float
    dr_estimate: float
    outcome_r2: float
    outcome_rmse: float
    if_variance: float
    if_tail_ratio: float
    score_mean: float
    score_p_value: float
    fresh_draw_coverage: float
    draws_per_prompt: int


@dataclass
class DRDiagnostics:
    """Doubly robust specific diagnostics."""

    per_policy: Dict[str, PolicyDRDiagnostics]
    cross_fitted: bool
    n_folds: int
    worst_if_tail: float
    best_outcome_r2: float
    worst_outcome_r2: float
    max_score_z: float  # For TMLE orthogonality
    status: DiagnosticStatus

    @classmethod
    def from_metadata(cls, dr_metadata: Dict[str, Any]) -> "DRDiagnostics":
        """Create from DR metadata dict."""
        per_policy = {}
        worst_if_tail = 0.0
        best_r2 = -1.0
        worst_r2 = 1.0
        max_score_z = 0.0

        for policy, diag_dict in dr_metadata.items():
            if isinstance(diag_dict, dict):
                per_policy[policy] = PolicyDRDiagnostics(
                    dm_estimate=diag_dict.get("dm_mean", 0.0),
                    ips_correction=diag_dict.get("ips_corr_mean", 0.0),
                    dr_estimate=diag_dict.get("dr_estimate", 0.0),
                    outcome_r2=diag_dict.get("r2_oof", 0.0),
                    outcome_rmse=diag_dict.get("residual_rmse", 0.0),
                    if_variance=diag_dict.get("if_var", 0.0),
                    if_tail_ratio=diag_dict.get("if_tail_ratio_99_5", 0.0),
                    score_mean=diag_dict.get("score_mean", 0.0),
                    score_p_value=diag_dict.get("score_p", 1.0),
                    fresh_draw_coverage=(
                        1.0 if diag_dict.get("coverage_ok", True) else 0.0
                    ),
                    draws_per_prompt=diag_dict.get("draws_per_prompt", 0),
                )

                # Track extremes
                worst_if_tail = max(worst_if_tail, per_policy[policy].if_tail_ratio)
                best_r2 = max(best_r2, per_policy[policy].outcome_r2)
                worst_r2 = min(worst_r2, per_policy[policy].outcome_r2)
                max_score_z = max(max_score_z, abs(diag_dict.get("score_z", 0.0)))

        # Determine status
        if worst_if_tail > 1000 or worst_r2 < 0:
            status = DiagnosticStatus.CRITICAL
        elif worst_if_tail > 100 or worst_r2 < 0.1:
            status = DiagnosticStatus.WARNING
        else:
            status = DiagnosticStatus.GOOD

        # Get cross-fitting info from first policy (should be same for all)
        first_diag = next(iter(dr_metadata.values())) if dr_metadata else {}

        return cls(
            per_policy=per_policy,
            cross_fitted=first_diag.get("cross_fitted", True),
            n_folds=first_diag.get("unique_folds", 5),
            worst_if_tail=worst_if_tail,
            best_outcome_r2=best_r2,
            worst_outcome_r2=worst_r2,
            max_score_z=max_score_z,
            status=status,
        )

    def validate(self) -> List[str]:
        """Validate DR diagnostics."""
        issues = []
        for policy, diag in self.per_policy.items():
            if not (0 <= diag.score_p_value <= 1):
                issues.append(f"Invalid p-value for {policy}: {diag.score_p_value}")
            if diag.outcome_r2 > 1:
                issues.append(f"R² > 1 for {policy}: {diag.outcome_r2}")
        return issues


# ============= Calibration Diagnostics =============


@dataclass
class CalibrationDiagnostics:
    """Judge calibration diagnostics."""

    method: str  # "isotonic", "cross_fit"
    n_oracle: int
    oracle_coverage: float
    rmse: float
    r2: float
    coverage_at_01: float
    oof_rmse: Optional[float] = None
    oof_r2: Optional[float] = None

    @classmethod
    def from_result(cls, cal_result: Any) -> "CalibrationDiagnostics":
        """Create from CalibrationResult."""
        return cls(
            method=(
                "cross_fit"
                if cal_result.calibrator and hasattr(cal_result.calibrator, "cv_models")
                else "isotonic"
            ),
            n_oracle=cal_result.n_oracle,
            oracle_coverage=(
                cal_result.n_oracle / len(cal_result.calibrated_scores)
                if len(cal_result.calibrated_scores) > 0
                else 0.0
            ),
            rmse=cal_result.calibration_rmse,
            r2=getattr(cal_result, "r2", 0.0),
            coverage_at_01=cal_result.coverage_at_01,
            oof_rmse=getattr(cal_result, "oof_rmse", None),
            oof_r2=getattr(cal_result, "oof_r2", None),
        )

    def validate(self) -> List[str]:
        """Validate calibration diagnostics."""
        issues = []
        if self.r2 > 1 or self.r2 < -1:
            issues.append(f"Invalid R²: {self.r2}")
        if self.coverage_at_01 < 0 or self.coverage_at_01 > 1:
            issues.append(f"Invalid coverage: {self.coverage_at_01}")
        return issues


# ============= Oracle Diagnostics =============


@dataclass
class OracleDiagnostics:
    """Oracle fidelity diagnostics."""

    r2_to_oracle: float  # R²(g(S) → Y)
    r2_to_reward: float  # R²(g(S) → R)
    surrogate_gap_mean: float
    surrogate_gap_std: float
    sufficiency_p_value: float
    sufficiency_violated: bool
    plateau_resolution: float
    n_oracle_samples: int

    def validate(self) -> List[str]:
        """Validate oracle diagnostics."""
        issues = []
        if self.r2_to_oracle > 1:
            issues.append(f"R² to oracle > 1: {self.r2_to_oracle}")
        if not (0 <= self.sufficiency_p_value <= 1):
            issues.append(f"Invalid p-value: {self.sufficiency_p_value}")
        return issues


# ============= Main Diagnostic Suite =============


@dataclass
class DiagnosticSuite:
    """Unified container for all diagnostic types.

    This is the single source of truth for all diagnostics in CJE.
    Stored in EstimationResult.diagnostics (not metadata).
    """

    # Core diagnostics (always present)
    data: DataDiagnostics
    estimation: EstimationDiagnostics

    # Conditional diagnostics (present when applicable)
    weights: Optional[WeightDiagnostics] = None
    calibration: Optional[CalibrationDiagnostics] = None
    dr: Optional[DRDiagnostics] = None
    oracle: Optional[OracleDiagnostics] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    estimator_type: str = ""
    version: str = "1.0"

    def overall_status(self) -> DiagnosticStatus:
        """Compute overall health status from all diagnostics."""
        statuses = []

        if self.weights:
            statuses.append(self.weights.status)
        if self.dr:
            statuses.append(self.dr.status)

        # Return worst status
        if any(s == DiagnosticStatus.CRITICAL for s in statuses):
            return DiagnosticStatus.CRITICAL
        elif any(s == DiagnosticStatus.WARNING for s in statuses):
            return DiagnosticStatus.WARNING
        return DiagnosticStatus.GOOD

    def validate(self) -> List[str]:
        """Run self-consistency checks, return any issues."""
        issues = []

        # Validate each component
        issues.extend(self.data.validate())
        issues.extend(self.estimation.validate())

        if self.weights:
            issues.extend(self.weights.validate())
        if self.calibration:
            issues.extend(self.calibration.validate())
        if self.dr:
            issues.extend(self.dr.validate())
        if self.oracle:
            issues.extend(self.oracle.validate())

        # Cross-component validation
        if self.weights and self.estimation:
            # Check that we have weight diagnostics for all estimated policies
            for policy in self.estimation.estimates:
                if policy != "base" and policy not in self.weights.per_policy:
                    issues.append(
                        f"Missing weight diagnostics for estimated policy: {policy}"
                    )

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary for serialization."""
        result = {
            "version": self.version,
            "timestamp": self.timestamp,
            "estimator_type": self.estimator_type,
            "overall_status": str(self.overall_status()),
            "data": asdict(self.data),
            "estimation": asdict(self.estimation),
        }

        if self.weights:
            result["weights"] = asdict(self.weights)
        if self.calibration:
            result["calibration"] = asdict(self.calibration)
        if self.dr:
            result["dr"] = asdict(self.dr)
        if self.oracle:
            result["oracle"] = asdict(self.oracle)

        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "DIAGNOSTIC SUMMARY",
            "=" * 60,
            f"Estimator: {self.estimator_type}",
            f"Status: {self.overall_status()}",
            f"Timestamp: {self.timestamp}",
            "",
        ]

        # Data summary
        lines.extend(
            [
                "DATA:",
                f"  Samples: {self.data.n_valid}/{self.data.n_total} valid",
                f"  Policies: {', '.join(self.data.policies)}",
                f"  Oracle coverage: {self.data.oracle_coverage:.1%}",
                "",
            ]
        )

        # Estimation summary
        lines.extend(
            [
                "ESTIMATION:",
                f"  Method: {self.estimation.method}",
                f"  Best policy: {self.estimation.best_policy()}",
                f"  Convergence: {'Yes' if self.estimation.convergence else 'No'}",
                "",
            ]
        )

        # Weight summary
        if self.weights:
            lines.extend(
                [
                    "WEIGHTS:",
                    f"  Overall ESS: {self.weights.overall_ess_fraction:.1%}",
                    f"  Worst tail ratio: {self.weights.worst_tail_ratio:.1f} ({self.weights.worst_policy})",
                    f"  Status: {self.weights.status}",
                    "",
                ]
            )

        # DR summary
        if self.dr:
            lines.extend(
                [
                    "DOUBLY ROBUST:",
                    f"  Cross-fitted: {'Yes' if self.dr.cross_fitted else 'No'} ({self.dr.n_folds} folds)",
                    f"  Outcome R²: [{self.dr.worst_outcome_r2:.3f}, {self.dr.best_outcome_r2:.3f}]",
                    f"  Worst IF tail: {self.dr.worst_if_tail:.1f}",
                    f"  Max |score z|: {self.dr.max_score_z:.2f}",
                    "",
                ]
            )

        # Validation issues
        issues = self.validate()
        if issues:
            lines.extend(
                ["VALIDATION ISSUES:", *[f"  - {issue}" for issue in issues], ""]
            )

        lines.append("=" * 60)
        return "\n".join(lines)


# ============= Diagnostic Manager =============


class DiagnosticManager:
    """Central diagnostic orchestrator for CJE.

    Coordinates computation of all diagnostics and maintains history.
    """

    def __init__(self) -> None:
        self.history: List[DiagnosticSuite] = []

    def compute_suite(
        self,
        estimator: Any,
        dataset: Optional[Any] = None,
        calibration_result: Optional[Any] = None,
        include_oracle: bool = False,
    ) -> DiagnosticSuite:
        """Compute complete diagnostic suite for an estimator.

        Args:
            estimator: The fitted estimator
            dataset: The dataset used (if None, will try to get from sampler)
            calibration_result: Optional calibration result
            include_oracle: Whether to compute oracle diagnostics

        Returns:
            Complete DiagnosticSuite
        """
        # Get dataset from sampler if not provided
        if dataset is None and hasattr(estimator, "sampler"):
            if hasattr(estimator.sampler, "dataset"):
                dataset = estimator.sampler.dataset
            elif hasattr(estimator.sampler, "_dataset"):
                dataset = estimator.sampler._dataset

        # Data diagnostics
        if dataset:
            n_total = (
                dataset.n_samples
                if hasattr(dataset, "n_samples")
                else len(dataset.samples)
            )
            samples = dataset.samples if hasattr(dataset, "samples") else []
        else:
            n_total = 0
            samples = []

        # Get valid samples from sampler if available
        if hasattr(estimator, "sampler"):
            n_valid = estimator.sampler.n_valid_samples
            policies = list(estimator.sampler.target_policies)
        else:
            n_valid = n_total
            policies = (
                dataset.target_policies
                if dataset and hasattr(dataset, "target_policies")
                else []
            )

        data_diag = DataDiagnostics(
            n_total=n_total,
            n_valid=n_valid,
            n_policies=len(policies),
            policies=policies,
            filter_rate=1.0 - (n_valid / n_total) if n_total > 0 else 0.0,
            has_rewards=(
                any(s.reward is not None for s in samples) if samples else False
            ),
            has_oracle_labels=(
                any("oracle_label" in s.metadata for s in samples) if samples else False
            ),
            oracle_coverage=(
                sum(1 for s in samples if "oracle_label" in s.metadata) / n_total
                if n_total > 0 and samples
                else 0.0
            ),
        )

        # Estimation diagnostics
        if hasattr(estimator, "_results"):
            results = estimator._results
            est_diag = EstimationDiagnostics(
                method=estimator.__class__.__name__.lower().replace("estimator", ""),
                estimates={p: results.estimates[i] for i, p in enumerate(policies)},
                standard_errors={
                    p: results.standard_errors[i] for i, p in enumerate(policies)
                },
                confidence_intervals={},  # TODO: compute from results
                n_samples_used={p: n_valid for p in policies},
                convergence=True,
                iterations=1,
            )
        else:
            est_diag = EstimationDiagnostics(
                method=estimator.__class__.__name__,
                estimates={},
                standard_errors={},
                confidence_intervals={},
                n_samples_used={},
            )

        # Weight diagnostics (if applicable)
        weight_diag = None
        if hasattr(estimator, "get_weights"):
            weight_diag = WeightDiagnostics.from_estimator(estimator)

        # Calibration diagnostics
        cal_diag = None
        if calibration_result is not None:
            cal_diag = CalibrationDiagnostics.from_result(calibration_result)

        # DR diagnostics
        dr_diag = None
        if (
            hasattr(estimator, "_results")
            and "dr_diagnostics" in estimator._results.metadata
        ):
            dr_diag = DRDiagnostics.from_metadata(
                estimator._results.metadata["dr_diagnostics"]
            )

        # Oracle diagnostics (if requested and available)
        oracle_diag = None
        if include_oracle and data_diag.has_oracle_labels:
            oracle_diag = self._compute_oracle_diagnostics(estimator, dataset)

        # Create suite
        suite = DiagnosticSuite(
            data=data_diag,
            estimation=est_diag,
            weights=weight_diag,
            calibration=cal_diag,
            dr=dr_diag,
            oracle=oracle_diag,
            estimator_type=estimator.__class__.__name__,
        )

        # Add to history
        self.history.append(suite)

        return suite

    def _compute_oracle_diagnostics(
        self, estimator: Any, dataset: Any
    ) -> Optional[OracleDiagnostics]:
        """Compute oracle fidelity diagnostics."""
        # TODO: Implement oracle diagnostic computation
        # This would use the oracle_diagnostics module
        return None

    def export_history(self, format: str = "json") -> str:
        """Export diagnostic history."""
        if format == "json":
            import json

            return json.dumps([suite.to_dict() for suite in self.history], indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")
