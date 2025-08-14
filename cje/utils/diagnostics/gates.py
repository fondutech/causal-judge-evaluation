"""
Automated diagnostic gates for CJE estimation pipeline.

Implements Section 9.5 of the CJE paper: automated stop/ship decisions
based on diagnostic thresholds to ensure estimation quality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GateStatus(Enum):
    """Status of a diagnostic gate check."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class GateResult:
    """Result from a single gate check."""

    name: str
    status: GateStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    threshold: Optional[float] = None
    observed: Optional[float] = None

    @property
    def passed(self) -> bool:
        """Check if gate passed (PASS or WARN)."""
        return self.status != GateStatus.FAIL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "threshold": self.threshold,
            "observed": self.observed,
        }


@dataclass
class GateReport:
    """Aggregated report from all gates."""

    overall_status: GateStatus
    gate_results: List[GateResult]
    summary: str
    n_passed: int = 0
    n_warned: int = 0
    n_failed: int = 0

    @classmethod
    def from_results(cls, results: List[GateResult]) -> "GateReport":
        """Create report from gate results."""
        n_passed = sum(1 for r in results if r.status == GateStatus.PASS)
        n_warned = sum(1 for r in results if r.status == GateStatus.WARN)
        n_failed = sum(1 for r in results if r.status == GateStatus.FAIL)

        # Overall status: FAIL if any fail, WARN if any warn, else PASS
        if n_failed > 0:
            overall_status = GateStatus.FAIL
            summary = f"❌ {n_failed} gate(s) failed"
        elif n_warned > 0:
            overall_status = GateStatus.WARN
            summary = f"⚠️ {n_warned} gate(s) warned"
        else:
            overall_status = GateStatus.PASS
            summary = f"✅ All {n_passed} gate(s) passed"

        return cls(
            overall_status=overall_status,
            gate_results=results,
            summary=summary,
            n_passed=n_passed,
            n_warned=n_warned,
            n_failed=n_failed,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_status": self.overall_status.value,
            "summary": self.summary,
            "n_passed": self.n_passed,
            "n_warned": self.n_warned,
            "n_failed": self.n_failed,
            "gate_results": [r.to_dict() for r in self.gate_results],
        }

    def format_terminal(self, verbose: bool = False) -> str:
        """Format report for terminal display."""
        lines = []

        # Header with color codes
        status_color = {
            GateStatus.PASS: "\033[92m",  # Green
            GateStatus.WARN: "\033[93m",  # Yellow
            GateStatus.FAIL: "\033[91m",  # Red
        }
        reset = "\033[0m"

        lines.append("\n" + "=" * 60)
        lines.append(f"{status_color[self.overall_status]}DIAGNOSTIC GATES{reset}")
        lines.append("=" * 60)
        lines.append(self.summary)
        lines.append("")

        # Individual gates
        for result in self.gate_results:
            icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[result.status.value]
            color = status_color[result.status]

            line = f"{icon} {color}{result.name}{reset}: {result.message}"
            if result.observed is not None and result.threshold is not None:
                line += f" (observed={result.observed:.3f}, threshold={result.threshold:.3f})"
            lines.append(line)

            if verbose and result.details:
                for key, value in result.details.items():
                    lines.append(f"    {key}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)


class DiagnosticGate(ABC):
    """Base class for diagnostic gates."""

    def __init__(
        self,
        name: str,
        enabled: bool = True,
        fail_threshold: Optional[float] = None,
        warn_threshold: Optional[float] = None,
    ):
        """Initialize gate.

        Args:
            name: Gate name for reporting
            enabled: Whether gate is active
            fail_threshold: Threshold for failing
            warn_threshold: Threshold for warning (if different from fail)
        """
        self.name = name
        self.enabled = enabled
        self.fail_threshold = fail_threshold
        self.warn_threshold = warn_threshold

    @abstractmethod
    def check(self, diagnostics: Dict[str, Any]) -> GateResult:
        """Check if gate passes given diagnostics.

        Args:
            diagnostics: Dictionary of diagnostic results

        Returns:
            GateResult with status and details
        """
        pass

    def _threshold_check(
        self, value: float, higher_is_better: bool = True
    ) -> GateStatus:
        """Helper to check value against thresholds."""
        if self.fail_threshold is not None:
            if higher_is_better:
                if value < self.fail_threshold:
                    return GateStatus.FAIL
            else:
                if value > self.fail_threshold:
                    return GateStatus.FAIL

        if self.warn_threshold is not None:
            if higher_is_better:
                if value < self.warn_threshold:
                    return GateStatus.WARN
            else:
                if value > self.warn_threshold:
                    return GateStatus.WARN

        return GateStatus.PASS


# ========== Standard Gates ==========


class OverlapGate(DiagnosticGate):
    """Check for sufficient overlap (ESS and tail behavior)."""

    def __init__(
        self,
        min_ess: float = 1000,
        min_tail_index: float = 2.0,
        enabled: bool = True,
    ):
        """Initialize overlap gate.

        Args:
            min_ess: Minimum effective sample size
            min_tail_index: Minimum Hill tail index (α < 2 means infinite variance)
            enabled: Whether gate is active
        """
        super().__init__(name="Overlap", enabled=enabled)
        self.min_ess = min_ess
        self.min_tail_index = min_tail_index

    def check(self, diagnostics: Dict[str, Any]) -> GateResult:
        """Check overlap diagnostics."""
        if not self.enabled:
            return GateResult(
                name=self.name,
                status=GateStatus.PASS,
                message="Gate disabled",
            )

        # Check if weight diagnostics exist
        if "weight_diagnostics" not in diagnostics:
            return GateResult(
                name=self.name,
                status=GateStatus.WARN,
                message="Weight diagnostics not available",
            )

        weight_diags = diagnostics["weight_diagnostics"]

        # Check ESS for each policy
        min_observed_ess = float("inf")
        worst_policy = None

        for policy, diags in weight_diags.items():
            if "ess" in diags:
                ess = diags["ess"]
                if ess < min_observed_ess:
                    min_observed_ess = ess
                    worst_policy = policy

        # Check tail index
        min_tail_index_observed = float("inf")
        worst_tail_policy = None

        for policy, diags in weight_diags.items():
            if "tail_index" in diags:
                tail_idx = diags["tail_index"]
                if tail_idx < min_tail_index_observed:
                    min_tail_index_observed = tail_idx
                    worst_tail_policy = policy

        # Determine status
        failures = []
        warnings = []

        if min_observed_ess < self.min_ess:
            if min_observed_ess < self.min_ess * 0.5:  # Less than half threshold
                failures.append(
                    f"ESS too low: {min_observed_ess:.0f} < {self.min_ess:.0f} (policy: {worst_policy})"
                )
            else:
                warnings.append(
                    f"ESS marginal: {min_observed_ess:.0f} < {self.min_ess:.0f} (policy: {worst_policy})"
                )

        if min_tail_index_observed < self.min_tail_index:
            if min_tail_index_observed < 1.5:  # Very heavy tails
                failures.append(
                    f"Extremely heavy tails: α={min_tail_index_observed:.2f} < 2 (policy: {worst_tail_policy})"
                )
            elif min_tail_index_observed < 2.0:
                warnings.append(
                    f"Heavy tails: α={min_tail_index_observed:.2f} < 2 may cause infinite variance"
                )

        # Determine overall status
        if failures:
            return GateResult(
                name=self.name,
                status=GateStatus.FAIL,
                message="; ".join(failures),
                details={
                    "min_ess": min_observed_ess,
                    "min_tail_index": min_tail_index_observed,
                    "worst_ess_policy": worst_policy,
                    "worst_tail_policy": worst_tail_policy,
                },
                threshold=self.min_ess,
                observed=min_observed_ess,
            )
        elif warnings:
            return GateResult(
                name=self.name,
                status=GateStatus.WARN,
                message="; ".join(warnings),
                details={
                    "min_ess": min_observed_ess,
                    "min_tail_index": min_tail_index_observed,
                },
            )
        else:
            return GateResult(
                name=self.name,
                status=GateStatus.PASS,
                message=f"Sufficient overlap (ESS={min_observed_ess:.0f}, α={min_tail_index_observed:.2f})",
                details={
                    "min_ess": min_observed_ess,
                    "min_tail_index": min_tail_index_observed,
                },
            )


class JudgeGate(DiagnosticGate):
    """Check judge calibration and stability."""

    def __init__(
        self,
        max_drift: float = 0.05,
        max_ece: float = 0.1,
        min_r2: float = 0.5,
        enabled: bool = True,
    ):
        """Initialize judge gate.

        Args:
            max_drift: Maximum allowed Kendall τ drift
            max_ece: Maximum expected calibration error
            min_r2: Minimum R² for calibration fit
            enabled: Whether gate is active
        """
        super().__init__(name="Judge Quality", enabled=enabled)
        self.max_drift = max_drift
        self.max_ece = max_ece
        self.min_r2 = min_r2

    def check(self, diagnostics: Dict[str, Any]) -> GateResult:
        """Check judge diagnostics."""
        if not self.enabled:
            return GateResult(
                name=self.name,
                status=GateStatus.PASS,
                message="Gate disabled",
            )

        warnings = []
        failures = []
        details = {}

        # Check drift if available
        if "stability_diagnostics" in diagnostics:
            stability = diagnostics["stability_diagnostics"]

            if "drift_detection" in stability:
                drift = stability["drift_detection"]
                if drift.get("has_drift", False):
                    max_tau_change = drift.get("max_tau_change", 0)
                    details["max_drift"] = max_tau_change

                    if abs(max_tau_change) > self.max_drift * 2:
                        failures.append(
                            f"Severe drift detected: Δτ={max_tau_change:.3f}"
                        )
                    elif abs(max_tau_change) > self.max_drift:
                        warnings.append(f"Drift detected: Δτ={max_tau_change:.3f}")

            # Check calibration
            if "calibration" in stability:
                cal = stability["calibration"]
                ece = cal.get("ece", 0)
                details["ece"] = ece

                if ece > self.max_ece * 1.5:
                    failures.append(
                        f"Poor calibration: ECE={ece:.3f} > {self.max_ece:.3f}"
                    )
                elif ece > self.max_ece:
                    warnings.append(f"Calibration degraded: ECE={ece:.3f}")

        # Check calibration R² if available
        if "calibration_result" in diagnostics:
            cal_result = diagnostics["calibration_result"]
            if hasattr(cal_result, "r2_score"):
                r2 = cal_result.r2_score
                details["calibration_r2"] = r2

                if r2 < self.min_r2 * 0.5:
                    failures.append(f"Very poor calibration fit: R²={r2:.3f}")
                elif r2 < self.min_r2:
                    warnings.append(f"Weak calibration fit: R²={r2:.3f}")

        # Determine status
        if failures:
            return GateResult(
                name=self.name,
                status=GateStatus.FAIL,
                message="; ".join(failures),
                details=details,
            )
        elif warnings:
            return GateResult(
                name=self.name,
                status=GateStatus.WARN,
                message="; ".join(warnings),
                details=details,
            )
        else:
            return GateResult(
                name=self.name,
                status=GateStatus.PASS,
                message="Judge quality acceptable",
                details=details,
            )


class OrthogonalityGate(DiagnosticGate):
    """Check DR orthogonality condition."""

    def __init__(
        self,
        max_score: float = 0.01,
        enabled: bool = True,
    ):
        """Initialize orthogonality gate.

        Args:
            max_score: Maximum allowed orthogonality score (should be near 0)
            enabled: Whether gate is active
        """
        super().__init__(name="DR Orthogonality", enabled=enabled)
        self.max_score = max_score

    def check(self, diagnostics: Dict[str, Any]) -> GateResult:
        """Check orthogonality diagnostics."""
        if not self.enabled:
            return GateResult(
                name=self.name,
                status=GateStatus.PASS,
                message="Gate disabled",
            )

        # Check if orthogonality scores exist
        if "orthogonality_scores" not in diagnostics:
            return GateResult(
                name=self.name,
                status=GateStatus.PASS,
                message="Not applicable (not using DR)",
            )

        orth_scores = diagnostics["orthogonality_scores"]

        # Find worst orthogonality violation
        max_violation = 0
        worst_policy = None

        for policy, scores in orth_scores.items():
            if "score" in scores:
                score = abs(scores["score"])
                if score > max_violation:
                    max_violation = score
                    worst_policy = policy

        # Check if significant
        details = {
            "max_violation": max_violation,
            "worst_policy": worst_policy,
        }

        if max_violation > self.max_score * 10:
            return GateResult(
                name=self.name,
                status=GateStatus.FAIL,
                message=f"Severe orthogonality violation: {max_violation:.4f} (policy: {worst_policy})",
                details=details,
                threshold=self.max_score,
                observed=max_violation,
            )
        elif max_violation > self.max_score:
            return GateResult(
                name=self.name,
                status=GateStatus.WARN,
                message=f"Orthogonality condition marginal: {max_violation:.4f}",
                details=details,
                threshold=self.max_score,
                observed=max_violation,
            )
        else:
            return GateResult(
                name=self.name,
                status=GateStatus.PASS,
                message=f"Orthogonality satisfied: {max_violation:.4f} < {self.max_score:.4f}",
                details=details,
            )


class MultiplicityGate(DiagnosticGate):
    """Check for multiple testing issues."""

    def __init__(
        self,
        min_policies_for_fdr: int = 5,
        fdr_alpha: float = 0.05,
        enabled: bool = True,
    ):
        """Initialize multiplicity gate.

        Args:
            min_policies_for_fdr: Minimum policies to trigger FDR control
            fdr_alpha: FDR level for Benjamini-Hochberg
            enabled: Whether gate is active
        """
        super().__init__(name="Multiplicity Control", enabled=enabled)
        self.min_policies_for_fdr = min_policies_for_fdr
        self.fdr_alpha = fdr_alpha

    def check(self, diagnostics: Dict[str, Any]) -> GateResult:
        """Check multiplicity diagnostics."""
        if not self.enabled:
            return GateResult(
                name=self.name,
                status=GateStatus.PASS,
                message="Gate disabled",
            )

        # Check number of policies
        n_policies = diagnostics.get("n_policies", 0)

        if n_policies < self.min_policies_for_fdr:
            return GateResult(
                name=self.name,
                status=GateStatus.PASS,
                message=f"FDR not needed ({n_policies} < {self.min_policies_for_fdr} policies)",
                details={"n_policies": n_policies},
            )

        # Check if FDR was applied
        if "fdr_results" not in diagnostics:
            return GateResult(
                name=self.name,
                status=GateStatus.WARN,
                message=f"FDR control recommended for {n_policies} policies",
                details={
                    "n_policies": n_policies,
                    "recommendation": "Apply Benjamini-Hochberg correction",
                },
            )

        fdr = diagnostics["fdr_results"]
        n_significant = fdr.get("n_significant", 0)

        return GateResult(
            name=self.name,
            status=GateStatus.PASS,
            message=f"FDR control applied: {n_significant}/{n_policies} significant at {self.fdr_alpha:.0%}",
            details={
                "n_policies": n_policies,
                "n_significant": n_significant,
                "fdr_level": self.fdr_alpha,
            },
        )


class NormalityGate(DiagnosticGate):
    """Check normality of influence functions for valid inference."""

    def __init__(
        self,
        min_shapiro_p: float = 0.01,
        max_qq_deviation: float = 0.1,
        enabled: bool = True,
    ):
        """Initialize normality gate.

        Args:
            min_shapiro_p: Minimum p-value for Shapiro-Wilk test
            max_qq_deviation: Maximum deviation from normal Q-Q line
            enabled: Whether gate is active
        """
        super().__init__(name="Inference Validity", enabled=enabled)
        self.min_shapiro_p = min_shapiro_p
        self.max_qq_deviation = max_qq_deviation

    def check(self, diagnostics: Dict[str, Any]) -> GateResult:
        """Check normality diagnostics."""
        if not self.enabled:
            return GateResult(
                name=self.name,
                status=GateStatus.PASS,
                message="Gate disabled",
            )

        # Check if normality diagnostics exist
        if "normality_tests" not in diagnostics:
            return GateResult(
                name=self.name,
                status=GateStatus.PASS,
                message="Normality tests not performed",
            )

        norm_tests = diagnostics["normality_tests"]

        failures: List[str] = []
        warnings: List[str] = []

        for policy, tests in norm_tests.items():
            if "shapiro_p" in tests:
                p_val = tests["shapiro_p"]
                if p_val < self.min_shapiro_p:
                    warnings.append(f"{policy}: non-normal (p={p_val:.3f})")

            if "qq_max_deviation" in tests:
                deviation = tests["qq_max_deviation"]
                if deviation > self.max_qq_deviation:
                    warnings.append(f"{policy}: Q-Q deviation={deviation:.3f}")

        if failures:
            return GateResult(
                name=self.name,
                status=GateStatus.FAIL,
                message="; ".join(failures),
            )
        elif warnings:
            return GateResult(
                name=self.name,
                status=GateStatus.WARN,
                message="Non-normality detected; consider robust inference",
                details={"warnings": warnings},
            )
        else:
            return GateResult(
                name=self.name,
                status=GateStatus.PASS,
                message="Influence functions approximately normal",
            )


# ========== Gate Runner ==========


class GateRunner:
    """Orchestrates diagnostic gates."""

    def __init__(
        self,
        gates: Optional[List[DiagnosticGate]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize gate runner.

        Args:
            gates: List of gates to run (uses defaults if None)
            config: Configuration dictionary for gates
        """
        if gates is None:
            gates = self._default_gates(config)
        self.gates = gates
        self.config = config or {}

    def _default_gates(self, config: Optional[Dict[str, Any]]) -> List[DiagnosticGate]:
        """Create default set of gates."""
        config = config or {}

        gates: List[DiagnosticGate] = []

        # Overlap gate
        overlap_config = config.get("overlap", {})
        gates.append(
            OverlapGate(
                min_ess=overlap_config.get("min_ess", 1000),
                min_tail_index=overlap_config.get("min_tail_index", 2.0),
                enabled=overlap_config.get("enabled", True),
            )
        )

        # Judge gate
        judge_config = config.get("judge", {})
        gates.append(
            JudgeGate(
                max_drift=judge_config.get("max_drift", 0.05),
                max_ece=judge_config.get("max_ece", 0.1),
                min_r2=judge_config.get("min_r2", 0.5),
                enabled=judge_config.get("enabled", True),
            )
        )

        # Orthogonality gate (for DR)
        orth_config = config.get("orthogonality", {})
        gates.append(
            OrthogonalityGate(
                max_score=orth_config.get("max_score", 0.01),
                enabled=orth_config.get("enabled", True),
            )
        )

        # Multiplicity gate
        mult_config = config.get("multiplicity", {})
        gates.append(
            MultiplicityGate(
                min_policies_for_fdr=mult_config.get("min_policies", 5),
                fdr_alpha=mult_config.get("fdr_alpha", 0.05),
                enabled=mult_config.get("enabled", True),
            )
        )

        # Normality gate
        norm_config = config.get("normality", {})
        gates.append(
            NormalityGate(
                min_shapiro_p=norm_config.get("min_shapiro_p", 0.01),
                max_qq_deviation=norm_config.get("max_qq_deviation", 0.1),
                enabled=norm_config.get("enabled", False),  # Off by default
            )
        )

        return gates

    def run_all(
        self,
        diagnostics: Dict[str, Any],
        fail_fast: bool = False,
    ) -> GateReport:
        """Run all gates and create report.

        Args:
            diagnostics: Dictionary of diagnostic results
            fail_fast: Stop at first failure

        Returns:
            GateReport with aggregated results
        """
        results = []

        for gate in self.gates:
            if not gate.enabled:
                continue

            try:
                result = gate.check(diagnostics)
                results.append(result)

                # Log gate result
                if result.status == GateStatus.FAIL:
                    logger.warning(f"Gate {gate.name} failed: {result.message}")
                elif result.status == GateStatus.WARN:
                    logger.info(f"Gate {gate.name} warned: {result.message}")
                else:
                    logger.debug(f"Gate {gate.name} passed")

                # Fail fast if requested
                if fail_fast and result.status == GateStatus.FAIL:
                    break

            except Exception as e:
                logger.error(f"Gate {gate.name} error: {e}")
                results.append(
                    GateResult(
                        name=gate.name,
                        status=GateStatus.WARN,
                        message=f"Gate check failed: {str(e)}",
                    )
                )

        return GateReport.from_results(results)

    def should_proceed(self, report: GateReport) -> bool:
        """Determine if estimation should proceed given gate results.

        Args:
            report: Gate report

        Returns:
            True if safe to proceed (no failures)
        """
        return report.overall_status != GateStatus.FAIL


def run_diagnostic_gates(
    diagnostics: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> GateReport:
    """Convenience function to run standard gates.

    Args:
        diagnostics: Diagnostic results to check
        config: Gate configuration
        verbose: Whether to print results

    Returns:
        GateReport with results
    """
    runner = GateRunner(config=config)
    report = runner.run_all(diagnostics)

    if verbose:
        print(report.format_terminal(verbose=verbose))

    return report
