"""Tests for diagnostic gates (Phase 4)."""

import numpy as np
import pytest
from typing import Dict, Any, List

from cje.utils.diagnostics.gates import (
    GateStatus,
    GateResult,
    GateReport,
    DiagnosticGate,
    OverlapGate,
    JudgeGate,
    OrthogonalityGate,
    MultiplicityGate,
    NormalityGate,
    GateRunner,
    run_diagnostic_gates,
)


class TestGateResult:
    """Test GateResult dataclass."""

    def test_gate_result_creation(self) -> None:
        """Test creating gate results."""
        result = GateResult(
            name="TestGate",
            status=GateStatus.PASS,
            message="All checks passed",
            threshold=1.0,
            observed=1.5,
        )

        assert result.name == "TestGate"
        assert result.status == GateStatus.PASS
        assert result.passed is True

        # Test fail result
        fail_result = GateResult(
            name="TestGate",
            status=GateStatus.FAIL,
            message="Check failed",
        )
        assert fail_result.passed is False

    def test_gate_result_serialization(self) -> None:
        """Test converting to dict."""
        result = GateResult(
            name="TestGate",
            status=GateStatus.WARN,
            message="Warning",
            details={"key": "value"},
        )

        d = result.to_dict()
        assert d["name"] == "TestGate"
        assert d["status"] == "WARN"
        assert d["details"]["key"] == "value"


class TestGateReport:
    """Test GateReport aggregation."""

    def test_report_creation(self) -> None:
        """Test creating report from results."""
        results = [
            GateResult("Gate1", GateStatus.PASS, "OK"),
            GateResult("Gate2", GateStatus.WARN, "Warning"),
            GateResult("Gate3", GateStatus.FAIL, "Failed"),
        ]

        report = GateReport.from_results(results)

        assert report.overall_status == GateStatus.FAIL
        assert report.n_passed == 1
        assert report.n_warned == 1
        assert report.n_failed == 1
        assert "failed" in report.summary.lower()

    def test_report_all_pass(self) -> None:
        """Test report when all gates pass."""
        results = [
            GateResult("Gate1", GateStatus.PASS, "OK"),
            GateResult("Gate2", GateStatus.PASS, "OK"),
        ]

        report = GateReport.from_results(results)

        assert report.overall_status == GateStatus.PASS
        assert report.n_passed == 2
        assert report.n_warned == 0
        assert report.n_failed == 0

    def test_report_formatting(self) -> None:
        """Test terminal formatting."""
        results = [
            GateResult(
                "TestGate",
                GateStatus.FAIL,
                "Threshold exceeded",
                threshold=1.0,
                observed=0.5,
            ),
        ]

        report = GateReport.from_results(results)
        formatted = report.format_terminal(verbose=False)

        assert "DIAGNOSTIC GATES" in formatted
        assert "TestGate" in formatted
        assert "Threshold exceeded" in formatted


class TestOverlapGate:
    """Test overlap diagnostic gate."""

    def test_sufficient_overlap(self) -> None:
        """Test when overlap is sufficient."""
        gate = OverlapGate(min_ess=100, min_tail_index=2.0)

        diagnostics = {
            "weight_diagnostics": {
                "policy1": {"ess": 150, "tail_index": 2.5},
                "policy2": {"ess": 200, "tail_index": 3.0},
            }
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.PASS
        assert "Sufficient overlap" in result.message

    def test_low_ess(self) -> None:
        """Test when ESS is too low."""
        gate = OverlapGate(min_ess=1000, min_tail_index=2.0)

        diagnostics = {
            "weight_diagnostics": {
                "policy1": {"ess": 500, "tail_index": 2.5},
            }
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.WARN
        assert "marginal" in result.message.lower()

    def test_very_low_ess(self) -> None:
        """Test when ESS is critically low."""
        gate = OverlapGate(min_ess=1000, min_tail_index=2.0)

        diagnostics = {
            "weight_diagnostics": {
                "policy1": {"ess": 100, "tail_index": 2.5},
            }
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.FAIL
        assert "too low" in result.message.lower()

    def test_heavy_tails(self) -> None:
        """Test when tail index indicates heavy tails."""
        gate = OverlapGate(min_ess=100, min_tail_index=2.0)

        diagnostics = {
            "weight_diagnostics": {
                "policy1": {"ess": 150, "tail_index": 1.8},
            }
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.WARN
        assert "Heavy tails" in result.message

    def test_extreme_tails(self) -> None:
        """Test when tail index is extremely low."""
        gate = OverlapGate(min_ess=100, min_tail_index=2.0)

        diagnostics = {
            "weight_diagnostics": {
                "policy1": {"ess": 150, "tail_index": 1.2},
            }
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.FAIL
        assert "Extremely heavy tails" in result.message


class TestJudgeGate:
    """Test judge quality gate."""

    def test_no_drift(self) -> None:
        """Test when no drift is detected."""
        gate = JudgeGate(max_drift=0.05, max_ece=0.1)

        diagnostics = {
            "stability_diagnostics": {
                "drift_detection": {"has_drift": False},
                "calibration": {"ece": 0.05},
            }
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.PASS

    def test_drift_detected(self) -> None:
        """Test when drift is detected."""
        gate = JudgeGate(max_drift=0.05, max_ece=0.1)

        diagnostics = {
            "stability_diagnostics": {
                "drift_detection": {
                    "has_drift": True,
                    "max_tau_change": 0.08,
                },
            }
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.WARN
        assert "Drift detected" in result.message

    def test_severe_drift(self) -> None:
        """Test when severe drift is detected."""
        gate = JudgeGate(max_drift=0.05, max_ece=0.1)

        diagnostics = {
            "stability_diagnostics": {
                "drift_detection": {
                    "has_drift": True,
                    "max_tau_change": 0.15,
                },
            }
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.FAIL
        assert "Severe drift" in result.message

    def test_poor_calibration(self) -> None:
        """Test when calibration is poor."""
        gate = JudgeGate(max_ece=0.1)

        diagnostics = {
            "stability_diagnostics": {
                "calibration": {"ece": 0.2},
            }
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.FAIL
        assert "Poor calibration" in result.message


class TestOrthogonalityGate:
    """Test DR orthogonality gate."""

    def test_orthogonality_satisfied(self) -> None:
        """Test when orthogonality is satisfied."""
        gate = OrthogonalityGate(max_score=0.01)

        diagnostics = {
            "orthogonality_scores": {
                "policy1": {"score": 0.005},
                "policy2": {"score": 0.003},
            }
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.PASS
        assert "satisfied" in result.message.lower()

    def test_orthogonality_marginal(self) -> None:
        """Test when orthogonality is marginal."""
        gate = OrthogonalityGate(max_score=0.01)

        diagnostics = {
            "orthogonality_scores": {
                "policy1": {"score": 0.02},
            }
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.WARN
        assert "marginal" in result.message.lower()

    def test_orthogonality_violated(self) -> None:
        """Test when orthogonality is severely violated."""
        gate = OrthogonalityGate(max_score=0.01)

        diagnostics = {
            "orthogonality_scores": {
                "policy1": {"score": 0.2},
            }
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.FAIL
        assert "Severe" in result.message

    def test_not_applicable(self) -> None:
        """Test when DR is not being used."""
        gate = OrthogonalityGate()

        diagnostics: Dict[str, Any] = {}  # No orthogonality scores

        result = gate.check(diagnostics)
        assert result.status == GateStatus.PASS
        assert "Not applicable" in result.message


class TestMultiplicityGate:
    """Test multiplicity control gate."""

    def test_few_policies(self) -> None:
        """Test when too few policies for FDR."""
        gate = MultiplicityGate(min_policies_for_fdr=5)

        diagnostics = {"n_policies": 3}

        result = gate.check(diagnostics)
        assert result.status == GateStatus.PASS
        assert "not needed" in result.message.lower()

    def test_fdr_applied(self) -> None:
        """Test when FDR is properly applied."""
        gate = MultiplicityGate(min_policies_for_fdr=5, fdr_alpha=0.05)

        diagnostics = {
            "n_policies": 10,
            "fdr_results": {
                "n_significant": 3,
            },
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.PASS
        assert "FDR control applied" in result.message
        assert "3/10" in result.message

    def test_fdr_missing(self) -> None:
        """Test when FDR should be applied but isn't."""
        gate = MultiplicityGate(min_policies_for_fdr=5)

        diagnostics = {"n_policies": 10}

        result = gate.check(diagnostics)
        assert result.status == GateStatus.WARN
        assert "recommended" in result.message.lower()


class TestNormalityGate:
    """Test normality gate for inference validity."""

    def test_normal_data(self) -> None:
        """Test when data is approximately normal."""
        gate = NormalityGate(min_shapiro_p=0.01, max_qq_deviation=0.1)

        diagnostics = {
            "normality_tests": {
                "policy1": {"shapiro_p": 0.5, "qq_max_deviation": 0.05},
            }
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.PASS
        assert "approximately normal" in result.message.lower()

    def test_non_normal(self) -> None:
        """Test when data is non-normal."""
        gate = NormalityGate(min_shapiro_p=0.01, max_qq_deviation=0.1)

        diagnostics = {
            "normality_tests": {
                "policy1": {"shapiro_p": 0.005},
            }
        }

        result = gate.check(diagnostics)
        assert result.status == GateStatus.WARN
        assert "Non-normality" in result.message


class TestGateRunner:
    """Test gate orchestration."""

    def test_default_gates(self) -> None:
        """Test runner with default gates."""
        runner = GateRunner()

        assert len(runner.gates) >= 4  # At least 4 default gates
        assert any(isinstance(g, OverlapGate) for g in runner.gates)
        assert any(isinstance(g, JudgeGate) for g in runner.gates)

    def test_custom_config(self) -> None:
        """Test runner with custom configuration."""
        config = {
            "overlap": {"min_ess": 500, "enabled": True},
            "judge": {"enabled": False},
        }

        runner = GateRunner(config=config)

        # Check overlap gate configured
        overlap_gate = next(g for g in runner.gates if isinstance(g, OverlapGate))
        assert overlap_gate.min_ess == 500

        # Check judge gate disabled
        judge_gate = next(g for g in runner.gates if isinstance(g, JudgeGate))
        assert judge_gate.enabled is False

    def test_run_all_pass(self) -> None:
        """Test running gates when all pass."""
        runner = GateRunner()

        diagnostics = {
            "weight_diagnostics": {
                "policy1": {"ess": 2000, "tail_index": 3.0},
            },
            "n_policies": 2,
        }

        report = runner.run_all(diagnostics)

        assert report.overall_status in [GateStatus.PASS, GateStatus.WARN]
        assert report.n_failed == 0

    def test_fail_fast(self) -> None:
        """Test fail-fast mode."""

        # Create custom gates that will fail
        class AlwaysFailGate(DiagnosticGate):
            def check(self, diagnostics: Dict[str, Any]) -> GateResult:
                return GateResult(
                    name=self.name,
                    status=GateStatus.FAIL,
                    message="Always fails",
                )

        gates: List[DiagnosticGate] = [
            AlwaysFailGate("Gate1"),
            AlwaysFailGate("Gate2"),
            AlwaysFailGate("Gate3"),
        ]

        runner = GateRunner(gates=gates)
        report = runner.run_all({}, fail_fast=True)

        # Should stop after first failure
        assert len(report.gate_results) == 1
        assert report.n_failed == 1

    def test_should_proceed(self) -> None:
        """Test decision logic."""
        runner = GateRunner()

        # Should proceed when passing
        pass_report = GateReport(
            overall_status=GateStatus.PASS,
            gate_results=[],
            summary="All passed",
        )
        assert runner.should_proceed(pass_report) is True

        # Should proceed with warnings
        warn_report = GateReport(
            overall_status=GateStatus.WARN,
            gate_results=[],
            summary="Warnings",
        )
        assert runner.should_proceed(warn_report) is True

        # Should not proceed with failures
        fail_report = GateReport(
            overall_status=GateStatus.FAIL,
            gate_results=[],
            summary="Failed",
        )
        assert runner.should_proceed(fail_report) is False


class TestConvenienceFunction:
    """Test convenience function."""

    def test_run_diagnostic_gates(self) -> None:
        """Test the convenience function."""
        diagnostics = {
            "weight_diagnostics": {
                "policy1": {"ess": 1500, "tail_index": 2.5},
            },
            "n_policies": 3,
        }

        report = run_diagnostic_gates(
            diagnostics,
            config={"overlap": {"min_ess": 1000}},
            verbose=False,
        )

        assert isinstance(report, GateReport)
        assert report.overall_status in [GateStatus.PASS, GateStatus.WARN]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_diagnostics(self) -> None:
        """Test with empty diagnostics."""
        runner = GateRunner()
        report = runner.run_all({})

        # Should handle gracefully
        assert isinstance(report, GateReport)

    def test_gate_exception(self) -> None:
        """Test when a gate throws an exception."""

        class BrokenGate(DiagnosticGate):
            def check(self, diagnostics: Dict[str, Any]) -> GateResult:
                raise ValueError("Broken gate")

        runner = GateRunner(gates=[BrokenGate("Broken")])
        report = runner.run_all({})

        # Should catch and report as warning
        assert report.n_failed == 0
        assert report.n_warned == 1
        assert "failed" in report.gate_results[0].message.lower()

    def test_disabled_gate(self) -> None:
        """Test disabled gates are skipped."""
        gate = OverlapGate(enabled=False)
        result = gate.check({})

        assert result.status == GateStatus.PASS
        assert "disabled" in result.message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
