"""Tests for HERA (Hellinger–ESS Raw Audit)."""

import numpy as np
import pytest
from cje.diagnostics.hera import (
    hera_hellinger,
    hera_ess,
    hera_threshold,
    hera_audit,
    hera_drill_down,
    hera_audit_weights,
    HERAMetrics,
)


class TestHERAHellinger:
    """Test HERA's Hellinger affinity computation."""

    def test_perfect_overlap(self) -> None:
        """Test uniform weights (log-ratios = 0)."""
        delta_log = np.zeros(1000)
        h = hera_hellinger(delta_log)
        assert abs(h - 1.0) < 0.01

    def test_no_overlap(self) -> None:
        """Test catastrophic mismatch."""
        # Most log-ratios very negative, a few very positive
        delta_log = np.concatenate([np.full(950, -6.0), np.full(50, 3.0)])
        h = hera_hellinger(delta_log)
        assert h < 0.3  # Poor overlap

    def test_moderate_shift(self) -> None:
        """Test moderate distribution shift."""
        np.random.seed(42)
        delta_log = np.random.normal(0.5, 0.5, 1000)
        h = hera_hellinger(delta_log)
        assert 0.85 < h <= 1.0  # Good overlap with small shift

    def test_empty_input(self) -> None:
        """Test empty array handling."""
        assert hera_hellinger(np.array([])) == 0.0

    def test_numerical_stability(self) -> None:
        """Test with extreme log values."""
        delta_log = np.array([-100, 100, 0])  # Very extreme
        h = hera_hellinger(delta_log)
        assert 0 <= h <= 1.0  # Should stay bounded


class TestHERAESS:
    """Test HERA's ESS computation."""

    def test_uniform_weights(self) -> None:
        """Test ESS for uniform weights."""
        delta_log = np.zeros(1000)
        e = hera_ess(delta_log)
        assert abs(e - 1.0) < 0.01

    def test_concentrated_weights(self) -> None:
        """Test ESS with weight concentration."""
        # 99% near zero, 1% large
        delta_log = np.concatenate([np.full(990, -5.0), np.full(10, 5.0)])
        e = hera_ess(delta_log)
        assert e < 0.15  # Very low ESS

    def test_log_normal_weights(self) -> None:
        """Test with log-normal distribution."""
        np.random.seed(42)
        delta_log = np.random.normal(0, 1.0, 1000)
        e = hera_ess(delta_log)
        # Theory: ESS ≈ 1/(1 + exp(σ²) - 1) ≈ 0.37 for σ=1
        assert 0.3 < e < 0.45

    def test_empty_input(self) -> None:
        """Test empty array handling."""
        assert hera_ess(np.array([])) == 0.0


class TestHERAThreshold:
    """Test HERA's auto-tuning."""

    def test_basic_formula(self) -> None:
        """Test threshold computation."""
        # For n=5000, δ=0.03 (±3%)
        threshold = hera_threshold(5000, 0.03)
        expected = 0.9604 / (5000 * 0.03**2)
        assert abs(threshold - expected) < 0.001
        assert abs(threshold - 0.213) < 0.01  # Should be ~21.3%

    def test_larger_sample(self) -> None:
        """Test with larger n."""
        # For n=100000, δ=0.01 (±1%)
        threshold = hera_threshold(100000, 0.01)
        expected = 0.9604 / (100000 * 0.01**2)
        assert abs(threshold - expected) < 0.001
        assert abs(threshold - 0.096) < 0.001  # Should be ~9.6%

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        assert hera_threshold(0, 0.01) == 0.10  # Default
        assert hera_threshold(1000, 0) == 0.10  # Default
        assert hera_threshold(-1000, 0.01) == 0.10  # Default


class TestHERAAudit:
    """Test HERA's audit function."""

    def test_critical_audit(self) -> None:
        """Test HERA critical gate."""
        # Very poor overlap
        delta_log = np.concatenate([np.full(950, -10.0), np.full(50, 2.0)])
        metrics = hera_audit(delta_log)

        assert metrics.hera_status == "critical"
        assert "AUDIT FAILED" in metrics.recommendation
        assert not metrics.passes_audit
        # Either H < 0.20 OR E < 0.10 triggers critical
        assert metrics.hellinger_affinity < 0.35 or metrics.ess_raw_fraction < 0.10

    def test_warning_audit(self) -> None:
        """Test HERA warning gate."""
        # Create a scenario that triggers warning
        delta_log = np.concatenate(
            [
                np.full(850, -1.0),  # Most have moderate underweighting
                np.full(150, 3.5),  # Some have high overweighting
            ]
        )
        metrics = hera_audit(delta_log)

        assert metrics.hera_status == "warning"
        assert "WARNING" in metrics.recommendation
        assert metrics.passes_audit  # Warning still passes
        assert 0.10 < metrics.ess_raw_fraction < 0.20

    def test_ok_audit(self) -> None:
        """Test HERA approval."""
        # Slight shift only
        np.random.seed(42)
        delta_log = np.random.normal(0, 0.3, 1000)
        metrics = hera_audit(delta_log)

        assert metrics.hera_status == "ok"
        assert "APPROVED" in metrics.recommendation
        assert metrics.passes_audit
        assert metrics.hellinger_affinity > 0.35
        assert metrics.ess_raw_fraction > 0.20

    def test_auto_tuning_integration(self) -> None:
        """Test auto-tuning in HERA audit."""
        delta_log = np.zeros(5000)  # Perfect overlap
        metrics = hera_audit(delta_log, n_samples=5000, target_ci_halfwidth=0.03)

        assert metrics.auto_tuned_threshold is not None
        assert abs(metrics.auto_tuned_threshold - 0.213) < 0.01
        # Should meet threshold with perfect overlap
        assert metrics.ess_raw_fraction >= metrics.auto_tuned_threshold

    def test_summary_format(self) -> None:
        """Test HERA summary string."""
        delta_log = np.random.normal(0, 1.0, 1000)
        metrics = hera_audit(delta_log)
        summary = metrics.summary()

        assert "HERA:" in summary
        assert "H=" in summary
        assert "E=" in summary
        assert metrics.hera_status.upper() in summary


class TestHERADrillDown:
    """Test HERA's drill-down diagnostics."""

    def test_drill_down_by_index(self) -> None:
        """Test binned HERA analysis."""
        np.random.seed(42)
        n = 1000

        # Create varying overlap by index
        index = np.linspace(0, 1, n)
        # Worse overlap at higher index values
        delta_log = -2.0 * index + np.random.normal(0, 0.5, n)

        drill_down = hera_drill_down(delta_log, index, n_bins=5)

        assert "bins" in drill_down
        assert len(drill_down["bins"]) == 5

        # Check that overlap degrades with index
        bins = drill_down["bins"]
        first_bin_h = bins[0]["hellinger"]
        last_bin_h = bins[-1]["hellinger"]
        assert first_bin_h > last_bin_h

        # Check status assignment (first bin should be better than last)
        assert bins[0]["status"] in ["ok", "warning"]
        # Last bin has worse overlap but may still be "ok" due to normalization
        # Just check it's not better than first
        first_status_rank = ["ok", "warning", "critical"].index(bins[0]["status"])
        last_status_rank = ["ok", "warning", "critical"].index(bins[-1]["status"])
        assert last_status_rank >= first_status_rank  # Last is same or worse

    def test_empty_drill_down(self) -> None:
        """Test drill-down with empty input."""
        drill_down = hera_drill_down(np.array([]), np.array([]))
        assert drill_down["bins"] == []


class TestHERAWeights:
    """Test HERA with importance weights (not log-ratios)."""

    def test_weights_to_audit(self) -> None:
        """Test HERA audit from importance weights."""
        # Create weights with poor overlap
        weights = np.concatenate([np.full(950, 0.1), np.full(50, 19.0)])

        metrics = hera_audit_weights(weights)

        # Should detect poor/critical overlap
        assert metrics.hera_status in ["warning", "critical"]
        assert metrics.hellinger_affinity < 0.6
        assert metrics.ess_raw_fraction < 0.2

    def test_zero_weights(self) -> None:
        """Test handling of zero weights."""
        weights = np.array([0.0, 1.0, 2.0, 0.0, 3.0])
        metrics = hera_audit_weights(weights)

        # Should handle zeros gracefully
        assert 0 <= metrics.hellinger_affinity <= 1
        assert 0 <= metrics.ess_raw_fraction <= 1


def test_hera_production_scenario() -> None:
    """Test HERA with realistic scenario from the paper."""
    np.random.seed(42)

    # Simulate realistic scenario: two policies with moderate overlap
    # Logging policy: normal(0, 1)
    # Target policy: normal(0.5, 1.2)
    n = 4900
    x = np.random.normal(0, 1, n)

    # Log importance ratios
    delta_log = -0.5 * ((x - 0.5) ** 2 / 1.2**2 - x**2) + np.log(1.0 / 1.2)

    # Run HERA audit with auto-tuning for ±3% CI
    metrics = hera_audit(delta_log, n_samples=n, target_ci_halfwidth=0.03)

    # Should be in warning/ok range
    assert metrics.hera_status in ["warning", "ok"]

    # Auto-tuned threshold should be ~21.8% for n=4900, δ=0.03
    assert metrics.auto_tuned_threshold is not None
    assert abs(metrics.auto_tuned_threshold - 0.218) < 0.02

    # Log the results
    print(f"\nHERA Production Test:")
    print(f"  H (Hellinger): {metrics.hellinger_affinity:.1%}")
    print(f"  E (ESS): {metrics.ess_raw_fraction:.1%}")
    print(f"  Status: {metrics.hera_status}")
    print(f"  Auto-threshold: {metrics.auto_tuned_threshold:.1%}")
    print(f"  Recommendation: {metrics.recommendation}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
