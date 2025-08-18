"""Tests for overlap metrics (Hellinger affinity and related diagnostics)."""

import numpy as np
import pytest
from cje.diagnostics.overlap import (
    hellinger_affinity,
    compute_auto_tuned_threshold,
    compute_overlap_metrics,
    diagnose_overlap_problems,
    OverlapMetrics,
)


class TestHellingerAffinity:
    """Test Hellinger affinity computation."""

    def test_perfect_overlap(self) -> None:
        """Test that uniform weights give affinity ≈ 1."""
        weights = np.ones(1000)
        affinity = hellinger_affinity(weights)
        assert abs(affinity - 1.0) < 0.01

    def test_no_overlap(self) -> None:
        """Test catastrophic mismatch."""
        # Most weights near zero, a few huge
        weights = np.concatenate(
            [np.full(950, 0.001), np.full(50, 19.95)]  # Mean still 1
        )
        affinity = hellinger_affinity(weights)
        assert affinity < 0.5  # Poor overlap

    def test_bimodal_weights(self) -> None:
        """Test bimodal distribution."""
        # 90% low, 10% high (mean 1)
        weights = np.concatenate([np.full(900, 0.1), np.full(100, 9.1)])
        affinity = hellinger_affinity(weights)
        # Should be around 0.9*sqrt(0.1) + 0.1*sqrt(9.1) ≈ 0.58
        assert 0.4 < affinity < 0.7

    def test_log_normal_weights(self) -> None:
        """Test log-normal weights (common in practice)."""
        np.random.seed(42)
        # LogNormal with mean 1
        sigma = 1.0
        weights = np.random.lognormal(-(sigma**2) / 2, sigma, 1000)
        weights = weights / weights.mean()  # Ensure mean 1

        affinity = hellinger_affinity(weights)
        # For σ=1, theory says affinity ≈ exp(-σ²/8) ≈ 0.88
        assert 0.8 < affinity < 0.95

    def test_empty_input(self) -> None:
        """Test empty array handling."""
        assert np.isnan(hellinger_affinity(np.array([])))

    def test_negative_weights(self) -> None:
        """Test handling of invalid negative weights."""
        weights = np.array([1, 2, -1, 3])  # Has negative
        affinity = hellinger_affinity(weights)
        # Should ignore the negative weight
        assert not np.isnan(affinity)

    def test_numerical_stability(self) -> None:
        """Test with extreme values."""
        weights = np.array([1e-10, 1e10, 1.0])
        affinity = hellinger_affinity(weights)
        assert 0 <= affinity <= 1  # Should stay in bounds


class TestAutoTunedThreshold:
    """Test auto-tuning of ESS thresholds."""

    def test_basic_calculation(self) -> None:
        """Test threshold calculation for typical values."""
        # For n=10000, target ±1% CI
        threshold = compute_auto_tuned_threshold(10000, 0.01)
        # Should be 0.9604/(10000*0.01²) = 0.9604/1 ≈ 0.9604
        assert abs(threshold - 0.9604) < 0.001

    def test_larger_sample(self) -> None:
        """Test with larger sample size."""
        # For n=100000, target ±1% CI
        threshold = compute_auto_tuned_threshold(100000, 0.01)
        # Should be 0.9604/(100000*0.01²) = 0.9604/10 ≈ 0.09604
        assert abs(threshold - 0.09604) < 0.001

    def test_looser_target(self) -> None:
        """Test with looser CI target."""
        # For n=10000, target ±2% CI
        threshold = compute_auto_tuned_threshold(10000, 0.02)
        # Should be 0.9604/(10000*0.04) = 0.9604/400 ≈ 0.2401
        assert abs(threshold - 0.2401) < 0.001

    def test_warning_level(self) -> None:
        """Test warning level (half of critical)."""
        critical = compute_auto_tuned_threshold(10000, 0.01, "critical")
        warning = compute_auto_tuned_threshold(10000, 0.01, "warning")
        assert abs(warning - critical / 2) < 0.001

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        # Invalid inputs should return default
        assert compute_auto_tuned_threshold(0, 0.01) == 0.1
        assert compute_auto_tuned_threshold(1000, 0) == 0.1
        assert compute_auto_tuned_threshold(-1000, 0.01) == 0.1


class TestOverlapMetrics:
    """Test comprehensive overlap metrics computation."""

    def test_good_overlap(self) -> None:
        """Test metrics for good overlap."""
        weights = np.ones(1000) + np.random.normal(0, 0.1, 1000)
        weights = np.maximum(weights, 0.1)  # Keep positive
        weights = weights / weights.mean()

        metrics = compute_overlap_metrics(weights)

        assert metrics.overlap_quality == "good"
        assert metrics.hellinger_affinity > 0.5
        assert metrics.ess_fraction > 0.8
        assert metrics.recommended_method == "ips"
        assert metrics.can_calibrate

    def test_poor_overlap(self) -> None:
        """Test metrics for poor overlap."""
        # Create weights with poor overlap
        weights = np.concatenate([np.full(900, 0.01), np.full(100, 9.91)])

        metrics = compute_overlap_metrics(weights)

        assert metrics.overlap_quality in ["poor", "marginal"]
        assert metrics.hellinger_affinity < 0.5
        assert metrics.ess_fraction < 0.2
        assert metrics.recommended_method in ["calibrated-ips", "dr", "refuse"]

    def test_catastrophic_overlap(self) -> None:
        """Test metrics for catastrophic overlap."""
        # Extreme mismatch
        weights = np.concatenate([np.full(990, 0.001), np.full(10, 99.01)])

        metrics = compute_overlap_metrics(weights)

        assert metrics.overlap_quality == "catastrophic"
        assert metrics.hellinger_affinity < 0.2
        assert metrics.recommended_method == "refuse"
        assert not metrics.can_calibrate  # Too far gone

    def test_heavy_tails(self) -> None:
        """Test detection of heavy tails."""
        np.random.seed(42)
        # Create power-law tailed weights
        n = 100
        weights = 1 / (np.arange(1, n + 1) ** 0.8)  # Heavy tail
        weights = weights / weights.mean()

        metrics = compute_overlap_metrics(weights, compute_tail_index=True)

        if metrics.tail_index is not None:
            # Should detect heavy tails
            assert metrics.tail_index < 3
            if metrics.tail_index < 2:
                assert metrics.recommended_method == "dr"

    def test_auto_tuning(self) -> None:
        """Test auto-tuning integration."""
        weights = np.ones(10000) * 1.0

        metrics = compute_overlap_metrics(
            weights, target_ci_halfwidth=0.01, auto_tune_threshold=True
        )

        assert metrics.auto_tuned_threshold is not None
        # Should be 0.9604/(10000*0.01²) = 0.9604
        assert abs(metrics.auto_tuned_threshold - 0.9604) < 0.001

    def test_confidence_penalty(self) -> None:
        """Test confidence interval penalty calculation."""
        # Perfect overlap
        weights = np.ones(1000)
        metrics = compute_overlap_metrics(weights)
        assert abs(metrics.confidence_penalty - 1.0) < 0.1

        # 25% ESS should give 2x penalty
        weights = np.concatenate([np.full(750, 0.333), np.full(250, 2.999)])
        metrics = compute_overlap_metrics(weights)
        # ESS ≈ 25%, penalty ≈ 1/sqrt(0.25) = 2
        assert 1.5 < metrics.confidence_penalty < 2.5


class TestDiagnoseOverlapProblems:
    """Test diagnostic message generation."""

    def test_good_overlap_message(self) -> None:
        """Test message for good overlap."""
        metrics = OverlapMetrics(
            hellinger_affinity=0.85,
            ess_fraction=0.75,
            tail_index=3.5,
            overlap_quality="good",
            efficiency_loss=0.25,
            can_calibrate=True,
            recommended_method="ips",
            confidence_penalty=1.15,
        )

        should_proceed, msg = diagnose_overlap_problems(metrics, verbose=False)

        assert should_proceed
        assert "Good overlap" in msg
        assert "85%" in msg

    def test_catastrophic_message(self) -> None:
        """Test message for catastrophic overlap."""
        metrics = OverlapMetrics(
            hellinger_affinity=0.15,
            ess_fraction=0.02,
            tail_index=None,
            overlap_quality="catastrophic",
            efficiency_loss=0.98,
            can_calibrate=False,
            recommended_method="refuse",
            confidence_penalty=7.0,
        )

        should_proceed, msg = diagnose_overlap_problems(metrics, verbose=False)

        assert not should_proceed
        assert "Catastrophic" in msg
        assert "fundamentally incompatible" in msg
        assert "98%" in msg  # efficiency loss

    def test_heavy_tail_warning(self) -> None:
        """Test heavy tail warning."""
        metrics = OverlapMetrics(
            hellinger_affinity=0.6,
            ess_fraction=0.4,
            tail_index=1.8,
            overlap_quality="marginal",
            efficiency_loss=0.6,
            can_calibrate=True,
            recommended_method="dr",
            confidence_penalty=1.6,
        )

        should_proceed, msg = diagnose_overlap_problems(metrics, verbose=False)

        assert should_proceed
        assert "Heavy tails" in msg
        assert "α=1.8" in msg
        assert "doubly-robust" in msg.lower()


def test_integration_with_real_weights() -> None:
    """Test with realistic importance weights."""
    np.random.seed(42)

    # Simulate realistic scenario: two policies with moderate overlap
    n = 1000
    # Logging policy: normal(0, 1)
    # Target policy: normal(0.5, 1.2)
    x = np.random.normal(0, 1, n)

    # Importance weights based on likelihood ratio
    weights = np.exp(-0.5 * ((x - 0.5) ** 2 / 1.2**2 - x**2)) * (1.0 / 1.2)
    weights = weights / weights.mean()

    # Compute metrics
    metrics = compute_overlap_metrics(weights, target_ci_halfwidth=0.01)

    # Should show moderate to good overlap
    assert 0.3 < metrics.hellinger_affinity <= 1.0  # Upper bound can be 1
    assert 0.2 < metrics.ess_fraction < 0.8
    assert metrics.overlap_quality in ["marginal", "good"]

    # Diagnose
    should_proceed, msg = diagnose_overlap_problems(metrics, verbose=False)
    assert should_proceed

    # Check that recommendation makes sense
    if metrics.ess_fraction < 0.3:
        assert metrics.recommended_method in ["calibrated-ips", "dr"]
    else:
        assert metrics.recommended_method == "ips"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
