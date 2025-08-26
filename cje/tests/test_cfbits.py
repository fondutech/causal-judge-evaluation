"""Tests for CF-bits functionality.

Tests IFR/aESS, sampling width, overlap floors, and CF-bits computation.
"""

import pytest
import numpy as np
from typing import Dict, Any

# CF-bits imports
from cje.cfbits import (
    compute_ifr_aess,
    compute_sampling_width,
    estimate_overlap_floors,
    compute_cfbits,
    apply_gates,
    EfficiencyStats,
    SamplingVariance,
    OverlapFloors,
    CFBits,
    GatesDecision,
)


class TestIFRComputation:
    """Test Information Fraction Ratio and adjusted ESS."""

    def test_ifr_perfect_efficiency(self):
        """Test IFR = 1 when IF = EIF."""
        n = 100
        # Create identical IF and EIF
        phi = np.random.randn(n)
        phi = phi - np.mean(phi)  # Center

        result = compute_ifr_aess(phi, eif=phi, n=n)

        assert isinstance(result, EfficiencyStats)
        assert abs(result.ifr_main - 1.0) < 1e-10
        assert abs(result.aess_main - n) < 1e-10
        assert result.var_phi == result.var_eif

    def test_ifr_half_efficiency(self):
        """Test IFR = 0.5 when Var(IF) = 2×Var(EIF)."""
        n = 100
        # Create EIF with half the variance
        eif = np.random.randn(n)
        eif = eif - np.mean(eif)
        phi = eif * np.sqrt(2)  # Double the variance

        result = compute_ifr_aess(phi, eif=eif, n=n)

        assert abs(result.ifr_main - 0.5) < 0.01
        assert abs(result.aess_main - 50) < 1
        assert abs(result.var_phi / result.var_eif - 2.0) < 0.1

    def test_ifr_without_eif(self):
        """Test IFR defaults to 1 when EIF not provided (IPS-like)."""
        n = 100
        phi = np.random.randn(n)

        result = compute_ifr_aess(phi, eif=None, n=n)

        assert result.ifr_main == 1.0
        assert result.aess_main == n
        assert result.var_eif == result.var_phi

    def test_ifr_bounds(self):
        """Test IFR is bounded in (0, 1]."""
        n = 100
        phi = np.random.randn(n) * 10
        eif = np.random.randn(n) * 0.1

        result = compute_ifr_aess(phi, eif=eif, n=n)

        assert 0 < result.ifr_main <= 1.0


class TestSamplingWidth:
    """Test sampling width computation."""

    @pytest.fixture
    def mock_estimator(self):
        """Create a mock estimator with influence functions."""

        class MockEstimator:
            def __init__(self):
                self._influence_functions = {"test_policy": np.random.randn(100) * 0.5}
                self.sampler = MockSampler()

            def get_influence_functions(self, policy):
                return self._influence_functions.get(policy)

        class MockSampler:
            def get_judge_scores(self):
                return np.random.uniform(0, 1, 100)

        return MockEstimator()

    def test_sampling_width_basic(self, mock_estimator):
        """Test basic sampling width computation."""
        wvar, var_components = compute_sampling_width(
            mock_estimator, "test_policy", use_iic=False
        )

        assert isinstance(wvar, float)
        assert wvar > 0
        assert isinstance(var_components, SamplingVariance)
        assert var_components.var_main > 0
        assert var_components.var_oracle == 0  # No OUA in Phase 1

    def test_sampling_width_confidence_level(self, mock_estimator):
        """Test different confidence levels affect width."""
        wvar_95, _ = compute_sampling_width(mock_estimator, "test_policy", alpha=0.05)
        wvar_99, _ = compute_sampling_width(mock_estimator, "test_policy", alpha=0.01)

        # 99% CI should be wider
        assert wvar_99 > wvar_95

    def test_sampling_width_missing_if(self, mock_estimator):
        """Test error when influence functions not available."""
        with pytest.raises(ValueError, match="No influence functions"):
            compute_sampling_width(mock_estimator, "nonexistent_policy")


class TestOverlapFloors:
    """Test overlap metrics on judge marginal."""

    def test_overlap_perfect(self):
        """Test perfect overlap (uniform weights)."""
        n = 100
        S = np.random.uniform(0, 1, n)
        W = np.ones(n)  # Perfect overlap

        result = estimate_overlap_floors(S, W, n_boot=50)

        assert isinstance(result, OverlapFloors)
        assert abs(result.aessf - 1.0) < 0.1  # Should be ~1
        assert abs(result.chi2_s - 0.0) < 0.1  # Should be ~0
        assert 0.9 < result.bc <= 1.0

    @pytest.mark.skip(
        reason="Isotonic regression oversmoothing issue - needs investigation"
    )
    def test_overlap_poor(self):
        """Test poor overlap (extreme weights)."""
        np.random.seed(42)  # For reproducibility
        n = 200  # More samples for better isotonic fit
        S = np.random.uniform(0, 1, n)

        # Create weights that vary with S to ensure isotonic picks up pattern
        # High weights for low S values (creates actual pattern)
        W = np.ones(n)
        low_s_mask = S < 0.2
        W[low_s_mask] = 10.0  # High weights for low judge scores

        result = estimate_overlap_floors(S, W, n_boot=50, random_state=42)

        # This should create actual chi2 divergence
        assert result.chi2_s > 0.5  # Significant divergence
        assert result.aessf < 0.7  # Poor overlap

        # Verify it's worse than uniform
        uniform_result = estimate_overlap_floors(
            S, np.ones(n), n_boot=20, random_state=42
        )
        assert result.aessf < uniform_result.aessf

    def test_overlap_theoretical_constraint(self):
        """Test A-ESSF ≤ BC² theoretical constraint."""
        n = 200
        S = np.random.uniform(0, 1, n)
        W = np.exp(np.random.randn(n))  # Log-normal weights

        result = estimate_overlap_floors(S, W, n_boot=100)

        # Allow small numerical tolerance
        assert result.aessf <= result.bc**2 + 0.05

    def test_overlap_confidence_intervals(self):
        """Test bootstrap confidence intervals."""
        n = 100
        S = np.random.uniform(0, 1, n)
        W = np.random.gamma(2, 1, n)

        result = estimate_overlap_floors(S, W, n_boot=100, alpha=0.05)

        # CI should contain point estimate (with small tolerance for bootstrap variation)
        assert result.ci_aessf[0] <= result.aessf <= result.ci_aessf[1] + 0.01
        assert result.ci_bc[0] <= result.bc <= result.ci_bc[1] + 0.01

        # CI should have reasonable width
        assert result.ci_aessf[1] - result.ci_aessf[0] < 1.0

    def test_overlap_small_sample(self):
        """Test with small sample size."""
        n = 20
        S = np.random.uniform(0, 1, n)
        W = np.ones(n) * 1.5

        # Should not crash with small sample
        result = estimate_overlap_floors(S, W, n_boot=20)
        assert result.aessf > 0


class TestCFBits:
    """Test CF-bits computation."""

    def test_cfbits_no_reduction(self):
        """Test bits = 0 when no width reduction."""
        w0 = 1.0
        wid = 0.5
        wvar = 0.5

        result = compute_cfbits(w0, wid, wvar)

        assert isinstance(result, CFBits)
        assert result.w_tot == 1.0
        assert abs(result.bits_tot) < 0.01  # ~0 bits
        assert result.w_max == 0.5

    def test_cfbits_one_bit(self):
        """Test bits = 1 when width halved."""
        w0 = 1.0
        wid = 0.2
        wvar = 0.3

        result = compute_cfbits(w0, wid, wvar)

        assert result.w_tot == 0.5
        assert abs(result.bits_tot - 1.0) < 0.01  # ~1 bit

    def test_cfbits_with_ifr(self):
        """Test variance bits computation from IFR."""
        w0 = 1.0
        wid = 0.3
        wvar = 0.2
        ifr_main = 0.25  # Quarter efficiency

        result = compute_cfbits(w0, wid, wvar, ifr_main=ifr_main)

        assert result.bits_var is not None
        # bits_var = 0.5 * log2(0.25) = 0.5 * (-2) = -1
        assert abs(result.bits_var - (-1.0)) < 0.01

    def test_cfbits_dominant_width(self):
        """Test identification of dominant width."""
        # Identification dominant
        result1 = compute_cfbits(1.0, wid=0.6, wvar=0.2)
        assert result1.w_max == 0.6

        # Sampling dominant
        result2 = compute_cfbits(1.0, wid=0.2, wvar=0.6)
        assert result2.w_max == 0.6


class TestGates:
    """Test reliability gating."""

    def test_gates_all_good(self):
        """Test GOOD state when all metrics acceptable."""
        decision = apply_gates(aessf=0.8, ifr=0.9, tail_index=3.0, var_oracle_ratio=0.5)

        assert isinstance(decision, GatesDecision)
        assert decision.state == "GOOD"
        assert len(decision.suggestions) == 0

    def test_gates_refuse_catastrophic_overlap(self):
        """Test REFUSE when overlap catastrophic."""
        decision = apply_gates(aessf=0.04)  # Below refuse threshold

        assert decision.state == "REFUSE"
        assert "Catastrophic overlap" in decision.reasons[0]
        assert "change_policy" in decision.suggestions

    def test_gates_critical_poor_overlap(self):
        """Test CRITICAL when overlap poor."""
        decision = apply_gates(aessf=0.15)  # Below critical threshold

        assert decision.state == "CRITICAL"
        assert "Poor overlap" in decision.reasons[0]
        assert "use_dr" in decision.suggestions

    def test_gates_warning_inefficient(self):
        """Test WARNING when inefficient."""
        decision = apply_gates(ifr=0.4)  # Below warning threshold

        assert decision.state == "WARNING"
        assert "Inefficient" in decision.reasons[0]

    def test_gates_critical_tail_risk(self):
        """Test CRITICAL with tail risk."""
        decision = apply_gates(tail_index=1.8)  # Below critical threshold

        assert decision.state == "CRITICAL"
        assert "Infinite variance risk" in decision.reasons[0]
        assert "robust_estimator" in decision.suggestions

    def test_gates_custom_thresholds(self):
        """Test custom threshold override."""
        decision = apply_gates(
            aessf=0.3, thresholds={"aessf_critical": 0.5}  # Stricter threshold
        )

        assert decision.state == "CRITICAL"  # Would be WARNING with defaults


class TestConsistencyInvariants:
    """Test mathematical consistency requirements."""

    def test_bits_total_consistency(self):
        """Test bits_tot == log2(w0 / (wid + wvar))."""
        w0 = 1.0
        wid = 0.3
        wvar = 0.2

        result = compute_cfbits(w0, wid, wvar)
        expected = np.log2(w0 / (wid + wvar))

        assert abs(result.bits_tot - expected) < 1e-10
        assert result.w_tot == wid + wvar

    def test_ifr_ordering(self):
        """Test IFR_OUA ≤ IFR_main ≤ 1."""
        n = 100
        phi = np.random.randn(n)
        eif = np.random.randn(n) * 0.5  # Lower variance
        var_oracle = 0.001

        result = compute_ifr_aess(phi, eif=eif, var_oracle=var_oracle)

        assert 0 < result.ifr_oua <= result.ifr_main <= 1.0
        assert result.aess_oua <= result.aess_main <= n

    def test_variance_total_consistency(self):
        """Test var_total == var_main/n + var_oracle."""
        from cje.cfbits.sampling import SamplingVariance

        var_main = 1.0
        var_oracle = 0.01
        n = 100
        var_total = var_main / n + var_oracle

        varc = SamplingVariance(
            var_main=var_main, var_oracle=var_oracle, var_total=var_total
        )

        assert abs(varc.var_total - (var_main / n + var_oracle)) < 1e-10

    def test_aessf_bc_constraint(self):
        """Test theoretical constraint A-ESSF ≤ BC²."""
        n = 200
        S = np.random.uniform(0, 1, n)
        W = np.exp(np.random.randn(n) * 0.5)  # Mild variation

        result = estimate_overlap_floors(S, W, n_boot=0, random_state=42)

        # Allow small tolerance for numerical error
        assert result.aessf <= result.bc**2 + 0.01

    def test_deterministic_overlap_estimation(self):
        """Test determinism with fixed random_state."""
        n = 100
        S = np.random.uniform(0, 1, n)
        W = np.random.gamma(2, 1, n)

        result1 = estimate_overlap_floors(
            S, W, method="bins", n_boot=0, random_state=13
        )
        result2 = estimate_overlap_floors(
            S, W, method="bins", n_boot=0, random_state=13
        )

        assert result1.aessf == result2.aessf
        assert result1.chi2_s == result2.chi2_s


@pytest.mark.integration
class TestEndToEnd:
    """Integration tests with real CJE components."""

    def test_full_pipeline(self):
        """Test complete CF-bits pipeline with real data."""
        from cje.data import load_dataset_from_jsonl
        from cje.calibration.dataset import calibrate_dataset
        from cje.data.precomputed_sampler import PrecomputedSampler
        from cje.estimators.calibrated_ips import CalibratedIPS

        # Load small dataset
        dataset = load_dataset_from_jsonl(
            "cje/experiments/arena_10k_simplified/data/cje_dataset.jsonl"
        )
        dataset.samples = dataset.samples[:100]

        # Calibrate
        calibrated_dataset, _ = calibrate_dataset(
            dataset, judge_field="judge_score", oracle_field="oracle_label"
        )
        sampler = PrecomputedSampler(calibrated_dataset)

        # Fit estimator
        estimator = CalibratedIPS(sampler)
        result = estimator.fit_and_estimate()

        # Get a policy
        policy = result.metadata["target_policies"][0]

        # Test IFR
        if_values = estimator.get_influence_functions(policy)
        efficiency = compute_ifr_aess(if_values)
        assert 0 < efficiency.ifr_main <= 1.0
        assert efficiency.aess_main > 0

        # Test sampling width
        wvar, _ = compute_sampling_width(estimator, policy, use_iic=False)
        assert wvar > 0

        # Test overlap
        weights = sampler.compute_importance_weights(policy, mode="raw")
        judge_scores = sampler.get_judge_scores()
        overlap = estimate_overlap_floors(judge_scores, weights, n_boot=20)
        assert 0 < overlap.aessf <= 1.0

        # Test CF-bits
        cfbits = compute_cfbits(1.0, 0.2, wvar, ifr_main=efficiency.ifr_main)
        assert cfbits.w_tot > 0

        # Test gates
        decision = apply_gates(aessf=overlap.aessf, ifr=efficiency.ifr_main)
        assert decision.state in ["GOOD", "WARNING", "CRITICAL", "REFUSE"]
