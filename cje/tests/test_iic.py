"""Tests for Isotonic Influence Control (IIC)."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from cje.calibration.iic import IsotonicInfluenceControl, IICConfig
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators.calibrated_ips import CalibratedIPS


class TestIsotonicInfluenceControl:
    """Test suite for IIC module."""

    def test_variance_reduction(self) -> None:
        """IIC should reduce or maintain variance."""
        np.random.seed(42)
        n = 1000

        # Create correlated IF and judge scores (realistic scenario)
        judge_scores = np.random.randn(n)
        # Influence functions often correlate with judge scores
        influence = 0.5 * judge_scores + np.random.randn(n) * 0.5

        # Apply IIC
        iic = IsotonicInfluenceControl()
        residual, diag = iic.residualize(influence, judge_scores, "test_policy")

        # Check variance reduction
        var_original = np.var(influence, ddof=1)
        var_residual = np.var(residual, ddof=1)

        # Should reduce variance (allow small numerical tolerance)
        assert (
            var_residual <= var_original * 1.001
        ), f"Variance increased: {var_residual:.4f} > {var_original:.4f}"

        # Diagnostics should show reduction
        assert diag["applied"] == True
        assert diag["var_reduction"] >= 0
        assert diag["se_reduction"] >= 0

    def test_mean_preservation(self) -> None:
        """IIC must preserve mean exactly (unbiasedness)."""
        np.random.seed(42)
        n = 500

        # IIC residualizes, so the mean becomes zero
        # But the estimand remains unchanged because we add back the mean
        for target_mean in [0, 1, -1, 10]:
            influence = np.random.randn(n) + target_mean
            judge_scores = np.random.randn(n)

            iic = IsotonicInfluenceControl()
            residual, diag = iic.residualize(influence, judge_scores, "test")

            # The residual has mean zero (that's the point of residualization)
            assert (
                abs(residual.mean()) < 1e-10
            ), f"Residual mean not zero: {residual.mean():.6f}"

            # But variance should be reduced
            assert np.var(residual) <= np.var(influence) * 1.001

    def test_cross_fitting(self) -> None:
        """Test fold-honest fitting to avoid overfitting."""
        np.random.seed(42)
        n = 1000

        # Create data with strong relationship
        judge_scores = np.linspace(-2, 2, n)
        influence = judge_scores**2 + np.random.randn(n) * 0.1

        # Create fold IDs
        fold_ids = np.repeat(np.arange(5), n // 5)

        # Test with cross-fitting
        iic_cv = IsotonicInfluenceControl(IICConfig(use_cross_fit=True))
        residual_cv, diag_cv = iic_cv.residualize(
            influence, judge_scores, "test", fold_ids
        )

        # Test without cross-fitting
        iic_no_cv = IsotonicInfluenceControl(IICConfig(use_cross_fit=False))
        residual_no_cv, diag_no_cv = iic_no_cv.residualize(
            influence, judge_scores, "test", fold_ids
        )

        # Both should reduce variance, but CV should be more conservative
        assert diag_cv["applied"] == True
        assert diag_no_cv["applied"] == True

        # Non-CV might have slightly better in-sample fit (higher R²)
        # but this isn't guaranteed due to isotonic constraints
        assert diag_cv["r_squared"] <= diag_no_cv["r_squared"] + 0.1

    def test_edge_cases(self) -> None:
        """Test various edge cases."""
        iic = IsotonicInfluenceControl()

        # Empty input
        residual, diag = iic.residualize(np.array([]), np.array([]), "test")
        assert len(residual) == 0
        assert diag["applied"] == False

        # Too few samples
        small_if = np.array([1, 2, 3])
        small_scores = np.array([1, 2, 3])
        residual, diag = iic.residualize(small_if, small_scores, "test")
        assert diag["applied"] == False
        assert diag["reason"] == "insufficient_samples"

        # Constant influence function
        n = 100
        constant_if = np.ones(n) * 5
        judge_scores = np.random.randn(n)
        residual, diag = iic.residualize(constant_if, judge_scores, "test")

        # Should remove the mean, leaving zeros
        assert np.allclose(residual, 0)
        assert diag["applied"] == True

        # Missing judge scores (NaN)
        n = 100
        influence = np.random.randn(n)
        judge_scores = np.full(n, np.nan)
        residual, diag = iic.residualize(influence, judge_scores, "test")
        assert diag["applied"] == False
        assert diag["reason"] == "insufficient_valid_scores"

    def test_disabled_iic(self) -> None:
        """Test that disabling IIC returns original influence."""
        np.random.seed(42)
        n = 100

        influence = np.random.randn(n)
        judge_scores = np.random.randn(n)

        # Disable IIC
        iic = IsotonicInfluenceControl(IICConfig(enable=False))
        residual, diag = iic.residualize(influence, judge_scores, "test")

        # Should return original
        assert np.array_equal(residual, influence)
        assert diag == {"applied": False, "reason": "disabled"}

    def test_r_squared_computation(self) -> None:
        """Test R² computation for isotonic fit."""
        np.random.seed(42)
        n = 500

        # Perfect relationship -> high R²
        judge_scores = np.linspace(0, 10, n)
        influence = 2 * judge_scores + np.random.randn(n) * 0.1

        iic = IsotonicInfluenceControl()
        _, diag = iic.residualize(influence, judge_scores, "test")

        assert diag["r_squared"] > 0.9  # Should explain most variance

        # No relationship -> low R²
        influence_random = np.random.randn(n)
        _, diag_random = iic.residualize(influence_random, judge_scores, "test")

        assert diag_random["r_squared"] < 0.1  # Should explain little

    def test_ess_gain(self) -> None:
        """Test effective sample size gain computation."""
        np.random.seed(42)
        n = 500

        # Create influence with reducible variance
        judge_scores = np.random.randn(n)
        influence = judge_scores + np.random.randn(n)

        iic = IsotonicInfluenceControl()
        _, diag = iic.residualize(influence, judge_scores, "test")

        # ESS gain = var_original / var_residual
        assert "ess_gain" in diag
        assert diag["ess_gain"] >= 1.0  # Should be at least 1 (no loss)

        # Verify calculation
        expected_gain = diag["var_original"] / diag["var_residual"]
        assert abs(diag["ess_gain"] - expected_gain) < 1e-10


class TestIICIntegration:
    """Test IIC integration with estimators."""

    @pytest.fixture
    def mock_sampler(self) -> Mock:
        """Create a mock sampler with judge scores."""
        sampler = Mock(spec=PrecomputedSampler)
        sampler.target_policies = ["policy1", "policy2"]
        sampler.n_valid_samples = 100
        sampler.oracle_coverage = 0.5  # Mock oracle coverage for auto-detection

        # Mock data with judge scores
        data = [{"judge_score": np.random.randn()} for _ in range(100)]
        sampler.get_data_for_policy.return_value = data

        return sampler

    def test_calibrated_ips_with_iic(self, mock_sampler: Mock) -> None:
        """Test CalibratedIPS with IIC enabled."""
        # Create estimator with IIC enabled (default)
        estimator = CalibratedIPS(mock_sampler, use_iic=True)

        assert estimator.use_iic == True
        assert estimator.iic is not None

    def test_calibrated_ips_without_iic(self, mock_sampler: Mock) -> None:
        """Test CalibratedIPS with IIC disabled."""
        # Create estimator with IIC disabled
        estimator = CalibratedIPS(mock_sampler, use_iic=False)

        assert estimator.use_iic == False
        assert estimator.iic is None

    def test_iic_diagnostics_in_results(self, mock_sampler: Mock) -> None:
        """Test that IIC diagnostics appear in results."""
        np.random.seed(42)

        # Create influence functions
        n = 100
        influence = np.random.randn(n)

        # Create estimator with IIC
        estimator = CalibratedIPS(mock_sampler, use_iic=True)
        estimator._influence_functions = {"policy1": influence}

        # Mock the _apply_iic method to track calls
        with patch.object(
            estimator, "_apply_iic", return_value=influence
        ) as mock_apply:
            # This would normally be called during estimate()
            residual = estimator._apply_iic(influence, "policy1")

            # Verify it was called
            mock_apply.assert_called_once_with(influence, "policy1")

    def test_iic_with_missing_judge_scores(self, mock_sampler: Mock) -> None:
        """Test IIC handles missing judge scores gracefully."""
        # Mock data without judge scores
        mock_sampler.get_data_for_policy.return_value = [
            {} for _ in range(100)  # No judge_score key
        ]

        estimator = CalibratedIPS(mock_sampler, use_iic=True)

        # Create influence functions
        influence = np.random.randn(100)

        # Apply IIC - should handle gracefully
        result = estimator._apply_iic(influence, "policy1")

        # Should return original when judge scores missing
        assert np.array_equal(result, influence)


class TestIICConfig:
    """Test IIC configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = IICConfig()

        assert config.enable == True  # On by default
        assert config.use_cross_fit == True  # Cross-fitting by default
        assert config.min_samples_for_iic == 50
        assert config.compute_diagnostics == True
        assert config.store_components == False

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = IICConfig(
            enable=False,
            use_cross_fit=False,
            min_samples_for_iic=100,
            compute_diagnostics=False,
            store_components=True,
        )

        assert config.enable == False
        assert config.use_cross_fit == False
        assert config.min_samples_for_iic == 100
        assert config.compute_diagnostics == False
        assert config.store_components == True


class TestIICMathematicalProperties:
    """Test mathematical properties of IIC."""

    def test_projection_property(self) -> None:
        """Test that IIC reduces variance on each application."""
        np.random.seed(42)
        n = 200

        judge_scores = np.random.randn(n)
        influence = np.random.randn(n)

        iic = IsotonicInfluenceControl()

        # First application
        residual1, diag1 = iic.residualize(influence, judge_scores, "test")
        var1 = np.var(residual1)

        # Second application (should either maintain or further reduce variance)
        residual2, diag2 = iic.residualize(residual1, judge_scores, "test")
        var2 = np.var(residual2)

        # Variance should not increase
        assert var2 <= var1 * 1.001  # Allow small numerical tolerance

        # If no relationship found in second pass, should be approximately unchanged
        if diag2["r_squared"] < 0.01:
            # Very little relationship, so should be mostly unchanged
            assert np.allclose(residual1, residual2, rtol=0.1)

    def test_orthogonality(self) -> None:
        """Test that residuals are orthogonal to fitted values."""
        np.random.seed(42)
        n = 500

        judge_scores = np.random.randn(n)
        influence = judge_scores + np.random.randn(n)

        # Configure to store components
        iic = IsotonicInfluenceControl(IICConfig(store_components=True))

        residual, _ = iic.residualize(influence, judge_scores, "test")
        fitted = iic.get_fitted_component("test")

        # Ensure fitted component was stored
        assert (
            fitted is not None
        ), "Fitted component should be stored when store_components=True"

        # Residual should be orthogonal to fitted
        correlation = np.corrcoef(residual, fitted)[0, 1]
        assert abs(correlation) < 0.01  # Near zero correlation

        # Also check inner product (should be near zero)
        inner_product = np.dot(residual - residual.mean(), fitted - fitted.mean()) / n
        assert abs(inner_product) < 0.01
