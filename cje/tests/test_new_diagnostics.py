"""Tests for new diagnostic features: Hill tail index, orthogonality score, DM-IPS decomposition."""

import numpy as np
import pytest
from typing import Dict, Any

from cje.utils.diagnostics import (
    hill_tail_index,
    hill_tail_index_stable,
    compute_orthogonality_score,
    compute_dm_ips_decomposition,
)


class TestHillTailIndex:
    """Test Hill tail index estimator."""

    def test_hill_tail_index_light_tails(self) -> None:
        """Test with light-tailed (Gaussian) data."""
        np.random.seed(42)
        weights = np.abs(np.random.normal(1.0, 0.2, 1000))

        # Light tails should give high index (>3)
        tail_index = hill_tail_index(weights)
        assert tail_index > 2.5, f"Expected light tails (>2.5), got {tail_index}"

    def test_hill_tail_index_heavy_tails(self) -> None:
        """Test with heavy-tailed (Pareto) data."""
        np.random.seed(42)
        # Generate Pareto with tail index ~1.5
        weights = np.random.pareto(1.5, 1000) + 1

        # Heavy tails should give low index
        tail_index = hill_tail_index(weights)
        assert 1.0 < tail_index < 2.5, f"Expected heavy tails (1-2.5), got {tail_index}"

    def test_hill_tail_index_small_sample(self) -> None:
        """Test with too few samples."""
        weights = np.array([1.0, 2.0, 3.0])

        # Should return inf for tiny samples
        tail_index = hill_tail_index(weights, min_k=10)
        assert np.isinf(tail_index), "Expected inf for small sample"

    def test_hill_tail_index_stable(self) -> None:
        """Test stable Hill estimator with multiple k values."""
        np.random.seed(42)
        weights = np.abs(np.random.normal(1.0, 0.3, 500))

        result = hill_tail_index_stable(weights)

        assert "estimate" in result
        assert "min" in result
        assert "max" in result
        assert "std" in result

        # Check reasonable range
        assert result["min"] <= result["estimate"] <= result["max"]
        assert result["std"] >= 0


class TestOrthogonalityScore:
    """Test orthogonality score computation."""

    def test_orthogonality_perfect(self) -> None:
        """Test when orthogonality is perfect (residuals uncorrelated with weights)."""
        np.random.seed(42)
        n = 100

        # Perfect orthogonality: E[W * (R - q)] = 0
        weights = np.ones(n)  # Uniform weights
        rewards = np.random.normal(0.5, 0.1, n)
        predictions = rewards.copy()  # Perfect predictions

        result = compute_orthogonality_score(weights, rewards, predictions)

        assert abs(result["score"]) < 1e-10, "Expected near-zero score"
        assert result["passes_test"], "Should pass orthogonality test"

    def test_orthogonality_violated(self) -> None:
        """Test when orthogonality is violated."""
        np.random.seed(42)
        n = 100

        # Create correlation between weights and residuals
        weights = np.random.uniform(0.5, 2.0, n)
        rewards = weights * 0.5 + np.random.normal(0, 0.01, n)  # Correlated
        predictions = np.zeros(n)  # Bad predictions

        result = compute_orthogonality_score(weights, rewards, predictions)

        assert abs(result["score"]) > 0.1, "Expected non-zero score"
        assert result["p_value"] < 0.05, "Should be significant"

    def test_orthogonality_with_ci(self) -> None:
        """Test confidence interval computation."""
        np.random.seed(42)
        n = 200

        weights = np.random.uniform(0.8, 1.2, n)
        rewards = np.random.normal(0.5, 0.1, n)
        predictions = rewards + np.random.normal(0, 0.05, n)

        result = compute_orthogonality_score(
            weights, rewards, predictions, return_ci=True
        )

        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] < result["score"] < result["ci_upper"]


class TestDMIPSDecomposition:
    """Test DM-IPS decomposition."""

    def test_decomposition_basic(self) -> None:
        """Test basic decomposition properties."""
        np.random.seed(42)
        n = 100

        g_hat = np.random.normal(0.6, 0.1, n)  # DM predictions
        weights = np.ones(n)  # Uniform weights
        rewards = np.random.normal(0.5, 0.1, n)
        q_hat = np.random.normal(0.5, 0.05, n)  # Outcome model predictions

        result = compute_dm_ips_decomposition(g_hat, weights, rewards, q_hat)

        # Check all fields present
        assert "dm_component" in result
        assert "ips_augmentation" in result
        assert "total" in result
        assert "dm_contribution" in result
        assert "ips_contribution" in result

        # Check decomposition adds up
        expected_total = result["dm_component"] + result["ips_augmentation"]
        assert abs(result["total"] - expected_total) < 1e-10

        # Check contributions sum to 1
        contrib_sum = result["dm_contribution"] + result["ips_contribution"]
        assert abs(contrib_sum - 1.0) < 1e-10

    def test_decomposition_dm_dominates(self) -> None:
        """Test when DM component dominates."""
        np.random.seed(42)
        n = 100

        g_hat = np.ones(n) * 0.8  # Strong DM signal
        weights = np.ones(n)
        rewards = np.random.normal(0.5, 0.01, n)
        q_hat = rewards - 0.001  # Very good outcome model

        result = compute_dm_ips_decomposition(g_hat, weights, rewards, q_hat)

        # DM should dominate
        assert result["dm_contribution"] > 0.9, "DM should dominate"
        assert abs(result["ips_augmentation"]) < 0.01, "IPS should be small"

    def test_decomposition_ips_dominates(self) -> None:
        """Test when IPS component dominates."""
        np.random.seed(42)
        n = 100

        g_hat = np.zeros(n)  # No DM signal
        weights = np.ones(n) * 2.0  # Strong weights
        rewards = np.ones(n) * 0.5
        q_hat = np.zeros(n)  # Bad outcome model

        result = compute_dm_ips_decomposition(g_hat, weights, rewards, q_hat)

        # IPS should dominate
        assert result["ips_contribution"] > 0.9, "IPS should dominate"
        assert abs(result["dm_component"]) < 0.01, "DM should be small"


class TestIntegrationWithDR:
    """Test that new diagnostics integrate with DR estimators."""

    def test_dr_includes_new_diagnostics(self) -> None:
        """Test that DR estimator includes new diagnostics in metadata."""
        pytest.skip(
            "Requires full DR setup with fresh draws - tested in integration tests"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
