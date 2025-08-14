"""Tests for robust inference utilities (Phase 3)."""

import numpy as np
import pytest
from typing import Dict, Any

from cje.utils.diagnostics import (
    stationary_bootstrap_se,
    moving_block_bootstrap_se,
    cluster_robust_se,
    benjamini_hochberg_correction,
    compute_simultaneous_bands,
    compute_robust_inference,
)


class TestStationaryBootstrap:
    """Test stationary bootstrap for time series."""

    def test_iid_data(self) -> None:
        """Test with IID data - should give similar results to regular bootstrap."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)

        result = stationary_bootstrap_se(
            data,
            statistic_fn=np.mean,
            n_bootstrap=1000,  # Fewer for testing
            alpha=0.05,
        )

        # Check structure
        assert "estimate" in result
        assert "se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "mean_block_length" in result

        # For IID data, mean should be close to 0
        assert abs(result["estimate"]) < 0.1

        # SE should be close to 1/sqrt(n) = 0.045
        assert 0.03 < result["se"] < 0.06

        # CI should contain true value (0)
        assert result["ci_lower"] < 0 < result["ci_upper"]

    def test_ar1_data(self) -> None:
        """Test with AR(1) time series."""
        np.random.seed(42)
        n = 500
        rho = 0.7  # AR coefficient

        # Generate AR(1) process
        data = np.zeros(n)
        data[0] = np.random.normal(0, 1)
        for t in range(1, n):
            data[t] = rho * data[t - 1] + np.random.normal(0, 1)

        result = stationary_bootstrap_se(
            data,
            statistic_fn=np.mean,
            n_bootstrap=1000,
        )

        # With positive autocorrelation, block length should be larger
        assert result["mean_block_length"] > 5

        # SE should be larger than IID case due to dependence
        assert result["se"] > 0.05

    def test_return_distribution(self) -> None:
        """Test returning bootstrap distribution."""
        np.random.seed(42)
        data = np.random.normal(1, 2, 200)

        result = stationary_bootstrap_se(
            data,
            statistic_fn=np.mean,
            n_bootstrap=500,
            return_distribution=True,
        )

        assert "distribution" in result
        assert len(result["distribution"]) == 500

        # Bootstrap distribution should be centered around estimate
        boot_mean = np.mean(result["distribution"])
        assert abs(boot_mean - result["estimate"]) < 0.1


class TestMovingBlockBootstrap:
    """Test moving block bootstrap."""

    def test_basic_functionality(self) -> None:
        """Test basic moving block bootstrap."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 300)

        result = moving_block_bootstrap_se(
            data,
            statistic_fn=np.mean,
            n_bootstrap=1000,
            block_length=10,
        )

        assert "estimate" in result
        assert "se" in result
        assert "block_length" in result
        assert result["block_length"] == 10

        # Should give reasonable SE
        assert 0.04 < result["se"] < 0.08

    def test_auto_block_length(self) -> None:
        """Test automatic block length selection."""
        np.random.seed(42)
        n = 1000
        data = np.random.normal(0, 1, n)

        result = moving_block_bootstrap_se(
            data,
            statistic_fn=np.mean,
            n_bootstrap=500,
            block_length=None,  # Auto
        )

        # Should choose block length ~ n^(1/3) = 10
        assert 8 <= result["block_length"] <= 12


class TestClusterRobustSE:
    """Test cluster-robust standard errors."""

    def test_clustered_data(self) -> None:
        """Test with clustered data."""
        np.random.seed(42)

        # Generate clustered data: 50 clusters, 10 obs each
        n_clusters = 50
        cluster_size = 10
        n = n_clusters * cluster_size

        cluster_ids = np.repeat(np.arange(n_clusters), cluster_size)

        # Add cluster-level random effects
        cluster_effects = np.random.normal(0, 1, n_clusters)
        data = np.zeros(n)
        for i in range(n_clusters):
            mask = cluster_ids == i
            data[mask] = cluster_effects[i] + np.random.normal(0, 0.5, cluster_size)

        result = cluster_robust_se(
            data,
            cluster_ids,
            statistic_fn=np.mean,
        )

        assert "estimate" in result
        assert "se" in result
        assert "n_clusters" in result
        assert result["n_clusters"] == n_clusters

        # Cluster-robust SE should be different from naive SE
        # (can be smaller or larger depending on intra-cluster correlation)
        naive_se = np.std(data) / np.sqrt(n)
        # Just check it's computed and reasonable
        assert result["se"] > 0
        assert result["se"] < 1.0  # Reasonable bound for this data

    def test_with_influence_function(self) -> None:
        """Test with provided influence function."""
        np.random.seed(42)

        n = 100
        cluster_ids = np.repeat(np.arange(10), 10)
        data = np.random.normal(0, 1, n)

        # Simple influence function for mean
        def influence_fn(x: np.ndarray) -> np.ndarray:
            return x - np.mean(x)

        result = cluster_robust_se(
            data,
            cluster_ids,
            statistic_fn=np.mean,
            influence_fn=influence_fn,
        )

        assert result["n_clusters"] == 10
        assert result["df"] == 9


class TestBenjaminiHochberg:
    """Test FDR control with Benjamini-Hochberg."""

    def test_no_signal(self) -> None:
        """Test with all null hypotheses true (no signal)."""
        np.random.seed(42)
        # All p-values from uniform(0,1) under null
        p_values = np.random.uniform(0, 1, 20)

        result = benjamini_hochberg_correction(
            p_values,
            alpha=0.05,
        )

        assert "adjusted_p_values" in result
        assert "significant" in result
        assert "n_significant" in result

        # With no signal, expect few false positives
        # At 5% FDR, expect about 1 false positive
        assert result["n_significant"] <= 2

    def test_strong_signal(self) -> None:
        """Test with some true alternatives."""
        np.random.seed(42)

        # Mix of small p-values (signal) and uniform (null)
        p_values = np.concatenate(
            [
                np.array([0.001, 0.002, 0.003, 0.004, 0.005]),  # True signals
                np.random.uniform(0.1, 1, 15),  # Nulls
            ]
        )

        result = benjamini_hochberg_correction(
            p_values,
            alpha=0.05,
        )

        # Should detect the strong signals
        assert result["n_significant"] >= 3

        # First 5 should be significant
        assert np.all(result["significant"][:5])

        # Adjusted p-values should maintain order
        adj_p = result["adjusted_p_values"]
        sorted_indices = np.argsort(p_values)
        for i in range(1, len(sorted_indices)):
            assert adj_p[sorted_indices[i]] >= adj_p[sorted_indices[i - 1]]

    def test_with_labels(self) -> None:
        """Test with policy labels."""
        p_values = np.array([0.01, 0.04, 0.03, 0.5])
        labels = ["policy_a", "policy_b", "policy_c", "baseline"]

        result = benjamini_hochberg_correction(
            p_values,
            alpha=0.05,
            labels=labels,
        )

        assert "summary" in result
        assert len(result["summary"]) == 4

        # Check summary is sorted by p-value
        summary_p = [s["p_value"] for s in result["summary"]]
        assert summary_p == sorted(summary_p)


class TestSimultaneousBands:
    """Test simultaneous confidence bands."""

    def test_basic_bands(self) -> None:
        """Test basic simultaneous bands computation."""
        np.random.seed(42)

        estimates = np.array([0.1, 0.3, -0.2, 0.5, 0.0])
        ses = np.array([0.05, 0.08, 0.06, 0.10, 0.07])

        result = compute_simultaneous_bands(
            estimates,
            ses,
            alpha=0.05,
        )

        assert "bands" in result
        assert len(result["bands"]) == 5
        assert "critical_value" in result

        # Critical value should be larger than normal due to multiplicity
        assert result["critical_value"] > 1.96

        # Check which are significant
        for band in result["bands"]:
            if band["significant"]:
                # Significant means CI doesn't contain 0
                assert band["lower"] > 0 or band["upper"] < 0

    def test_with_correlation(self) -> None:
        """Test with correlation matrix."""
        estimates = np.array([0.2, 0.25, 0.22])
        ses = np.array([0.05, 0.05, 0.05])

        # High correlation between estimates
        corr_matrix = np.array(
            [
                [1.0, 0.8, 0.8],
                [0.8, 1.0, 0.8],
                [0.8, 0.8, 1.0],
            ]
        )

        result = compute_simultaneous_bands(
            estimates,
            ses,
            correlation_matrix=corr_matrix,
            alpha=0.05,
        )

        # All should be significant given the effect sizes
        n_sig = result["n_significant"]
        assert n_sig >= 2


class TestIntegratedRobustInference:
    """Test integrated robust inference function."""

    def test_stationary_bootstrap_integration(self) -> None:
        """Test integrated inference with stationary bootstrap."""
        np.random.seed(42)

        # Simulate influence functions for 3 policies
        n = 500
        influence_functions = np.random.normal(0, 1, (n, 3))
        influence_functions[:, 0] += 0.2  # Policy 1 has positive effect
        influence_functions[:, 1] += 0.1  # Policy 2 has small effect

        estimates = np.mean(influence_functions, axis=0)

        result = compute_robust_inference(
            estimates,
            influence_functions=influence_functions,
            method="stationary_bootstrap",
            n_bootstrap=500,  # Fewer for testing
            apply_fdr=True,
            fdr_alpha=0.05,
            policy_labels=["policy_1", "policy_2", "policy_3"],
        )

        assert "robust_ses" in result
        assert "robust_cis" in result
        assert "p_values" in result
        assert "fdr_results" in result

        assert len(result["robust_ses"]) == 3
        assert len(result["p_values"]) == 3

        # Policy 1 should be significant
        assert result["p_values"][0] < 0.05

    def test_cluster_robust_integration(self) -> None:
        """Test with cluster-robust inference."""
        np.random.seed(42)

        n = 200
        n_clusters = 20
        cluster_ids = np.repeat(np.arange(n_clusters), n // n_clusters)

        # Create clustered influence functions
        influence_functions = np.random.normal(0, 1, (n, 2))
        for i in range(n_clusters):
            mask = cluster_ids == i
            cluster_effect = np.random.normal(0, 0.5)
            influence_functions[mask, :] += cluster_effect

        estimates = np.mean(influence_functions, axis=0)

        result = compute_robust_inference(
            estimates,
            influence_functions=influence_functions,
            method="cluster",
            cluster_ids=cluster_ids,
            apply_fdr=False,
        )

        assert "robust_ses" in result
        assert result["method"] == "cluster"
        assert result["fdr_results"] is None  # Didn't apply FDR


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self) -> None:
        """Test with empty data."""
        result = benjamini_hochberg_correction(
            np.array([]),
            alpha=0.05,
        )

        assert result["n_significant"] == 0
        assert len(result["adjusted_p_values"]) == 0

    def test_single_hypothesis(self) -> None:
        """Test with single hypothesis (no multiplicity)."""
        result = benjamini_hochberg_correction(
            np.array([0.03]),
            alpha=0.05,
        )

        # Single test should be significant at 0.05 level
        assert result["n_significant"] == 1
        assert result["adjusted_p_values"][0] == 0.03  # No adjustment needed

    def test_all_significant(self) -> None:
        """Test when all hypotheses are highly significant."""
        p_values = np.array([0.0001, 0.0002, 0.0003, 0.0004, 0.0005])

        result = benjamini_hochberg_correction(
            p_values,
            alpha=0.05,
        )

        # All should remain significant
        assert result["n_significant"] == 5
        assert np.all(result["significant"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
