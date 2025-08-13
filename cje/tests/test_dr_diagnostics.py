"""Tests for DR diagnostics functionality."""

import numpy as np
import pytest
from unittest.mock import Mock
from typing import Dict, Any

from cje.utils.dr_diagnostics import (
    compute_dr_policy_diagnostics,
    compute_dr_diagnostics_all,
    format_dr_diagnostic_summary,
    DRPolicyDiagnostics,
)
from cje.data.models import EstimationResult


class TestDRPolicyDiagnostics:
    """Test the DR diagnostic computation."""

    def test_basic_diagnostics_computation(self) -> None:
        """Test basic diagnostic computation with synthetic data."""
        np.random.seed(42)
        n = 100

        # Create synthetic data
        weights = np.ones(n)  # Mean-one weights
        rewards = np.random.uniform(0, 1, n)
        g_logged = rewards + np.random.normal(0, 0.1, n)  # Good predictions
        g_logged = np.clip(g_logged, 0, 1)
        g_fresh = rewards.mean() * np.ones(n)  # Simple DM estimate
        dr_estimate = 0.5
        se = 0.05

        # Compute diagnostics
        diag = compute_dr_policy_diagnostics(
            weights=weights,
            rewards=rewards,
            g_logged=g_logged,
            g_fresh=g_fresh,
            dr_estimate=dr_estimate,
            se=se,
            draws_per_prompt=10,
            n_folds=5,
        )

        # Check basic properties
        assert isinstance(diag, DRPolicyDiagnostics)
        assert diag.n == n
        assert diag.dr_estimate == dr_estimate
        assert diag.se == se
        assert diag.cross_fitted
        assert diag.unique_folds == 5
        assert diag.draws_per_prompt == 10

        # Check metrics are reasonable
        assert 0 <= diag.r2_oof <= 1
        assert diag.residual_rmse >= 0
        assert diag.if_var >= 0
        assert diag.if_p95 >= 0
        assert diag.if_p99 >= diag.if_p95

    def test_orthogonality_check_tmle(self) -> None:
        """Test that TMLE achieves orthogonality (score_mean ≈ 0)."""
        np.random.seed(42)
        n = 1000

        # Create data where W*(R - g) has mean zero (orthogonal)
        weights = np.random.exponential(1, n)
        weights = weights / weights.mean()  # Mean-one
        rewards = np.random.uniform(0, 1, n)

        # Make g_logged such that E[W*(R - g)] = 0
        # This simulates successful TMLE targeting
        residuals = rewards - rewards.mean()
        # Adjust residuals to be orthogonal to weights
        score = weights * residuals
        adjustment = score.mean() / weights.mean()
        g_logged = rewards - residuals + adjustment
        g_logged = np.clip(g_logged, 0, 1)

        g_fresh = rewards.mean() * np.ones(n)
        dr_estimate = 0.5
        se = 0.05

        diag = compute_dr_policy_diagnostics(
            weights=weights,
            rewards=rewards,
            g_logged=g_logged,
            g_fresh=g_fresh,
            dr_estimate=dr_estimate,
            se=se,
        )

        # TMLE should achieve near-zero score mean
        assert abs(diag.score_mean) < 0.05  # Should be close to 0
        assert diag.score_p > 0.05  # Should not reject null of mean=0

    def test_heavy_tail_detection(self) -> None:
        """Test detection of heavy-tailed influence functions."""
        np.random.seed(42)
        n = 100

        # Create data with heavy-tailed IF
        weights = np.ones(n)
        rewards = np.random.uniform(0, 1, n)
        g_logged = rewards.copy()
        g_fresh = rewards.mean() * np.ones(n)

        # Spike a few IF contributions
        weights[:5] = 100  # Create extreme weights

        dr_estimate = 0.5
        se = 0.05

        diag = compute_dr_policy_diagnostics(
            weights=weights,
            rewards=rewards,
            g_logged=g_logged,
            g_fresh=g_fresh,
            dr_estimate=dr_estimate,
            se=se,
        )

        # Should detect heavy tails
        assert diag.if_tail_ratio_99_5 > 10  # Large tail ratio
        assert diag.if_top1_share > 0.1  # Top 1% has significant mass

    def test_fresh_draw_variance(self) -> None:
        """Test fresh draw variance tracking."""
        np.random.seed(42)
        n = 50

        weights = np.ones(n)
        rewards = np.random.uniform(0, 1, n)
        g_logged = rewards.copy()
        g_fresh = rewards.mean() * np.ones(n)

        # Create varying fresh draw variances
        fresh_var_per_prompt = np.random.exponential(0.1, n)

        diag = compute_dr_policy_diagnostics(
            weights=weights,
            rewards=rewards,
            g_logged=g_logged,
            g_fresh=g_fresh,
            dr_estimate=0.5,
            se=0.05,
            fresh_draw_var_per_prompt=fresh_var_per_prompt,
            draws_per_prompt=20,
        )

        # Check variance is tracked correctly
        assert diag.g_fresh_draw_var_mean == pytest.approx(fresh_var_per_prompt.mean())
        assert diag.draws_per_prompt == 20


class TestDRDiagnosticAggregation:
    """Test aggregation and formatting of DR diagnostics."""

    def test_compute_dr_diagnostics_all(self) -> None:
        """Test aggregation across multiple policies."""
        # Create mock estimation result with diagnostics
        dr_diags = {
            "policy1": {
                "dm_mean": 0.5,
                "ips_corr_mean": 0.05,
                "if_tail_ratio_99_5": 10.0,
                "r2_oof": 0.8,
                "score_mean": 0.01,
                "score_z": 0.5,
            },
            "policy2": {
                "dm_mean": 0.6,
                "ips_corr_mean": -0.02,
                "if_tail_ratio_99_5": 20.0,
                "r2_oof": 0.7,
                "score_mean": -0.02,
                "score_z": -1.0,
            },
        }

        mock_result = Mock(spec=EstimationResult)
        mock_result.metadata = {"dr_diagnostics": dr_diags}
        mock_result.method = "tmle"

        # Compute aggregated diagnostics
        aggregated = compute_dr_diagnostics_all(mock_result, per_policy=True)

        # Check aggregation
        assert aggregated["n_policies"] == 2
        assert aggregated["policies"] == ["policy1", "policy2"]
        assert aggregated["worst_if_tail_ratio"] == 20.0
        assert aggregated["best_r2_oof"] == 0.8
        assert aggregated["worst_r2_oof"] == 0.7
        assert "tmle_score_abs_mean" in aggregated
        assert aggregated["tmle_max_score_z"] == 1.0

    def test_format_diagnostic_summary(self) -> None:
        """Test formatting of diagnostic summary table."""
        diagnostics = {
            "per_policy": {
                "policy1": {
                    "dm_mean": 0.5,
                    "ips_corr_mean": 0.05,
                    "dr_estimate": 0.55,
                    "se": 0.02,
                    "score_mean": 0.001,
                    "score_se": 0.002,
                    "score_p": 0.95,
                    "residual_rmse": 0.1,
                    "if_tail_ratio_99_5": 15.5,
                },
            },
            "worst_if_tail_ratio": 15.5,
            "best_r2_oof": 0.85,
            "worst_r2_oof": 0.75,
        }

        # Format summary
        summary = format_dr_diagnostic_summary(diagnostics)

        # Check output contains key information
        assert "DR DIAGNOSTICS SUMMARY" in summary
        assert "policy1" in summary
        assert "0.500" in summary  # DM mean
        assert "0.050" in summary  # IPS correction
        assert "15.5" in summary  # Tail ratio
        assert "RMSE" in summary

    def test_empty_diagnostics_handling(self) -> None:
        """Test handling of empty diagnostics."""
        # Empty diagnostics
        empty_diags: Dict[str, Any] = {}
        summary = format_dr_diagnostic_summary(empty_diags)
        assert "No DR diagnostics available" in summary

        # Missing per_policy
        partial_diags = {"policies": ["p1"]}
        summary = format_dr_diagnostic_summary(partial_diags)
        assert "No DR diagnostics available" in summary


class TestDRDiagnosticIntegration:
    """Integration tests with actual estimators."""

    def test_dr_estimator_produces_diagnostics(self) -> None:
        """Test that DR estimators produce expected diagnostics structure."""
        from cje.data.models import Sample, Dataset
        from cje.data.precomputed_sampler import PrecomputedSampler
        from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample
        from cje.estimators.dr_base import DREstimator

        # Create minimal dataset
        samples = []
        for i in range(10):
            sample = Sample(
                prompt_id=f"p{i}",
                prompt=f"prompt {i}",
                response=f"response {i}",
                base_policy_logprob=-10.0,
                target_policy_logprobs={"target": -8.0},
                reward=0.5,
                metadata={"judge_score": 0.5, "cv_fold": i % 5},
            )
            samples.append(sample)

        dataset = Dataset(samples=samples, target_policies=["policy_a"])
        sampler = PrecomputedSampler(dataset)

        # Create estimator
        estimator = DREstimator(sampler, n_folds=5)
        estimator.fit()

        # Create fresh draws
        fresh_samples = []
        for i in range(10):
            for j in range(5):  # 5 draws per prompt
                fresh_sample = FreshDrawSample(
                    prompt_id=f"p{i}",
                    response=f"fresh {j}",
                    judge_score=0.5 + np.random.normal(0, 0.1),
                    target_policy="policy_a",
                    draw_idx=j,
                    fold_id=i % 5,
                )
                fresh_samples.append(fresh_sample)

        fresh_dataset = FreshDrawDataset(
            samples=fresh_samples,
            target_policy="target",
            draws_per_prompt=5,
        )

        estimator.add_fresh_draws("target", fresh_dataset)

        # Run estimation
        result = estimator.estimate()

        # Check diagnostics are present
        assert "dr_diagnostics" in result.metadata
        assert "dr_overview" in result.metadata
        assert "dr_calibration_data" in result.metadata

        # Check structure
        dr_diags = result.metadata["dr_diagnostics"]
        assert "target" in dr_diags

        target_diag = dr_diags["target"]
        assert "dm_mean" in target_diag
        assert "ips_corr_mean" in target_diag
        assert "score_mean" in target_diag
        assert "if_tail_ratio_99_5" in target_diag
        assert "r2_oof" in target_diag
