"""Test multi-policy support in uncertainty-aware CJE."""

import pytest
import numpy as np
from typing import List, Dict, Any

from cje.uncertainty import (
    JudgeScore,
    UncertaintyAwareDRCPO,
    create_uncertainty_report,
)
from cje.uncertainty.judge import MockUncertaintyJudge
from cje.uncertainty.estimator import UncertaintyEstimatorConfig
from cje.uncertainty.results import MultiPolicyUncertaintyResult


class TestMultiPolicyUncertainty:
    """Test suite for multi-policy uncertainty-aware evaluation."""

    def test_multi_policy_estimation(self) -> None:
        """Test basic multi-policy estimation with uncertainty."""
        np.random.seed(42)
        n_samples = 200
        n_policies = 3

        # Generate mock judge scores
        judge = MockUncertaintyJudge(base_score=0.7, base_variance=0.04)
        samples = [
            {"context": f"Question {i}", "response": f"Answer {i}"}
            for i in range(n_samples)
        ]
        judge_scores = judge.score_batch(samples)

        # Generate oracle rewards
        oracle_rewards = np.array(
            [np.clip(s.mean + np.random.normal(0, 0.1), 0, 1) for s in judge_scores]
        )

        # Generate importance weights for multiple policies
        # Each policy has different distribution shift
        importance_weights = np.zeros((n_samples, n_policies))
        for policy_idx in range(n_policies):
            shift = policy_idx * 0.5  # Increasing shift
            log_weights = np.random.normal(shift, 1.0, n_samples)
            importance_weights[:, policy_idx] = np.exp(log_weights)

        # Normalize weights
        for policy_idx in range(n_policies):
            importance_weights[:, policy_idx] /= np.mean(
                importance_weights[:, policy_idx]
            )

        # Configure and run estimator
        config = UncertaintyEstimatorConfig(
            k_folds=3,
            use_variance_shrinkage=True,
            shrinkage_method="optimal",
        )

        estimator = UncertaintyAwareDRCPO(config)
        result = estimator.fit(
            X=None,
            judge_scores=judge_scores,
            oracle_rewards=oracle_rewards,
            importance_weights=importance_weights,
            policy_names=["Baseline", "Improved_v1", "Improved_v2"],
        )

        # Verify result structure
        assert isinstance(result, MultiPolicyUncertaintyResult)
        assert result.n_policies == 3
        assert result.n_samples == n_samples

        # Check policy names
        assert result.get_policy("Baseline") is not None
        assert result.get_policy("Improved_v1") is not None
        assert result.get_policy("Improved_v2") is not None

        # Check estimates are reasonable
        estimates = result.get_estimates()
        assert len(estimates) == 3
        assert all(0 <= e <= 1 for e in estimates)

        # Check standard errors
        ses = result.get_standard_errors()
        assert len(ses) == 3
        assert all(se > 0 for se in ses)

        # Check confidence intervals
        cis = result.get_confidence_intervals()
        assert len(cis) == 3
        for i, (lower, upper) in enumerate(cis):
            assert lower < estimates[i] < upper

    def test_policy_comparison(self) -> None:
        """Test pairwise policy comparison with uncertainty."""
        np.random.seed(123)
        n_samples = 300

        # Create two policies with known difference
        judge_scores = []
        oracle_rewards = []

        # Policy 1: Lower quality
        for i in range(n_samples // 2):
            score = JudgeScore(mean=0.4 + np.random.normal(0, 0.1), variance=0.03)
            judge_scores.append(score)
            oracle_rewards.append(np.clip(score.mean + np.random.normal(0, 0.08), 0, 1))

        # Policy 2: Higher quality
        for i in range(n_samples // 2):
            score = JudgeScore(mean=0.7 + np.random.normal(0, 0.1), variance=0.03)
            judge_scores.append(score)
            oracle_rewards.append(np.clip(score.mean + np.random.normal(0, 0.08), 0, 1))

        oracle_rewards_array = np.array(oracle_rewards)

        # Create importance weights favoring each policy
        weights = np.zeros((n_samples, 2))
        weights[: n_samples // 2, 0] = 2.0  # Policy 1 oversampled in first half
        weights[n_samples // 2 :, 0] = 0.5
        weights[: n_samples // 2, 1] = 0.5  # Policy 2 oversampled in second half
        weights[n_samples // 2 :, 1] = 2.0

        # Run estimation
        config = UncertaintyEstimatorConfig(k_folds=3)
        estimator = UncertaintyAwareDRCPO(config)
        result = estimator.fit(
            X=None,
            judge_scores=judge_scores,
            oracle_rewards=oracle_rewards_array,
            importance_weights=weights,
            policy_names=["Policy_A", "Policy_B"],
        )

        # Compare policies
        comparison = result.pairwise_comparison("Policy_B", "Policy_A")

        # Policy B should be significantly better
        assert comparison["favors"] == "Policy_B"
        assert comparison["difference"] > 0.2  # Substantial difference
        assert comparison["p_value"] < 0.05  # Statistically significant
        assert comparison["significant_at_0.05"] is True

    def test_variance_decomposition_multi_policy(self) -> None:
        """Test variance decomposition for multiple policies."""
        np.random.seed(456)
        n_samples = 150

        # Create scenarios with different uncertainty patterns
        judge_scores = []

        # High uncertainty samples
        for i in range(50):
            judge_scores.append(
                JudgeScore(
                    mean=np.random.uniform(0.4, 0.6),
                    variance=np.random.uniform(0.08, 0.15),  # High uncertainty
                )
            )

        # Low uncertainty samples
        for i in range(100):
            judge_scores.append(
                JudgeScore(
                    mean=np.random.uniform(0.6, 0.8),
                    variance=np.random.uniform(0.01, 0.03),  # Low uncertainty
                )
            )

        oracle_rewards = np.array([s.mean for s in judge_scores])

        # Two policies with different weight patterns
        weights = np.zeros((n_samples, 2))

        # Policy 1: Uniform weights
        weights[:, 0] = 1.0

        # Policy 2: High weights on high-uncertainty samples
        weights[:50, 1] = 3.0  # High weight on uncertain samples
        weights[50:, 1] = 0.5  # Low weight on certain samples

        # Run estimation
        config = UncertaintyEstimatorConfig(
            k_folds=3,
            use_variance_shrinkage=True,
        )
        estimator = UncertaintyAwareDRCPO(config)
        result = estimator.fit(
            X=None,
            judge_scores=judge_scores,
            oracle_rewards=oracle_rewards,
            importance_weights=weights,
            policy_names=["Uniform", "HighUncertaintyFocus"],
        )

        # Check variance decomposition
        uniform_policy = result.get_policy("Uniform")
        high_unc_policy = result.get_policy("HighUncertaintyFocus")

        # High uncertainty focus should have more judge variance contribution
        assert uniform_policy is not None
        assert high_unc_policy is not None
        uniform_decomp = uniform_policy.estimate.variance_decomposition
        high_unc_decomp = high_unc_policy.estimate.variance_decomposition

        assert high_unc_decomp.judge_pct > uniform_decomp.judge_pct
        assert high_unc_decomp.judge > uniform_decomp.judge

        # Both should have valid percentages
        assert abs(uniform_decomp.eif_pct + uniform_decomp.judge_pct - 100) < 0.1
        assert abs(high_unc_decomp.eif_pct + high_unc_decomp.judge_pct - 100) < 0.1

    def test_policy_ranking(self) -> None:
        """Test policy ranking functionality."""
        np.random.seed(789)
        n_samples = 100

        # Create judge scores
        judge_scores = [
            JudgeScore(mean=np.random.uniform(0.3, 0.8), variance=0.04)
            for _ in range(n_samples)
        ]
        oracle_rewards = np.array([s.mean for s in judge_scores])

        # Create 4 policies with known quality ordering
        weights = np.zeros((n_samples, 4))

        # Manipulate weights to create expected ordering
        for i in range(n_samples):
            if judge_scores[i].mean > 0.7:  # High quality samples
                weights[i, 0] = 0.5  # Policy 0: Low on high quality
                weights[i, 1] = 1.0  # Policy 1: Medium
                weights[i, 2] = 1.5  # Policy 2: High
                weights[i, 3] = 2.0  # Policy 3: Highest on high quality
            else:  # Low quality samples
                weights[i, 0] = 2.0  # Policy 0: High on low quality
                weights[i, 1] = 1.5
                weights[i, 2] = 1.0
                weights[i, 3] = 0.5  # Policy 3: Low on low quality

        # Run estimation
        config = UncertaintyEstimatorConfig(k_folds=3)
        estimator = UncertaintyAwareDRCPO(config)
        result = estimator.fit(
            X=None,
            judge_scores=judge_scores,
            oracle_rewards=oracle_rewards,
            importance_weights=weights,
            policy_names=["Worst", "Below_Average", "Above_Average", "Best"],
        )

        # Check ranking
        ranking = result.rank_policies()

        # Best should be first, Worst should be last
        assert ranking[0] == "Best"
        assert ranking[-1] == "Worst"

        # Verify estimates follow expected order
        estimates = result.get_estimates()
        best_idx = 3
        worst_idx = 0
        assert estimates[best_idx] > estimates[worst_idx]

    def test_result_summary(self) -> None:
        """Test result summary generation."""
        np.random.seed(999)

        # Create simple test case
        judge_scores = [JudgeScore(mean=0.6, variance=0.05) for _ in range(50)]
        oracle_rewards = np.array([0.6] * 50)
        weights = np.ones((50, 2))

        config = UncertaintyEstimatorConfig(k_folds=2)
        estimator = UncertaintyAwareDRCPO(config)
        result = estimator.fit(
            X=None,
            judge_scores=judge_scores,
            oracle_rewards=oracle_rewards,
            importance_weights=weights,
            policy_names=["Control", "Treatment"],
        )

        # Get summary
        summary = result.summary()

        # Check summary contains key information
        assert "Multi-Policy Uncertainty-Aware Results" in summary
        assert "Control" in summary
        assert "Treatment" in summary
        assert "ESS:" in summary
        assert "Variance:" in summary
        assert "Rankings:" in summary

    def test_standard_result_compatibility(self) -> None:
        """Test conversion to standard EstimationResult."""
        np.random.seed(111)

        # Create simple test case
        judge_scores = [JudgeScore(mean=0.5, variance=0.03) for _ in range(30)]
        oracle_rewards = np.array([0.5] * 30)
        weights = np.ones((30, 1))

        config = UncertaintyEstimatorConfig(k_folds=2)
        estimator = UncertaintyAwareDRCPO(config)
        uncertainty_result = estimator.fit(
            X=None,
            judge_scores=judge_scores,
            oracle_rewards=oracle_rewards,
            importance_weights=weights,
            policy_names=["SinglePolicy"],
        )

        # Convert to standard result
        standard_result = uncertainty_result.to_standard_result()

        # Check compatibility
        assert hasattr(standard_result, "v_hat")
        assert hasattr(standard_result, "se")
        assert hasattr(standard_result, "n")
        assert standard_result.n == 30
        assert len(standard_result.v_hat) == 1
        assert len(standard_result.se) == 1


@pytest.mark.parametrize("n_policies", [1, 2, 5, 10])
def test_various_policy_counts(n_policies: int) -> None:
    """Test with different numbers of policies."""
    np.random.seed(200)
    n_samples = 100

    # Create data
    judge_scores = [
        JudgeScore(mean=np.random.uniform(0.4, 0.7), variance=0.04)
        for _ in range(n_samples)
    ]
    oracle_rewards = np.array([s.mean for s in judge_scores])

    # Random weights for each policy
    weights = np.random.exponential(1.0, size=(n_samples, n_policies))

    # Policy names
    policy_names = [f"Policy_{i}" for i in range(n_policies)]

    # Run estimation
    config = UncertaintyEstimatorConfig(k_folds=2)
    estimator = UncertaintyAwareDRCPO(config)
    result = estimator.fit(
        X=None,
        judge_scores=judge_scores,
        oracle_rewards=oracle_rewards,
        importance_weights=weights,
        policy_names=policy_names,
    )

    # Verify correct number of policies
    assert result.n_policies == n_policies
    assert len(result.get_estimates()) == n_policies
    assert len(result.get_standard_errors()) == n_policies
    assert len(result.rank_policies()) == n_policies
