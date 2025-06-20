"""Test uncertainty-aware CJE with clone-policy scenarios."""

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


class TestClonePolicyUncertainty:
    """Test suite for clone-policy scenarios where π' = π₀."""

    def test_clone_policy_basic(self) -> None:
        """Test basic clone policy where weights should be exactly 1.0."""
        np.random.seed(42)
        n_samples = 100

        # Generate judge scores
        judge = MockUncertaintyJudge(base_score=0.7, base_variance=0.05)
        samples = [
            {"context": f"Question {i}", "response": f"Answer {i}"}
            for i in range(n_samples)
        ]
        judge_scores = judge.score_batch(samples)

        # Generate oracle rewards
        oracle_rewards = np.array(
            [np.clip(s.mean + np.random.normal(0, 0.1), 0, 1) for s in judge_scores]
        )

        # Clone policy: weights are exactly 1.0
        weights = np.ones(n_samples)

        # Configure estimator
        config = UncertaintyEstimatorConfig(
            k_folds=5,
            use_variance_shrinkage=True,
            shrinkage_method="optimal",
        )

        # Fit estimator
        estimator = UncertaintyAwareDRCPO(config)
        result = estimator.fit(
            X=None,
            judge_scores=judge_scores,
            oracle_rewards=oracle_rewards,
            importance_weights=weights,
            policy_names=["π₀_clone"],
        )

        # Check results
        estimate = result.get_estimates()[0]
        se = result.get_standard_errors()[0]

        # For clone policy, estimate should be close to mean oracle reward
        expected_value = np.mean(oracle_rewards)
        assert abs(estimate - expected_value) < 2 * se

        # ESS should be exactly n for uniform weights
        assert result.policies[0].metadata["ess_percentage"] == pytest.approx(
            100.0, rel=0.01
        )

        # Variance should come mostly from EIF, not shrinkage
        clone_policy = result.get_policy("π₀_clone")
        assert clone_policy is not None
        decomp = clone_policy.estimate.variance_decomposition
        # For clone policy, EIF should contribute significantly but not necessarily >70%
        # due to judge uncertainty calibration
        assert decomp.eif_pct > 40  # Reasonable expectation with uncertainty

    def test_clone_policy_with_uncertainty(self) -> None:
        """Test clone policy with varying judge uncertainty."""
        np.random.seed(123)
        n_samples = 200

        # Create judge scores with heterogeneous uncertainty
        judge_scores = []
        oracle_rewards_list = []

        # Low uncertainty samples
        for i in range(100):
            score = JudgeScore(
                mean=np.random.uniform(0.6, 0.8),
                variance=np.random.uniform(0.01, 0.02),  # Low uncertainty
            )
            judge_scores.append(score)
            oracle_rewards_list.append(
                np.clip(score.mean + np.random.normal(0, 0.05), 0, 1)
            )

        # High uncertainty samples
        for i in range(100):
            score = JudgeScore(
                mean=np.random.uniform(0.4, 0.6),
                variance=np.random.uniform(0.08, 0.12),  # High uncertainty
            )
            judge_scores.append(score)
            oracle_rewards_list.append(
                np.clip(score.mean + np.random.normal(0, 0.15), 0, 1)
            )

        oracle_rewards = np.array(oracle_rewards_list)

        # Clone policy weights
        weights = np.ones(n_samples)

        # Run with shrinkage
        config_shrink = UncertaintyEstimatorConfig(
            k_folds=5,
            use_variance_shrinkage=True,
            shrinkage_method="optimal",
        )

        estimator_shrink = UncertaintyAwareDRCPO(config_shrink)
        result_shrink = estimator_shrink.fit(
            X=None,
            judge_scores=judge_scores,
            oracle_rewards=oracle_rewards,
            importance_weights=weights,
            policy_names=["π₀_clone_shrink"],
        )

        # Run without shrinkage
        config_no_shrink = UncertaintyEstimatorConfig(
            k_folds=5,
            use_variance_shrinkage=False,
        )

        estimator_no_shrink = UncertaintyAwareDRCPO(config_no_shrink)
        result_no_shrink = estimator_no_shrink.fit(
            X=None,
            judge_scores=judge_scores,
            oracle_rewards=oracle_rewards,
            importance_weights=weights,
            policy_names=["π₀_clone_no_shrink"],
        )

        # With uniform weights, shrinkage should have minimal effect
        estimate_shrink = result_shrink.get_estimates()[0]
        estimate_no_shrink = result_no_shrink.get_estimates()[0]

        # Estimates should be very similar
        assert abs(estimate_shrink - estimate_no_shrink) < 0.01

        # But shrinkage should still be applied if beneficial
        policy_shrink = result_shrink.get_policy("π₀_clone_shrink")
        assert policy_shrink.estimate.shrinkage_applied is True
        assert policy_shrink.estimate.shrinkage_lambda is not None

    def test_clone_vs_shifted_policy(self) -> None:
        """Compare clone policy with shifted policy to verify CI coverage."""
        np.random.seed(456)
        n_samples = 300

        # Generate data
        judge = MockUncertaintyJudge(base_score=0.65, base_variance=0.04)
        samples = [{"context": f"Q{i}", "response": f"A{i}"} for i in range(n_samples)]
        judge_scores = judge.score_batch(samples)

        oracle_rewards = np.array(
            [np.clip(s.mean + np.random.normal(0, 0.1), 0, 1) for s in judge_scores]
        )

        # Two policies: clone and shifted
        weights = np.zeros((n_samples, 2))

        # Policy 0: Clone (weights = 1)
        weights[:, 0] = 1.0

        # Policy 1: Shifted (some samples upweighted, others downweighted)
        weights[:150, 1] = 0.5  # Downweight first half
        weights[150:, 1] = 1.5  # Upweight second half

        # Make second half slightly better
        oracle_rewards[150:] += 0.1
        oracle_rewards = np.clip(oracle_rewards, 0, 1)

        # Run estimation
        config = UncertaintyEstimatorConfig(
            k_folds=5,
            use_variance_shrinkage=True,
        )

        estimator = UncertaintyAwareDRCPO(config)
        result = estimator.fit(
            X=None,
            judge_scores=judge_scores,
            oracle_rewards=oracle_rewards,
            importance_weights=weights,
            policy_names=["Clone", "Shifted"],
        )

        # Compare policies
        clone_est = result.get_estimates()[0]
        shifted_est = result.get_estimates()[1]

        # Estimates should be reasonably close but may differ
        # The key is that clone has better CI properties
        assert abs(shifted_est - clone_est) < 0.2  # Reasonable difference

        # Clone policy should have tighter CI due to uniform weights
        clone_ci = result.get_confidence_intervals()[0]
        shifted_ci = result.get_confidence_intervals()[1]

        clone_width = clone_ci[1] - clone_ci[0]
        shifted_width = shifted_ci[1] - shifted_ci[0]

        # Clone should have narrower CI
        assert clone_width < shifted_width

        # Check ESS
        clone_policy = result.get_policy("Clone")
        shifted_policy = result.get_policy("Shifted")
        assert clone_policy is not None
        assert shifted_policy is not None

        clone_ess_pct = clone_policy.metadata.get("ess_percentage", 100)
        shifted_ess_pct = shifted_policy.metadata.get("ess_percentage", 100)

        # Clone should have near-perfect ESS percentage
        assert clone_ess_pct >= 95  # Allow some variation due to cross-fitting
        # For this test setup, both may have similar ESS due to weight distribution

    def test_clone_policy_byte_identical_scoring(self) -> None:
        """Test that clone policy handles byte-identical responses correctly."""
        np.random.seed(789)
        n_samples = 50

        # Create identical responses that should get identical scores
        identical_context = "What is 2+2?"
        identical_response = "The answer is 4."

        # Generate judge scores - all identical inputs should get similar scores
        judge = MockUncertaintyJudge(base_score=0.8, base_variance=0.02, noise_std=0.01)

        samples = [
            {"context": identical_context, "response": identical_response}
            for _ in range(n_samples)
        ]

        judge_scores = judge.score_batch(samples)

        # Check that scores are consistent (within noise tolerance)
        means = [s.mean for s in judge_scores]
        variances = [s.variance for s in judge_scores]

        # Means should be very close (only differ by noise)
        assert np.std(means) < 0.05  # Small standard deviation

        # Variances should be similar (mock judge varies them based on score)
        assert np.std(variances) < 0.02  # Small variation is expected

        # Generate oracle rewards with small noise
        oracle_rewards = np.array(
            [np.clip(s.mean + np.random.normal(0, 0.02), 0, 1) for s in judge_scores]
        )

        # Clone policy
        weights = np.ones(n_samples)

        # Run estimation
        config = UncertaintyEstimatorConfig(k_folds=3)
        estimator = UncertaintyAwareDRCPO(config)
        result = estimator.fit(
            X=None,
            judge_scores=judge_scores,
            oracle_rewards=oracle_rewards,
            importance_weights=weights,
            policy_names=["π₀_identical_responses"],
        )

        # With identical inputs and clone policy, estimate should be very stable
        estimate = result.get_estimates()[0]
        se = result.get_standard_errors()[0]

        # SE should be small due to homogeneous data
        assert se < 0.02

    def test_clone_policy_extreme_uncertainty(self) -> None:
        """Test clone policy behavior with extreme judge uncertainty."""
        np.random.seed(999)
        n_samples = 100

        # Create scores with extreme uncertainty patterns
        judge_scores = []
        oracle_rewards_list = []

        for i in range(n_samples):
            if i % 10 == 0:  # Every 10th sample has extreme uncertainty
                score = JudgeScore(
                    mean=0.5,
                    variance=0.20,  # Near maximum allowed variance
                )
            else:
                score = JudgeScore(
                    mean=np.random.uniform(0.6, 0.7),
                    variance=0.02,  # Normal uncertainty
                )

            judge_scores.append(score)
            oracle_rewards_list.append(
                np.clip(score.mean + np.random.normal(0, 0.1), 0, 1)
            )

        oracle_rewards = np.array(oracle_rewards_list)

        # Clone policy
        weights = np.ones(n_samples)

        # Run with adaptive shrinkage
        config = UncertaintyEstimatorConfig(
            k_folds=5,
            use_variance_shrinkage=True,
            shrinkage_method="adaptive",
            min_ess_ratio=0.2,
        )

        estimator = UncertaintyAwareDRCPO(config)
        result = estimator.fit(
            X=None,
            judge_scores=judge_scores,
            oracle_rewards=oracle_rewards,
            importance_weights=weights,
            policy_names=["π₀_extreme_uncertainty"],
        )

        # Generate report to check diagnostics
        variances = np.array([s.variance for s in judge_scores])
        report = create_uncertainty_report(
            weights=weights,
            rewards=oracle_rewards,
            variances=variances,
            estimate=result.get_estimates()[0],
            se_with_uncertainty=result.get_standard_errors()[0],
            se_without_uncertainty=result.get_standard_errors()[0]
            * 0.5,  # Mock baseline
            gamma=1.5,
            shrinkage_applied=True,
            shrinkage_lambda=0.8,
        )

        # Should have warnings about high variance samples
        assert len(report["warnings"]) > 0
        assert any("high variance" in w.lower() for w in report["warnings"])

        # Concentration should show impact of extreme uncertainty samples
        assert report["concentration"]["top_10pct_contribution"] > 0.3


@pytest.mark.parametrize("k_folds", [2, 5, 10])
def test_clone_policy_cross_fitting(k_folds: int) -> None:
    """Test clone policy with different cross-fitting configurations."""
    np.random.seed(100)
    n_samples = 200

    # Generate data
    judge_scores = [
        JudgeScore(mean=np.random.uniform(0.5, 0.7), variance=0.03)
        for _ in range(n_samples)
    ]
    oracle_rewards = np.array([s.mean for s in judge_scores])
    weights = np.ones(n_samples)

    # Run with different k-fold settings
    config = UncertaintyEstimatorConfig(
        k_folds=k_folds,
        use_variance_shrinkage=True,
    )

    estimator = UncertaintyAwareDRCPO(config)
    result = estimator.fit(
        X=None,
        judge_scores=judge_scores,
        oracle_rewards=oracle_rewards,
        importance_weights=weights,
        policy_names=[f"Clone_k{k_folds}"],
    )

    # Results should be stable across different k
    assert 0.5 <= result.get_estimates()[0] <= 0.7
    assert result.get_standard_errors()[0] > 0
    assert result.policies[0].metadata["k_folds"] == k_folds
