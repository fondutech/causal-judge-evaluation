"""Test the refactored uncertainty-aware CJE implementation."""

import pytest
import numpy as np
from typing import List, Dict, Any

from cje.uncertainty import (
    JudgeScore,
    UncertaintyAPIJudge,
    DeterministicJudge,
    UncertaintyAwareDRCPO,
    create_uncertainty_report,
)
from cje.uncertainty.judge import MockUncertaintyJudge, UncertaintyJudgeConfig
from cje.uncertainty.estimator import UncertaintyEstimatorConfig


class TestUncertaintyRefactor:
    """Test suite for the clean uncertainty implementation."""

    def test_judge_score_validation(self) -> None:
        """Test JudgeScore validation."""
        # Valid score
        score = JudgeScore(mean=0.7, variance=0.05)
        assert score.mean == 0.7
        assert score.variance == 0.05

        # Invalid mean
        with pytest.raises(ValueError, match="must be in"):
            JudgeScore(mean=1.5, variance=0.05)

        # Invalid variance
        with pytest.raises(ValueError, match="cannot be negative"):
            JudgeScore(mean=0.5, variance=-0.1)

        # Variance too large
        with pytest.raises(ValueError, match="too large"):
            JudgeScore(mean=0.5, variance=0.3)

    def test_mock_uncertainty_judge(self) -> None:
        """Test mock judge for development."""
        judge = MockUncertaintyJudge(base_score=0.8, base_variance=0.02)

        # Single scoring
        score = judge.score("What is 2+2?", "4")
        assert isinstance(score, JudgeScore)
        assert 0.6 <= score.mean <= 1.0  # With noise
        assert score.variance > 0

        # Batch scoring
        samples = [
            {"context": f"Question {i}", "response": f"Answer {i}"} for i in range(5)
        ]
        scores = judge.score_batch(samples)
        assert len(scores) == 5
        assert all(isinstance(s, JudgeScore) for s in scores)

    def test_deterministic_judge_wrapper(self) -> None:
        """Test wrapping legacy judges."""

        # Mock legacy judge
        class LegacyJudge:
            def score(self, context: str, response: str) -> float:
                return 0.85

            def score_batch(self, samples: List[Dict[str, str]]) -> List[float]:
                return [0.85] * len(samples)

        # Wrap it
        wrapped = DeterministicJudge(LegacyJudge())

        # Test single score
        score = wrapped.score("context", "response")
        assert isinstance(score, JudgeScore)
        assert score.mean == 0.85
        assert score.variance == 0.0  # Deterministic

        # Test batch
        samples = [{"context": "q", "response": "a"}] * 3
        scores = wrapped.score_batch(samples)
        assert len(scores) == 3
        assert all(s.variance == 0.0 for s in scores)

    def test_uncertainty_aware_estimation(self) -> None:
        """Test the uncertainty-aware estimator."""
        np.random.seed(42)
        n_samples = 100

        # Generate mock data
        judge = MockUncertaintyJudge(base_score=0.7, base_variance=0.05)
        samples = [{"context": f"Q{i}", "response": f"A{i}"} for i in range(n_samples)]

        # Get judge scores
        judge_scores = judge.score_batch(samples)

        # Mock oracle rewards (correlated with judge scores)
        oracle_rewards = np.array(
            [np.clip(s.mean + np.random.normal(0, 0.1), 0, 1) for s in judge_scores]
        )

        # Mock importance weights
        weights = np.exp(np.random.normal(0, 0.5, n_samples))
        weights = weights / np.mean(weights)

        # Configure estimator
        config = UncertaintyEstimatorConfig(
            k_folds=3,
            use_variance_shrinkage=True,
            shrinkage_method="optimal",
        )

        # Fit estimator
        estimator = UncertaintyAwareDRCPO(config)
        result = estimator.fit(
            X=None,  # Not used
            judge_scores=judge_scores,
            oracle_rewards=oracle_rewards,
            importance_weights=weights,
            policy_names=["target_policy"],
        )

        # Check results
        estimates = result.get_estimates()
        se = result.get_standard_errors()
        assert len(estimates) == 1
        assert len(se) == 1
        assert se[0] > 0  # Should have uncertainty

        # Check metadata
        assert result.estimator_type == "UncertaintyAwareDRCPO"
        assert "ess_percentage" in result.global_metadata

    def test_variance_calibration_integration(self) -> None:
        """Test variance calibration in the pipeline."""
        from cje.uncertainty.calibration import calibrate_variance_gamma

        np.random.seed(123)
        n = 50

        # Overconfident judge scenario
        true_values = np.random.uniform(0.3, 0.7, n)
        judge_scores = [
            JudgeScore(
                mean=np.clip(true_values[i] + np.random.normal(0, 0.15), 0, 1),
                variance=0.01,  # Too confident
            )
            for i in range(n)
        ]

        # Compute gamma
        gamma = calibrate_variance_gamma(judge_scores, true_values)

        # Should detect overconfidence
        assert gamma > 1.5  # Judge underestimates uncertainty

    def test_uncertainty_report(self) -> None:
        """Test comprehensive uncertainty reporting."""
        np.random.seed(456)
        n = 100

        # Create data with varying uncertainty
        weights = np.abs(np.random.lognormal(0, 1, n))
        rewards = np.random.uniform(0.4, 0.8, n)
        variances = np.random.exponential(0.03, n)

        # High variance for high-weight samples
        high_weight_idx = np.argsort(weights)[-10:]
        variances[high_weight_idx] *= 5

        # Generate report
        report = create_uncertainty_report(
            weights=weights,
            rewards=rewards,
            variances=variances,
            estimate=0.65,
            se_with_uncertainty=0.08,
            se_without_uncertainty=0.05,
            gamma=2.1,
            shrinkage_applied=True,
            shrinkage_lambda=0.5,
        )

        # Check report structure
        assert "summary" in report
        assert "variance_decomposition" in report
        assert "variance_statistics" in report
        assert "calibration" in report
        assert "concentration" in report
        assert "recommendations" in report

        # Check specific values
        assert report["summary"]["se_increase_pct"] == pytest.approx(60.0, rel=0.01)
        assert report["calibration"]["gamma"] == 2.1
        assert report["shrinkage"]["applied"] is True
        assert report["shrinkage"]["lambda"] == 0.5

    def test_end_to_end_workflow(self) -> None:
        """Test complete uncertainty-aware workflow."""
        np.random.seed(789)

        # 1. Create uncertainty-aware judge
        judge = MockUncertaintyJudge(
            base_score=0.75,
            base_variance=0.04,
            noise_std=0.1,
        )

        # 2. Score samples
        n_samples = 200
        samples = [
            {"context": f"Question {i}", "response": f"Response {i}"}
            for i in range(n_samples)
        ]
        judge_scores = judge.score_batch(samples)

        # 3. Generate mock oracle labels
        oracle_rewards = np.array(
            [np.clip(s.mean + np.random.normal(0, 0.12), 0, 1) for s in judge_scores]
        )

        # 4. Generate importance weights (some extreme)
        log_weights = np.random.normal(0, 1.5, n_samples)
        log_weights[::20] += 3  # Some high weights
        weights = np.exp(log_weights)

        # 5. Configure and run estimator
        config = UncertaintyEstimatorConfig(
            k_folds=5,
            use_variance_shrinkage=True,
            shrinkage_method="adaptive",
            min_ess_ratio=0.15,
            calibrate_variance=True,
            compute_diagnostics=True,
        )

        estimator = UncertaintyAwareDRCPO(config)
        result = estimator.fit(
            X=None,
            judge_scores=judge_scores,
            oracle_rewards=oracle_rewards,
            importance_weights=weights,
            policy_names=["π_target"],
        )

        # 6. Verify results
        assert result.estimator_type == "UncertaintyAwareDRCPO"
        estimates = result.get_estimates()
        se = result.get_standard_errors()
        assert len(estimates) == 1
        assert 0 <= estimates[0] <= 1
        assert se[0] > 0

        # Check confidence interval
        ci_lower, ci_upper = result.get_confidence_intervals()[0]
        assert ci_lower < estimates[0] < ci_upper

        # 7. Generate detailed report
        variances = np.array([s.variance for s in judge_scores])
        report = create_uncertainty_report(
            weights=weights,
            rewards=oracle_rewards,
            variances=variances,
            estimate=estimates[0],
            se_with_uncertainty=se[0],
            se_without_uncertainty=se[0] * 0.7,  # Mock comparison
            gamma=1.8,
            shrinkage_applied=True,
            shrinkage_lambda=0.3,
        )

        # Verify report has recommendations
        assert len(report["recommendations"]) > 0
        assert "warnings" in report


@pytest.mark.parametrize("shrinkage_method", ["optimal", "adaptive", "fixed"])
def test_shrinkage_methods(shrinkage_method: str) -> None:
    """Test different shrinkage methods."""
    np.random.seed(100)
    n = 50

    # Create test data
    judge_scores = [
        JudgeScore(
            mean=np.random.uniform(0.4, 0.8), variance=np.random.exponential(0.05)
        )
        for _ in range(n)
    ]

    oracle_rewards = np.array([s.mean for s in judge_scores])
    weights = np.exp(np.random.normal(0, 0.8, n))

    # Configure estimator
    config = UncertaintyEstimatorConfig(
        k_folds=3,
        use_variance_shrinkage=True,
        shrinkage_method=shrinkage_method,
        fixed_shrinkage_lambda=1.0,
    )

    # Run estimation
    estimator = UncertaintyAwareDRCPO(config)
    result = estimator.fit(
        X=None,
        judge_scores=judge_scores,
        oracle_rewards=oracle_rewards,
        importance_weights=weights,
    )

    # Should produce valid results
    estimates = result.get_estimates()
    se = result.get_standard_errors()
    assert estimates[0] > 0
    assert se[0] > 0
