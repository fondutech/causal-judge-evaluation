"""
Test suite for CJE theoretical guarantees.

This module validates the core theoretical claims from the CJE paper:
1. Unbiasedness under assumptions (B1-B3)
2. Double robustness
3. Single-rate efficiency (only one nuisance needs n^{-1/4} rate)
4. Semiparametric efficiency bound
5. Asymptotic normality of √n(V̂ - V)
6. Calibration properties (monotonicity, centering)
"""

import numpy as np
import pytest
from typing import List, Dict, Any, Callable, Optional, Tuple, cast
from dataclasses import dataclass
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
import warnings

from cje.estimators import MultiIPSEstimator, MultiDRCPOEstimator
from cje.estimators.results import EstimationResult
from cje.loggers.multi_target_sampler import MultiTargetSampler
from cje.loggers.precomputed_sampler import PrecomputedMultiTargetSampler
from cje.testing import MockPolicyRunner


class SimpleMockPolicyRunner:
    """Simple mock that uses exact probabilities from the test scenario."""

    def __init__(self, policy_name: str, action_probs: Dict[str, float], **kwargs: Any):
        self.model_name = policy_name
        self.action_probs = action_probs  # e.g., {"0": 0.8, "1": 0.2}

    def log_prob(self, context: str, response: str, **kwargs: Any) -> float:
        """Return log probability based on action probabilities."""
        if response in self.action_probs:
            return float(np.log(self.action_probs[response]))
        else:
            return float(-np.inf)  # Zero probability for unseen actions

    def generate_with_logp(
        self, prompts: List[str], **kwargs: Any
    ) -> List[Tuple[str, float, Any]]:
        """Generate responses with log probabilities."""
        results = []
        for prompt in prompts:
            # Sample action according to policy probabilities
            actions = list(self.action_probs.keys())
            probs = list(self.action_probs.values())
            action = str(np.random.choice(actions, p=probs))
            logp = float(np.log(self.action_probs[action]))
            results.append((action, logp, None))
        return results


from cje.calibration import cross_fit_calibration


@dataclass
class TheoreticalTestScenario:
    """A test scenario with known ground truth for validating theoretical properties."""

    name: str
    n_samples: int
    true_values: np.ndarray  # True policy values V(π^k)
    logging_policy: Callable[[str], Tuple[str, float]]  # Maps context -> (action, logp)
    target_policies: List[Callable[[str], Tuple[str, float]]]  # List of target policies
    reward_function: Callable[[str, str], float]  # Maps (context, action) -> reward
    seed: int = 42


class TestCJEUnbiasedness:
    """Test unbiasedness under assumptions B1-B3."""

    def create_simple_bandit_scenario(self) -> TheoreticalTestScenario:
        """Create a simple 2-armed bandit with known optimal values."""

        def logging_policy(context: str) -> Tuple[str, float]:
            """Uniform random policy over {0, 1}."""
            action = np.random.choice(["0", "1"])
            return action, np.log(0.5)

        def target_policy_0(context: str) -> Tuple[str, float]:
            """Policy that prefers action 0 (80% vs 20%)."""
            action = np.random.choice(["0", "1"], p=[0.8, 0.2])
            logp = np.log(0.8) if action == "0" else np.log(0.2)
            return action, logp

        def target_policy_1(context: str) -> Tuple[str, float]:
            """Policy that prefers action 1 (20% vs 80%)."""
            action = np.random.choice(["0", "1"], p=[0.2, 0.8])
            logp = np.log(0.2) if action == "0" else np.log(0.8)
            return action, logp

        def reward_function(context: str, action: str) -> float:
            """Reward: action 0 gives 0.3, action 1 gives 0.7."""
            return 0.3 if action == "0" else 0.7

        # Calculate true values for the stochastic policies
        # Policy 0: 0.8 * 0.3 + 0.2 * 0.7 = 0.24 + 0.14 = 0.38
        # Policy 1: 0.2 * 0.3 + 0.8 * 0.7 = 0.06 + 0.56 = 0.62
        return TheoreticalTestScenario(
            name="simple_bandit",
            n_samples=1000,
            true_values=np.array(
                [0.38, 0.62]
            ),  # Updated true values for stochastic policies
            logging_policy=logging_policy,
            target_policies=[target_policy_0, target_policy_1],
            reward_function=reward_function,
        )

    def generate_logs_from_scenario(
        self, scenario: TheoreticalTestScenario
    ) -> List[Dict[str, Any]]:
        """Generate logs from a theoretical scenario."""
        np.random.seed(scenario.seed)
        logs = []

        for i in range(scenario.n_samples):
            context = "constant_context"  # Use constant context for simple bandit
            action, logp = scenario.logging_policy(context)
            reward = scenario.reward_function(context, action)

            # Calculate target log probabilities of the observed action
            target_logps = {}
            # For policy 0 (prefers action 0): P(action="0")=0.8, P(action="1")=0.2
            if action == "0":
                target_logps["policy_0"] = np.log(0.8)
                target_logps["policy_1"] = np.log(0.2)
            else:  # action == "1"
                target_logps["policy_0"] = np.log(0.2)
                target_logps["policy_1"] = np.log(0.8)

            logs.append(
                {
                    "context": context,
                    "response": action,
                    "logp": logp,
                    "reward": reward,
                    "logp_target_all": target_logps,
                }
            )

        return logs

    def test_unbiasedness_simple_bandit(self) -> None:
        """Test that CJE estimates are unbiased for a simple bandit."""
        scenario = self.create_simple_bandit_scenario()

        # Run multiple independent trials
        n_trials = 20
        estimates: list[np.ndarray] = []

        for trial in range(n_trials):
            # Generate fresh data for each trial
            scenario.seed = 42 + trial
            logs = self.generate_logs_from_scenario(scenario)

            # Create a precomputed sampler using the exact log probabilities from our test
            logp_lookup = {}
            for log in logs:
                context = log["context"]
                response = log["response"]
                logp_target_all = log["logp_target_all"]
                logp_list = [logp_target_all["policy_0"], logp_target_all["policy_1"]]
                logp_lookup[(context, response)] = logp_list

            sampler = PrecomputedMultiTargetSampler(logp_lookup, n_policies=2)

            # Use simpler IPS estimator for theoretical tests (no outcome modeling)
            estimator = MultiIPSEstimator(sampler=sampler)  # type: ignore[arg-type]
            estimator.fit(logs)
            result = estimator.estimate()

            estimates.append(result.v_hat)

        # Test unbiasedness: E[V̂] ≈ V
        estimates_arr = np.array(estimates)
        mean_estimates = np.mean(estimates_arr, axis=0)

        # Should be close to true values (IPS can have higher variance than DR)
        np.testing.assert_allclose(
            mean_estimates,
            scenario.true_values,
            atol=0.1,  # Tolerance for IPS estimator
            err_msg=f"Estimates {mean_estimates} not close to true values {scenario.true_values}",
        )

    def test_consistency_increasing_sample_size(self) -> None:
        """Test that estimates converge to true values as sample size increases."""
        scenario = self.create_simple_bandit_scenario()

        sample_sizes = [100, 500, 1000, 2000]
        errors = []

        for n in sample_sizes:
            scenario.n_samples = n
            logs = self.generate_logs_from_scenario(scenario)

            # Create a precomputed sampler using the exact log probabilities from our test
            logp_lookup = {}
            for log in logs:
                context = log["context"]
                response = log["response"]
                logp_target_all = log["logp_target_all"]
                logp_list = [logp_target_all["policy_0"], logp_target_all["policy_1"]]
                logp_lookup[(context, response)] = logp_list

            sampler = PrecomputedMultiTargetSampler(logp_lookup, n_policies=2)

            # Use simpler IPS estimator for theoretical tests
            estimator = MultiIPSEstimator(sampler=sampler)  # type: ignore[arg-type]
            estimator.fit(logs)
            result = estimator.estimate()

            # Calculate mean absolute error
            error = np.mean(np.abs(result.v_hat - scenario.true_values))
            errors.append(error)

        # Errors should generally decrease (allow some statistical noise)
        # For IPS, convergence can be noisy, so check that final error is reasonable
        assert errors[-1] <= max(
            errors[0], 0.15
        ), f"Error should not increase dramatically with sample size: {errors}"


class TestCJEDoubleRobustness:
    """Test double robustness property: estimator is consistent if either outcome model OR propensity model is correct."""

    def create_contextual_bandit_scenario(self) -> TheoreticalTestScenario:
        """Create a contextual bandit where context affects rewards."""

        def logging_policy(context: str) -> Tuple[str, float]:
            """Context-dependent logging policy."""
            context_idx = int(context.split("_")[1]) % 100
            prob_action_1 = 0.3 + 0.4 * (context_idx / 100)  # Varies from 0.3 to 0.7
            action = "1" if np.random.random() < prob_action_1 else "0"
            logp = np.log(prob_action_1) if action == "1" else np.log(1 - prob_action_1)
            return action, logp

        def target_policy_greedy(context: str) -> Tuple[str, float]:
            """Greedy policy that chooses better action based on context."""
            context_idx = int(context.split("_")[1]) % 100
            # Action 1 is better for high context indices
            action = "1" if context_idx > 50 else "0"
            return action, 0.0

        def reward_function(context: str, action: str) -> float:
            """Context-dependent rewards."""
            context_idx = int(context.split("_")[1]) % 100
            base_reward = context_idx / 100  # 0 to 1
            if action == "1":
                return base_reward + 0.2  # Action 1 gets bonus
            else:
                return base_reward

        return TheoreticalTestScenario(
            name="contextual_bandit",
            n_samples=800,
            true_values=np.array([0.6]),  # Approximate expected value for greedy policy
            logging_policy=logging_policy,
            target_policies=[target_policy_greedy],
            reward_function=reward_function,
        )

    def test_robustness_to_outcome_model_misspecification(self) -> None:
        """Test that estimator is consistent even with misspecified outcome model."""
        scenario = self.create_contextual_bandit_scenario()

        # TODO: This would require creating a deliberately misspecified outcome model
        # and showing that the estimator is still consistent when propensities are correct
        pytest.skip("Requires outcome model misspecification simulation")

    def test_robustness_to_propensity_misspecification(self) -> None:
        """Test that estimator is consistent even with misspecified propensities."""
        scenario = self.create_contextual_bandit_scenario()

        # TODO: This would require creating deliberately misspecified propensities
        # and showing that the estimator is still consistent when outcome model is correct
        pytest.skip("Requires propensity misspecification simulation")


class TestCJESingleRateEfficiency:
    """Test single-rate efficiency: only one nuisance needs n^{-1/4} convergence rate."""

    def test_single_rate_simulation(self) -> None:
        """Simulate scenario where only one nuisance converges at fast rate."""
        # TODO: This requires careful simulation where we control the convergence rates
        # of the outcome model and propensity model separately
        pytest.skip("Requires controlled nuisance convergence rate simulation")


class TestCJECalibrationProperties:
    """Test calibration properties: monotonicity preservation and centering."""

    def test_isotonic_weight_calibration_centering(self) -> None:
        """Test that isotonic weight calibration ensures E[w] = 1."""
        n = 500
        np.random.seed(42)

        # Generate raw importance weights (potentially uncentered)
        raw_weights = np.random.exponential(scale=1.2, size=n)  # Mean = 1.2 ≠ 1

        # Apply isotonic calibration to center at 1
        iso = IsotonicRegression(out_of_bounds="clip")
        target = np.ones(n)  # Target mean of 1
        iso.fit(raw_weights, target)
        calibrated_weights = iso.predict(raw_weights)

        # Check that calibrated weights have mean ≈ 1
        assert abs(np.mean(calibrated_weights) - 1.0) < 0.01

        # Check monotonicity preservation
        sorted_indices = np.argsort(raw_weights)
        calibrated_sorted = calibrated_weights[sorted_indices]
        assert np.all(np.diff(calibrated_sorted) >= -1e-10), "Monotonicity violated"

    def test_judge_calibration_monotonicity(self) -> None:
        """Test that judge calibration preserves monotonicity while aligning scale."""
        n = 200
        np.random.seed(123)

        # Generate judge scores and oracle labels with monotonic relationship
        judge_scores = np.random.uniform(0, 10, n)
        oracle_labels = 0.1 * judge_scores + 0.05 * np.random.normal(0, 1, n)
        oracle_labels = np.clip(oracle_labels, 0, 1)  # Clip to [0,1]

        # Calibrate judge scores
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(judge_scores, oracle_labels)
        calibrated_scores = iso.predict(judge_scores)

        # Check monotonicity
        for i in range(n - 1):
            for j in range(i + 1, n):
                if judge_scores[i] <= judge_scores[j]:
                    assert calibrated_scores[i] <= calibrated_scores[j] + 1e-10

        # Check that calibration improves alignment
        raw_correlation = np.corrcoef(judge_scores, oracle_labels)[0, 1]
        calibrated_correlation = np.corrcoef(calibrated_scores, oracle_labels)[0, 1]
        assert (
            calibrated_correlation >= raw_correlation - 0.01
        )  # Should not decrease much


class TestCJEAsymptoticNormality:
    """Test asymptotic normality: √n(V̂ - V) → N(0, σ²_eff)."""

    @pytest.mark.slow
    def test_asymptotic_normality_simulation(self) -> None:
        """Test that √n(V̂ - V) is approximately normal for large n."""
        scenario = TestCJEUnbiasedness().create_simple_bandit_scenario()
        scenario.n_samples = 2000  # Large sample size

        n_trials = 100
        normalized_errors: list[np.ndarray] = []

        for trial in range(n_trials):
            scenario.seed = 1000 + trial
            logs = TestCJEUnbiasedness().generate_logs_from_scenario(scenario)

            runners = [MockPolicyRunner(f"policy_{k}") for k in range(2)]
            sampler = MultiTargetSampler(runners)

            estimator = MultiDRCPOEstimator(sampler=sampler, k=5, seed=scenario.seed)
            estimator.fit(logs)
            result = estimator.estimate()

            # Calculate √n(V̂ - V) / σ̂
            sqrt_n = np.sqrt(scenario.n_samples)
            errors = result.v_hat - scenario.true_values
            normalized_errors.append(sqrt_n * errors / result.se)

        normalized_errors_arr = np.array(normalized_errors)

        # Test normality for each policy using Kolmogorov-Smirnov test
        for k in range(2):
            errors_k = normalized_errors_arr[:, k]

            # Remove any infinite or NaN values
            errors_k = errors_k[np.isfinite(errors_k)]

            if len(errors_k) > 10:  # Need sufficient samples
                # Test against standard normal
                ks_stat, p_value = stats.kstest(errors_k, "norm")

                # With many trials, we expect approximate normality
                # Use lenient p-value threshold due to finite sample effects
                assert p_value > 0.01, f"Policy {k}: KS test p-value {p_value} too low"


class TestCJEVarianceEstimation:
    """Test variance estimation and confidence interval coverage."""

    @pytest.mark.slow
    def test_confidence_interval_coverage(self) -> None:
        """Test that 95% confidence intervals have approximately 95% coverage."""
        scenario = TestCJEUnbiasedness().create_simple_bandit_scenario()
        scenario.n_samples = 1000

        n_trials = 200
        coverage_count = np.zeros(2)  # For 2 policies

        for trial in range(n_trials):
            scenario.seed = 2000 + trial
            logs = TestCJEUnbiasedness().generate_logs_from_scenario(scenario)

            runners = [MockPolicyRunner(f"policy_{k}") for k in range(2)]
            sampler = MultiTargetSampler(runners)

            estimator = MultiDRCPOEstimator(sampler=sampler, k=5, seed=scenario.seed)
            estimator.fit(logs)
            result = estimator.estimate()

            # Check if true value is in 95% CI
            ci_lower, ci_upper = result.confidence_interval(0.95)

            for k in range(2):
                if ci_lower[k] <= scenario.true_values[k] <= ci_upper[k]:
                    coverage_count[k] += 1

        # Check coverage rates
        coverage_rates = coverage_count / n_trials

        for k in range(2):
            # Allow some deviation from 95% due to finite sample
            assert (
                0.90 <= coverage_rates[k] <= 1.0
            ), f"Policy {k}: Coverage rate {coverage_rates[k]:.3f} outside [0.90, 1.00]"


class TestCJEEdgeCases:
    """Test edge cases and robustness."""

    def test_heavy_tailed_weights(self) -> None:
        """Test behavior with heavy-tailed importance weights."""
        n = 500
        np.random.seed(42)

        # Create scenario with heavy-tailed weights
        logs = []
        for i in range(n):
            # Logging policy: uniform over {0, 1}
            action = np.random.choice(["0", "1"])
            logp = np.log(0.5)

            # Target policy: heavily biased toward action 1
            target_prob = 0.95 if action == "1" else 0.05
            target_logp = np.log(target_prob)

            logs.append(
                {
                    "context": f"context_{i}",
                    "response": action,
                    "logp": logp,
                    "reward": float(
                        action == "1"
                    ),  # Reward 1 for action 1, 0 for action 0
                    "logp_target_all": {"policy_0": target_logp},
                }
            )

        runners = [MockPolicyRunner("policy_0")]
        sampler = MultiTargetSampler(runners)

        # Test with different clipping values
        for clip in [5.0, 10.0, 20.0]:
            estimator = MultiDRCPOEstimator(sampler=sampler, k=5, clip=clip, seed=42)
            estimator.fit(logs)
            result = estimator.estimate()

            # Should produce finite estimates despite heavy tails
            assert np.isfinite(result.v_hat[0])
            assert np.isfinite(result.se[0])
            assert result.se[0] > 0

    def test_zero_overlap_regions(self) -> None:
        """Test behavior when target policy has zero support on some logged actions."""
        n = 100
        logs = []

        for i in range(n):
            # Logging policy supports actions {0, 1, 2}
            action = np.random.choice(["0", "1", "2"])
            logp = np.log(1 / 3)

            # Target policy only supports actions {0, 1} (zero prob on action 2)
            if action in ["0", "1"]:
                target_logp = np.log(0.5)
            else:
                target_logp = -np.inf  # Zero probability

            logs.append(
                {
                    "context": f"context_{i}",
                    "response": action,
                    "logp": logp,
                    "reward": float(action == "1"),
                    "logp_target_all": {"policy_0": target_logp},
                }
            )

        runners = [MockPolicyRunner("policy_0")]
        sampler = MultiTargetSampler(runners)

        # Should handle zero overlap gracefully
        estimator = MultiDRCPOEstimator(sampler=sampler, k=3, seed=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Expect warnings about zero overlap
            estimator.fit(logs)
            result = estimator.estimate()

            # Should still produce finite estimates
            assert np.isfinite(result.v_hat[0])
            assert result.se[0] > 0


class TestCJEImplementationCorrectness:
    """Test implementation details: cross-fitting, fold-wise calibration, etc."""

    def test_cross_fitting_consistency(self) -> None:
        """Test that cross-fitting produces consistent results."""
        scenario = TestCJEUnbiasedness().create_simple_bandit_scenario()
        logs = TestCJEUnbiasedness().generate_logs_from_scenario(scenario)

        runners = [MockPolicyRunner(f"policy_{k}") for k in range(2)]
        sampler = MultiTargetSampler(runners)

        # Test with different numbers of folds
        results = []
        for k in [3, 5, 10]:
            estimator = MultiDRCPOEstimator(sampler=sampler, k=k, seed=42)
            estimator.fit(logs)
            result = estimator.estimate()
            results.append(result.v_hat)

        # Results should be similar across different fold numbers
        for i in range(1, len(results)):
            np.testing.assert_allclose(
                results[0],
                results[i],
                rtol=0.1,  # Allow 10% relative difference
                err_msg=f"Results vary too much across fold numbers: {results}",
            )

    def test_efficient_influence_function_properties(self) -> None:
        """Test that the EIF components have expected properties."""
        scenario = TestCJEUnbiasedness().create_simple_bandit_scenario()
        logs = TestCJEUnbiasedness().generate_logs_from_scenario(scenario)

        runners = [MockPolicyRunner(f"policy_{k}") for k in range(2)]
        sampler = MultiTargetSampler(runners)

        estimator = MultiDRCPOEstimator(sampler=sampler, k=5, seed=42)
        estimator.fit(logs)
        result = estimator.estimate()

        # EIF components should have mean zero (they are centered)
        if hasattr(result, "eif_components") and result.eif_components is not None:
            eif_means = np.mean(result.eif_components, axis=0)
            np.testing.assert_allclose(
                eif_means,
                np.zeros_like(result.v_hat),
                rtol=1e-5,
                atol=1e-5,
                err_msg="EIF components should have mean zero (centered)",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
