"""
Property-based testing for CJE.

This module tests fundamental mathematical properties and reliability invariants
that should hold for the CJE framework regardless of specific data.
"""

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple, Generator
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.extra.numpy import arrays

from cje.estimators import MultiDRCPOEstimator, MultiIPSEstimator
from cje.loggers.multi_target_sampler import MultiTargetSampler
from cje.testing import MockPolicyRunner


@pytest.fixture(scope="session")
def temp_work_dir() -> Generator[str, None, None]:
    """Provide a temporary work directory for DRCPO estimators and clean up afterward."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir
    # Cleanup also happens in project root as a fallback
    target_samples_file = Path("target_samples.jsonl")
    if target_samples_file.exists():
        target_samples_file.unlink()


class ExactLogProbMockPolicyRunner:
    """Simple mock that returns exact log probabilities for reliability testing."""

    def __init__(self, model_name: str, logp: float):
        self.model_name = model_name
        self.logp = logp

    def log_prob(self, context: str, response: str, **kwargs: Any) -> float:
        """Return the exact log probability specified at initialization."""
        return self.logp

    def generate_with_logp(
        self, prompts: List[str], **kwargs: Any
    ) -> List[Tuple[str, float, Any]]:
        """Generate responses with exact log probabilities."""
        return [(prompt, self.logp, None) for prompt in prompts]


# Hypothesis strategies for generating test data
@st.composite
def valid_log_entry(draw: Any) -> Dict[str, Any]:
    """Generate a valid log entry."""
    context = draw(st.text(min_size=1, max_size=50))
    response = draw(st.text(min_size=1, max_size=50))
    logp = draw(
        st.floats(min_value=-20.0, max_value=0.0, allow_nan=False, allow_infinity=False)
    )
    reward = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )

    # Target log probabilities
    n_policies = draw(st.integers(min_value=1, max_value=3))
    target_logps = {}
    for i in range(n_policies):
        target_logp = draw(
            st.floats(
                min_value=-20.0, max_value=0.0, allow_nan=False, allow_infinity=False
            )
        )
        target_logps[f"policy_{i}"] = target_logp

    return {
        "context": context,
        "response": response,
        "logp": logp,
        "reward": reward,
        "logp_target_all": target_logps,
    }


@st.composite
def valid_dataset(
    draw: Any, min_size: int = 10, max_size: int = 100
) -> Tuple[List[Dict[str, Any]], int]:
    """Generate a valid dataset."""
    n_samples = draw(st.integers(min_value=min_size, max_value=max_size))
    n_policies = draw(st.integers(min_value=1, max_value=3))

    logs = []
    for i in range(n_samples):
        log = draw(valid_log_entry())
        # Ensure consistent number of policies
        target_logps = {}
        for j in range(n_policies):
            target_logp = draw(
                st.floats(
                    min_value=-20.0,
                    max_value=0.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            target_logps[f"policy_{j}"] = target_logp
        log["logp_target_all"] = target_logps
        logs.append(log)

    return logs, n_policies


class TestFundamentalProperties:
    """Test fundamental mathematical properties that must always hold."""

    @given(test_data=valid_dataset(min_size=20, max_size=50))
    @settings(
        max_examples=5,
        deadline=30000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_output_validity(
        self, test_data: Tuple[List[Dict[str, Any]], int], temp_work_dir: str
    ) -> None:
        """Test that estimator outputs are always valid."""
        logs, n_policies = test_data

        runners = [MockPolicyRunner(f"policy_{i}") for i in range(n_policies)]
        sampler = MultiTargetSampler(runners)

        estimator = MultiDRCPOEstimator(
            sampler=sampler, k=3, seed=42, work_dir=temp_work_dir
        )
        estimator.fit(logs)
        result = estimator.estimate()

        # Basic output validity
        assert result.n_policies == n_policies
        assert result.v_hat.shape == (n_policies,)
        assert result.se.shape == (n_policies,)

        # All values must be finite
        assert np.all(np.isfinite(result.v_hat))
        assert np.all(np.isfinite(result.se))

        # Standard errors must be non-negative (can be 0 for degenerate cases)
        assert np.all(result.se >= 0)

        # Covariance matrix must be positive semi-definite
        if result.covariance_matrix is not None:
            eigenvals = np.linalg.eigvalsh(result.covariance_matrix)
            assert np.all(eigenvals >= -1e-10), f"Negative eigenvalues: {eigenvals}"

    def test_identical_policies_reliability(self, temp_work_dir: str) -> None:
        """CRITICAL RELIABILITY TEST: When target equals logging policy, weights should be 1."""
        logs: List[Dict[str, Any]] = []

        for i in range(50):
            logp = np.log(0.5)
            logs.append(
                {
                    "context": f"context_{i}",
                    "response": "action",
                    "logp": logp,
                    "reward": np.random.uniform(0, 1),
                    "logp_target_all": {"policy_0": logp},  # Identical policies
                }
            )

        # Use exact mock to guarantee identical log probabilities
        runners = [ExactLogProbMockPolicyRunner("policy_0", logp=np.log(0.5))]
        sampler = MultiTargetSampler(runners)

        estimator = MultiDRCPOEstimator(
            sampler=sampler,
            k=2,  # Minimum k=2 for cross-fitting
            seed=42,
            calibrate_weights=False,
            work_dir=temp_work_dir,
        )
        estimator.fit(logs)
        result = estimator.estimate()

        # When policies are identical, weights MUST be 1
        assert estimator.W is not None
        assert np.allclose(
            estimator.W, 1.0, atol=1e-8
        ), f"Weights not all 1: {estimator.W[:5]}"

        # Estimate should match empirical mean
        empirical_mean = np.mean([log["reward"] for log in logs])
        assert abs(result.v_hat[0] - empirical_mean) < 0.1

    def test_multiple_policies(self, temp_work_dir: str) -> None:
        """Test that multiple target policies work correctly."""
        logs: List[Dict[str, Any]] = []
        n_samples = 50

        # Different policies with different probabilities
        policies = {
            "policy_0": np.log(0.3),  # Low probability
            "policy_1": np.log(0.5),  # Medium probability (same as logging)
            "policy_2": np.log(0.7),  # High probability
        }

        logp_log = np.log(0.5)  # Logging policy

        for i in range(n_samples):
            logs.append(
                {
                    "context": f"context_{i}",
                    "response": "action",
                    "logp": logp_log,
                    "reward": np.random.uniform(0, 1),
                    "logp_target_all": policies,
                }
            )

        runners = [
            ExactLogProbMockPolicyRunner(name, logp=logp)
            for name, logp in policies.items()
        ]
        sampler = MultiTargetSampler(runners)

        estimator = MultiDRCPOEstimator(
            sampler=sampler,
            k=2,  # Minimum k=2 for cross-fitting
            seed=42,
            calibrate_weights=False,
            work_dir=temp_work_dir,
        )
        estimator.fit(logs)
        result = estimator.estimate()

        # Check weights are correct for each policy
        assert estimator.W is not None
        for i, (name, logp) in enumerate(policies.items()):
            expected_weight = np.exp(logp - logp_log)
            actual_weights = estimator.W[:, i]
            assert np.allclose(
                actual_weights, expected_weight, atol=1e-8
            ), f"Weights for {name} incorrect"

    @given(test_seed=st.integers(min_value=0, max_value=1000))
    @settings(
        max_examples=3, suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_reproducibility(self, test_seed: int, temp_work_dir: str) -> None:
        """Test that same seed produces identical results."""
        logs: List[Dict[str, Any]] = []
        for i in range(30):
            logs.append(
                {
                    "context": f"context_{i}",
                    "response": str(i % 2),
                    "logp": np.log(0.5),
                    "reward": 0.6,
                    "logp_target_all": {
                        "policy_0": np.log(0.8) if i % 2 == 1 else np.log(0.2)
                    },
                }
            )

        runners = [MockPolicyRunner("policy_0")]
        sampler = MultiTargetSampler(runners)

        # Two identical runs
        estimator1 = MultiDRCPOEstimator(
            sampler=sampler, k=3, seed=test_seed, work_dir=temp_work_dir
        )
        estimator1.fit(logs)
        result1 = estimator1.estimate()

        estimator2 = MultiDRCPOEstimator(
            sampler=sampler, k=3, seed=test_seed, work_dir=temp_work_dir
        )
        estimator2.fit(logs)
        result2 = estimator2.estimate()

        # Results must be identical
        np.testing.assert_allclose(result1.v_hat, result2.v_hat, rtol=1e-15)
        np.testing.assert_allclose(result1.se, result2.se, rtol=1e-15)


class TestErrorHandling:
    """Test graceful handling of edge cases and errors."""

    def test_empty_logs(self, temp_work_dir: str) -> None:
        """Test behavior with empty log list."""
        runners = [MockPolicyRunner("policy_0")]
        sampler = MultiTargetSampler(runners)
        estimator = MultiDRCPOEstimator(
            sampler=sampler, k=2, seed=42, work_dir=temp_work_dir
        )

        with pytest.raises(ValueError, match="empty logs"):
            estimator.fit([])

    def test_missing_required_fields(self, temp_work_dir: str) -> None:
        """Test behavior with malformed log entries."""
        incomplete_logs = [
            {
                "context": "test",
                "response": "test",
                # Missing logp, reward, logp_target_all
            }
        ]

        runners = [MockPolicyRunner("policy_0")]
        sampler = MultiTargetSampler(runners)
        estimator = MultiDRCPOEstimator(
            sampler=sampler, k=2, seed=42, work_dir=temp_work_dir
        )

        with pytest.raises(KeyError):
            estimator.fit(incomplete_logs)

    def test_minimal_dataset(self, temp_work_dir: str) -> None:
        """Test behavior with minimal valid dataset."""
        logs: List[Dict[str, Any]] = [
            {
                "context": "context_0",
                "response": "action",
                "logp": np.log(0.5),
                "reward": 0.5,
                "logp_target_all": {"policy_0": np.log(0.6)},
            },
            {
                "context": "context_1",
                "response": "action",
                "logp": np.log(0.5),
                "reward": 0.7,
                "logp_target_all": {"policy_0": np.log(0.6)},
            },
        ]

        runners = [MockPolicyRunner("policy_0")]
        sampler = MultiTargetSampler(runners)
        estimator = MultiDRCPOEstimator(
            sampler=sampler, k=2, seed=42, work_dir=temp_work_dir
        )

        # Should not crash
        estimator.fit(logs)
        result = estimator.estimate()

        assert np.isfinite(result.v_hat[0])
        assert result.se[0] > 0

    def test_extreme_weights(self, temp_work_dir: str) -> None:
        """Test numerical stability with extreme importance weights."""
        logs: List[Dict[str, Any]] = []

        for i in range(50):
            if i < 5:
                # Some samples with extreme weights
                logp_logging = np.log(0.001)  # Very low logging probability
                logp_target = np.log(0.5)
            else:
                # Normal samples
                logp_logging = np.log(0.5)
                logp_target = np.log(0.5)

            logs.append(
                {
                    "context": f"context_{i}",
                    "response": "action",
                    "logp": logp_logging,
                    "reward": np.random.uniform(0, 1),
                    "logp_target_all": {"policy_0": logp_target},
                }
            )

        runners = [MockPolicyRunner("policy_0")]
        sampler = MultiTargetSampler(runners)
        estimator = MultiDRCPOEstimator(
            sampler=sampler, k=3, clip=20.0, seed=42, work_dir=temp_work_dir
        )

        # Should handle gracefully
        estimator.fit(logs)
        result = estimator.estimate()

        assert np.isfinite(result.v_hat[0])
        assert result.se[0] > 0


class TestPerformance:
    """Test performance and scalability properties."""

    @pytest.mark.slow
    def test_large_dataset_performance(self, temp_work_dir: str) -> None:
        """Test that large datasets complete in reasonable time."""
        import time

        # Large dataset
        n_samples = 1000
        logs: List[Dict[str, Any]] = []

        for i in range(n_samples):
            logs.append(
                {
                    "context": f"context_{i}",
                    "response": f"response_{i % 10}",
                    "logp": np.log(0.1 + 0.8 * np.random.uniform()),
                    "reward": np.random.uniform(0, 1),
                    "logp_target_all": {
                        "policy_0": np.log(0.1 + 0.8 * np.random.uniform())
                    },
                }
            )

        runners = [MockPolicyRunner("policy_0")]
        sampler = MultiTargetSampler(runners)
        estimator = MultiDRCPOEstimator(
            sampler=sampler, k=5, seed=42, work_dir=temp_work_dir
        )

        start_time = time.time()
        estimator.fit(logs)
        result = estimator.estimate()
        elapsed = time.time() - start_time

        # Should complete in reasonable time (less than 30 seconds)
        assert elapsed < 30.0, f"Too slow: {elapsed:.2f}s for {n_samples} samples"

        # Should still produce valid results
        assert np.isfinite(result.v_hat[0])
        assert result.se[0] > 0

    @pytest.mark.slow
    def test_many_policies_scalability(self, temp_work_dir: str) -> None:
        """Test behavior with many target policies."""
        n_policies = 10
        n_samples = 100

        logs: List[Dict[str, Any]] = []
        for i in range(n_samples):
            target_logps = {}
            for j in range(n_policies):
                target_logps[f"policy_{j}"] = np.log(0.1 + 0.8 * np.random.uniform())

            logs.append(
                {
                    "context": f"context_{i}",
                    "response": "response",
                    "logp": np.log(0.5),
                    "reward": np.random.uniform(0, 1),
                    "logp_target_all": target_logps,
                }
            )

        runners = [MockPolicyRunner(f"policy_{i}") for i in range(n_policies)]
        sampler = MultiTargetSampler(runners)
        estimator = MultiDRCPOEstimator(
            sampler=sampler, k=3, seed=42, work_dir=temp_work_dir
        )

        # Should handle many policies
        estimator.fit(logs)
        result = estimator.estimate()

        assert result.n_policies == n_policies
        assert result.v_hat.shape == (n_policies,)
        assert np.all(np.isfinite(result.v_hat))
        assert np.all(result.se > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
