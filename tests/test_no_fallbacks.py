"""
Tests to ensure no fallback values are used in CJE.

These tests verify that the new implementation never uses
arbitrary fallback values that could corrupt results.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from cje.types import LogProbResult, LogProbStatus, SampleResult, BatchResult
from cje.loggers.base_policy import BasePolicy
from cje.loggers.multi_target_sampler import MultiTargetSampler


class TestNoFallbackValues:
    """Ensure no fallback values are ever used."""

    def test_log_prob_result_forces_explicit_handling(self) -> None:
        """Test that LogProbResult forces explicit error handling."""
        # Success case
        result = LogProbResult(status=LogProbStatus.SUCCESS, value=-10.5, attempts=1)
        assert result.is_valid
        assert result.unwrap() == -10.5

        # Failure case
        failed = LogProbResult(
            status=LogProbStatus.API_ERROR, error="Connection failed", attempts=3
        )
        assert not failed.is_valid

        # unwrap() should raise on failure
        with pytest.raises(ValueError, match="Cannot unwrap invalid"):
            failed.unwrap()

        # unwrap_or() logs warning but returns default
        with patch("logging.getLogger") as mock_logger:
            default = failed.unwrap_or(-999.0)
            assert default == -999.0
            # Verify warning was logged

    def test_no_fallback_constants_exist(self) -> None:
        """Verify dangerous fallback constants are removed."""
        # These imports should fail or not contain fallbacks
        from cje.utils import error_handling

        # These dangerous constants should not exist
        assert not hasattr(error_handling, "FALLBACK_LOG_PROB")
        assert not hasattr(error_handling, "FALLBACK_PROBABILITY")
        assert not hasattr(error_handling, "FALLBACK_SCORE")

        # The dangerous safe_call should not exist
        assert not hasattr(error_handling, "safe_call")

    def test_base_policy_returns_result_not_float(self) -> None:
        """Test that policies return LogProbResult, not raw floats."""

        class MockPolicy(BasePolicy):
            def _compute_log_prob_impl(self, context: str, response: str) -> float:
                if "fail" in context:
                    raise RuntimeError("API error")
                return -15.5

        policy = MockPolicy("test", "test-model")

        # Success case
        result = policy.compute_log_prob("hello", "world")
        assert isinstance(result, LogProbResult)
        assert result.is_valid
        assert result.value == -15.5

        # Failure case
        result = policy.compute_log_prob("fail", "world")
        assert isinstance(result, LogProbResult)
        assert not result.is_valid
        assert result.value is None  # NOT -100.0 or 0.0!
        assert result.error is not None

    def test_multi_target_sampler_no_fallbacks(self) -> None:
        """Test MultiTargetSampler doesn't use fallback values."""

        class SuccessPolicy(BasePolicy):
            def _compute_log_prob_impl(self, context: str, response: str) -> float:
                return -10.0

        class FailPolicy(BasePolicy):
            def _compute_log_prob_impl(self, context: str, response: str) -> float:
                raise RuntimeError("Always fails")

        # Set up policies
        base = SuccessPolicy("base", "base-model")
        target1 = SuccessPolicy("target1", "target1-model")
        target2 = FailPolicy("target2", "target2-model")

        sampler = MultiTargetSampler(
            policies=[base, target1, target2], base_policy_name="base"
        )

        # Process a sample
        result = sampler.process_sample("test_1", "context", "response")

        assert isinstance(result, SampleResult)
        assert result.policy_results["base"].is_valid
        assert result.policy_results["target1"].is_valid
        assert not result.policy_results["target2"].is_valid

        # Check importance weights
        assert result.importance_weights["base"] == 1.0
        assert result.importance_weights["target1"] == 1.0  # Same log prob
        assert result.importance_weights["target2"] is None  # Failed - not -100!

    def test_importance_weights_with_base_failure(self) -> None:
        """Test that base policy failure results in all None weights."""

        class FailPolicy(BasePolicy):
            def _compute_log_prob_impl(self, context: str, response: str) -> float:
                raise RuntimeError("Fails")

        base = FailPolicy("base", "base-model")
        target = FailPolicy("target", "target-model")

        sampler = MultiTargetSampler(policies=[base, target], base_policy_name="base")

        log_prob_results = {
            "base": LogProbResult(status=LogProbStatus.API_ERROR, error="Failed"),
            "target": LogProbResult(status=LogProbStatus.SUCCESS, value=-10.0),
        }

        weights = sampler.compute_importance_weights(log_prob_results)

        # All weights should be None when base fails
        assert weights["base"] is None
        assert weights["target"] is None

    def test_batch_processing_continues_despite_failures(self) -> None:
        """Test that batch processing doesn't stop on failures."""

        class SometimesFailPolicy(BasePolicy):
            def _compute_log_prob_impl(self, context: str, response: str) -> float:
                if "fail" in context:
                    raise RuntimeError("Requested failure")
                return -12.0

        policy = SometimesFailPolicy("test", "test-model")
        sampler = MultiTargetSampler(policies=[policy], base_policy_name="test")

        # Process batch with some failures
        samples = [
            ("s1", "good context", "response"),
            ("s2", "fail context", "response"),
            ("s3", "another good", "response"),
        ]

        batch_result = sampler.process_batch(samples, show_progress=False)

        assert isinstance(batch_result, BatchResult)
        assert batch_result.num_samples == 3
        assert batch_result.num_complete == 2
        assert batch_result.num_failed == 1

        # Check individual results
        assert batch_result.results[0].all_valid
        assert not batch_result.results[1].all_valid
        assert batch_result.results[2].all_valid

    def test_no_silent_corruption(self) -> None:
        """Test that we never silently corrupt importance weights."""

        class TestPolicy(BasePolicy):
            def __init__(self, name: str, log_prob: float):
                super().__init__(name, f"{name}-model")
                self.log_prob = log_prob

            def _compute_log_prob_impl(self, context: str, response: str) -> float:
                return self.log_prob

        # Set up scenario that would cause corruption with fallbacks
        base = TestPolicy("base", -12.0)
        target = TestPolicy("target", -15.0)

        sampler = MultiTargetSampler(policies=[base, target], base_policy_name="base")

        # Normal case - correct weight
        result = sampler.process_sample("test", "context", "response")
        expected_weight = np.exp(-15.0 - (-12.0))  # exp(-3) ≈ 0.0498
        assert abs(result.importance_weights["target"] - expected_weight) < 0.0001

        # Now simulate what would happen with old fallback system
        # If target returned -100.0 fallback, weight would be exp(-88) ≈ 0
        # This should NEVER happen in our new system

        # Force a failure
        target._compute_log_prob_impl = Mock(side_effect=RuntimeError("API failed"))  # type: ignore[method-assign]
        result2 = sampler.process_sample("test2", "context", "response")

        # Weight should be None, not some corrupted value
        assert result2.importance_weights["target"] is None

        # NOT exp(-100 - (-12)) which would be ~6e-39
        assert result2.importance_weights["target"] != pytest.approx(6e-39, abs=1e-40)

    def test_cache_stores_results_not_floats(self) -> None:
        """Test that cache stores LogProbResult objects."""

        class CacheTestPolicy(BasePolicy):
            def __init__(self) -> None:
                super().__init__("test", "test-model")
                self.call_count = 0

            def _compute_log_prob_impl(self, context: str, response: str) -> float:
                self.call_count += 1
                return -8.0

        from cje.loggers.api_policy import APIPolicyRunner

        # This is a mock test since we'd need a real API implementation
        # The key point is that the cache stores LogProbResult, not floats

    def test_retry_logic_without_fallbacks(self) -> None:
        """Test retry logic works without returning fallback values."""

        class RetryTestPolicy(BasePolicy):
            def __init__(self) -> None:
                super().__init__("test", "test-model", max_retries=3)
                self.attempt = 0

            def _compute_log_prob_impl(self, context: str, response: str) -> float:
                self.attempt += 1
                if self.attempt < 3:
                    raise RuntimeError("429 Rate limited")
                return -5.0

        policy = RetryTestPolicy()
        result = policy.compute_log_prob("test", "response")

        assert result.is_valid
        assert result.value == -5.0
        assert result.attempts == 3
        assert policy.attempt == 3


def test_no_fallback_in_importance_calculation() -> None:
    """Ensure importance weight calculation never uses fallback values."""

    # This would be the old dangerous calculation with fallback
    def bad_importance_weight(target_logp: float, base_logp: float) -> float:
        # If target_logp was -100.0 (fallback), this corrupts everything
        return float(np.exp(target_logp - base_logp))

    # Demonstrate the corruption
    base_logp = -12.0
    fallback_logp = -100.0  # Old fallback value

    corrupted_weight = bad_importance_weight(fallback_logp, base_logp)
    assert corrupted_weight == pytest.approx(6.05e-39)  # Completely wrong!

    # Our new system would never allow this
    # It would return None instead


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
