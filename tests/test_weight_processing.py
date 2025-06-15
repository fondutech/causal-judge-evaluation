"""
Unit tests for weight processing pipeline components.

This module tests the critical stages of importance weight computation
to prevent regressions in numerical stability and correctness.
"""

import numpy as np
import pytest
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, MagicMock

from cje.loggers.multi_target_sampler import MultiTargetSampler


class TestWeightProcessingPipeline:
    """Test the multi-stage weight processing pipeline."""

    def create_mock_runner(self, log_probs: Dict[Tuple[str, str], float]) -> Mock:
        """Create a mock policy runner with predefined log probabilities."""
        runner = Mock()

        def mock_log_prob(context: str, response: str, **kwargs: Any) -> float:
            return log_probs.get((context, response), -100.0)

        runner.log_prob = Mock(side_effect=mock_log_prob)
        runner.model_name = "mock_policy"
        return runner

    def test_hard_clipping_at_boundaries(self) -> None:
        """Test that log ratios are clipped exactly at ±20."""
        # Create scenarios with extreme log probability differences
        behavior_logp = -50.0  # Very unlikely under behavior policy
        target_logp = 10.0  # Much more likely under target

        # This would create log ratio of 60, which should be clipped to 20
        log_probs = {("test", "response"): target_logp}
        runner = self.create_mock_runner(log_probs)

        sampler = MultiTargetSampler([runner], log_ratio_clip=20.0)

        # Compute importance weights
        weights, stats = sampler.importance_weights_matrix(
            ["test"],
            ["response"],
            [behavior_logp],
            stabilize=False,  # Test clipping without stabilization
            return_stats=True,
        )

        # Check that log ratio was clipped
        expected_log_ratio = target_logp - behavior_logp  # Would be 60
        clipped_log_ratio = 20.0  # Should be clipped to this
        expected_weight = np.exp(clipped_log_ratio)

        assert np.isclose(
            weights[0, 0], expected_weight, rtol=1e-6
        ), f"Weight should be exp(20) due to clipping, got {weights[0, 0]}"

    def test_float64_overflow_prevention(self) -> None:
        """Test that float64 casting prevents overflow in weight computation."""
        # Create extreme but valid scenarios
        behavior_logp = -30.0
        target_logp = 30.0  # Log ratio of 60 before clipping

        log_probs = {("test", "response"): target_logp}
        runner = self.create_mock_runner(log_probs)

        sampler = MultiTargetSampler([runner], log_ratio_clip=20.0)

        # Multiple samples to ensure matrix operations
        n_samples = 100
        contexts = ["test"] * n_samples
        responses = ["response"] * n_samples
        behavior_logps = [behavior_logp] * n_samples

        weights, stats = sampler.importance_weights_matrix(
            contexts, responses, behavior_logps, stabilize=True, return_stats=True
        )

        # Check no infinities or NaNs
        assert not np.any(np.isinf(weights)), "Weights contain infinities"
        assert not np.any(np.isnan(weights)), "Weights contain NaNs"
        assert weights.dtype == np.float64, "Weights should be float64"

    def test_soft_stabilization_preserves_diversity(self) -> None:
        """Test that soft stabilization preserves relative weight differences."""
        # Create three policies with different preferences
        log_probs_1 = {("ctx", "A"): -1.0, ("ctx", "B"): -5.0}
        log_probs_2 = {("ctx", "A"): -2.0, ("ctx", "B"): -3.0}
        log_probs_3 = {("ctx", "A"): -15.0, ("ctx", "B"): -0.5}  # More extreme

        runners = [
            self.create_mock_runner(log_probs_1),
            self.create_mock_runner(log_probs_2),
            self.create_mock_runner(log_probs_3),
        ]

        sampler = MultiTargetSampler(runners)

        # Test on samples where policies disagree
        contexts = ["ctx", "ctx"]
        responses = ["A", "B"]
        behavior_logps = [-2.5, -2.5]  # Uniform behavior policy

        weights, stats = sampler.importance_weights_matrix(
            contexts, responses, behavior_logps, stabilize=True, return_stats=True
        )

        # Check that weights are different across policies
        for i in range(len(contexts)):
            weights_for_sample = weights[i, :]
            unique_weights = np.unique(weights_for_sample)
            assert (
                len(unique_weights) > 1
            ), f"Stabilization collapsed all weights to same value for sample {i}"

        # Check relative ordering is preserved
        # For response "A": policy 1 > policy 2 > policy 3
        assert (
            weights[0, 0] > weights[0, 1] > weights[0, 2]
        ), "Relative preference ordering not preserved for response A"

        # For response "B": policy 3 > policy 2 > policy 1
        assert (
            weights[1, 2] > weights[1, 1] > weights[1, 0]
        ), "Relative preference ordering not preserved for response B"

    def test_ess_calculation_correctness(self) -> None:
        """Test that ESS is calculated correctly and reported accurately."""
        # Create a scenario with varying weights to properly test ESS
        # Policy 1: matches behavior on half samples, different on other half
        # Policy 2: always matches behavior (ESS ≈ n)

        log_probs_varying = {
            ("ctx1", "resp1"): -2.0,  # Matches behavior
            ("ctx2", "resp2"): -10.0,  # Very different from behavior
        }
        log_probs_matching = {
            ("ctx1", "resp1"): -2.0,  # Always matches
            ("ctx2", "resp2"): -2.0,  # Always matches
        }

        runners = [
            self.create_mock_runner(log_probs_varying),
            self.create_mock_runner(log_probs_matching),
        ]

        sampler = MultiTargetSampler(runners)

        # Create mixed samples
        n_samples = 100
        contexts = ["ctx1"] * 50 + ["ctx2"] * 50
        responses = ["resp1"] * 50 + ["resp2"] * 50
        behavior_logps = [-2.0] * n_samples  # Uniform behavior

        weights, stats = sampler.importance_weights_matrix(
            contexts,
            responses,
            behavior_logps,
            stabilize=False,  # No stabilization for exact ESS test
            return_stats=True,
        )

        # Check ESS values
        assert "ess_values" in stats, "ESS values should be in stats"
        assert "ess_percentage" in stats, "ESS percentage should be in stats"

        ess_values = stats["ess_values"]
        assert len(ess_values) == 2, "Should have ESS for each policy"

        # Policy 2 has perfect overlap, ESS should be ≈ n
        assert (
            ess_values[1] > 0.95 * n_samples
        ), f"ESS for matching policy should be near {n_samples}, got {ess_values[1]}"

        # Policy 1 has varying weights, ESS should be < n
        # With half weights = 1 and half weights = exp(-8) ≈ 0.0003
        # ESS will be much lower than n
        assert (
            ess_values[0] < 0.6 * n_samples
        ), f"ESS for varying policy should be less than {n_samples}, got {ess_values[0]}"

    def test_teacher_forcing_regression(self) -> None:
        """Test that identical policies produce identical weights (teacher forcing bug)."""
        # Create two identical policies
        log_probs = {
            ("context1", "response1"): -1.5,
            ("context2", "response2"): -2.0,
            ("context3", "response3"): -3.0,
        }

        runner1 = self.create_mock_runner(log_probs)
        runner2 = self.create_mock_runner(log_probs)  # Identical

        sampler = MultiTargetSampler([runner1, runner2])

        contexts = ["context1", "context2", "context3"]
        responses = ["response1", "response2", "response3"]
        behavior_logps = [-2.0, -2.5, -2.0]

        weights, stats = sampler.importance_weights_matrix(
            contexts, responses, behavior_logps, stabilize=True, return_stats=True
        )

        # Check that identical policies have identical weights
        np.testing.assert_array_almost_equal(
            weights[:, 0],
            weights[:, 1],
            decimal=10,
            err_msg="Identical policies should produce identical weights",
        )

    def test_weight_statistics_collection(self) -> None:
        """Test that weight statistics are collected correctly."""
        # Create a scenario with known clipping behavior
        log_probs_mild = {("ctx", "resp"): -2.0}
        log_probs_extreme = {("ctx", "resp"): 25.0}  # Will trigger clipping

        runners = [
            self.create_mock_runner(log_probs_mild),
            self.create_mock_runner(log_probs_extreme),
        ]

        sampler = MultiTargetSampler(runners, log_ratio_clip=20.0)

        contexts = ["ctx"] * 10
        responses = ["resp"] * 10
        behavior_logps = [-3.0] * 10

        weights, stats = sampler.importance_weights_matrix(
            contexts, responses, behavior_logps, stabilize=True, return_stats=True
        )

        # Check all expected statistics are present
        assert "weight_range" in stats, "Should report weight range"
        assert "ess_values" in stats, "Should report ESS values"
        assert "ess_percentage" in stats, "Should report ESS percentage"
        assert "n_clipped" in stats, "Should report number of clipped weights"
        assert "clip_fraction" in stats, "Should report clipping fraction"
        assert "stabilization_applied" in stats, "Should report stabilization status"

        # Verify weight range is reasonable
        min_w, max_w = stats["weight_range"]
        assert min_w > 0, "Min weight should be positive"
        assert max_w < np.exp(21), "Max weight should be bounded by clipping"

        # Check clipping was detected for extreme policy
        assert stats["n_clipped"] > 0, "Should detect clipped weights"
        assert stats["clip_fraction"] > 0, "Should report non-zero clip fraction"

    def test_stabilization_trigger_conditions(self) -> None:
        """Test when stabilization is triggered vs not triggered."""
        # Case 1: Mild differences, no stabilization needed
        log_probs_mild = {("ctx", "resp"): -2.0}
        runner_mild = self.create_mock_runner(log_probs_mild)
        sampler_mild = MultiTargetSampler([runner_mild])

        weights_mild, stats_mild = sampler_mild.importance_weights_matrix(
            ["ctx"],
            ["resp"],
            [-3.0],  # Small difference of 1.0
            stabilize=True,
            return_stats=True,
        )

        # Case 2: Large differences, stabilization needed
        log_probs_extreme = {("ctx", "resp"): 15.0}
        runner_extreme = self.create_mock_runner(log_probs_extreme)
        sampler_extreme = MultiTargetSampler([runner_extreme])

        weights_extreme, stats_extreme = sampler_extreme.importance_weights_matrix(
            ["ctx"],
            ["resp"],
            [-5.0],  # Large difference of 20.0
            stabilize=True,
            return_stats=True,
        )

        # Check stabilization behavior
        assert not stats_mild.get(
            "stabilization_applied", False
        ), "Stabilization should not be applied for small differences"

        assert stats_extreme.get(
            "stabilization_applied", False
        ), "Stabilization should be applied for large differences"


class TestEdgeCases:
    """Test edge cases in weight processing."""

    def test_empty_inputs(self) -> None:
        """Test handling of empty inputs."""
        runner = Mock()
        runner.log_prob = Mock(return_value=-1.0)
        sampler = MultiTargetSampler([runner])

        with pytest.raises(ValueError):
            sampler.importance_weights_matrix([], [], [], stabilize=True)

    def test_mismatched_lengths(self) -> None:
        """Test handling of mismatched input lengths."""
        runner = Mock()
        runner.log_prob = Mock(return_value=-1.0)
        sampler = MultiTargetSampler([runner])

        with pytest.raises(ValueError):
            sampler.importance_weights_matrix(
                ["ctx1", "ctx2"],  # 2 contexts
                ["resp1"],  # 1 response
                [-1.0, -2.0],  # 2 behavior logps
                stabilize=True,
            )

    def test_numerical_edge_cases(self) -> None:
        """Test handling of numerical edge cases."""
        log_probs = {
            ("ctx", "resp_inf"): float("-inf"),  # Zero probability
            ("ctx", "resp_normal"): -2.0,
        }

        runner = Mock()
        runner.log_prob = Mock(
            side_effect=lambda c, r, **kw: log_probs.get((c, r), -100.0)
        )
        sampler = MultiTargetSampler([runner])

        contexts = ["ctx", "ctx"]
        responses = ["resp_inf", "resp_normal"]
        behavior_logps = [-3.0, -3.0]

        weights, stats = sampler.importance_weights_matrix(
            contexts, responses, behavior_logps, stabilize=True, return_stats=True
        )

        # Check handling of -inf log probability
        # After clipping to -20, weight = exp(-20 - (-3)) = exp(-17) ≈ 4e-8
        assert (
            weights[0, 0] < 1e-6
        ), "Weight should be very small for -inf log probability"
        assert not np.isnan(weights[0, 0]), "Should not produce NaN"
        assert not np.isinf(weights[0, 0]), "Should not produce inf"
        assert weights[1, 0] > 0, "Normal weight should be positive"
