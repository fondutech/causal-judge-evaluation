"""Test the fix for target policy stage teacher forcing bug."""

import pytest
import numpy as np
from typing import List, Dict, Any

from cje.pipeline.stages.target_policy import TargetPolicyStage
from cje.loggers.multi_target_sampler import MultiTargetSampler


class MockPolicyRunner:
    """Mock policy runner for testing."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def log_prob(self, context: str, response: str) -> float:
        """Return a deterministic log prob based on context and response."""
        # Simple deterministic function for testing
        return -len(context) * 0.1 - len(response) * 0.05


class TestTargetPolicyFix:
    """Test that the target policy stage correctly uses teacher forcing."""

    def test_target_policy_uses_responses(self, tmp_path):
        """Test that target policy stage extracts and uses responses."""
        # Create mock data with contexts and responses
        rows = [
            {
                "context": "What is 2+2?",
                "response": "The answer is 4.",
                "logp": -5.0,
            },
            {
                "context": "What is the capital of France?",
                "response": "The capital of France is Paris.",
                "logp": -8.0,
            },
        ]

        # Create mock sampler with two policies
        runners = [
            MockPolicyRunner("policy_0"),
            MockPolicyRunner("policy_1"),
        ]
        sampler = MultiTargetSampler(runners)

        # Create target policy stage
        stage = TargetPolicyStage(work_dir=tmp_path)

        # Run the fixed _compute_logprobs method
        result_rows = stage._compute_logprobs(rows, sampler, num_policies=2)

        # Verify results
        assert len(result_rows) == 2

        for i, row in enumerate(result_rows):
            # Check that log probabilities were computed
            assert "logp_target_all" in row
            assert len(row["logp_target_all"]) == 2  # Two policies

            # Verify the log probs are reasonable
            for logp in row["logp_target_all"]:
                assert isinstance(logp, float)
                assert logp < 0  # Log probs should be negative

            # Verify original data is preserved
            assert row["context"] == rows[i]["context"]
            assert row["response"] == rows[i]["response"]
            assert row["logp"] == rows[i]["logp"]

    def test_target_policy_validates_responses(self, tmp_path):
        """Test that target policy stage validates response field exists."""
        # Create data WITHOUT responses
        rows = [
            {
                "context": "What is 2+2?",
                "logp": -5.0,
            },
        ]

        # Create mock sampler
        runners = [MockPolicyRunner("policy_0")]
        sampler = MultiTargetSampler(runners)

        # Create target policy stage
        stage = TargetPolicyStage(work_dir=tmp_path)

        # Should raise error about missing response field
        with pytest.raises(ValueError, match="missing 'response' field"):
            stage._compute_logprobs(rows, sampler, num_policies=1)

    def test_logp_matrix_shape(self):
        """Test that MultiTargetSampler.logp_matrix returns correct shape."""
        # Create sampler with 3 policies
        runners = [MockPolicyRunner(f"policy_{i}") for i in range(3)]
        sampler = MultiTargetSampler(runners)

        # Test data
        contexts = ["Context 1", "Context 2", "Context 3", "Context 4"]
        responses = ["Response 1", "Response 2", "Response 3", "Response 4"]

        # Get log prob matrix
        logp_matrix = sampler.logp_matrix(contexts, responses)

        # Verify shape
        assert isinstance(logp_matrix, np.ndarray)
        assert logp_matrix.shape == (4, 3)  # 4 samples, 3 policies

        # Verify all values are negative (log probs)
        assert np.all(logp_matrix < 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
