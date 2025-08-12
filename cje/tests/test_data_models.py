"""Test pydantic data models."""

from cje.data.models import Sample, Dataset
import numpy as np


def test_sample() -> None:
    """Test Sample model creation and validation."""

    # Valid sample
    sample = Sample(
        prompt="What is machine learning?",
        response="ML is...",
        reward=0.8,
        base_policy_logprob=-5.0,
        target_policy_logprobs={"pi_a": -4.0, "pi_b": -6.0},
    )

    assert sample.reward == 0.8
    assert sample.get_importance_weight("pi_a") == np.exp(-4.0 - (-5.0))
    print("✓ Valid sample created")

    # Test invalid base logprob
    try:
        Sample(
            prompt="test",
            response="response",
            reward=0.5,
            base_policy_logprob=1.0,  # Invalid: positive
            target_policy_logprobs={},
        )
        assert False, "Should have raised error"
    except ValueError as e:
        assert "must be <= 0" in str(e)
        print("✓ Caught invalid base_policy_logprob")

    # Test invalid reward
    try:
        Sample(
            prompt="test",
            response="response",
            reward=1.5,  # Invalid: > 1
            base_policy_logprob=-5.0,
            target_policy_logprobs={"pi_test": 2.0},  # Invalid
        )
        assert False, "Should have raised error"
    except ValueError as e:
        assert "must be <= 0" in str(e)
        print("✓ Caught invalid target logprob")


def test_dataset() -> None:
    """Test Dataset model creation and validation."""

    samples = [
        Sample(
            prompt=f"prompt{i}",
            response=f"response{i}",
            reward=0.5 + i * 0.1,
            base_policy_logprob=-10.0,
            target_policy_logprobs={"pi_a": -9.0, "pi_b": -11.0},
        )
        for i in range(5)
    ]

    dataset = Dataset(samples=samples, target_policies=["pi_a", "pi_b"])
    assert dataset.n_samples == 5
    assert len(dataset.target_policies) == 2
    print("✓ Dataset created and tested")


if __name__ == "__main__":
    test_sample()
    test_dataset()
    print("\nAll data model tests passed! ✨")
