"""Test pydantic data models."""

from cje_simplified.data.models import Sample, Dataset, WeightCalibrationConfig
import numpy as np


def test_sample_validation():
    """Test Sample model validation."""

    # Valid sample
    sample = Sample(
        prompt="What is 2+2?",
        response="4",
        reward=0.9,
        base_logprob=-5.0,
        target_logprobs={"pi_good": -4.0, "pi_bad": -8.0},
    )

    assert sample.reward == 0.9
    assert sample.get_importance_weight("pi_good") == np.exp(-4.0 - (-5.0))
    print("✓ Valid sample created")

    # Test invalid log prob
    try:
        Sample(
            prompt="test",
            response="test",
            reward=0.5,
            base_logprob=1.0,  # Invalid: positive
            target_logprobs={},
        )
        assert False, "Should have raised error"
    except ValueError as e:
        assert "must be <= 0" in str(e)
        print("✓ Caught invalid base_logprob")

    # Test invalid reward
    try:
        Sample(
            prompt="test",
            response="test",
            reward=1.5,  # Invalid: > 1
            base_logprob=-5.0,
            target_logprobs={},
        )
        assert False, "Should have raised error"
    except ValueError:
        print("✓ Caught invalid reward")


def test_dataset():
    """Test Dataset model."""

    samples = [
        Sample(
            prompt=f"Q{i}",
            response=f"A{i}",
            reward=0.7 + 0.01 * i,
            base_logprob=-10.0,
            target_logprobs={"pi_a": -9.0, "pi_b": -11.0},
        )
        for i in range(5)
    ]

    dataset = Dataset(samples=samples, target_policies=["pi_a", "pi_b"])

    assert dataset.n_samples == 5

    # Test filtering
    valid_a = dataset.filter_valid_samples("pi_a")
    assert len(valid_a) == 5

    # Test summary
    summary = dataset.summary()
    assert summary["n_samples"] == 5
    assert "pi_a" in summary["valid_samples_per_policy"]
    print("✓ Dataset created and tested")


def test_config():
    """Test configuration model."""

    config = WeightCalibrationConfig(k_folds=10, clip_weight=50.0)
    assert config.k_folds == 10
    assert config.target_mean == 1.0  # Default

    # Test validation
    try:
        WeightCalibrationConfig(k_folds=1)  # Invalid: < 2
        assert False
    except ValueError:
        print("✓ Config validation works")


if __name__ == "__main__":
    print("Testing data models...\n")
    test_sample_validation()
    test_dataset()
    test_config()
    print("\nAll data model tests passed! ✨")
