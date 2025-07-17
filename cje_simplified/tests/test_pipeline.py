"""Test the complete pipeline with simulated data."""

from cje_simplified import (
    PrecomputedSampler,
    CalibratedIPS,
    DatasetFactory,
)
import json
import os
from pathlib import Path


def test_pipeline_with_edge_cases(tmp_path: Path) -> None:
    """Test pipeline with various edge cases."""
    # Create test data with edge cases - need at least 10 oracle samples
    test_data = []

    # Add 12 samples with oracle labels to meet minimum requirement
    for i in range(12):
        test_data.append(
            {
                "prompt": f"Question {i}",
                "response": f"Answer {i}",
                "judge_score": 5.0 + (i % 5),  # Scores from 5-9
                "oracle_label": 0.3 + 0.05 * i,  # Oracle labels from 0.3-0.85
                "base_policy_logprob": -10.0 - i,
                "target_policy_logprobs": {"policy_a": -8.0 - i, "policy_b": -12.0 - i},
            }
        )

    # Add a few more without oracle labels
    for i in range(12, 15):
        test_data.append(
            {
                "prompt": f"Question {i}",
                "response": f"Answer {i}",
                "judge_score": 6.0 + (i % 3),
                "base_policy_logprob": -10.0 - i,
                "target_policy_logprobs": {"policy_a": -8.0 - i, "policy_b": -12.0 - i},
            }
        )

    # Write test data
    temp_file = tmp_path / "test_data.jsonl"
    with open(temp_file, "w") as f:
        for record in test_data:
            f.write(json.dumps(record) + "\n")

    # Load and calibrate data using factory
    factory = DatasetFactory()
    dataset, stats = factory.create_from_jsonl_with_calibration(temp_file)

    # Test that calibration worked
    assert stats["n_oracle"] == 12
    assert stats["n_total"] == 15
    assert all(0 <= sample.reward <= 1 for sample in dataset.samples)

    # Run estimation
    sampler = PrecomputedSampler(dataset)
    estimator = CalibratedIPS(sampler)
    results = estimator.fit_and_estimate()

    # Basic checks
    assert len(results.estimates) == 2  # Two target policies
    assert len(results.standard_errors) == 2
    assert results.method == "calibrated_ips"

    print("Full pipeline test passed!")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_pipeline_with_edge_cases(Path(tmp_dir))
