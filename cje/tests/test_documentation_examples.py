"""Test that documentation examples actually work.

This helps ensure our documentation stays accurate as the code evolves.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import numpy as np

from cje import (
    analyze_dataset,
    load_dataset_from_jsonl,
    calibrate_dataset,
    PrecomputedSampler,
    CalibratedIPS,
    Sample,
    Dataset,
    export_results_json,
    export_results_csv,
)


def create_test_data_file(n_samples: int = 50) -> str:
    """Create a temporary test data file matching our documentation format."""
    samples = []

    # Create samples with enough oracle labels for calibration
    for i in range(n_samples):
        sample: Dict[str, Any] = {
            "prompt": f"What is {i}?",
            "response": f"The answer is {i}.",
            "base_policy_logprob": -10.0 - i * 0.1,
            "target_policy_logprobs": {
                "gpt4": -9.0 - i * 0.1,
                "claude": -9.5 - i * 0.1,
            },
            "metadata": {
                "judge_score": 0.5 + (i / n_samples) * 0.4,
            },
        }

        # Add oracle labels to first 20 samples (enough for calibration)
        if i < 20:
            sample["metadata"]["oracle_label"] = 0.6 + (i / 20) * 0.3

        samples.append(sample)

    # Write to temp file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for sample in samples:
        temp_file.write(json.dumps(sample) + "\n")
    temp_file.close()

    return temp_file.name


def test_readme_quick_start_high_level_api() -> None:
    """Test the high-level API example from README."""
    # Create test data
    data_file = create_test_data_file()

    try:
        # This is the exact code from README.md Quick Start section
        from cje import analyze_dataset

        results = analyze_dataset(data_file, estimator="calibrated-ips")

        # Verify it works
        assert results is not None
        # best_policy() returns an index, not a name
        best_idx = results.best_policy()
        assert isinstance(best_idx, int)
        assert 0 <= best_idx < 2  # Valid index for 2 policies
        assert len(results.estimates) == 2  # Two target policies
        assert len(results.standard_errors) == 2

    finally:
        Path(data_file).unlink()


def test_readme_quick_start_lower_level_api() -> None:
    """Test the lower-level API example from README."""
    # Create test data
    data_file = create_test_data_file()

    try:
        # This is the exact code from README.md (lower-level API)
        from cje import (
            load_dataset_from_jsonl,
            calibrate_dataset,
            PrecomputedSampler,
            CalibratedIPS,
        )

        dataset = load_dataset_from_jsonl(data_file)
        calibrated_dataset, cal_result = calibrate_dataset(
            dataset,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,  # For DR
        )
        sampler = PrecomputedSampler(calibrated_dataset)
        estimator = CalibratedIPS(
            sampler,
            calibrator=cal_result.calibrator,  # For DR-aware stacking
        )
        results = estimator.fit_and_estimate()

        # Verify it works
        assert results is not None
        assert len(results.estimates) == 2

    finally:
        Path(data_file).unlink()


def test_index_rst_quick_start() -> None:
    """Test the quick start example from docs/index.rst."""
    # Create test data
    data_file = create_test_data_file()

    try:
        # This is from docs/index.rst Quick Start section
        from cje import analyze_dataset

        # One-line analysis with automatic calibration
        results = analyze_dataset(data_file, estimator="calibrated-ips")

        # Get unbiased policy estimates
        best = results.best_policy()
        estimates = results.estimates

        assert isinstance(best, int)  # best_policy() returns index
        assert 0 <= best < 2  # Valid index
        assert len(estimates) == 2

    finally:
        Path(data_file).unlink()


def test_getting_started_high_level_example() -> None:
    """Test the high-level API example from getting_started.rst."""
    data_file = create_test_data_file()

    try:
        # From docs/getting_started.rst Example: High-Level API
        from cje import analyze_dataset

        # One-line analysis
        results = analyze_dataset(data_file, estimator="calibrated-ips")

        # Check results
        best_idx = results.best_policy()
        assert isinstance(best_idx, int)  # best_policy() returns index
        assert 0 <= best_idx < 2  # Valid index
        assert results.estimates is not None
        assert results.standard_errors is not None

        # Export results
        from cje import export_results_json

        temp_results = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        export_results_json(results, temp_results.name)

        # Verify export worked
        assert Path(temp_results.name).exists()
        Path(temp_results.name).unlink()

    finally:
        Path(data_file).unlink()


def test_data_format_example() -> None:
    """Test that the data format example from docs works."""
    # This is the exact JSON from data_format.rst
    sample_data = {
        "prompt": "What is machine learning?",
        "response": "Machine learning is...",
        "base_policy_logprob": -35.704,
        "target_policy_logprobs": {"gpt4": -32.456, "claude": -33.789},
        "metadata": {"judge_score": 0.85, "oracle_label": 0.90},
    }

    # Write to file
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    temp_file.write(json.dumps(sample_data) + "\n")
    temp_file.close()

    try:
        # Load it
        dataset = load_dataset_from_jsonl(temp_file.name)

        # Verify it loaded correctly
        assert len(dataset.samples) == 1
        assert dataset.samples[0].prompt == "What is machine learning?"
        assert dataset.samples[0].base_policy_logprob == -35.704

    finally:
        Path(temp_file.name).unlink()


def test_export_formats_example() -> None:
    """Test the export formats example from README."""
    data_file = create_test_data_file()

    try:
        # From README Export Formats section
        from cje import analyze_dataset, export_results_json, export_results_csv

        # Analyze
        results = analyze_dataset(data_file)

        # Export to JSON (includes full metadata and diagnostics)
        json_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        export_results_json(results, json_file.name)

        # Export to CSV (tabular format for analysis)
        csv_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        export_results_csv(results, csv_file.name)

        # Verify files were created
        assert Path(json_file.name).exists()
        assert Path(csv_file.name).exists()

        # Clean up
        Path(json_file.name).unlink()
        Path(csv_file.name).unlink()

    finally:
        Path(data_file).unlink()


def test_prompt_id_optional() -> None:
    """Test that prompt_id is truly optional as documented."""
    # Create data WITHOUT prompt_id
    sample_data = {
        "prompt": "What is 2+2?",
        "response": "4",
        "base_policy_logprob": -10.0,
        "target_policy_logprobs": {"model_v2": -9.0},
        "metadata": {"judge_score": 0.9},
    }

    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    temp_file.write(json.dumps(sample_data) + "\n")
    temp_file.close()

    try:
        # Should load without error
        dataset = load_dataset_from_jsonl(temp_file.name)

        # Should have auto-generated prompt_id
        assert dataset.samples[0].prompt_id is not None
        assert dataset.samples[0].prompt_id.startswith("prompt_")

    finally:
        Path(temp_file.name).unlink()


if __name__ == "__main__":
    # Run tests
    test_readme_quick_start_high_level_api()
    test_readme_quick_start_lower_level_api()
    test_index_rst_quick_start()
    test_getting_started_high_level_example()
    test_data_format_example()
    test_export_formats_example()
    test_prompt_id_optional()

    print("✅ All documentation examples work!")
