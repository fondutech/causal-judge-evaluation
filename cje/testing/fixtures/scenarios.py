"""
Scenario fixtures for CJE testing.

Provides complete test scenarios that combine data, configurations, and expected
outcomes for comprehensive testing of CJE pipeline components.
"""

import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, NamedTuple, Callable
import pytest

from .data import scenario_1_data, scenario_2_data, scenario_3_data, create_test_dataset
from .configs import basic_config, multi_policy_config, temperature_sweep_config


class TestScenario(NamedTuple):
    """Complete test scenario with data, config, and expected outcomes."""

    name: str
    description: str
    data: List[Dict[str, Any]]
    config: Dict[str, Any]
    expected_results: Dict[str, Any]
    setup_functions: List[Callable[[], None]]
    cleanup_functions: List[Callable[[], None]]


def quick_test_scenario() -> TestScenario:
    """
    Quick test scenario for fast validation.

    Minimal dataset and configuration for rapid testing during development.
    """
    data = scenario_2_data(size=3)
    config = basic_config(dataset_size=3)

    expected_results = {
        "should_complete": True,
        "min_samples": 3,
        "expected_policies": 1,
        "expected_files": ["logs.jsonl", "result.json"],
        "max_runtime_seconds": 30,
    }

    return TestScenario(
        name="quick_test",
        description="Quick test with minimal data for fast validation",
        data=data,
        config=config,
        expected_results=expected_results,
        setup_functions=[],
        cleanup_functions=[],
    )


def full_pipeline_scenario() -> TestScenario:
    """
    Full pipeline test scenario.

    Complete test with realistic data size and configuration.
    """
    data = scenario_2_data(size=15)
    config = basic_config(dataset_size=15)

    expected_results = {
        "should_complete": True,
        "min_samples": 15,
        "expected_policies": 1,
        "expected_files": ["logs.jsonl", "result.json"],
        "expected_estimator": "DRCPO",
        "max_runtime_seconds": 120,
        "min_v_hat": -1.0,  # Reasonable bounds
        "max_v_hat": 1.0,
    }

    return TestScenario(
        name="full_pipeline",
        description="Complete pipeline test with realistic data",
        data=data,
        config=config,
        expected_results=expected_results,
        setup_functions=[],
        cleanup_functions=[],
    )


def multi_policy_scenario() -> TestScenario:
    """
    Multi-policy comparison scenario.

    Tests multi-policy evaluation with temperature sweep.
    """
    data = scenario_2_data(size=12)
    config = multi_policy_config(num_policies=3, dataset_size=12)

    expected_results = {
        "should_complete": True,
        "min_samples": 12,
        "expected_policies": 3,
        "expected_files": ["logs.jsonl", "result.json"],
        "expected_estimator": "DRCPO",
        "max_runtime_seconds": 180,
        "should_have_policy_comparison": True,
        "expected_policy_names": [
            "policy_temp_0.1",
            "policy_temp_0.5",
            "policy_temp_1.0",
        ],
    }

    return TestScenario(
        name="multi_policy",
        description="Multi-policy evaluation with temperature sweep",
        data=data,
        config=config,
        expected_results=expected_results,
        setup_functions=[],
        cleanup_functions=[],
    )


def judge_comparison_scenario() -> TestScenario:
    """
    Judge comparison scenario.

    Tests different judge configurations and their impact on results.
    """
    data = scenario_2_data(size=10)

    # Create multiple configurations with different judges
    base_config = basic_config(dataset_size=10)

    # We'll test the same data with different judge configurations
    judge_configs = {
        "strict_judge": {
            **base_config,
            "judge": {
                "provider": "mock",
                "model_name": "mock-judge",
                "template": "strict_judge",
            },
        },
        "lenient_judge": {
            **base_config,
            "judge": {
                "provider": "mock",
                "model_name": "mock-judge",
                "template": "lenient_judge",
            },
        },
        "creative_judge": {
            **base_config,
            "judge": {
                "provider": "mock",
                "model_name": "mock-judge",
                "template": "creative_judge",
            },
        },
    }

    expected_results = {
        "should_complete": True,
        "min_samples": 10,
        "expected_judges": 3,
        "should_show_judge_differences": True,
        "max_runtime_seconds": 90,
        "expected_score_variance": True,  # Different judges should give different scores
    }

    return TestScenario(
        name="judge_comparison",
        description="Compare different judge configurations",
        data=data,
        config=judge_configs,  # Multiple configs
        expected_results=expected_results,
        setup_functions=[],
        cleanup_functions=[],
    )


def error_handling_scenario() -> TestScenario:
    """
    Error handling scenario.

    Tests graceful handling of various error conditions.
    """
    # Create problematic data
    problematic_data: List[Dict[str, Any]] = [
        {
            "uid": "error_1",
            "context": "",
            "response": "response to empty context",
        },  # Empty context
        {
            "uid": "error_2",
            "context": "normal context",
            "response": "",
        },  # Empty response
        {
            "uid": "error_3",
            "context": "normal context",
            "response": "normal response",
            "logp": "invalid",
        },  # Invalid logp
        {
            "uid": "error_4",
            "context": "normal context",
            "response": "normal response",
            "y_true": "invalid",
        },  # Invalid y_true
        {
            "uid": "good_1",
            "context": "good context",
            "response": "good response",
            "logp": -10.5,
            "y_true": 0.8,
        },  # Good sample
    ]

    config = basic_config(dataset_size=5)

    expected_results = {
        "should_handle_errors_gracefully": True,
        "min_valid_samples": 1,  # At least one good sample
        "max_runtime_seconds": 60,
        "should_log_warnings": True,
        "should_not_crash": True,
    }

    return TestScenario(
        name="error_handling",
        description="Test graceful handling of data quality issues",
        data=problematic_data,
        config=config,
        expected_results=expected_results,
        setup_functions=[],
        cleanup_functions=[],
    )


def scenario_1_pipeline() -> TestScenario:
    """
    Scenario 1 pipeline test (context only).

    Tests complete pipeline with context-only data where CJE generates responses.
    """
    data = scenario_1_data(size=8)
    config = basic_config(dataset_size=8)

    expected_results = {
        "should_complete": True,
        "min_samples": 8,
        "should_generate_responses": True,
        "should_use_judge_for_scoring": True,
        "expected_files": ["logs.jsonl", "result.json"],
        "max_runtime_seconds": 90,
    }

    return TestScenario(
        name="scenario_1_pipeline",
        description="Complete Scenario 1 pipeline (context only)",
        data=data,
        config=config,
        expected_results=expected_results,
        setup_functions=[],
        cleanup_functions=[],
    )


def scenario_3_pipeline() -> TestScenario:
    """
    Scenario 3 pipeline test (pre-computed target data).

    Tests pipeline with pre-computed target policy responses.
    """
    data = scenario_3_data(size=6)
    config = basic_config(dataset_size=6)

    expected_results = {
        "should_complete": True,
        "min_samples": 6,
        "should_use_precomputed_targets": True,
        "expected_files": ["logs.jsonl", "result.json"],
        "max_runtime_seconds": 60,
    }

    return TestScenario(
        name="scenario_3_pipeline",
        description="Complete Scenario 3 pipeline (pre-computed targets)",
        data=data,
        config=config,
        expected_results=expected_results,
        setup_functions=[],
        cleanup_functions=[],
    )


def performance_test_scenario() -> TestScenario:
    """
    Performance test scenario.

    Tests with larger dataset to validate performance characteristics.
    """
    data = scenario_2_data(size=50)
    config = basic_config(dataset_size=50)

    expected_results = {
        "should_complete": True,
        "min_samples": 50,
        "max_runtime_seconds": 300,  # 5 minutes
        "expected_files": ["logs.jsonl", "result.json"],
        "should_be_reasonably_fast": True,
    }

    return TestScenario(
        name="performance_test",
        description="Performance test with larger dataset",
        data=data,
        config=config,
        expected_results=expected_results,
        setup_functions=[],
        cleanup_functions=[],
    )


def temperature_effect_scenario() -> TestScenario:
    """
    Temperature effect analysis scenario.

    Tests how different temperatures affect results with detailed analysis.
    """
    data = scenario_2_data(size=20)
    temperatures = [0.1, 0.5, 1.0, 1.5]
    config = temperature_sweep_config(temperatures=temperatures)
    config["dataset"]["sample_limit"] = 20

    expected_results = {
        "should_complete": True,
        "min_samples": 20,
        "expected_policies": len(temperatures),
        "should_show_temperature_effects": True,
        "expected_policy_ordering": True,  # Conservative should differ from creative
        "max_runtime_seconds": 200,
    }

    return TestScenario(
        name="temperature_effect",
        description="Analyze temperature effects on policy performance",
        data=data,
        config=config,
        expected_results=expected_results,
        setup_functions=[],
        cleanup_functions=[],
    )


# Helper functions for scenario management
def setup_scenario_files(scenario: TestScenario, work_dir: str) -> Dict[str, str]:
    """
    Set up files needed for a test scenario.

    Args:
        scenario: Test scenario to set up
        work_dir: Working directory for files

    Returns:
        Dictionary mapping file types to file paths
    """
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    file_paths = {}

    # Create data file
    data_file = work_path / "test_data.jsonl"
    with open(data_file, "w") as f:
        for sample in scenario.data:
            f.write(json.dumps(sample) + "\n")
    file_paths["data"] = str(data_file)

    # Update config to point to data file
    if isinstance(scenario.config, dict):
        scenario.config["dataset"]["name"] = str(data_file)

    return file_paths


def cleanup_scenario_files(file_paths: Dict[str, str]) -> None:
    """
    Clean up files created for a test scenario.

    Args:
        file_paths: Dictionary of file paths to clean up
    """
    for file_path in file_paths.values():
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception:
            pass  # Ignore cleanup errors


def validate_scenario_results(
    scenario: TestScenario, actual_results: Dict[str, Any]
) -> List[str]:
    """
    Validate that scenario results match expectations.

    Args:
        scenario: Test scenario with expectations
        actual_results: Actual results from running the scenario

    Returns:
        List of validation errors (empty if all expectations met)
    """
    errors = []
    expected = scenario.expected_results

    # Check completion
    if expected.get("should_complete") and not actual_results.get("completed", False):
        errors.append("Scenario should have completed successfully")

    # Check sample count
    if "min_samples" in expected:
        actual_samples = actual_results.get("sample_count", 0)
        if actual_samples < expected["min_samples"]:
            errors.append(
                f"Expected at least {expected['min_samples']} samples, got {actual_samples}"
            )

    # Check runtime
    if "max_runtime_seconds" in expected:
        actual_runtime = actual_results.get("runtime_seconds", float("inf"))
        if actual_runtime > expected["max_runtime_seconds"]:
            errors.append(
                f"Runtime exceeded limit: {actual_runtime}s > {expected['max_runtime_seconds']}s"
            )

    # Check files
    if "expected_files" in expected:
        actual_files = actual_results.get("output_files", [])
        for expected_file in expected["expected_files"]:
            if expected_file not in actual_files:
                errors.append(f"Expected output file missing: {expected_file}")

    return errors


# Pytest fixtures for scenarios
@pytest.fixture
def quick_test() -> TestScenario:
    """Pytest fixture for quick test scenario."""
    return quick_test_scenario()


@pytest.fixture
def full_pipeline() -> TestScenario:
    """Pytest fixture for full pipeline scenario."""
    return full_pipeline_scenario()


@pytest.fixture
def multi_policy() -> TestScenario:
    """Pytest fixture for multi-policy scenario."""
    return multi_policy_scenario()


@pytest.fixture
def judge_comparison() -> TestScenario:
    """Pytest fixture for judge comparison scenario."""
    return judge_comparison_scenario()


@pytest.fixture
def error_handling() -> TestScenario:
    """Pytest fixture for error handling scenario."""
    return error_handling_scenario()


# Export all scenario functions
__all__ = [
    "TestScenario",
    "quick_test_scenario",
    "full_pipeline_scenario",
    "multi_policy_scenario",
    "judge_comparison_scenario",
    "error_handling_scenario",
    "scenario_1_pipeline",
    "scenario_3_pipeline",
    "performance_test_scenario",
    "temperature_effect_scenario",
    "setup_scenario_files",
    "cleanup_scenario_files",
    "validate_scenario_results",
]
