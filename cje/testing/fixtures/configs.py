"""
Configuration fixtures for CJE testing.

Provides common test configurations for different testing scenarios,
including basic configs, multi-policy setups, and specialized configurations.
"""

import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import pytest


def basic_config(
    work_dir: Optional[str] = None, dataset_size: int = 5, use_mock_judge: bool = True
) -> Dict[str, Any]:
    """
    Create a basic CJE configuration for simple testing.

    Args:
        work_dir: Working directory (creates temp if None)
        dataset_size: Size of test dataset to create
        use_mock_judge: Whether to use mock judge for testing

    Returns:
        Basic CJE configuration dictionary
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp()

    config = {
        "paths": {"work_dir": work_dir},
        "dataset": {
            "name": "mock_dataset",
            "split": "test",
            "sample_limit": dataset_size,
        },
        "logging_policy": {
            "model_name": "sshleifer/tiny-gpt2",
            "provider": "hf",
            "temperature": 0.7,
            "max_new_tokens": 50,
        },
        "target_policies": [
            {
                "name": "target",
                "model_name": "sshleifer/tiny-gpt2",
                "provider": "hf",
                "temperature": 0.1,
                "mc_samples": 3,
            }
        ],
        "judge": {
            "provider": "mock",
            "model_name": "mock-judge",
            "template": "quick_judge",
        },
        "estimator": {"name": "DRCPO", "k": 2},
    }

    return config


def multi_policy_config(
    work_dir: Optional[str] = None, num_policies: int = 3, dataset_size: int = 10
) -> Dict[str, Any]:
    """
    Create a multi-policy configuration for testing policy comparison.

    Args:
        work_dir: Working directory (creates temp if None)
        num_policies: Number of target policies to create
        dataset_size: Size of test dataset

    Returns:
        Multi-policy CJE configuration
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp()

    # Create policies with different temperatures
    target_policies = []
    temperatures = [0.1, 0.5, 1.0, 1.5, 2.0][:num_policies]

    for i, temp in enumerate(temperatures):
        policy = {
            "name": f"policy_temp_{temp}",
            "model_name": "sshleifer/tiny-gpt2",
            "provider": "hf",
            "temperature": temp,
            "mc_samples": 3,
        }
        target_policies.append(policy)

    config = {
        "paths": {"work_dir": work_dir},
        "dataset": {
            "name": "mock_dataset",
            "split": "test",
            "sample_limit": dataset_size,
        },
        "logging_policy": {
            "model_name": "sshleifer/tiny-gpt2",
            "provider": "hf",
            "temperature": 0.7,
            "max_new_tokens": 50,
        },
        "target_policies": target_policies,
        "judge": {
            "provider": "mock",
            "model_name": "mock-judge",
            "template": "quick_judge",
        },
        "estimator": {"name": "DRCPO", "k": 3},
    }

    return config


def temperature_sweep_config(
    base_model: str = "sshleifer/tiny-gpt2",
    temperatures: List[float] = [0.1, 0.7, 1.2],
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create configuration for temperature sweep testing.

    Args:
        base_model: Base model to use for all policies
        temperatures: List of temperatures to test
        work_dir: Working directory

    Returns:
        Temperature sweep configuration
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp()

    target_policies = []
    for temp in temperatures:
        policy = {
            "name": f"temp_{temp}",
            "model_name": base_model,
            "provider": "hf",
            "temperature": temp,
            "mc_samples": 3,
        }
        target_policies.append(policy)

    config = {
        "paths": {"work_dir": work_dir},
        "dataset": {"name": "mock_dataset", "split": "test", "sample_limit": 8},
        "logging_policy": {
            "model_name": base_model,
            "provider": "hf",
            "temperature": 0.5,  # Middle ground for logging
            "max_new_tokens": 50,
        },
        "target_policies": target_policies,
        "judge": {
            "provider": "mock",
            "model_name": "mock-judge",
            "template": "quick_judge",
        },
        "estimator": {"name": "IPS", "k": 2},
    }

    return config


def api_only_config(work_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Create configuration using only API-based models.

    Args:
        work_dir: Working directory

    Returns:
        API-only configuration (uses mocks for testing)
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp()

    config = {
        "paths": {"work_dir": work_dir},
        "dataset": {"name": "mock_dataset", "split": "test", "sample_limit": 6},
        "logging_policy": {
            "model_name": "gpt-4o-mini",
            "provider": "openai",
            "temperature": 0.7,
            "max_new_tokens": 100,
        },
        "target_policies": [
            {
                "name": "conservative",
                "model_name": "gpt-4o-mini",
                "provider": "openai",
                "temperature": 0.1,
                "mc_samples": 3,
            },
            {
                "name": "creative",
                "model_name": "claude-3-haiku-20240307",
                "provider": "anthropic",
                "temperature": 1.0,
                "mc_samples": 3,
            },
        ],
        "judge": {
            "provider": "mock",
            "model_name": "mock-judge",
            "template": "quick_judge",
        },
        "estimator": {"name": "SNIPS", "k": 2},
    }

    return config


def local_only_config(work_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Create configuration using only local models.

    Args:
        work_dir: Working directory

    Returns:
        Local-only configuration
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp()

    config = {
        "paths": {"work_dir": work_dir},
        "dataset": {"name": "mock_dataset", "split": "test", "sample_limit": 5},
        "logging_policy": {
            "model_name": "sshleifer/tiny-gpt2",
            "provider": "hf",
            "temperature": 0.7,
            "max_new_tokens": 50,
            "device": "cpu",
        },
        "target_policies": [
            {
                "name": "conservative_local",
                "model_name": "sshleifer/tiny-gpt2",
                "provider": "hf",
                "temperature": 0.1,
                "mc_samples": 2,
                "device": "cpu",
            },
            {
                "name": "creative_local",
                "model_name": "sshleifer/tiny-gpt2",
                "provider": "hf",
                "temperature": 1.2,
                "mc_samples": 2,
                "device": "cpu",
            },
        ],
        "judge": {
            "provider": "mock",
            "model_name": "mock-judge",
            "template": "quick_judge",
        },
        "estimator": {"name": "IPS", "k": 2},
    }

    return config


def minimal_config(work_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Create minimal configuration for fast testing.

    Args:
        work_dir: Working directory

    Returns:
        Minimal configuration with smallest possible settings
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp()

    config = {
        "paths": {"work_dir": work_dir},
        "dataset": {
            "name": "mock_dataset",
            "split": "test",
            "sample_limit": 2,  # Minimal dataset
        },
        "logging_policy": {
            "model_name": "mock-tiny",
            "provider": "hf",
            "temperature": 0.5,
            "max_new_tokens": 10,  # Very short responses
        },
        "target_policies": [
            {
                "name": "minimal_target",
                "model_name": "mock-tiny",
                "provider": "hf",
                "temperature": 0.2,
                "mc_samples": 1,  # Single sample
            }
        ],
        "judge": {
            "provider": "mock",
            "model_name": "mock-judge",
            "template": "quick_judge",
        },
        "estimator": {"name": "IPS", "k": 1},  # Minimal k
    }

    return config


def error_testing_config(work_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Create configuration designed to test error handling.

    Args:
        work_dir: Working directory

    Returns:
        Configuration with potential error conditions
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp()

    config = {
        "paths": {"work_dir": work_dir},
        "dataset": {
            "name": "mock_error_dataset",  # Non-existent dataset
            "split": "test",
            "sample_limit": 3,
        },
        "logging_policy": {
            "model_name": "nonexistent-model",  # Non-existent model
            "provider": "hf",
            "temperature": 0.5,
            "max_new_tokens": 20,
        },
        "target_policies": [
            {
                "name": "error_target",
                "model_name": "another-nonexistent-model",
                "provider": "hf",
                "temperature": 0.1,
                "mc_samples": 2,
            }
        ],
        "judge": {
            "provider": "mock",
            "model_name": "mock-judge",
            "template": "quick_judge",
        },
        "estimator": {"name": "InvalidEstimator", "k": 2},  # Invalid estimator
    }

    return config


# Pytest fixtures for configurations
@pytest.fixture
def basic_test_config() -> Dict[str, Any]:
    """Pytest fixture for basic configuration."""
    return basic_config()


@pytest.fixture
def multi_policy_test_config() -> Dict[str, Any]:
    """Pytest fixture for multi-policy configuration."""
    return multi_policy_config()


@pytest.fixture
def temperature_sweep_test_config() -> Dict[str, Any]:
    """Pytest fixture for temperature sweep configuration."""
    return temperature_sweep_config()


@pytest.fixture
def api_test_config() -> Dict[str, Any]:
    """Pytest fixture for API-only configuration."""
    return api_only_config()


@pytest.fixture
def local_test_config() -> Dict[str, Any]:
    """Pytest fixture for local-only configuration."""
    return local_only_config()


@pytest.fixture
def minimal_test_config() -> Dict[str, Any]:
    """Pytest fixture for minimal configuration."""
    return minimal_config()


# Configuration validation helpers
def validate_config_structure(config: Dict[str, Any]) -> List[str]:
    """
    Validate that a configuration has required structure.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required top-level keys
    required_keys = [
        "paths",
        "dataset",
        "logging_policy",
        "target_policies",
        "judge",
        "estimator",
    ]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key: {key}")

    # Check paths structure
    if "paths" in config:
        if "work_dir" not in config["paths"]:
            errors.append("Missing paths.work_dir")

    # Check dataset structure
    if "dataset" in config:
        dataset = config["dataset"]
        if "name" not in dataset:
            errors.append("Missing dataset.name")

    # Check logging policy structure
    if "logging_policy" in config:
        policy = config["logging_policy"]
        required_policy_keys = ["model_name", "provider"]
        for key in required_policy_keys:
            if key not in policy:
                errors.append(f"Missing logging_policy.{key}")

    # Check target policies structure
    if "target_policies" in config:
        if not isinstance(config["target_policies"], list):
            errors.append("target_policies must be a list")
        elif len(config["target_policies"]) == 0:
            errors.append("target_policies cannot be empty")
        else:
            for i, policy in enumerate(config["target_policies"]):
                if "model_name" not in policy:
                    errors.append(f"target_policies[{i}] missing model_name")
                if "provider" not in policy:
                    errors.append(f"target_policies[{i}] missing provider")

    # Check judge structure
    if "judge" in config:
        judge = config["judge"]
        if "provider" not in judge:
            errors.append("Missing judge.provider")
        if "model_name" not in judge:
            errors.append("Missing judge.model_name")

    # Check estimator structure
    if "estimator" in config:
        estimator = config["estimator"]
        if "name" not in estimator:
            errors.append("Missing estimator.name")

    return errors


# Export all configuration functions
__all__ = [
    "basic_config",
    "multi_policy_config",
    "temperature_sweep_config",
    "api_only_config",
    "local_only_config",
    "minimal_config",
    "error_testing_config",
    "validate_config_structure",
]
