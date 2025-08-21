"""Shared test fixtures and utilities for CJE test suite.

This file is automatically loaded by pytest and provides common fixtures
and utilities used across multiple test files.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from cje.data.models import Sample, Dataset, EstimationResult
from cje.data.fresh_draws import FreshDrawSample, FreshDrawDataset


# ============================================================================
# Standard Test Datasets
# ============================================================================


@pytest.fixture
def basic_dataset() -> Dataset:
    """Create a basic dataset with judge scores and rewards.

    Returns:
        Dataset with 20 samples, all required fields populated.
    """
    samples = []
    for i in range(20):
        sample = Sample(
            prompt_id=f"test_{i}",
            prompt=f"Test question {i}",
            response=f"Test response {i}",
            reward=0.5 + 0.3 * np.sin(i / 3),  # Varies between 0.2 and 0.8
            base_policy_logprob=-10.0 - i * 0.1,
            target_policy_logprobs={
                "policy_a": -9.0 - i * 0.1,
                "policy_b": -11.0 - i * 0.1,
            },
            metadata={
                "judge_score": 0.5 + 0.3 * np.sin(i / 3),
            },
        )
        samples.append(sample)

    return Dataset(
        samples=samples,
        target_policies=["policy_a", "policy_b"],
    )


@pytest.fixture
def dataset_with_oracle() -> Dataset:
    """Create dataset with 50% oracle label coverage for calibration.

    Returns:
        Dataset with 20 samples, 10 having oracle labels.
    """
    samples = []
    for i in range(20):
        judge_score = 0.5 + 0.3 * np.sin(i / 3)
        oracle_label = judge_score + 0.05 * np.random.normal() if i < 10 else None

        sample = Sample(
            prompt_id=f"test_{i}",
            prompt=f"Question {i}",
            response=f"Answer {i}",
            reward=None,  # Will be set by calibration
            base_policy_logprob=-10.0 - i * 0.1,
            target_policy_logprobs={
                "improved": -9.0 - i * 0.1,
                "worse": -11.0 - i * 0.1,
            },
            metadata={
                "judge_score": float(np.clip(judge_score, 0, 1)),
                "oracle_label": (
                    float(np.clip(oracle_label, 0, 1))
                    if oracle_label is not None
                    else None
                ),
            },
        )
        samples.append(sample)

    return Dataset(
        samples=samples,
        target_policies=["improved", "worse"],
    )


@pytest.fixture
def dataset_for_dr() -> Dataset:
    """Create dataset suitable for DR estimation with cross-fit folds.

    Returns:
        Dataset with 25 samples, including cv_fold assignments.
    """
    samples = []
    for i in range(25):
        sample = Sample(
            prompt_id=f"dr_test_{i}",
            prompt=f"DR question {i}",
            response=f"DR answer {i}",
            reward=0.6 + 0.2 * np.random.normal(),
            base_policy_logprob=-12.0 + np.random.normal(),
            target_policy_logprobs={
                "target": -10.0 + np.random.normal(),
            },
            metadata={
                "judge_score": 0.6 + 0.2 * np.random.normal(),
                "cv_fold": i % 5,  # 5-fold cross-validation
            },
        )
        samples.append(sample)

    return Dataset(
        samples=samples,
        target_policies=["target"],
        metadata={"cross_fitted": True, "n_folds": 5},
    )


@pytest.fixture
def synthetic_fresh_draws() -> Dict[str, FreshDrawDataset]:
    """Create synthetic fresh draws for DR estimation.

    Returns:
        Dict mapping policy names to FreshDrawDataset objects.
    """
    fresh_draws = {}

    for policy in ["improved", "worse"]:
        samples = []
        for prompt_id in range(20):
            for draw_idx in range(3):  # 3 draws per prompt
                # Make "improved" policy have higher scores
                base_score = 0.7 if policy == "improved" else 0.3
                score = base_score + 0.1 * np.random.normal()

                sample = FreshDrawSample(
                    prompt_id=f"test_{prompt_id}",
                    target_policy=policy,
                    judge_score=float(np.clip(score, 0, 1)),
                    draw_idx=draw_idx,
                    fold_id=prompt_id % 5,
                )
                samples.append(sample)

        fresh_draws[policy] = FreshDrawDataset(
            samples=samples,
            target_policy=policy,
            draws_per_prompt=3,
        )

    return fresh_draws


# ============================================================================
# Test Data Factories
# ============================================================================


def create_test_samples(
    n_samples: int = 10,
    with_oracle: bool = False,
    oracle_coverage: float = 0.5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Create test sample data as raw dictionaries.

    Args:
        n_samples: Number of samples to create
        with_oracle: Whether to include oracle labels
        oracle_coverage: Fraction of samples with oracle labels
        seed: Random seed for reproducibility

    Returns:
        List of sample dictionaries suitable for PrecomputedSampler.
    """
    np.random.seed(seed)
    samples = []

    n_oracle = int(n_samples * oracle_coverage) if with_oracle else 0

    for i in range(n_samples):
        judge_score = 0.5 + 0.3 * np.sin(i / 3)

        metadata: Dict[str, Any] = {
            "judge_score": float(judge_score),
        }

        if with_oracle and i < n_oracle:
            oracle = judge_score + 0.05 * np.random.normal()
            metadata["oracle_label"] = float(np.clip(oracle, 0, 1))

        sample = {
            "prompt": f"Question {i}",
            "response": f"Answer {i}",
            "reward": float(judge_score),  # Use judge score as reward by default
            "base_policy_logprob": -10.0 - i * 0.1,
            "target_policy_logprobs": {
                "policy_a": -9.0 - i * 0.1,
                "policy_b": -11.0 - i * 0.1,
            },
            "metadata": metadata,
        }

        samples.append(sample)

    return samples


# ============================================================================
# Assertion Helpers
# ============================================================================


def assert_valid_estimation_result(
    result: EstimationResult,
    n_policies: int,
    check_diagnostics: bool = False,
) -> None:
    """Standard assertions for estimation results.

    Args:
        result: EstimationResult to validate
        n_policies: Expected number of policies
        check_diagnostics: Whether to check for diagnostics
    """
    # Check basic structure
    assert result is not None
    assert len(result.estimates) == n_policies
    assert len(result.standard_errors) == n_policies

    # Check values are reasonable
    assert not np.any(np.isnan(result.estimates)), "Estimates contain NaN"
    assert not np.any(np.isnan(result.standard_errors)), "Standard errors contain NaN"
    assert np.all(result.estimates >= 0), "Estimates should be non-negative"
    assert np.all(
        result.estimates <= 1
    ), "Estimates should be <= 1 for rewards in [0,1]"
    assert np.all(result.standard_errors >= 0), "Standard errors should be non-negative"

    # Check method is specified
    assert result.method is not None

    # Check diagnostics if requested
    if check_diagnostics:
        assert result.diagnostics is not None
        assert result.diagnostics.summary() is not None


def assert_weights_calibrated(
    weights: np.ndarray,
    target_mean: float = 1.0,
    tolerance: float = 0.01,
) -> None:
    """Assert that importance weights are properly calibrated.

    Args:
        weights: Array of importance weights
        target_mean: Expected mean (usually 1.0 for Hajek weights)
        tolerance: Tolerance for mean comparison
    """
    assert weights is not None
    assert len(weights) > 0
    assert not np.any(np.isnan(weights)), "Weights contain NaN"
    assert np.all(weights >= 0), "Weights should be non-negative"

    # Check mean is close to target
    mean_weight = np.mean(weights)
    assert (
        abs(mean_weight - target_mean) < tolerance
    ), f"Weight mean {mean_weight:.3f} not close to target {target_mean}"

    # Check not all weights are identical (unless n=1)
    if len(weights) > 1:
        assert not np.allclose(
            weights, weights[0]
        ), "All weights are identical, suggesting no calibration"


def assert_dataset_valid(dataset: Dataset) -> None:
    """Assert that a dataset is valid for CJE analysis.

    Args:
        dataset: Dataset to validate
    """
    assert dataset is not None
    assert len(dataset.samples) > 0, "Dataset has no samples"
    assert dataset.target_policies is not None
    assert len(dataset.target_policies) > 0, "Dataset has no target policies"

    # Check all samples have required fields
    for sample in dataset.samples:
        assert sample.prompt is not None
        assert sample.response is not None
        assert sample.base_policy_logprob is not None
        assert sample.target_policy_logprobs is not None

        # Check target policies match
        for policy in dataset.target_policies:
            assert (
                policy in sample.target_policy_logprobs
            ), f"Policy {policy} missing from sample target_policy_logprobs"


# ============================================================================
# Test Markers and Configuration
# ============================================================================


def pytest_configure(config: Any) -> None:
    """Register custom markers for test categorization."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require API credentials"
    )
