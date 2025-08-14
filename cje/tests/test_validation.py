"""Tests for data validation utilities."""

from typing import Any, Dict, List
import pytest
from cje.data.validation import validate_cje_data, validate_for_precomputed_sampler


def test_validate_empty_data() -> None:
    """Test validation with empty data."""
    is_valid, issues = validate_cje_data([])
    assert not is_valid
    assert "Data is empty" in issues[0]


def test_validate_missing_core_fields() -> None:
    """Test validation with missing core fields."""
    data = [{"prompt": "test", "response": "answer"}]
    is_valid, issues = validate_cje_data(data)
    assert not is_valid
    assert any("prompt_id" in issue for issue in issues)
    assert any("base_policy_logprob" in issue for issue in issues)
    assert any("target_policy_logprobs" in issue for issue in issues)


def test_validate_complete_data_with_reward() -> None:
    """Test validation with complete data including rewards."""
    data = [
        {
            "prompt_id": "test_001",
            "prompt": "What is 2+2?",
            "response": "4",
            "base_policy_logprob": -10.5,
            "target_policy_logprobs": {"gpt4": -8.3},
            "reward": 0.85,
        }
    ]
    is_valid, issues = validate_cje_data(data, reward_field="reward")
    assert is_valid
    assert len(issues) == 0


def test_validate_data_with_judge_scores_only() -> None:
    """Test validation with judge scores but no oracle labels - should fail."""
    data = [
        {
            "prompt_id": "test_001",
            "prompt": "What is 2+2?",
            "response": "4",
            "base_policy_logprob": -10.5,
            "target_policy_logprobs": {"gpt4": -8.3},
            "metadata": {"judge_score": 0.85},
        }
    ]
    is_valid, issues = validate_cje_data(data, judge_field="judge_score")
    assert not is_valid  # Judge scores alone are NOT valid without oracle labels
    assert any("oracle labels for calibration" in issue for issue in issues)

    # Also test with oracle_field specified but no data
    is_valid, issues = validate_cje_data(
        data, judge_field="judge_score", oracle_field="oracle_label"
    )
    assert not is_valid
    assert any("No valid oracle labels found" in issue for issue in issues)


def test_validate_insufficient_oracle_samples() -> None:
    """Test validation with too few oracle samples."""
    data: List[Dict[str, Any]] = []
    for i in range(15):
        record: Dict[str, Any] = {
            "prompt_id": f"test_{i}",
            "prompt": f"Question {i}",
            "response": f"Answer {i}",
            "base_policy_logprob": -10.5,
            "target_policy_logprobs": {"gpt4": -8.3},
            "metadata": {"judge_score": 0.85},
        }
        # Only add oracle labels to first 5 samples
        if i < 5:
            metadata = record["metadata"]
            assert isinstance(metadata, dict)
            metadata["oracle_label"] = 0.9
        data.append(record)

    is_valid, issues = validate_cje_data(
        data, judge_field="judge_score", oracle_field="oracle_label"
    )
    assert not is_valid
    assert any("Too few oracle samples (5)" in issue for issue in issues)


def test_validate_sufficient_oracle_samples() -> None:
    """Test validation with sufficient oracle samples."""
    data: List[Dict[str, Any]] = []
    for i in range(100):
        record: Dict[str, Any] = {
            "prompt_id": f"test_{i}",
            "prompt": f"Question {i}",
            "response": f"Answer {i}",
            "base_policy_logprob": -10.5,
            "target_policy_logprobs": {"gpt4": -8.3},
            "metadata": {"judge_score": 0.85},
        }
        # Add oracle labels to first 50 samples
        if i < 50:
            metadata = record["metadata"]
            assert isinstance(metadata, dict)
            metadata["oracle_label"] = 0.9
        data.append(record)

    is_valid, issues = validate_cje_data(
        data, judge_field="judge_score", oracle_field="oracle_label"
    )
    assert is_valid
    assert len(issues) == 0


def test_validate_for_precomputed_sampler_no_rewards() -> None:
    """Test PrecomputedSampler validation without rewards."""
    data = [
        {
            "prompt_id": "test_001",
            "prompt": "What is 2+2?",
            "response": "4",
            "base_policy_logprob": -10.5,
            "target_policy_logprobs": {"gpt4": -8.3},
            "metadata": {"judge_score": 0.85},
        }
    ]
    is_valid, issues = validate_for_precomputed_sampler(data)
    assert not is_valid
    assert any(
        "PrecomputedSampler requires 'reward' field" in issue for issue in issues
    )


def test_validate_for_precomputed_sampler_with_rewards() -> None:
    """Test PrecomputedSampler validation with rewards."""
    data = [
        {
            "prompt_id": "test_001",
            "prompt": "What is 2+2?",
            "response": "4",
            "base_policy_logprob": -10.5,
            "target_policy_logprobs": {"gpt4": -8.3},
            "reward": 0.85,
        }
    ]
    is_valid, issues = validate_for_precomputed_sampler(data)
    assert is_valid
    assert len(issues) == 0


def test_validate_inconsistent_target_policies() -> None:
    """Test validation with inconsistent target policies."""
    data = [
        {
            "prompt_id": "test_001",
            "prompt": "Q1",
            "response": "A1",
            "base_policy_logprob": -10.5,
            "target_policy_logprobs": {"gpt4": -8.3, "claude": -9.1},
        },
        {
            "prompt_id": "test_002",
            "prompt": "Q2",
            "response": "A2",
            "base_policy_logprob": -11.2,
            "target_policy_logprobs": {"gpt4": -8.7},  # Missing claude
        },
    ]
    is_valid, issues = validate_cje_data(data)
    assert not is_valid
    assert any("Inconsistent target policies" in issue for issue in issues)
