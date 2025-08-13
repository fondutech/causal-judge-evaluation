"""Test suite for reward handling to prevent regressions."""

from typing import Optional, Any
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward_utils import (
    RewardSource,
    RewardConfig,
    determine_reward_config,
    should_recalibrate_for_estimator,
)
from validation import (
    validate_no_unnecessary_calibration,
    validate_reward_source,
)


class MockSample:
    """Mock sample for testing."""

    def __init__(
        self, reward: Optional[float] = None, metadata: Optional[dict] = None
    ) -> None:
        self.reward = reward
        self.metadata = metadata or {}


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, samples: list) -> None:
        self.samples = samples


def test_reward_source_enum() -> None:
    """Test that RewardSource enum works correctly."""
    assert RewardSource.NONE.value == "none"
    assert RewardSource.ORACLE_DIRECT.value == "oracle_direct"
    assert (
        str(RewardSource.ORACLE_DIRECT)
        == "Using oracle labels directly (100% coverage)"
    )

    # Test needs_calibration_for_dr
    assert RewardSource.NONE.needs_calibration_for_dr() == True
    assert RewardSource.ORACLE_DIRECT.needs_calibration_for_dr() == False
    assert RewardSource.CALIBRATED.needs_calibration_for_dr() == False


def test_determine_reward_config_precomputed() -> None:
    """Test detection of pre-computed rewards."""
    samples = [MockSample(reward=0.5) for _ in range(10)]
    dataset = MockDataset(samples)

    config = determine_reward_config(dataset, oracle_coverage=0.5, use_oracle=False)

    assert config.source == RewardSource.PRECOMPUTED
    assert config.needs_calibration == False


def test_determine_reward_config_oracle_direct() -> None:
    """Test detection of 100% oracle coverage."""
    samples = [MockSample(metadata={"oracle_label": 0.8}) for _ in range(10)]
    dataset = MockDataset(samples)

    # Test with oracle_coverage=1.0
    config = determine_reward_config(dataset, oracle_coverage=1.0, use_oracle=False)
    assert config.source == RewardSource.ORACLE_DIRECT

    # Test with use_oracle=True
    config = determine_reward_config(dataset, oracle_coverage=0.5, use_oracle=True)
    assert config.source == RewardSource.ORACLE_DIRECT


def test_determine_reward_config_needs_calibration() -> None:
    """Test detection of need for calibration."""
    # Mix of samples with and without oracle labels
    samples = [
        MockSample(metadata={"oracle_label": 0.8}) if i < 5 else MockSample(metadata={})
        for i in range(10)
    ]
    dataset = MockDataset(samples)

    config = determine_reward_config(dataset, oracle_coverage=0.5, use_oracle=False)

    # With only 50% having oracle labels, it should need calibration
    assert config.source == RewardSource.CALIBRATED
    assert config.needs_calibration == True
    assert config.calibration_params is not None


def test_validate_no_unnecessary_calibration_catches_mistake() -> None:
    """Test that validation catches unnecessary calibration."""
    # Create dataset with oracle labels as rewards
    samples = [
        MockSample(reward=0.8, metadata={"oracle_label": 0.8}) for _ in range(10)
    ]
    dataset = MockDataset(samples)

    # Should raise error
    with pytest.raises(ValueError, match="STOP.*already set to oracle"):
        validate_no_unnecessary_calibration(dataset, 1.0, None)


def test_validate_no_unnecessary_calibration_allows_valid() -> None:
    """Test that validation allows valid calibration."""
    # Dataset with no rewards yet
    samples = [MockSample(metadata={"oracle_label": 0.8}) for _ in range(10)]
    dataset = MockDataset(samples)

    # Should not raise
    validate_no_unnecessary_calibration(dataset, 0.5, None)


def test_validate_reward_source() -> None:
    """Test reward source detection."""
    # Test oracle_direct
    samples = [
        MockSample(reward=i / 30, metadata={"oracle_label": i / 30})
        for i in range(30)  # 30 unique values
    ]
    dataset = MockDataset(samples)
    assert validate_reward_source(dataset) == "oracle_direct"

    # Test calibrated (fewer unique values)
    samples = [
        MockSample(reward=int(i / 3) / 10, metadata={"oracle_label": i / 30})
        for i in range(30)  # 10 unique reward values, 30 oracle values
    ]
    dataset = MockDataset(samples)
    assert validate_reward_source(dataset) == "calibrated"

    # Test none
    samples = [MockSample() for _ in range(10)]
    dataset = MockDataset(samples)
    assert validate_reward_source(dataset) == "none"


def test_should_recalibrate_for_estimator() -> None:
    """Test recalibration decision logic."""
    # Oracle direct - never recalibrate
    config = RewardConfig(
        source=RewardSource.ORACLE_DIRECT, oracle_coverage=1.0, needs_calibration=False
    )
    assert should_recalibrate_for_estimator("mrdr", config, None) == False

    # Precomputed - never recalibrate
    config = RewardConfig(
        source=RewardSource.PRECOMPUTED, oracle_coverage=1.0, needs_calibration=False
    )
    assert should_recalibrate_for_estimator("tmle", config, None) == False

    # Calibrated - already done, don't recalibrate
    config = RewardConfig(
        source=RewardSource.CALIBRATED, oracle_coverage=0.5, needs_calibration=False
    )
    cal_result = MagicMock(calibrator=True)
    assert should_recalibrate_for_estimator("mrdr", config, cal_result) == False


def test_integration_full_oracle_workflow() -> None:
    """Test the full workflow with 100% oracle coverage."""
    # Create dataset with full oracle labels
    samples = [
        MockSample(metadata={"oracle_label": i / 25, "judge_score": i / 20})
        for i in range(25)
    ]
    dataset = MockDataset(samples)

    # Determine config
    config = determine_reward_config(dataset, oracle_coverage=1.0, use_oracle=False)
    assert config.source == RewardSource.ORACLE_DIRECT

    # Should not need recalibration for any estimator
    assert should_recalibrate_for_estimator("mrdr", config, None) == False
    assert should_recalibrate_for_estimator("tmle", config, None) == False
    assert should_recalibrate_for_estimator("dr-cpo", config, None) == False


if __name__ == "__main__":
    # Run tests manually
    test_reward_source_enum()
    print("✅ RewardSource enum tests passed")

    test_determine_reward_config_precomputed()
    test_determine_reward_config_oracle_direct()
    test_determine_reward_config_needs_calibration()
    print("✅ determine_reward_config tests passed")

    test_validate_no_unnecessary_calibration_catches_mistake()
    test_validate_no_unnecessary_calibration_allows_valid()
    test_validate_reward_source()
    print("✅ Validation tests passed")

    test_should_recalibrate_for_estimator()
    print("✅ Recalibration logic tests passed")

    test_integration_full_oracle_workflow()
    print("✅ Integration test passed")

    print("\n🎉 All tests passed!")
