"""Tests for the modular pipeline architecture."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from cje.pipeline import CJEPipeline, PipelineConfig
from cje.pipeline.stages import (
    DatasetStage,
    LoggingPolicyStage,
    JudgeStage,
    OracleStage,
    CalibrationStage,
    TargetPolicyStage,
)


def test_pipeline_config_creation() -> None:
    """Test that PipelineConfig can be created properly."""
    config = PipelineConfig(
        work_dir=Path("/tmp/test"),
        dataset_config={"name": "test_dataset", "split": "train"},
        logging_policy_config={"provider": "test", "model_name": "test_model"},
        judge_config={
            "provider": "test",
            "model_name": "test_judge",
        },
        calibration_config={"n_folds": 5},
        target_policies_config=[
            {"provider": "test", "model_name": "target1"},
            {"provider": "test", "model_name": "target2"},
        ],
        estimator_configs=[
            {"name": "ips", "params": {}},
            {"name": "dr", "params": {}},
        ],
    )

    assert config.work_dir == Path("/tmp/test")
    assert config.dataset_config["name"] == "test_dataset"
    assert len(config.target_policies_config) == 2
    assert len(config.estimator_configs) == 2


def test_dataset_stage_init() -> None:
    """Test DatasetStage initialization."""
    work_dir = Path("/tmp/test")
    stage = DatasetStage(work_dir)

    assert stage.work_dir == work_dir
    assert stage.console is not None


def test_pipeline_initialization() -> None:
    """Test CJEPipeline initialization."""
    config = PipelineConfig(
        work_dir=Path("/tmp/test"),
        dataset_config={"name": "test_dataset", "split": "train"},
        logging_policy_config={"provider": "test", "model_name": "test_model"},
        judge_config={"provider": "test", "model_name": "test_judge"},
        calibration_config={"n_folds": 5},
        target_policies_config=[{"provider": "test", "model_name": "target1"}],
        estimator_configs=[{"name": "ips", "params": {}}],
    )

    pipeline = CJEPipeline(config)

    assert pipeline.config == config
    assert pipeline.work_dir == config.work_dir
    assert isinstance(pipeline.dataset_stage, DatasetStage)
    assert isinstance(pipeline.logging_policy_stage, LoggingPolicyStage)
    assert isinstance(pipeline.judge_stage, JudgeStage)
    assert isinstance(pipeline.calibration_stage, CalibrationStage)
    assert isinstance(pipeline.target_policy_stage, TargetPolicyStage)
    assert pipeline.oracle_stage is None  # No oracle config


def test_pipeline_with_oracle() -> None:
    """Test CJEPipeline initialization with oracle."""
    config = PipelineConfig(
        work_dir=Path("/tmp/test"),
        dataset_config={"name": "test_dataset", "split": "train"},
        logging_policy_config={"provider": "test", "model_name": "test_model"},
        judge_config={"provider": "test", "model_name": "test_judge"},
        calibration_config={"n_folds": 5},
        target_policies_config=[{"provider": "test", "model_name": "target1"}],
        estimator_configs=[{"name": "ips", "params": {}}],
        oracle_config={
            "enabled": True,
            "provider": "test",
            "model_name": "oracle_model",
        },
    )

    pipeline = CJEPipeline(config)

    assert pipeline.oracle_stage is not None
    assert isinstance(pipeline.oracle_stage, OracleStage)


@patch("cje.pipeline.coordinator.get_estimator")
@patch("cje.data.load_dataset")
def test_pipeline_run_mock(
    mock_load_dataset: Mock, mock_get_estimator: Mock, tmp_path: Path
) -> None:
    """Test pipeline run with mocked components."""
    # Create a mock dataset
    mock_sample = Mock()
    mock_sample.uid = "test_uid"
    mock_sample.context = "test context"

    mock_dataset = Mock()
    mock_dataset.__iter__ = Mock(return_value=iter([mock_sample]))
    mock_dataset.__len__ = Mock(return_value=1)
    mock_load_dataset.return_value = mock_dataset

    # Create a mock estimator
    mock_estimator = Mock()
    mock_estimator.fit = Mock()
    mock_result = Mock()
    mock_result.estimates = [0.5]
    mock_result.standard_errors = [0.1]
    mock_estimator.estimate = Mock(return_value=mock_result)
    mock_get_estimator.return_value = mock_estimator

    # Create pipeline config
    config = PipelineConfig(
        work_dir=tmp_path,
        dataset_config={"name": "test_dataset", "split": "train"},
        logging_policy_config={"provider": "test", "model_name": "test_model"},
        judge_config={"provider": "test", "model_name": "test_judge"},
        calibration_config={"n_folds": 5},
        target_policies_config=[{"provider": "test", "model_name": "target1"}],
        estimator_configs=[{"name": "ips", "params": {}}],
    )

    # Create and run pipeline
    pipeline = CJEPipeline(config)

    # Mock stage methods
    with patch.object(pipeline.logging_policy_stage, "run") as mock_logging:
        with patch.object(pipeline.judge_stage, "run") as mock_judge:
            with patch.object(pipeline.calibration_stage, "run") as mock_calibration:
                with patch.object(pipeline.target_policy_stage, "run") as mock_target:
                    # Setup mock returns
                    mock_rows: List[Dict[str, Any]] = [
                        {
                            "uid": "test_uid",
                            "context": "test context",
                            "response": "test response",
                        }
                    ]
                    mock_logging.return_value = mock_rows
                    mock_judge.return_value = mock_rows
                    mock_calibration.return_value = mock_rows
                    import copy

                    mock_rows_with_logp = copy.deepcopy(mock_rows)
                    for row in mock_rows_with_logp:
                        row["logp_target_all"] = [0.5]
                        row["reward"] = 1.0
                    mock_target.return_value = mock_rows_with_logp

                    # Run pipeline
                    result = pipeline.run()

                    # Verify calls
                    assert mock_load_dataset.called
                    assert mock_logging.called
                    assert mock_judge.called
                    assert mock_calibration.called
                    assert mock_target.called
                    assert mock_get_estimator.called
                    assert mock_estimator.fit.called
                    assert mock_estimator.estimate.called

                    # Check results
                    assert "results" in result
                    assert "ips" in result["results"]
                    assert "execution_time" in result
                    assert "stage_times" in result
