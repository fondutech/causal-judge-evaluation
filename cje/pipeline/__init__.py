"""
CJE Pipeline - Modular experiment execution pipeline.

This package provides a clean, testable pipeline for running CJE experiments.
Each stage is isolated and can be tested independently.
"""

from .coordinator import CJEPipeline
from .config import PipelineConfig

# Also export stages for testing/customization
from .stages import (
    DatasetStage,
    LoggingPolicyStage,
    JudgeStage,
    OracleStage,
    CalibrationStage,
    TargetPolicyStage,
)

__all__ = [
    "CJEPipeline",
    "PipelineConfig",
    "DatasetStage",
    "LoggingPolicyStage",
    "JudgeStage",
    "OracleStage",
    "CalibrationStage",
    "TargetPolicyStage",
]
