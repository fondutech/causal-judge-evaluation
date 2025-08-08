"""Calibration utilities for CJE.

This module contains all calibration functionality:
- Optimized isotonic regression for weight calibration
- Judge score calibration to match oracle labels
- Dataset calibration workflows
"""

from .isotonic import (
    calibrate_to_target_mean,
)
from .judge import (
    JudgeCalibrator,
    calibrate_judge_scores,
    CalibrationResult,
)
from .dataset import (
    calibrate_dataset,
    calibrate_from_raw_data,
)

__all__ = [
    # Isotonic regression utilities
    "calibrate_to_target_mean",
    # Judge calibration
    "JudgeCalibrator",
    "calibrate_judge_scores",
    "CalibrationResult",
    # Dataset calibration
    "calibrate_dataset",
    "calibrate_from_raw_data",
]
