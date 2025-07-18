"""Calibration utilities for CJE.

This module contains all calibration functionality:
- Isotonic regression utilities for cross-fitting
- Judge score calibration to match oracle labels
- Dataset calibration workflows
"""

from .isotonic import (
    cross_fit_isotonic,
    calibrate_to_target_mean,
    compute_calibration_diagnostics,
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
    "cross_fit_isotonic",
    "calibrate_to_target_mean",
    "compute_calibration_diagnostics",
    # Judge calibration
    "JudgeCalibrator",
    "calibrate_judge_scores",
    "CalibrationResult",
    # Dataset calibration
    "calibrate_dataset",
    "calibrate_from_raw_data",
]
