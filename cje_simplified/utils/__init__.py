"""Utility functions for calibration and diagnostics.

This module contains:
- Calibration Utils: Shared isotonic regression and cross-fitting
- Judge Calibration: Map judge scores to business KPIs
- Weight Diagnostics: Debug importance sampling issues
"""

from .calibration_utils import (
    cross_fit_isotonic,
    fit_isotonic_with_cv,
    calibrate_to_target_mean,
    compute_calibration_diagnostics,
)
from .judge_calibration import (
    JudgeCalibrator,
    calibrate_judge_scores,
    CalibrationResult,
)
from .dataset_calibration import (
    calibrate_dataset,
    calibrate_from_raw_data,
)
from .weight_diagnostics import (
    diagnose_weights,
    create_weight_summary_table,
    detect_api_nondeterminism,
    WeightDiagnostics,
)

__all__ = [
    # Calibration utilities
    "cross_fit_isotonic",
    "fit_isotonic_with_cv",
    "calibrate_to_target_mean",
    "compute_calibration_diagnostics",
    # Judge calibration
    "JudgeCalibrator",
    "calibrate_judge_scores",
    "CalibrationResult",
    # Dataset calibration
    "calibrate_dataset",
    "calibrate_from_raw_data",
    # Weight diagnostics
    "diagnose_weights",
    "create_weight_summary_table",
    "detect_api_nondeterminism",
    "WeightDiagnostics",
]
