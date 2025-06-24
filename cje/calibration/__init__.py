"""Model calibration and uncertainty quantification tools.

This module provides calibration functionality for different stages of the CJE pipeline:

1. Judge Score Calibration:
   - cross_fit_calibration: Calibrates judge scores against oracle labels
   - fit_isotonic, plot_reliability: Basic isotonic calibration utilities

2. Importance Weight Calibration:
   - calibrate_weights_isotonic: Calibrates importance weights for estimators
   - apply_weight_calibration_pipeline: Full pipeline with clipping and stabilization

3. Outcome Model Calibration:
   - calibrate_outcome_model_isotonic: Calibrates predictions from outcome models

The calibration functions use isotonic regression to maintain monotonicity while
improving statistical calibration. Cross-fitting is used to prevent overfitting.
"""

from .isotonic import fit_isotonic, plot_reliability
from .cross_fit import cross_fit_calibration
from .weights import (
    calibrate_weights_isotonic,
    calibrate_outcome_model_isotonic,
    apply_weight_calibration_pipeline,
)

__all__ = [
    "fit_isotonic",
    "plot_reliability",
    "cross_fit_calibration",
    "calibrate_weights_isotonic",
    "calibrate_outcome_model_isotonic",
    "apply_weight_calibration_pipeline",
]
