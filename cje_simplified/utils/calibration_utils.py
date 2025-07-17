"""Shared calibration utilities for isotonic regression and cross-fitting."""

import numpy as np
from typing import Callable, Optional, Tuple
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold


def cross_fit_isotonic(
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
    random_seed: int = 42,
    out_of_bounds: str = "clip",
) -> np.ndarray:
    """Apply isotonic regression with cross-fitting to prevent overfitting.

    Each fold is calibrated using a model trained on the other k-1 folds.
    This ensures the calibration model never sees its own test data during training.

    Args:
        X: Input values to calibrate (e.g., scores or weights)
        y: Target values to calibrate towards
        k_folds: Number of cross-fitting folds (minimum 2)
        random_seed: Random seed for fold splitting
        out_of_bounds: How to handle predictions outside training range

    Returns:
        Calibrated values with same shape as X

    Example:
        # Calibrate importance weights to have mean 1
        calibrated_weights = cross_fit_isotonic(
            raw_weights,
            np.ones_like(raw_weights)
        )

        # Calibrate judge scores to oracle labels
        calibrated_scores = cross_fit_isotonic(
            judge_scores,
            oracle_labels
        )
    """
    n = len(X)
    k_folds = min(max(2, k_folds), n // 2)  # Ensure valid number of folds

    # Handle edge cases
    if n < 4 or k_folds < 2:
        # Too little data for cross-fitting, fit directly
        iso = IsotonicRegression(out_of_bounds=out_of_bounds)
        iso.fit(X, y)
        return iso.predict(X)

    # Initialize output
    calibrated = np.zeros(n)

    # Cross-fitting
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    for train_idx, test_idx in kf.split(X):
        # Fit isotonic regression on training folds
        iso = IsotonicRegression(out_of_bounds=out_of_bounds)
        iso.fit(X[train_idx], y[train_idx])

        # Calibrate test fold
        calibrated[test_idx] = iso.predict(X[test_idx])

    return calibrated


def fit_isotonic_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
    random_seed: int = 42,
    out_of_bounds: str = "clip",
) -> Tuple[np.ndarray, IsotonicRegression]:
    """Fit isotonic regression with cross-validation and return final model.

    Similar to cross_fit_isotonic but also returns a final model trained
    on all data for future predictions.

    Args:
        X: Input values to calibrate
        y: Target values
        k_folds: Number of cross-fitting folds
        random_seed: Random seed
        out_of_bounds: How to handle out-of-bounds predictions

    Returns:
        Tuple of (cross_fit_predictions, final_model)
    """
    # Get cross-fitted predictions
    cv_predictions = cross_fit_isotonic(X, y, k_folds, random_seed, out_of_bounds)

    # Fit final model on all data
    final_model = IsotonicRegression(out_of_bounds=out_of_bounds)
    final_model.fit(X, y)

    return cv_predictions, final_model


def calibrate_to_target_mean(
    values: np.ndarray,
    target_mean: float = 1.0,
    k_folds: int = 5,
    random_seed: int = 42,
) -> np.ndarray:
    """Calibrate values to have a specific mean using isotonic regression.

    Common use case: calibrating importance weights to have mean 1.0
    to ensure unbiasedness in importance sampling.

    Args:
        values: Values to calibrate (e.g., importance weights)
        target_mean: Desired mean after calibration
        k_folds: Number of cross-fitting folds
        random_seed: Random seed

    Returns:
        Calibrated values with mean approximately equal to target_mean

    Example:
        # Ensure importance weights have mean 1
        calibrated_weights = calibrate_to_target_mean(raw_weights, 1.0)
    """
    # Create target array with desired mean
    target = np.full_like(values, target_mean)

    # Apply isotonic calibration
    return cross_fit_isotonic(
        values, target, k_folds, random_seed, out_of_bounds="clip"
    )


def compute_calibration_diagnostics(
    predictions: np.ndarray, actuals: np.ndarray, coverage_threshold: float = 0.1
) -> dict:
    """Compute diagnostics for calibration quality.

    Args:
        predictions: Calibrated predictions
        actuals: True values
        coverage_threshold: Threshold for coverage calculation

    Returns:
        Dict with rmse, mae, coverage, and correlation
    """
    residuals = predictions - actuals

    return {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "coverage": float(np.mean(np.abs(residuals) <= coverage_threshold)),
        "correlation": float(np.corrcoef(predictions, actuals)[0, 1]),
    }
