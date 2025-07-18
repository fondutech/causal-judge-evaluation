"""Judge score calibration using isotonic regression.

Calibrates cheap LLM judge scores to actual business KPIs/oracle labels
using monotonic regression on a labeled subset.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from sklearn.isotonic import IsotonicRegression
from dataclasses import dataclass

from .isotonic import (
    cross_fit_isotonic,
    compute_calibration_diagnostics,
)


@dataclass
class CalibrationResult:
    """Result of judge calibration."""

    calibrated_scores: np.ndarray  # Calibrated scores for all data
    calibration_rmse: float  # RMSE on oracle subset
    coverage_at_01: float  # Fraction within ±0.1 of true label
    n_oracle: int  # Number of oracle samples used

    def summary(self) -> str:
        """Format calibration results."""
        return (
            f"Calibration Summary:\n"
            f"  Oracle samples: {self.n_oracle}\n"
            f"  RMSE: {self.calibration_rmse:.3f}\n"
            f"  Coverage (±0.1): {self.coverage_at_01:.1%}"
        )


class JudgeCalibrator:
    """Calibrate judge scores to oracle labels using isotonic regression.

    Uses cross-fitting on oracle subset to prevent overfitting, then
    applies calibration to all data.

    Args:
        k_folds: Number of cross-fitting folds for oracle data (default 5)
        random_seed: Random seed for reproducibility
    """

    def __init__(self, k_folds: int = 5, random_seed: int = 42):
        self.k_folds = max(2, k_folds)
        self.random_seed = random_seed
        self._final_calibrator: Optional[IsotonicRegression] = None

    def fit_transform(
        self,
        judge_scores: np.ndarray,
        oracle_labels: Optional[np.ndarray] = None,
        oracle_mask: Optional[np.ndarray] = None,
    ) -> CalibrationResult:
        """Calibrate judge scores using oracle labels.

        Args:
            judge_scores: Raw judge scores for all data
            oracle_labels: True labels for oracle subset (if oracle_mask provided)
            oracle_mask: Boolean mask indicating which samples have oracle labels

        Returns:
            CalibrationResult with calibrated scores and diagnostics

        Example:
            # With explicit mask
            calibrator = JudgeCalibrator()
            result = calibrator.fit_transform(
                judge_scores=all_scores,
                oracle_labels=oracle_values,
                oracle_mask=has_oracle_label
            )

            # Or with implicit mask (oracle_labels shorter than judge_scores)
            result = calibrator.fit_transform(
                judge_scores=all_scores,
                oracle_labels=oracle_subset_labels
            )
        """
        judge_scores = np.asarray(judge_scores)
        n_total = len(judge_scores)

        # Handle different input formats
        if oracle_mask is not None:
            # Explicit mask provided
            oracle_mask = np.asarray(oracle_mask, dtype=bool)
            if oracle_labels is None:
                raise ValueError("oracle_labels required when oracle_mask provided")
            oracle_labels = np.asarray(oracle_labels)

            # Extract oracle subset
            oracle_scores = judge_scores[oracle_mask]
            oracle_y = oracle_labels

        elif oracle_labels is not None and len(oracle_labels) < n_total:
            # Oracle labels provided for first n samples
            n_oracle = len(oracle_labels)
            oracle_scores = judge_scores[:n_oracle]
            oracle_y = np.asarray(oracle_labels)
            oracle_mask = np.zeros(n_total, dtype=bool)
            oracle_mask[:n_oracle] = True

        else:
            # All data has oracle labels (no holdout)
            oracle_scores = judge_scores
            oracle_y = (
                np.asarray(oracle_labels) if oracle_labels is not None else judge_scores
            )
            oracle_mask = np.ones(n_total, dtype=bool)

        n_oracle = len(oracle_y)

        if n_oracle < 10:
            raise ValueError(f"Too few oracle samples ({n_oracle}). Need at least 10.")

        # Initialize calibrated scores
        calibrated_scores = np.copy(judge_scores)

        # Cross-fit calibration on oracle subset
        if n_oracle < n_total:
            # Standard case: calibrate on oracle subset
            # Get cross-fitted predictions for oracle samples
            oracle_calibrated = cross_fit_isotonic(
                oracle_scores,
                oracle_y,
                k_folds=self.k_folds,
                random_seed=self.random_seed,
            )
            calibrated_scores[oracle_mask] = oracle_calibrated

            # Fit final model on all oracle data for calibrating non-oracle samples
            self._final_calibrator = IsotonicRegression(out_of_bounds="clip")
            self._final_calibrator.fit(oracle_scores, oracle_y)

            # Calibrate non-oracle samples
            non_oracle_mask = ~oracle_mask
            if np.any(non_oracle_mask):
                calibrated_scores[non_oracle_mask] = self._final_calibrator.predict(
                    judge_scores[non_oracle_mask]
                )
        else:
            # All data has labels: just cross-fit everything
            calibrated_scores = cross_fit_isotonic(
                judge_scores,
                oracle_y,
                k_folds=self.k_folds,
                random_seed=self.random_seed,
            )

            # Still fit a final model for potential future predictions
            self._final_calibrator = IsotonicRegression(out_of_bounds="clip")
            self._final_calibrator.fit(judge_scores, oracle_y)

        # Compute diagnostics on oracle subset using shared utility
        oracle_cal_scores = calibrated_scores[oracle_mask]
        diagnostics = compute_calibration_diagnostics(
            oracle_cal_scores, oracle_y, coverage_threshold=0.1
        )

        return CalibrationResult(
            calibrated_scores=calibrated_scores,
            calibration_rmse=diagnostics["rmse"],
            coverage_at_01=diagnostics["coverage"],
            n_oracle=n_oracle,
        )

    def transform(self, judge_scores: np.ndarray) -> np.ndarray:
        """Apply calibration to new judge scores.

        Args:
            judge_scores: Raw judge scores to calibrate

        Returns:
            Calibrated scores

        Raises:
            RuntimeError: If fit_transform() hasn't been called yet
        """
        if self._final_calibrator is None:
            raise RuntimeError("Must call fit_transform() before transform()")

        return self._final_calibrator.predict(judge_scores)


def calibrate_judge_scores(
    judge_scores: np.ndarray,
    oracle_labels: np.ndarray,
    oracle_mask: Optional[np.ndarray] = None,
    k_folds: int = 5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Convenience function for judge calibration.

    Args:
        judge_scores: Raw judge scores for all data
        oracle_labels: True labels for oracle subset
        oracle_mask: Optional boolean mask for oracle samples
        k_folds: Number of cross-fitting folds

    Returns:
        Tuple of (calibrated_scores, diagnostics_dict)

    Example:
        # Calibrate judge scores with 25% oracle labels
        cal_scores, stats = calibrate_judge_scores(
            judge_scores=all_judge_scores,
            oracle_labels=oracle_subset_labels[:1000]  # First 1000 have labels
        )

        print(f"Calibration RMSE: {stats['rmse']:.3f}")
        print(f"Coverage: {stats['coverage']:.1%}")
    """
    calibrator = JudgeCalibrator(k_folds=k_folds)
    result = calibrator.fit_transform(judge_scores, oracle_labels, oracle_mask)

    diagnostics = {
        "rmse": result.calibration_rmse,
        "coverage": result.coverage_at_01,
        "n_oracle": result.n_oracle,
    }

    return result.calibrated_scores, diagnostics
