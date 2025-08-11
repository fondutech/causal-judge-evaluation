"""Judge score calibration using isotonic regression.

Calibrates cheap LLM judge scores to actual business KPIs/oracle labels
using monotonic regression on a labeled subset.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from sklearn.isotonic import IsotonicRegression
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of judge calibration."""

    calibrated_scores: np.ndarray  # Calibrated scores for all data
    calibration_rmse: float  # RMSE on oracle subset
    coverage_at_01: float  # Fraction within ±0.1 of true label
    n_oracle: int  # Number of oracle samples used
    calibrator: Optional["JudgeCalibrator"] = None  # The fitted calibrator

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

    Args:
        random_seed: Random seed for reproducibility
    """

    def __init__(self, random_seed: int = 42):
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

        # Always fit the full model (for stable reward labels)
        self._final_calibrator = IsotonicRegression(out_of_bounds="clip")
        self._final_calibrator.fit(oracle_scores, oracle_y)

        # Apply calibration to all samples using full model
        calibrated_scores = self._final_calibrator.predict(judge_scores)

        # Compute diagnostics on oracle subset
        oracle_calibrated = calibrated_scores[oracle_mask]
        rmse = np.sqrt(np.mean((oracle_calibrated - oracle_y) ** 2))
        coverage_01 = np.mean(np.abs(oracle_calibrated - oracle_y) <= 0.1)

        # Log summary
        logger.info(
            f"Calibration complete: {n_oracle} oracle samples, "
            f"RMSE={rmse:.3f}, coverage@0.1={coverage_01:.1%}"
        )

        return CalibrationResult(
            calibrated_scores=calibrated_scores,
            calibration_rmse=float(rmse),
            coverage_at_01=float(coverage_01),
            n_oracle=n_oracle,
            calibrator=self,
        )

    def predict(self, judge_scores: np.ndarray) -> np.ndarray:
        """Apply calibration to new judge scores.

        Args:
            judge_scores: Judge scores to calibrate

        Returns:
            Calibrated scores
        """
        if self._final_calibrator is None:
            raise RuntimeError("Calibrator must be fitted before prediction")

        return self._final_calibrator.predict(np.asarray(judge_scores))


def calibrate_judge_scores(
    judge_scores: np.ndarray,
    oracle_labels: np.ndarray,
    oracle_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Convenience function for judge calibration.

    Args:
        judge_scores: Raw judge scores for all data
        oracle_labels: True labels for oracle subset
        oracle_mask: Optional boolean mask for oracle samples

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
    calibrator = JudgeCalibrator()
    result = calibrator.fit_transform(judge_scores, oracle_labels, oracle_mask)

    diagnostics = {
        "rmse": result.calibration_rmse,
        "coverage": result.coverage_at_01,
        "n_oracle": result.n_oracle,
    }

    return result.calibrated_scores, diagnostics
