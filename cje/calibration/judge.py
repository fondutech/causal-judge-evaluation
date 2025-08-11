"""Judge score calibration using isotonic regression.

Calibrates cheap LLM judge scores to actual business KPIs/oracle labels
using monotonic regression on a labeled subset.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from dataclasses import dataclass
import logging
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of judge calibration."""

    calibrated_scores: np.ndarray  # Calibrated scores for all data
    calibration_rmse: float  # RMSE on oracle subset
    coverage_at_01: float  # Fraction within ±0.1 of true label
    n_oracle: int  # Number of oracle samples used
    calibrator: Optional["JudgeCalibrator"] = None  # The fitted calibrator
    fold_ids: Optional[np.ndarray] = None  # CV fold assignment for each sample
    oof_rmse: Optional[float] = None  # Out-of-fold RMSE (if cross-fitted)
    oof_coverage_at_01: Optional[float] = None  # Out-of-fold coverage (if cross-fitted)

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
        self._fold_models: Dict[int, IsotonicRegression] = {}
        self._fold_ids: Optional[np.ndarray] = None
        self._n_folds: int = 5

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

        elif oracle_labels is not None:
            # All data has oracle labels (no holdout)
            oracle_labels = np.asarray(oracle_labels)
            if len(oracle_labels) != n_total:
                raise ValueError(
                    f"oracle_labels length ({len(oracle_labels)}) must match "
                    f"judge_scores length ({n_total}) or be shorter for partial labeling"
                )
            oracle_scores = judge_scores
            oracle_y = oracle_labels
            oracle_mask = np.ones(n_total, dtype=bool)
        else:
            # No oracle labels provided
            raise ValueError(
                "oracle_labels is required for calibration. "
                "Provide oracle labels for at least a subset of samples."
            )

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
            fold_ids=self._fold_ids,
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

    def fit_cv(
        self,
        judge_scores: np.ndarray,
        oracle_labels: Optional[np.ndarray] = None,
        oracle_mask: Optional[np.ndarray] = None,
        n_folds: int = 5,
    ) -> CalibrationResult:
        """Fit both global and cross-fitted calibration models.

        This method:
        1. Fits a global model f_all on all oracle data (for stable rewards)
        2. Fits per-fold models f^(-k) for cross-fitted predictions (for DR)
        3. Assigns fold IDs to all samples (labeled by CV, unlabeled by hash)

        Args:
            judge_scores: Raw judge scores for all data
            oracle_labels: True labels for oracle subset
            oracle_mask: Boolean mask indicating which samples have oracle labels
            n_folds: Number of CV folds

        Returns:
            CalibrationResult with both global and CV calibration
        """
        judge_scores = np.asarray(judge_scores)
        n_total = len(judge_scores)
        self._n_folds = n_folds

        # Handle different input formats (same as fit_transform)
        if oracle_mask is not None:
            oracle_mask = np.asarray(oracle_mask, dtype=bool)
            if oracle_labels is None:
                raise ValueError("oracle_labels required when oracle_mask provided")
            oracle_labels = np.asarray(oracle_labels)
            oracle_scores = judge_scores[oracle_mask]
            oracle_y = oracle_labels
        elif oracle_labels is not None and len(oracle_labels) < n_total:
            n_oracle = len(oracle_labels)
            oracle_scores = judge_scores[:n_oracle]
            oracle_y = np.asarray(oracle_labels)
            oracle_mask = np.zeros(n_total, dtype=bool)
            oracle_mask[:n_oracle] = True
        elif oracle_labels is not None:
            oracle_labels = np.asarray(oracle_labels)
            if len(oracle_labels) != n_total:
                raise ValueError(
                    f"oracle_labels length ({len(oracle_labels)}) must match "
                    f"judge_scores length ({n_total}) or be shorter for partial labeling"
                )
            oracle_scores = judge_scores
            oracle_y = oracle_labels
            oracle_mask = np.ones(n_total, dtype=bool)
        else:
            raise ValueError(
                "oracle_labels is required for calibration. "
                "Provide oracle labels for at least a subset of samples."
            )

        n_oracle = len(oracle_y)
        if n_oracle < n_folds * 2:
            raise ValueError(
                f"Too few oracle samples ({n_oracle}) for {n_folds}-fold CV. "
                f"Need at least {n_folds * 2}."
            )

        # Step 1: Fit global model (same as fit_transform)
        self._final_calibrator = IsotonicRegression(out_of_bounds="clip")
        self._final_calibrator.fit(oracle_scores, oracle_y)
        calibrated_scores = self._final_calibrator.predict(judge_scores)

        # Step 2: Assign fold IDs to all samples
        self._fold_ids = np.zeros(n_total, dtype=int)

        # Labeled samples: assign by KFold
        oracle_indices = np.where(oracle_mask)[0]
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        for fold_id, (_, test_idx) in enumerate(kf.split(oracle_indices)):
            fold_samples = oracle_indices[test_idx]
            self._fold_ids[fold_samples] = fold_id

        # Unlabeled samples: assign deterministically by stable hash
        unlabeled_mask = ~oracle_mask
        unlabeled_indices = np.where(unlabeled_mask)[0]
        if len(unlabeled_indices) > 0:

            def _fold_for_idx(i: int, seed: int, n_folds: int) -> int:
                """Stable hash-based fold assignment."""
                h = hashlib.blake2b(f"{i}-{seed}".encode(), digest_size=2)
                return int.from_bytes(h.digest(), "big") % n_folds

            for idx in unlabeled_indices:
                self._fold_ids[idx] = _fold_for_idx(int(idx), self.random_seed, n_folds)

        # Step 3: Fit per-fold models
        self._fold_models = {}
        for fold_id in range(n_folds):
            # Get training indices (all oracle samples NOT in this fold)
            train_mask = oracle_mask & (self._fold_ids != fold_id)
            train_scores = judge_scores[train_mask]
            # Get corresponding oracle labels for training samples
            oracle_fold_mask = self._fold_ids[oracle_mask] != fold_id
            train_labels = oracle_y[oracle_fold_mask]

            if len(train_scores) > 0:
                fold_model = IsotonicRegression(out_of_bounds="clip")
                fold_model.fit(train_scores, train_labels)
                self._fold_models[fold_id] = fold_model
            else:
                # Fallback to global model if not enough data
                self._fold_models[fold_id] = self._final_calibrator

        # Compute diagnostics with both global and OOF predictions
        oracle_calibrated = calibrated_scores[oracle_mask]
        rmse = np.sqrt(np.mean((oracle_calibrated - oracle_y) ** 2))
        coverage_01 = np.mean(np.abs(oracle_calibrated - oracle_y) <= 0.1)

        # Compute OOF diagnostics for oracle points
        oracle_oof = np.empty_like(oracle_y)
        for fold_id, model in self._fold_models.items():
            mask = self._fold_ids[oracle_mask] == fold_id
            if np.any(mask):
                oracle_oof[mask] = model.predict(oracle_scores[mask])

        rmse_oof = float(np.sqrt(np.mean((oracle_oof - oracle_y) ** 2)))
        coverage_01_oof = float(np.mean(np.abs(oracle_oof - oracle_y) <= 0.1))

        logger.info(
            f"CV Calibration complete: {n_oracle} oracle samples, {n_folds} folds, "
            f"RMSE={rmse:.3f} (OOF: {rmse_oof:.3f}), "
            f"coverage@0.1={coverage_01:.1%} (OOF: {coverage_01_oof:.1%})"
        )

        return CalibrationResult(
            calibrated_scores=calibrated_scores,
            calibration_rmse=float(rmse),
            coverage_at_01=float(coverage_01),
            n_oracle=n_oracle,
            calibrator=self,
            fold_ids=self._fold_ids,
            oof_rmse=rmse_oof,
            oof_coverage_at_01=coverage_01_oof,
        )

    def predict_all(self, judge_scores: np.ndarray) -> np.ndarray:
        """Predict using global model f_all (stable for rewards).

        Args:
            judge_scores: Judge scores to calibrate

        Returns:
            Globally calibrated scores
        """
        return self.predict(judge_scores)

    def predict_oof(self, judge_scores: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
        """Out-of-fold predictions using cross-fitted models.

        Args:
            judge_scores: Judge scores to calibrate
            fold_ids: Fold assignment for each score

        Returns:
            Cross-fitted calibrated scores
        """
        if not self._fold_models:
            raise RuntimeError("Must call fit_cv before predict_oof")

        judge_scores = np.asarray(judge_scores)
        fold_ids = np.asarray(fold_ids)

        predictions = np.zeros_like(judge_scores)
        for fold_id, model in self._fold_models.items():
            fold_mask = fold_ids == fold_id
            if np.any(fold_mask):
                predictions[fold_mask] = model.predict(judge_scores[fold_mask])

        return predictions


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
