"""Dataset factory for coordinating loading and calibration.

This module follows SOLID principles by using dependency injection
and separating concerns into focused classes.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .models import Dataset
from .loaders import DatasetLoader, DataSource, JsonlDataSource, InMemoryDataSource
from ..utils.judge_calibration import JudgeCalibrator


class DatasetFactory:
    """Factory for creating Datasets with optional judge calibration.

    Follows SOLID principles:
    - Single Responsibility: Coordinates loading and calibration
    - Open/Closed: Easy to extend with new loaders or calibrators
    - Dependency Injection: Takes loader and calibrator as dependencies
    """

    def __init__(
        self,
        loader: Optional[DatasetLoader] = None,
        calibrator: Optional[JudgeCalibrator] = None,
    ):
        """Initialize factory with optional custom loader and calibrator.

        Args:
            loader: DatasetLoader instance. If None, uses default.
            calibrator: JudgeCalibrator instance. If None, creates one when needed.
        """
        self.loader = loader or DatasetLoader()
        self.calibrator = calibrator

    def create_from_jsonl(
        self, file_path: str, target_policies: Optional[List[str]] = None
    ) -> Dataset:
        """Create Dataset from JSONL file.

        Args:
            file_path: Path to JSONL file
            target_policies: Optional list of target policy names

        Returns:
            Dataset instance
        """
        source = JsonlDataSource(file_path)
        return self.loader.load_from_source(source, target_policies)

    def create_from_data(
        self, data: List[Dict[str, Any]], target_policies: Optional[List[str]] = None
    ) -> Dataset:
        """Create Dataset from in-memory data.

        Args:
            data: List of dictionaries with data
            target_policies: Optional list of target policy names

        Returns:
            Dataset instance
        """
        source = InMemoryDataSource(data)
        return self.loader.load_from_source(source, target_policies)

    def create_with_calibration(
        self,
        source: DataSource,
        judge_score_field: str = "judge_score",
        oracle_label_field: Optional[str] = "oracle_label",
        k_folds: int = 5,
        target_policies: Optional[List[str]] = None,
    ) -> Tuple[Dataset, Dict[str, float]]:
        """Create Dataset with judge score calibration.

        Args:
            source: Data source to load from
            judge_score_field: Field containing raw judge scores
            oracle_label_field: Field containing oracle labels
            k_folds: Number of cross-fitting folds for calibration
            target_policies: List of target policy names

        Returns:
            Tuple of (Dataset instance, calibration_stats)
        """
        # Load raw data
        raw_data = source.load()

        # Extract judge scores and oracle labels
        judge_scores, oracle_labels, oracle_mask = self._extract_calibration_data(
            raw_data, judge_score_field, oracle_label_field
        )

        # Calibrate judge scores
        calibrator = self.calibrator or JudgeCalibrator(k_folds=k_folds)
        result = calibrator.fit_transform(judge_scores, oracle_labels, oracle_mask)

        # Add calibrated rewards to data
        calibrated_data = self._add_rewards_to_data(
            raw_data, result.calibrated_scores, judge_score_field, oracle_label_field
        )

        # Create Dataset from calibrated data
        calibrated_source = InMemoryDataSource(calibrated_data)
        dataset = self.loader.load_from_source(calibrated_source, target_policies)

        # Prepare calibration statistics
        stats = {
            "rmse": result.calibration_rmse,
            "coverage": result.coverage_at_01,
            "n_oracle": result.n_oracle,
            "n_total": len(raw_data),
        }

        return dataset, stats

    def create_from_jsonl_with_calibration(
        self,
        file_path: str,
        judge_score_field: str = "judge_score",
        oracle_label_field: Optional[str] = "oracle_label",
        k_folds: int = 5,
        target_policies: Optional[List[str]] = None,
    ) -> Tuple[Dataset, Dict[str, float]]:
        """Create Dataset from JSONL with calibration.

        Args:
            file_path: Path to JSONL file with raw judge scores
            judge_score_field: Field containing raw judge scores
            oracle_label_field: Field containing oracle labels
            k_folds: Number of cross-fitting folds for calibration
            target_policies: List of target policy names

        Returns:
            Tuple of (Dataset instance, calibration_stats)
        """
        source = JsonlDataSource(file_path)
        return self.create_with_calibration(
            source, judge_score_field, oracle_label_field, k_folds, target_policies
        )

    def create_from_data_with_calibration(
        self,
        data: List[Dict[str, Any]],
        judge_score_field: str = "judge_score",
        oracle_label_field: Optional[str] = "oracle_label",
        k_folds: int = 5,
        target_policies: Optional[List[str]] = None,
    ) -> Tuple[Dataset, Dict[str, float]]:
        """Create Dataset from in-memory data with calibration.

        Args:
            data: List of dictionaries with raw judge scores
            judge_score_field: Field containing raw judge scores
            oracle_label_field: Field containing oracle labels
            k_folds: Number of cross-fitting folds for calibration
            target_policies: List of target policy names

        Returns:
            Tuple of (Dataset instance, calibration_stats)
        """
        source = InMemoryDataSource(data)
        return self.create_with_calibration(
            source, judge_score_field, oracle_label_field, k_folds, target_policies
        )

    def _extract_calibration_data(
        self,
        data: List[Dict[str, Any]],
        judge_score_field: str,
        oracle_label_field: Optional[str],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Extract judge scores and oracle labels from raw data."""
        judge_scores = []
        oracle_labels = []
        oracle_mask = []

        for record in data:
            # Extract judge score
            judge_score = record.get(judge_score_field)
            if isinstance(judge_score, dict):
                judge_score = judge_score.get("mean", judge_score.get("value"))
            judge_scores.append(float(judge_score))

            # Check for oracle label
            oracle_label = (
                record.get(oracle_label_field) if oracle_label_field else None
            )
            if oracle_label is not None:
                oracle_labels.append(float(oracle_label))
                oracle_mask.append(True)
            else:
                oracle_mask.append(False)

        judge_scores_array = np.array(judge_scores)
        oracle_labels_array = np.array(oracle_labels) if oracle_labels else None
        oracle_mask_array = np.array(oracle_mask)

        return judge_scores_array, oracle_labels_array, oracle_mask_array

    def _add_rewards_to_data(
        self,
        data: List[Dict[str, Any]],
        calibrated_scores: np.ndarray,
        judge_score_field: str,
        oracle_label_field: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Add calibrated rewards to raw data."""
        calibrated_data = []
        for i, record in enumerate(data):
            record_copy = record.copy()
            record_copy["reward"] = float(calibrated_scores[i])

            # Preserve original fields in metadata
            metadata = record_copy.get("metadata", {})
            metadata["judge_score"] = record.get(judge_score_field)
            if oracle_label_field and oracle_label_field in record:
                metadata["oracle_label"] = record.get(oracle_label_field)
            record_copy["metadata"] = metadata

            calibrated_data.append(record_copy)

        return calibrated_data


# Convenience factory instance with default configuration
default_factory = DatasetFactory()
