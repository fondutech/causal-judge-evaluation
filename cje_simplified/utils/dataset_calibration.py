"""Dataset calibration utilities.

This module provides functions to calibrate datasets with judge scores
to match oracle labels, creating calibrated rewards for CJE analysis.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from ..data.models import Dataset, Sample
from ..data.loaders import DatasetLoader, InMemoryDataSource
from .judge_calibration import JudgeCalibrator, CalibrationResult


def calibrate_dataset(
    dataset: Dataset,
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    k_folds: int = 5,
) -> Tuple[Dataset, CalibrationResult]:
    """Calibrate judge scores in a dataset to match oracle labels.

    This function extracts judge scores and oracle labels from the dataset,
    calibrates the judge scores to match the oracle distribution, and returns
    a new dataset with calibrated rewards.

    Args:
        dataset: Dataset containing judge scores and oracle labels
        judge_field: Field name in metadata containing judge scores
        oracle_field: Field name in metadata containing oracle labels
        k_folds: Number of cross-fitting folds for calibration

    Returns:
        Tuple of (calibrated_dataset, calibration_result)

    Example:
        >>> # Load dataset with judge scores
        >>> dataset = load_dataset_from_jsonl("data.jsonl", reward_field="judge_score")
        >>>
        >>> # Calibrate judge scores to oracle labels
        >>> calibrated_dataset, stats = calibrate_dataset(
        ...     dataset,
        ...     judge_field="judge_score",
        ...     oracle_field="oracle_label"
        ... )
    """
    # Extract judge scores and oracle labels
    judge_scores = []
    oracle_labels = []
    oracle_mask = []

    for sample in dataset.samples:
        # Look for judge score in metadata first, then as reward
        if judge_field in sample.metadata:
            judge_score = sample.metadata[judge_field]
        elif judge_field == "reward":
            judge_score = sample.reward
        else:
            raise ValueError(f"Judge field '{judge_field}' not found in sample")

        judge_scores.append(float(judge_score))

        # Look for oracle label
        if oracle_field in sample.metadata:
            oracle_labels.append(float(sample.metadata[oracle_field]))
            oracle_mask.append(True)
        else:
            oracle_mask.append(False)

    # Convert to arrays
    judge_scores_array = np.array(judge_scores)
    oracle_labels_array = np.array(oracle_labels) if oracle_labels else None
    oracle_mask_array = np.array(oracle_mask)

    if not np.any(oracle_mask_array):
        raise ValueError(f"No oracle labels found in field '{oracle_field}'")

    # Calibrate judge scores
    calibrator = JudgeCalibrator(k_folds=k_folds)
    result = calibrator.fit_transform(
        judge_scores_array, oracle_labels_array, oracle_mask_array
    )

    # Create new samples with calibrated rewards
    calibrated_samples = []
    for i, sample in enumerate(dataset.samples):
        # Create new sample with calibrated reward
        new_metadata = sample.metadata.copy()
        new_metadata[judge_field] = judge_scores[i]  # Preserve original
        if oracle_mask[i]:
            new_metadata[oracle_field] = oracle_labels[oracle_mask_array][
                : np.sum(oracle_mask_array[: i + 1])
            ][-1]

        calibrated_sample = Sample(
            prompt=sample.prompt,
            response=sample.response,
            reward=float(result.calibrated_scores[i]),
            base_policy_logprob=sample.base_policy_logprob,
            target_policy_logprobs=sample.target_policy_logprobs,
            metadata=new_metadata,
        )
        calibrated_samples.append(calibrated_sample)

    # Create new dataset
    calibrated_dataset = Dataset(
        samples=calibrated_samples,
        target_policies=dataset.target_policies,
        metadata=dataset.metadata.copy(),
    )

    return calibrated_dataset, result


def calibrate_from_raw_data(
    data: List[Dict[str, Any]],
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    reward_field: str = "reward",
    k_folds: int = 5,
) -> Tuple[List[Dict[str, Any]], CalibrationResult]:
    """Calibrate judge scores in raw data to create calibrated rewards.

    This is a lower-level function that works with raw dictionaries
    instead of Dataset objects.

    Args:
        data: List of dictionaries containing judge scores and oracle labels
        judge_field: Field name containing judge scores
        oracle_field: Field name containing oracle labels
        reward_field: Field name to store calibrated rewards
        k_folds: Number of cross-fitting folds

    Returns:
        Tuple of (calibrated_data, calibration_result)
    """
    # Extract judge scores and oracle labels
    judge_scores = []
    oracle_labels = []
    oracle_mask = []

    for record in data:
        # Extract judge score
        judge_score = record.get(judge_field)
        if judge_score is None:
            raise ValueError(f"Judge field '{judge_field}' not found in record")

        if isinstance(judge_score, dict):
            judge_score = judge_score.get("mean", judge_score.get("value"))
        judge_scores.append(float(judge_score))

        # Check for oracle label
        oracle_label = record.get(oracle_field)
        if oracle_label is not None:
            oracle_labels.append(float(oracle_label))
            oracle_mask.append(True)
        else:
            oracle_mask.append(False)

    # Convert to arrays
    judge_scores_array = np.array(judge_scores)
    oracle_labels_array = np.array(oracle_labels) if oracle_labels else None
    oracle_mask_array = np.array(oracle_mask)

    if not np.any(oracle_mask_array):
        raise ValueError(f"No oracle labels found in field '{oracle_field}'")

    # Calibrate judge scores
    calibrator = JudgeCalibrator(k_folds=k_folds)
    result = calibrator.fit_transform(
        judge_scores_array, oracle_labels_array, oracle_mask_array
    )

    # Add calibrated rewards to data
    calibrated_data = []
    for i, record in enumerate(data):
        record_copy = record.copy()
        record_copy[reward_field] = float(result.calibrated_scores[i])

        # Preserve original fields in metadata if it exists
        metadata = record_copy.get("metadata", {})
        metadata[judge_field] = judge_scores[i]
        if oracle_mask[i]:
            metadata[oracle_field] = oracle_labels[oracle_mask_array][
                : np.sum(oracle_mask_array[: i + 1])
            ][-1]
        record_copy["metadata"] = metadata

        calibrated_data.append(record_copy)

    return calibrated_data, result
