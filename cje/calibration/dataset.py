"""Dataset calibration utilities.

This module provides functions to calibrate datasets with judge scores
to match oracle labels, creating calibrated rewards for CJE analysis.
"""

from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy
import numpy as np
from ..data.models import Dataset, Sample
from .judge import JudgeCalibrator, CalibrationResult


def calibrate_dataset(
    dataset: Dataset,
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    enable_cross_fit: bool = False,
    n_folds: int = 5,
) -> Tuple[Dataset, CalibrationResult]:
    """Calibrate judge scores in a dataset to match oracle labels.

    This function extracts judge scores and oracle labels from the dataset,
    calibrates the judge scores to match the oracle distribution, and returns
    a new dataset with calibrated rewards.

    Args:
        dataset: Dataset containing judge scores and oracle labels
        judge_field: Field name in metadata containing judge scores
        oracle_field: Field name in metadata containing oracle labels
        enable_cross_fit: If True, also fits cross-fitted models for DR
        n_folds: Number of CV folds (only used if enable_cross_fit=True)

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
    # Extract judge scores, oracle labels, and prompt_ids
    judge_scores = []
    oracle_labels = []
    oracle_mask = []
    prompt_ids = []

    # Forbid judge_field="reward" to avoid confusion
    if judge_field == "reward":
        raise ValueError(
            "judge_field='reward' is not allowed to avoid confusion between "
            "raw and calibrated values. Use a different field name in metadata."
        )

    for sample in dataset.samples:
        # Look for judge score in metadata
        if judge_field not in sample.metadata:
            raise ValueError(
                f"Judge field '{judge_field}' not found in sample metadata"
            )

        judge_score = sample.metadata[judge_field]
        judge_scores.append(float(judge_score))
        prompt_ids.append(sample.prompt_id)

        # Look for oracle label
        if (
            oracle_field in sample.metadata
            and sample.metadata[oracle_field] is not None
        ):
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
    calibrator = JudgeCalibrator()
    if enable_cross_fit:
        # Use cross-fitted calibration for DR support
        # Pass prompt_ids to enable unified fold system
        result = calibrator.fit_cv(
            judge_scores_array,
            oracle_labels_array,
            oracle_mask_array,
            n_folds,
            prompt_ids=prompt_ids,
        )
    else:
        # Use standard calibration (backward compatible)
        result = calibrator.fit_transform(
            judge_scores_array, oracle_labels_array, oracle_mask_array
        )

    # Create new samples with calibrated rewards
    calibrated_samples = []
    oracle_idx = 0
    for i, sample in enumerate(dataset.samples):
        # Create new sample with calibrated reward
        new_metadata = sample.metadata.copy()
        new_metadata[judge_field] = judge_scores[i]  # Preserve original
        if oracle_mask[i]:
            new_metadata[oracle_field] = oracle_labels[oracle_idx]
            oracle_idx += 1

        # Note: We no longer store cv_fold in metadata
        # Folds are computed on-demand from prompt_id using the unified system

        calibrated_sample = Sample(
            prompt_id=sample.prompt_id,
            prompt=sample.prompt,
            response=sample.response,
            reward=float(result.calibrated_scores[i]),
            base_policy_logprob=sample.base_policy_logprob,
            target_policy_logprobs=sample.target_policy_logprobs,
            metadata=new_metadata,
        )
        calibrated_samples.append(calibrated_sample)

    # Create new dataset with calibration info in metadata
    dataset_metadata = dataset.metadata.copy()

    # Add calibration summary for downstream diagnostics
    dataset_metadata["calibration_info"] = {
        "rmse": result.calibration_rmse,
        "coverage": result.coverage_at_01,
        "n_oracle": result.n_oracle,
        "n_total": len(judge_scores),
        "method": "cross_fitted_isotonic" if enable_cross_fit else "isotonic",
        "n_folds": n_folds if enable_cross_fit else None,
        "oof_rmse": result.oof_rmse if enable_cross_fit else None,
        "oof_coverage": result.oof_coverage_at_01 if enable_cross_fit else None,
    }

    calibrated_dataset = Dataset(
        samples=calibrated_samples,
        target_policies=dataset.target_policies,
        metadata=dataset_metadata,
    )

    return calibrated_dataset, result


def calibrate_from_raw_data(
    data: List[Dict[str, Any]],
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    reward_field: str = "reward",
) -> Tuple[List[Dict[str, Any]], CalibrationResult]:
    """Calibrate judge scores in raw data to create calibrated rewards.

    This is a lower-level function that works with raw dictionaries
    instead of Dataset objects.

    Args:
        data: List of dictionaries containing judge scores and oracle labels
        judge_field: Field name containing judge scores
        oracle_field: Field name containing oracle labels
        reward_field: Field name to store calibrated rewards

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
    calibrator = JudgeCalibrator()
    result = calibrator.fit_transform(
        judge_scores_array, oracle_labels_array, oracle_mask_array
    )

    # Add calibrated rewards to data
    calibrated_data = []
    for i, record in enumerate(data):
        record_copy = record.copy()
        record_copy[reward_field] = float(result.calibrated_scores[i])

        # Deep copy metadata to avoid mutating caller's nested dict
        metadata = deepcopy(record_copy.get("metadata", {}))
        metadata[judge_field] = judge_scores[i]
        if oracle_mask[i]:
            # Find the index of this oracle label
            oracle_idx = np.sum(oracle_mask_array[: i + 1]) - 1
            metadata[oracle_field] = (
                float(oracle_labels_array[oracle_idx])
                if oracle_labels_array is not None
                else None
            )
        record_copy["metadata"] = metadata

        calibrated_data.append(record_copy)

    return calibrated_data, result
