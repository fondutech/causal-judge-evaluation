"""Dataset calibration utilities.

This module provides functions to calibrate datasets with judge scores
to match oracle labels, creating calibrated rewards for CJE analysis.
"""

from typing import Dict, List, Any, Optional, Tuple, Literal, cast
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
    calibration_mode: Optional[str] = None,
    random_seed: int = 42,
) -> Tuple[Dataset, CalibrationResult]:
    """Calibrate judge scores in a dataset to match oracle labels.

    This function extracts judge scores and oracle labels from the dataset,
    calibrates the judge scores to match the oracle distribution, and returns
    a new dataset with calibrated rewards. By default, uses auto mode to
    automatically select between monotone and flexible calibration.

    Args:
        dataset: Dataset containing judge scores and oracle labels
        judge_field: Field name in metadata containing judge scores
        oracle_field: Field name in metadata containing oracle labels
        enable_cross_fit: If True, also fits cross-fitted models for DR
        n_folds: Number of CV folds (only used if enable_cross_fit=True)
        calibration_mode: Calibration mode ('auto', 'monotone', 'two_stage').
                         If None, defaults to 'auto' for cross-fit, 'monotone' otherwise.

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
            oracle_labels.append(np.nan)  # Placeholder for missing oracle
            oracle_mask.append(False)

    # Convert to arrays
    judge_scores_array = np.array(judge_scores)
    oracle_labels_array = np.array(
        oracle_labels
    )  # Now always same length as judge_scores
    oracle_mask_array = np.array(oracle_mask)

    if not np.any(oracle_mask_array):
        raise ValueError(f"No oracle labels found in field '{oracle_field}'")

    # Determine calibration mode
    if calibration_mode is None:
        # Default to auto for cross-fit (better for DR), monotone otherwise
        calibration_mode = "auto" if enable_cross_fit else "monotone"

    # Calibrate judge scores
    calibrator = JudgeCalibrator(
        calibration_mode=cast(
            Literal["monotone", "two_stage", "auto"], calibration_mode
        ),
        random_seed=random_seed,
    )
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
        "method": (
            "cross_fitted_isotonic" if enable_cross_fit else "isotonic"
        ),  # Will be updated below
        "n_folds": n_folds if enable_cross_fit else None,
        "oof_rmse": result.oof_rmse if enable_cross_fit else None,
        "oof_coverage": result.oof_coverage_at_01 if enable_cross_fit else None,
        "calibration_mode": calibration_mode,
    }

    # Store fold configuration for reproducibility
    dataset_metadata["n_folds"] = n_folds
    dataset_metadata["fold_seed"] = random_seed

    # Store selected calibration mode and update method field
    selected_mode: Optional[str] = calibration_mode  # Default to the requested mode
    if (
        hasattr(calibrator, "_flexible_calibrator")
        and calibrator._flexible_calibrator is not None
    ):
        selected_mode = calibrator._flexible_calibrator.selected_mode
        if selected_mode is not None:
            dataset_metadata["calibration_info"]["selected_mode"] = selected_mode
    elif calibration_mode == "auto" and hasattr(calibrator, "selected_mode"):
        selected_mode = calibrator.selected_mode
        if selected_mode is not None:
            dataset_metadata["calibration_info"]["selected_mode"] = selected_mode

    # Update method field to reflect actual calibration mode used
    if selected_mode:
        dataset_metadata["calibration_info"]["method"] = (
            f"cross_fitted_{selected_mode}" if enable_cross_fit else selected_mode
        )

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
    calibration_mode: Optional[Literal["auto", "monotone", "two_stage"]] = "monotone",
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], CalibrationResult]:
    """Calibrate judge scores in raw data to create calibrated rewards.

    This is a lower-level function that works with raw dictionaries
    instead of Dataset objects.

    Args:
        data: List of dictionaries containing judge scores and oracle labels
        judge_field: Field name containing judge scores
        oracle_field: Field name containing oracle labels
        reward_field: Field name to store calibrated rewards
        calibration_mode: Calibration mode ('auto', 'monotone', 'two_stage').
                         Defaults to 'monotone' for backward compatibility.

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
    oracle_labels_array = np.array(
        oracle_labels
    )  # Now always same length as judge_scores
    oracle_mask_array = np.array(oracle_mask)

    if not np.any(oracle_mask_array):
        raise ValueError(f"No oracle labels found in field '{oracle_field}'")

    # Calibrate judge scores
    calibrator = JudgeCalibrator(
        calibration_mode=cast(
            Literal["monotone", "two_stage", "auto"], calibration_mode
        ),
        random_seed=random_seed,
    )
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
