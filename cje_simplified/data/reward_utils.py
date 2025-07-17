"""Utilities for creating calibrated rewards from judge scores and oracle labels.

This module helps prepare data for CJE by converting raw judge scores to
calibrated rewards that align with business KPIs.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

from ..utils.judge_calibration import JudgeCalibrator


def create_calibrated_rewards(
    data: Union[List[Dict], str],
    oracle_labels: Optional[np.ndarray] = None,
    oracle_mask: Optional[np.ndarray] = None,
    judge_score_field: str = "judge_score",
    oracle_label_field: Optional[str] = "oracle_label",
    output_reward_field: str = "reward",
    k_folds: int = 5,
    inplace: bool = False,
) -> Tuple[List[Dict], Dict[str, float]]:
    """Create calibrated rewards from judge scores using oracle labels.

    This is the main utility for preparing data for CJE. It calibrates
    raw judge scores to match oracle KPIs, creating the 'reward' field
    needed by PrecomputedSampler.

    Args:
        data: Either a list of dicts or path to JSONL file
        oracle_labels: Array of oracle labels (if not in data)
        oracle_mask: Boolean mask for which samples have oracle labels
        judge_score_field: Field name containing raw judge scores
        oracle_label_field: Field name containing oracle labels (if in data)
        output_reward_field: Field name to store calibrated rewards
        k_folds: Number of cross-fitting folds for calibration
        inplace: If True, modify data in place. If False, create copies.

    Returns:
        Tuple of (data_with_rewards, calibration_stats)

    Example 1 - Oracle labels in separate array:
        data_with_rewards, stats = create_calibrated_rewards(
            data=raw_data,
            oracle_labels=kpi_values[:1000],  # First 1000 have labels
            judge_score_field="llm_score"
        )

    Example 2 - Oracle labels in data:
        data_with_rewards, stats = create_calibrated_rewards(
            "data.jsonl",
            oracle_label_field="human_rating",
            judge_score_field="gpt4_score"
        )
    """
    # Load data if path provided
    if isinstance(data, str):
        data = load_jsonl(data)

    # Create copies if not inplace
    if not inplace:
        data = [dict(record) for record in data]

    # Extract judge scores
    judge_scores = extract_field(data, judge_score_field)

    # Handle oracle labels from different sources
    if oracle_label_field and oracle_label_field in data[0]:
        # Oracle labels are in the data
        oracle_records = [
            r
            for r in data
            if oracle_label_field in r and r[oracle_label_field] is not None
        ]
        if oracle_records:
            oracle_indices = [
                i
                for i, r in enumerate(data)
                if oracle_label_field in r and r[oracle_label_field] is not None
            ]
            oracle_labels = np.array([r[oracle_label_field] for r in oracle_records])
            oracle_mask = np.zeros(len(data), dtype=bool)
            oracle_mask[oracle_indices] = True

    # Calibrate judge scores
    calibrator = JudgeCalibrator(k_folds=k_folds)
    result = calibrator.fit_transform(judge_scores, oracle_labels, oracle_mask)

    # Add calibrated rewards to data
    for i, record in enumerate(data):
        record[output_reward_field] = float(result.calibrated_scores[i])

    # Prepare calibration statistics
    stats = {
        "rmse": result.calibration_rmse,
        "coverage": result.coverage_at_01,
        "n_oracle": result.n_oracle,
        "n_total": len(data),
    }

    return data, stats


def extract_field(
    data: List[Dict[str, Any]], field: str, default: Optional[float] = None
) -> np.ndarray:
    """Extract a field from list of dicts, handling nested formats.

    Args:
        data: List of dictionaries
        field: Field name to extract
        default: Default value if field missing

    Returns:
        Numpy array of extracted values
    """
    values = []
    for record in data:
        value = record.get(field, default)

        # Handle nested score format
        if isinstance(value, dict):
            value = value.get("mean", value.get("value", default))

        if value is None and default is not None:
            value = default

        values.append(float(value))

    return np.array(values)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    with open(file_path, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")


def prepare_cje_data(
    raw_data_path: str,
    oracle_data_path: Optional[str] = None,
    output_path: str = "cje_ready_data.jsonl",
    judge_score_field: str = "judge_score",
    oracle_fraction: float = 0.25,
    k_folds: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """End-to-end data preparation for CJE.

    This utility handles the common case of preparing data with a
    subset of oracle labels for calibration.

    Args:
        raw_data_path: Path to JSONL with judge scores and log probs
        oracle_data_path: Optional separate file with oracle labels
        output_path: Where to save CJE-ready data
        judge_score_field: Field containing raw judge scores
        oracle_fraction: Fraction of data to use for calibration
        k_folds: Cross-fitting folds for calibration
        random_seed: Random seed for oracle subset selection

    Returns:
        Dictionary with preparation statistics

    Example:
        stats = prepare_cje_data(
            "raw_responses_with_scores.jsonl",
            oracle_data_path="human_labels_subset.jsonl",
            output_path="ready_for_cje.jsonl"
        )

        # Now ready to use:
        sampler = PrecomputedSampler.from_jsonl("ready_for_cje.jsonl")
    """
    # Load raw data
    data = load_jsonl(raw_data_path)
    n_total = len(data)

    # Load oracle labels if provided separately
    oracle_labels = None
    oracle_mask = None

    if oracle_data_path:
        oracle_data = load_jsonl(oracle_data_path)
        # Assume oracle data has matching order or has IDs to match
        # This is a simplified version - real implementation would match by ID
        n_oracle = min(len(oracle_data), n_total)
        oracle_labels = extract_field(oracle_data[:n_oracle], "label")
        oracle_mask = np.zeros(n_total, dtype=bool)
        oracle_mask[:n_oracle] = True
    else:
        # Select random subset for oracle labeling
        np.random.seed(random_seed)
        n_oracle = int(n_total * oracle_fraction)
        oracle_indices = np.random.choice(n_total, n_oracle, replace=False)
        oracle_mask = np.zeros(n_total, dtype=bool)
        oracle_mask[oracle_indices] = True

        # In this case, assume oracle labels are already in data
        # or will be added before calling this function

    # Create calibrated rewards
    data_with_rewards, cal_stats = create_calibrated_rewards(
        data,
        oracle_labels=oracle_labels,
        oracle_mask=oracle_mask,
        judge_score_field=judge_score_field,
        k_folds=k_folds,
    )

    # Validate all required fields
    required_fields = ["prompt", "response", "reward", "total_logprob", "target_logps"]
    sample = data_with_rewards[0]
    missing = [f for f in required_fields if f not in sample]
    if missing:
        raise ValueError(f"Data missing required fields for CJE: {missing}")

    # Save prepared data
    save_jsonl(data_with_rewards, output_path)

    # Return statistics
    return {
        "n_total": n_total,
        "n_oracle": cal_stats["n_oracle"],
        "calibration_rmse": cal_stats["rmse"],
        "calibration_coverage": cal_stats["coverage"],
        "output_path": output_path,
        "ready_for_cje": True,
    }


def add_rewards_to_existing_data(
    data_path: str,
    calibrator: JudgeCalibrator,
    judge_score_field: str = "judge_score",
    output_path: Optional[str] = None,
    output_reward_field: str = "reward",
) -> str:
    """Add calibrated rewards to existing data using pre-fitted calibrator.

    Useful when you've already calibrated on one dataset and want to
    apply the same calibration to new data.

    Args:
        data_path: Path to JSONL file with judge scores
        calibrator: Pre-fitted JudgeCalibrator
        judge_score_field: Field containing judge scores
        output_path: Where to save (defaults to data_path with .rewards suffix)
        output_reward_field: Field name for calibrated rewards

    Returns:
        Path to output file
    """
    # Load data
    data = load_jsonl(data_path)

    # Extract and calibrate scores
    judge_scores = extract_field(data, judge_score_field)
    calibrated_rewards = calibrator.transform(judge_scores)

    # Add rewards to data
    for i, record in enumerate(data):
        record[output_reward_field] = float(calibrated_rewards[i])

    # Save
    if output_path is None:
        path = Path(data_path)
        output_path = str(path.parent / f"{path.stem}.rewards{path.suffix}")

    save_jsonl(data, output_path)
    return output_path
