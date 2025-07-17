"""Utilities for judge calibration and data preparation.

This module provides utility functions for working with calibrated rewards.
Most data loading and calibration functionality has been moved to the Dataset class.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..utils.judge_calibration import JudgeCalibrator


def save_jsonl(data: List[Dict], file_path: str) -> None:
    """Save data to JSONL file."""
    with open(file_path, "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")


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
    # Load data using Dataset for consistency
    from .models import Dataset

    # Load with Dataset to get type safety, then extract raw data
    dataset = Dataset.from_jsonl(data_path)

    # Get judge scores - look for them in metadata first, then try the field
    judge_scores = []
    for sample in dataset.samples:
        if judge_score_field in sample.metadata:
            score = sample.metadata[judge_score_field]
        else:
            # This is a fallback - ideally scores should be in metadata
            score = None

        if score is None:
            raise ValueError(
                f"Judge score field '{judge_score_field}' not found in sample metadata"
            )

        if isinstance(score, dict):
            score = score.get("mean", score.get("value"))
        judge_scores.append(float(score))

    judge_scores = np.array(judge_scores)

    # Apply calibration
    calibrated_rewards = calibrator.transform(judge_scores)

    # Convert back to dict format and add rewards
    data = []
    for i, sample in enumerate(dataset.samples):
        record = {
            "prompt": sample.prompt,
            "response": sample.response,
            "base_policy_logprob": sample.base_policy_logprob,
            "target_policy_logprobs": sample.target_policy_logprobs,
            "metadata": sample.metadata,
            output_reward_field: float(calibrated_rewards[i]),
        }
        data.append(record)

    # Save
    if output_path is None:
        path = Path(data_path)
        output_path = str(path.parent / f"{path.stem}.rewards{path.suffix}")

    save_jsonl(data, output_path)
    return output_path
