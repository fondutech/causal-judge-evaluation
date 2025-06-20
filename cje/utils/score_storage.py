"""Utilities for storing and loading judge scores in unified format.

This module handles the transition from float scores to structured
JudgeScore objects in storage (JSONL files).
"""

import json
from typing import Dict, List, Any, Union, Optional
from pathlib import Path

from ..judge.schemas import JudgeScore, score_to_float, float_to_score


def serialize_score(score: Union[float, JudgeScore]) -> Union[float, Dict[str, float]]:
    """Serialize a score for storage.

    Args:
        score: Either a float (legacy) or JudgeScore

    Returns:
        Either float (for legacy compatibility) or dict with mean/variance
    """
    if isinstance(score, (int, float)):
        # Legacy format - just the float
        return float(score)
    elif isinstance(score, JudgeScore):
        # New format - structured data
        return {"mean": score.mean, "variance": score.variance}
    else:
        raise TypeError(f"Cannot serialize score of type {type(score)}")


def deserialize_score(data: Union[float, Dict[str, float]]) -> JudgeScore:
    """Deserialize a score from storage.

    Args:
        data: Either float (legacy) or dict with mean/variance

    Returns:
        JudgeScore object
    """
    if isinstance(data, (int, float)):
        # Legacy format - assume zero variance
        return JudgeScore(mean=float(data), variance=0.0)
    elif isinstance(data, dict):
        # New format
        return JudgeScore(
            mean=float(data["mean"]), variance=float(data.get("variance", 0.0))
        )
    else:
        raise TypeError(f"Cannot deserialize score from {type(data)}")


def update_row_with_score(
    row: Dict[str, Any], score: Union[float, JudgeScore], score_field: str = "score_raw"
) -> Dict[str, Any]:
    """Update a data row with a judge score.

    Maintains backward compatibility by storing both old and new formats.

    Args:
        row: Data row to update
        score: The judge score
        score_field: Field name for the score (default: "score_raw")

    Returns:
        Updated row
    """
    new_row = row.copy()

    if isinstance(score, JudgeScore):
        # Store structured format
        new_row[score_field] = serialize_score(score)
        # Also store float for backward compatibility
        new_row[f"{score_field}_float"] = float(score.mean)
        # Store variance separately for easy access
        new_row[f"{score_field}_variance"] = score.variance
    else:
        # Legacy float score
        new_row[score_field] = float(score)
        new_row[f"{score_field}_float"] = float(score)
        new_row[f"{score_field}_variance"] = 0.0

    return new_row


def extract_score_from_row(
    row: Dict[str, Any], score_field: str = "score_raw"
) -> JudgeScore:
    """Extract a JudgeScore from a data row.

    Handles both legacy float format and new structured format.

    Args:
        row: Data row
        score_field: Field name for the score

    Returns:
        JudgeScore object
    """
    if score_field not in row:
        # Try alternative field names
        if f"{score_field}_float" in row:
            # New format with separate fields
            return JudgeScore(
                mean=float(row[f"{score_field}_float"]),
                variance=float(row.get(f"{score_field}_variance", 0.0)),
            )
        else:
            raise KeyError(f"Score field '{score_field}' not found in row")

    # Deserialize from stored format
    return deserialize_score(row[score_field])


def migrate_jsonl_to_unified(
    input_path: Path,
    output_path: Optional[Path] = None,
    score_fields: List[str] = ["score_raw", "score_cal"],
) -> None:
    """Migrate a JSONL file to unified score format.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file (default: overwrite input)
        score_fields: List of score fields to migrate
    """
    if output_path is None:
        output_path = input_path

    # Read all rows
    rows = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # Migrate each row
    migrated_rows = []
    for row in rows:
        new_row = row.copy()

        for field in score_fields:
            if field in row:
                # Extract old score
                old_score = row[field]

                # Convert to JudgeScore
                if isinstance(old_score, dict) and "mean" in old_score:
                    # Already migrated
                    continue

                # Create JudgeScore and update row
                judge_score = float_to_score(float(old_score))
                new_row = update_row_with_score(new_row, judge_score, field)

        migrated_rows.append(new_row)

    # Write back
    with open(output_path, "w") as f:
        for row in migrated_rows:
            f.write(json.dumps(row) + "\n")


def is_unified_format(row: Dict[str, Any], score_field: str = "score_raw") -> bool:
    """Check if a row uses unified score format.

    Args:
        row: Data row
        score_field: Field to check

    Returns:
        True if using unified format (dict with mean/variance)
    """
    if score_field not in row:
        return False

    score_data = row[score_field]
    return isinstance(score_data, dict) and "mean" in score_data


class ScoreCompatibilityLayer:
    """Compatibility layer for transitioning to unified scores.

    Provides transparent access to scores regardless of storage format.
    """

    @staticmethod
    def get_score_value(row: Dict[str, Any], field: str = "score_raw") -> float:
        """Get score value (mean) from any format.

        Args:
            row: Data row
            field: Score field name

        Returns:
            Score mean value as float
        """
        if f"{field}_float" in row:
            # Prefer explicit float field if available
            return float(row[f"{field}_float"])

        score = extract_score_from_row(row, field)
        return float(score.mean)

    @staticmethod
    def get_score_variance(row: Dict[str, Any], field: str = "score_raw") -> float:
        """Get score variance from any format.

        Args:
            row: Data row
            field: Score field name

        Returns:
            Score variance (0 for legacy scores)
        """
        if f"{field}_variance" in row:
            # Prefer explicit variance field if available
            return float(row[f"{field}_variance"])

        score = extract_score_from_row(row, field)
        return float(score.variance)

    @staticmethod
    def get_score(row: Dict[str, Any], field: str = "score_raw") -> JudgeScore:
        """Get full JudgeScore from any format.

        Args:
            row: Data row
            field: Score field name

        Returns:
            JudgeScore object
        """
        return extract_score_from_row(row, field)
