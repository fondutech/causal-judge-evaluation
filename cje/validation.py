"""
Simplified validation utilities for CJE.

This module provides essential validation functions for the CJE pipeline.
"""

import logging
from typing import List, Dict, Any
from .utils.error_handling import ValidationError, require_not_empty

logger = logging.getLogger(__name__)


def validate_core_fields(
    rows: List[Dict[str, Any]], source_description: str = "data"
) -> None:
    """
    Validate that core fields required by estimators are present.

    Args:
        rows: List of data rows to validate
        source_description: Description of data source for error messages

    Raises:
        ValidationError: If required fields are missing
    """
    require_not_empty(rows, source_description)

    core_fields = ["logp", "context", "response"]

    for i, row in enumerate(rows):
        for field in core_fields:
            if field not in row:
                error_msg = f"Row {i} in {source_description} is missing essential field '{field}'."
                logger.error(error_msg)
                raise ValidationError(error_msg)


def validate_response_reward_correspondence(
    rows: List[Dict[str, Any]], source_description: str = "data"
) -> None:
    """
    Validate that responses and rewards correspond correctly.

    Args:
        rows: List of data rows to validate
        source_description: Description of data source for error messages

    Raises:
        ValidationError: If response-reward correspondence is invalid
    """
    for i, row in enumerate(rows):
        # Critical constraint: if y_true is provided, response must also be provided
        if "y_true" in row and row["y_true"] is not None:
            if "response" not in row or row["response"] is None:
                error_msg = (
                    f"Row {i} in {source_description} has y_true but missing response. "
                    f"Ground truth labels require corresponding responses."
                )
                logger.error(error_msg)
                raise ValidationError(error_msg)


def validate_pipeline_data(
    rows: List[Dict[str, Any]],
    source_description: str = "data",
    stage: str = "pre_estimation",
) -> None:
    """
    Comprehensive validation for different pipeline stages.

    Args:
        rows: List of data rows to validate
        source_description: Description of data source for error messages
        stage: Pipeline stage ("pre_estimation", "post_target_computation")

    Raises:
        ValidationError: If validation fails
    """
    require_not_empty(rows, source_description)

    logger.info(f"Validating {len(rows)} rows at stage: {stage}")

    # Always validate core fields
    validate_core_fields(rows, source_description)

    # Validate response-reward correspondence
    validate_response_reward_correspondence(rows, source_description)

    if stage == "pre_estimation":
        # Check if we have any rewards or if we'll rely on judge evaluation
        has_any_rewards = any(
            row.get("reward") is not None
            or row.get("y_true") is not None
            or row.get("score_cal") is not None
            for row in rows
        )

        if has_any_rewards:
            # Full validation including rewards
            validate_and_normalize_rewards(rows, source_description)
        else:
            # All rows will rely on judge evaluation - that's fine for Scenario 1
            logger.info("All rows will rely on judge evaluation for rewards")
            for row in rows:
                row["reward"] = None

    logger.info(f"Validation complete for {stage}")


def assign_rewards_with_priority(
    rows: List[Dict[str, Any]],
    source_description: str = "data",
    oracle_analysis_enabled: bool = False,
) -> None:
    """
    Centralized reward assignment with explicit priority order.

    Priority Order (highest to lowest):
    1. calibration_fallback=True: Use fallback-specific logic (raw scores + oracle labels)
    2. score_cal: Calibrated judge scores (primary for oracle analysis)
    3. reward: Existing reward field (if not None)
    4. y_true: Ground truth labels (if response exists)
    5. None: Will rely on judge evaluation

    Args:
        rows: List of data rows to process in-place
        source_description: Description of data source for logging
        oracle_analysis_enabled: Whether oracle analysis is active

    Raises:
        ValidationError: If reward assignment fails
    """
    logger.info(f"Assigning rewards with priority order for {len(rows)} rows")

    stats = {
        "calibration_fallback": 0,
        "score_cal": 0,
        "existing_reward": 0,
        "y_true": 0,
        "none_judge_needed": 0,
    }

    for i, row in enumerate(rows):
        # Priority 1: Calibration fallback (highest priority - never override)
        if row.get("calibration_fallback", False):
            # Fallback rewards were already set during calibration collapse handling
            # Verify they exist and don't override
            if "reward" not in row or row["reward"] is None:
                logger.warning(
                    f"Row {i}: calibration_fallback=True but no reward set, using 0.0"
                )
                row["reward"] = 0.0
            stats["calibration_fallback"] += 1
            continue

        # Priority 2: Calibrated scores (primary for oracle analysis)
        if "score_cal" in row and row["score_cal"] is not None:
            row["reward"] = float(row["score_cal"])
            stats["score_cal"] += 1
            continue

        # Priority 3: Existing reward field (if not None)
        if "reward" in row and row["reward"] is not None:
            # Keep existing reward, just ensure it's numeric
            try:
                row["reward"] = float(row["reward"])
                stats["existing_reward"] += 1
                continue
            except (ValueError, TypeError):
                logger.warning(
                    f"Row {i}: Invalid reward value {row['reward']}, falling back to y_true"
                )

        # Priority 4: Ground truth labels (if response exists)
        if "y_true" in row and row["y_true"] is not None:
            if "response" not in row or not row["response"]:
                logger.warning(
                    f"Row {i}: y_true exists but no response - clearing y_true"
                )
                row["y_true"] = None
                row["reward"] = None
            else:
                try:
                    row["reward"] = float(row["y_true"])
                    stats["y_true"] += 1
                    continue
                except (ValueError, TypeError):
                    logger.warning(f"Row {i}: Invalid y_true value {row['y_true']}")

        # Priority 5: None - will rely on judge evaluation
        row["reward"] = None
        stats["none_judge_needed"] += 1

    # Log assignment statistics
    logger.info(f"Reward assignment complete for {source_description}:")
    for source, count in stats.items():
        if count > 0:
            logger.info(f"  - {source}: {count} rows")

    # Validate oracle analysis requirements
    if oracle_analysis_enabled:
        calibrated_count = stats["score_cal"] + stats["calibration_fallback"]
        if calibrated_count == 0:
            raise ValidationError(
                "Oracle analysis enabled but no calibrated scores found. "
                "Ensure judge scoring and calibration completed successfully."
            )
        logger.info(
            f"Oracle analysis: {calibrated_count}/{len(rows)} rows have calibrated rewards"
        )


def validate_and_normalize_rewards(
    rows: List[Dict[str, Any]], source_description: str = "data"
) -> None:
    """
    Validate and normalize reward values using the centralized assignment logic.

    Args:
        rows: List of data rows to validate and normalize
        source_description: Description of data source for error messages

    Raises:
        ValidationError: If reward validation fails
    """
    try:
        assign_rewards_with_priority(rows)
        logger.info(f"Reward assignment completed for {len(rows)} rows")
    except Exception as e:
        raise ValidationError(f"Reward validation failed for {source_description}: {e}")


def validate_target_policy_computation(
    rows: List[Dict[str, Any]], source_description: str = "data"
) -> None:
    """
    Validate target policy computation results.

    Args:
        rows: List of data rows with target policy data
        source_description: Description of data source for error messages

    Raises:
        ValidationError: If target policy validation fails
    """
    require_not_empty(rows, source_description)

    for i, row in enumerate(rows):
        # Check for target policy log probabilities
        if "logp_target_all" not in row:
            error_msg = f"Row {i} in {source_description} missing target policy log probabilities"
            logger.error(error_msg)
            raise ValidationError(error_msg)

        logp_target = row["logp_target_all"]
        if not isinstance(logp_target, (list, dict)) or len(logp_target) == 0:
            error_msg = f"Row {i} in {source_description} has invalid target policy log probabilities"
            logger.error(error_msg)
            raise ValidationError(error_msg)

    logger.info(f"Target policy validation complete for {len(rows)} rows")
