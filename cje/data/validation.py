"""Data validation utilities for CJE.

This module provides functions to validate that input data has the required
fields for different CJE use cases.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_cje_data(
    data: List[Dict[str, Any]],
    reward_field: Optional[str] = None,
    judge_field: Optional[str] = None,
    oracle_field: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Validate that data has required fields for CJE analysis.

    This function checks that the data has the core required fields
    (prompt, response, base_policy_logprob, target_policy_logprobs)
    and appropriate evaluation fields (either reward or judge scores).

    Args:
        data: List of data records to validate
        reward_field: Field name containing pre-calibrated rewards
        judge_field: Field name containing judge scores
        oracle_field: Field name containing oracle labels

    Returns:
        Tuple of (is_valid, list_of_issues)

    Example:
        >>> data = load_jsonl("data.jsonl")
        >>> is_valid, issues = validate_cje_data(
        ...     data,
        ...     judge_field="judge_score",
        ...     oracle_field="oracle_label"
        ... )
        >>> if not is_valid:
        ...     for issue in issues:
        ...         print(f"⚠️  {issue}")
    """
    issues = []

    if not data:
        issues.append("Data is empty")
        return False, issues

    # Check core fields in first sample (assume homogeneous)
    sample = data[0]
    core_fields = [
        "prompt_id",
        "prompt",
        "response",
        "base_policy_logprob",
        "target_policy_logprobs",
    ]

    for field in core_fields:
        if field not in sample:
            issues.append(f"Missing required field: {field}")

    # Check that target_policy_logprobs is a dict
    if "target_policy_logprobs" in sample:
        if not isinstance(sample["target_policy_logprobs"], dict):
            issues.append("target_policy_logprobs must be a dictionary")
        elif not sample["target_policy_logprobs"]:
            issues.append("target_policy_logprobs cannot be empty")

    # Check evaluation fields
    has_reward = reward_field and reward_field in sample
    has_judge = judge_field and all(
        (
            judge_field in rec.get("metadata", {})
            if judge_field not in rec
            else judge_field in rec
        )
        for rec in data[: min(10, len(data))]  # Check first 10 samples
    )

    if not has_reward and not has_judge:
        issues.append(
            "No evaluation field found. Need either:\n"
            "  - A 'reward' field with pre-calibrated values, OR\n"
            "  - Judge scores in metadata for calibration"
        )

    # If using judge scores, oracle labels are REQUIRED for calibration
    if has_judge and not has_reward:
        # Judge scores without rewards require oracle labels for calibration
        if not oracle_field:
            issues.append(
                "Judge scores require oracle labels for calibration. "
                "Provide oracle_field parameter pointing to oracle labels."
            )
        else:
            oracle_count = sum(
                1
                for rec in data
                if oracle_field in rec.get("metadata", {})
                and rec["metadata"][oracle_field] is not None
            )

            if oracle_count == 0:
                issues.append(
                    f"No oracle labels found in field '{oracle_field}'. "
                    "Judge scores require oracle labels for calibration. "
                    "Need at least 10 samples with oracle labels (50-100 recommended)."
                )
            elif oracle_count < 10:
                issues.append(
                    f"Too few oracle samples ({oracle_count}). "
                    "Absolute minimum is 10 samples. "
                    "Strongly recommend 50-100+ for robust calibration."
                )
            elif oracle_count < 50:
                logger.warning(
                    f"Found {oracle_count} oracle samples. "
                    f"Consider adding more (50-100 recommended) for better calibration."
                )
            else:
                logger.info(f"Found {oracle_count} oracle samples for calibration")

    # Check data consistency across samples
    n_samples = len(data)
    valid_base_lp = sum(1 for rec in data if rec.get("base_policy_logprob") is not None)

    if valid_base_lp < n_samples:
        pct_missing = 100 * (n_samples - valid_base_lp) / n_samples
        issues.append(
            f"{n_samples - valid_base_lp}/{n_samples} samples "
            f"({pct_missing:.1f}%) have missing base_policy_logprob"
        )

    # Check target policies consistency
    if data and "target_policy_logprobs" in data[0]:
        first_policies = set(data[0]["target_policy_logprobs"].keys())
        inconsistent = []

        for i, rec in enumerate(data[1:11], 1):  # Check first 10
            if "target_policy_logprobs" in rec:
                policies = set(rec["target_policy_logprobs"].keys())
                if policies != first_policies:
                    inconsistent.append(i)

        if inconsistent:
            issues.append(
                f"Inconsistent target policies in samples {inconsistent}. "
                f"Expected: {first_policies}"
            )

    is_valid = len(issues) == 0
    return is_valid, issues


def validate_for_precomputed_sampler(
    data: List[Dict[str, Any]], reward_field: str = "reward"
) -> Tuple[bool, List[str]]:
    """Validate data specifically for PrecomputedSampler.

    PrecomputedSampler requires rewards to be already present,
    either as pre-calibrated values or from judge calibration.

    Args:
        data: List of data records
        reward_field: Field name containing rewards

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # First check basic CJE requirements including the reward field
    is_valid, base_issues = validate_cje_data(data, reward_field=reward_field)
    issues.extend(base_issues)

    # Check for rewards
    if not data:
        issues.append("Data is empty")
        return False, issues

    has_rewards = all(
        reward_field in rec and rec[reward_field] is not None
        for rec in data[: min(100, len(data))]
    )

    if not has_rewards:
        issues.append(
            f"PrecomputedSampler requires '{reward_field}' field. "
            "Either provide pre-calibrated rewards or use calibrate_dataset() first."
        )

    return len(issues) == 0, issues
