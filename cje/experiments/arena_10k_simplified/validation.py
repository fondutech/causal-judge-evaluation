"""Validation utilities to prevent common mistakes in CJE analysis."""

import logging
import numpy as np
from typing import Any, Optional

logger = logging.getLogger(__name__)


def validate_no_unnecessary_calibration(
    dataset: Any,
    oracle_coverage: float,
    cal_result: Optional[Any],
    oracle_field: str = "oracle_label",
) -> None:
    """Validate that we're not about to calibrate unnecessarily.

    This catches the dangerous case where we would calibrate oracle labels
    that are already being used as rewards, which would lose information.

    Raises:
        ValueError: If calibration would be a mistake
    """
    # Check if rewards already exist and match oracle labels
    if hasattr(dataset, "samples") and dataset.samples:
        rewards = [s.reward for s in dataset.samples if s.reward is not None]
        oracle_labels = [
            s.metadata.get(oracle_field)
            for s in dataset.samples
            if oracle_field in s.metadata and s.metadata[oracle_field] is not None
        ]

        if rewards and oracle_labels and len(rewards) == len(oracle_labels):
            # Check if rewards are already oracle labels
            rewards_array = np.array(rewards[: len(oracle_labels)])
            oracle_array = np.array(oracle_labels[: len(rewards)])

            if np.allclose(rewards_array, oracle_array, rtol=1e-10):
                raise ValueError(
                    "STOP: Dataset rewards are already set to oracle labels! "
                    "Calibration would lose information (25 unique values → 10). "
                    "This usually means oracle_coverage=1.0 was already handled. "
                    "Check the reward assignment logic."
                )

    # Warn if we're calibrating with 100% coverage
    if oracle_coverage >= 1.0 and cal_result is None:
        logger.warning(
            "About to calibrate with 100% oracle coverage. "
            "Consider using oracle labels directly as rewards instead."
        )


def validate_reward_source(dataset: Any) -> str:
    """Determine and validate the source of rewards in a dataset.

    Returns:
        One of: "none", "oracle_direct", "calibrated", "unknown"
    """
    if not hasattr(dataset, "samples") or not dataset.samples:
        return "none"

    rewards = [s.reward for s in dataset.samples if s.reward is not None]
    if not rewards:
        return "none"

    # Check if rewards match oracle labels (indicating direct use)
    oracle_labels = [
        s.metadata.get("oracle_label")
        for s in dataset.samples
        if "oracle_label" in s.metadata and s.metadata["oracle_label"] is not None
    ]

    if oracle_labels and len(rewards) == len(oracle_labels):
        rewards_array = np.array(rewards[: len(oracle_labels)])
        oracle_array = np.array(oracle_labels[: len(rewards)])

        # Check unique values
        unique_rewards = len(np.unique(rewards_array))
        unique_oracle = len(np.unique(oracle_array))

        if unique_rewards == unique_oracle and unique_rewards > 20:
            return "oracle_direct"  # Many unique values suggest direct oracle use
        elif unique_rewards < unique_oracle and unique_rewards <= 15:
            return "calibrated"  # Few unique values suggest calibration

    return "unknown"
