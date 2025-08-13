"""Utilities for reward handling in CJE experiments.

This module provides a clear abstraction for reward handling decisions,
preventing common mistakes like unnecessary calibration with 100% oracle coverage.

IMPORTANT: The main protection is the validate_no_unnecessary_calibration() function
which should be called before any calibration operation to prevent the mistake of
calibrating when oracle labels are already being used directly as rewards.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Any
import logging
import random
import numpy as np

from cje import calibrate_dataset

logger = logging.getLogger(__name__)


class RewardSource(Enum):
    """Explicit tracking of where rewards came from.

    This prevents ambiguity and makes the data flow clear.
    """

    NONE = "none"  # No rewards assigned yet
    PRECOMPUTED = "precomputed"  # Rewards already in dataset
    ORACLE_DIRECT = "oracle_direct"  # 100% oracle labels used directly
    CALIBRATED = "calibrated"  # Judge scores calibrated to partial oracle

    def needs_calibration_for_dr(self) -> bool:
        """Check if DR estimators need calibration for this reward source."""
        # Only NONE needs calibration; CALIBRATED already has it
        return self == RewardSource.NONE

    def __str__(self) -> str:
        descriptions = {
            RewardSource.NONE: "No rewards assigned",
            RewardSource.PRECOMPUTED: "Using pre-computed rewards",
            RewardSource.ORACLE_DIRECT: "Using oracle labels directly (100% coverage)",
            RewardSource.CALIBRATED: "Using calibrated rewards",
        }
        return descriptions.get(self, self.value)


@dataclass
class RewardConfig:
    """Configuration for reward handling.

    This makes the reward source explicit and prevents confusion.
    """

    source: RewardSource  # Explicit enum instead of string
    oracle_coverage: float
    needs_calibration: bool
    calibration_params: Optional[dict] = None

    @property
    def needs_cross_fitted_calibration(self) -> bool:
        """Check if cross-fitted calibration is needed for DR estimators."""
        return self.source == RewardSource.NONE and self.calibration_params is not None

    def __str__(self) -> str:
        return str(self.source)


def determine_reward_config(
    dataset: Any,
    oracle_coverage: float,
    use_oracle: bool,
    oracle_field: str = "oracle_label",
    judge_field: str = "judge_score",
) -> RewardConfig:
    """Determine the reward configuration based on dataset and parameters.

    This centralizes the logic for deciding how to handle rewards.

    Args:
        dataset: The loaded dataset
        oracle_coverage: Fraction of oracle labels to use (0-1)
        use_oracle: Force using oracle labels directly
        oracle_field: Field name for oracle labels
        judge_field: Field name for judge scores

    Returns:
        RewardConfig specifying how rewards should be handled
    """
    # Check if rewards already exist
    rewards_exist = sum(1 for s in dataset.samples if s.reward is not None)

    if rewards_exist > 0:
        logger.info(f"Found {rewards_exist} pre-computed rewards")
        return RewardConfig(
            source=RewardSource.PRECOMPUTED,
            oracle_coverage=1.0,  # Not applicable
            needs_calibration=False,
        )

    # Check oracle coverage
    oracle_count = sum(
        1
        for s in dataset.samples
        if oracle_field in s.metadata and s.metadata[oracle_field] is not None
    )
    actual_coverage = oracle_count / len(dataset.samples) if dataset.samples else 0

    # Decision logic
    if use_oracle or oracle_coverage >= 1.0 or actual_coverage >= 1.0:
        # Use oracle labels directly
        logger.info(
            f"Using oracle labels directly ({oracle_count}/{len(dataset.samples)} available)"
        )
        return RewardConfig(
            source=RewardSource.ORACLE_DIRECT,
            oracle_coverage=actual_coverage,
            needs_calibration=False,
        )
    else:
        # Need calibration
        logger.info(f"Will calibrate with {oracle_coverage:.0%} oracle coverage")
        return RewardConfig(
            source=RewardSource.CALIBRATED,
            oracle_coverage=oracle_coverage,
            needs_calibration=True,
            calibration_params={
                "judge_field": judge_field,
                "oracle_field": oracle_field,
                "enable_cross_fit": True,
                "n_folds": 5,
            },
        )


def apply_reward_config(
    dataset: Any,
    config: RewardConfig,
    oracle_field: str = "oracle_label",
) -> Tuple[Any, Optional[Any]]:
    """Apply the reward configuration to the dataset.

    Args:
        dataset: The dataset to process
        config: The reward configuration
        oracle_field: Field name for oracle labels

    Returns:
        Tuple of (processed_dataset, calibration_result)
    """
    if config.source == RewardSource.PRECOMPUTED:
        # Rewards already exist
        return dataset, None

    elif config.source == RewardSource.ORACLE_DIRECT:
        # Use oracle labels directly as rewards
        oracle_count = 0
        for sample in dataset.samples:
            if (
                oracle_field in sample.metadata
                and sample.metadata[oracle_field] is not None
            ):
                sample.reward = float(sample.metadata[oracle_field])
                oracle_count += 1
        logger.info(f"Assigned {oracle_count} oracle labels as rewards")
        return dataset, None

    elif config.source == RewardSource.CALIBRATED:
        # Perform calibration
        if config.oracle_coverage < 1.0:
            # Mask some oracle labels for partial coverage
            random.seed(42)
            np.random.seed(42)

            samples_with_oracle = [
                i
                for i, s in enumerate(dataset.samples)
                if oracle_field in s.metadata and s.metadata[oracle_field] is not None
            ]

            n_keep = max(2, int(len(samples_with_oracle) * config.oracle_coverage))
            keep_indices = set(
                random.sample(
                    samples_with_oracle, min(n_keep, len(samples_with_oracle))
                )
            )

            # Store original labels and mask others
            original_oracle_labels = {}
            for i, sample in enumerate(dataset.samples):
                if i not in keep_indices and oracle_field in sample.metadata:
                    original_oracle_labels[i] = sample.metadata[oracle_field]
                    sample.metadata[oracle_field] = None

        # Calibrate
        calibrated_dataset, cal_result = calibrate_dataset(
            dataset, **config.calibration_params
        )

        # Restore masked labels if any
        if config.oracle_coverage < 1.0:
            for i, original_value in original_oracle_labels.items():
                dataset.samples[i].metadata[oracle_field] = original_value

        return calibrated_dataset, cal_result

    else:
        raise ValueError(f"Unknown reward source: {config.source}")


def should_recalibrate_for_estimator(
    estimator_name: str,
    config: RewardConfig,
    cal_result: Optional[Any],
) -> bool:
    """Determine if an estimator needs (re)calibration.

    This prevents the mistake of recalibrating when using oracle labels directly.

    Args:
        estimator_name: Name of the estimator (e.g., "mrdr", "tmle")
        config: The reward configuration
        cal_result: Existing calibration result (if any)

    Returns:
        True if calibration is needed, False otherwise
    """
    # Never recalibrate if using oracle labels directly
    if config.source == RewardSource.ORACLE_DIRECT:
        return False

    # Never recalibrate if rewards were precomputed
    if config.source == RewardSource.PRECOMPUTED:
        return False

    # For DR estimators, check if we have cross-fitted calibration
    dr_estimators = {"dr-cpo", "mrdr", "tmle"}
    if estimator_name in dr_estimators:
        # Need calibration if we don't have a calibrator
        return config.needs_calibration and (
            cal_result is None or not cal_result.calibrator
        )

    # IPS estimators don't need calibration beyond rewards
    return False
