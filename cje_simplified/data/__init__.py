"""Data loading and preparation utilities.

This module contains:
- Data models: Pydantic models for type safety
- PrecomputedSampler: Load data with log probs and rewards
- Reward Utils: Convert judge scores to calibrated rewards
"""

from .precomputed_sampler import PrecomputedSampler
from .reward_utils import (
    create_calibrated_rewards,
    prepare_cje_data,
    add_rewards_to_existing_data,
)
from .models import (
    Sample,
    Dataset,
    EstimationResult,
    WeightCalibrationConfig,
    LogProbStatus,
    LogProbResult,
)

__all__ = [
    # Data loading
    "PrecomputedSampler",
    # Data models
    "Sample",
    "Dataset",
    "EstimationResult",
    "WeightCalibrationConfig",
    "LogProbStatus",
    "LogProbResult",
    # Utilities
    "create_calibrated_rewards",
    "prepare_cje_data",
    "add_rewards_to_existing_data",
]
