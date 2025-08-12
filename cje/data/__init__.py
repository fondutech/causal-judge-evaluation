"""Data loading and preparation utilities.

This module contains:
- Data models: Pydantic models for type safety
- PrecomputedSampler: Load data with log probs and rewards
- DatasetFactory: SOLID-compliant data loading with optional calibration
- DatasetLoader: Pure data loading functionality
- Reward Utils: Utility functions for calibrated rewards
"""

from .precomputed_sampler import PrecomputedSampler
from .reward_utils import (
    add_rewards_to_existing_data,
)
from .models import (
    Sample,
    Dataset,
    EstimationResult,
    LogProbStatus,
    LogProbResult,
)
from .factory import DatasetFactory, default_factory
from .loaders import DatasetLoader, JsonlDataSource, InMemoryDataSource

__all__ = [
    # Data loading
    "PrecomputedSampler",
    "DatasetFactory",
    "DatasetLoader",
    "default_factory",
    "JsonlDataSource",
    "InMemoryDataSource",
    # Data models
    "Sample",
    "Dataset",
    "EstimationResult",
    "LogProbStatus",
    "LogProbResult",
    # Utilities
    "add_rewards_to_existing_data",
]
