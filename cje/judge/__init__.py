"""Unified judge module with uncertainty-aware scoring.

All judges now return JudgeScore objects with mean and variance.
"""

# Base interfaces
from .judges import Judge, DeterministicJudge, ProbabilisticJudge, LegacyJudgeAdapter
from .schemas import (
    JudgeScore,
    JudgeEvaluation,
    DetailedJudgeEvaluation,
    JudgeResult,
    score_to_float,
    scores_to_floats,
    float_to_score,
    floats_to_scores,
)

# Factory
from .factory import JudgeFactory

# API judges
from .api_judge import APIJudge, DeterministicAPIJudge, MCAPIJudge

# Cached judge
from .cached_judge import CachedJudge

# Base configurations
from .base import JudgeConfig, APIJudgeConfig, LocalJudgeConfig, BaseJudge

__all__ = [
    # Core interfaces
    "Judge",
    "DeterministicJudge",
    "ProbabilisticJudge",
    "LegacyJudgeAdapter",
    # Schemas
    "JudgeScore",
    "JudgeEvaluation",
    "DetailedJudgeEvaluation",
    "JudgeResult",
    # Factory
    "JudgeFactory",
    # Implementations
    "APIJudge",
    "DeterministicAPIJudge",
    "MCAPIJudge",
    "CachedJudge",
    "BaseJudge",
    # Configurations
    "JudgeConfig",
    "APIJudgeConfig",
    "LocalJudgeConfig",
    # Utilities
    "score_to_float",
    "scores_to_floats",
    "float_to_score",
    "floats_to_scores",
]
