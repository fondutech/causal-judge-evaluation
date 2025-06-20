"""Unified judge module with uncertainty-aware scoring.

All judges now return JudgeScore objects with mean and variance.
"""

# Base interfaces
from .judges import Judge, DeterministicJudge, ProbabilisticJudge
from .schemas import (
    JudgeScore,
    JudgeEvaluation,
    DetailedJudgeEvaluation,
    JudgeResult,
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
]
