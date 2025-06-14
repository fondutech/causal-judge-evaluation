"""Judge module with structured output support."""

from .base import (
    BaseJudge,
    JudgeConfig,
    APIJudgeConfig,
    LocalJudgeConfig,
    CachedJudge,
)
from .judges import Judge
from .api_judge import APIJudge
from .local_judge import LocalJudge
from .factory import JudgeFactory
from .schemas import (
    JudgeScore,
    JudgeEvaluation,
    DetailedJudgeEvaluation,
    JudgeResult,
)

__all__ = [
    # Base classes
    "BaseJudge",
    "Judge",
    # Implementations
    "APIJudge",
    "LocalJudge",
    "CachedJudge",
    # Configuration
    "JudgeConfig",
    "APIJudgeConfig",
    "LocalJudgeConfig",
    # Factory
    "JudgeFactory",
    # Schemas
    "JudgeScore",
    "JudgeEvaluation",
    "DetailedJudgeEvaluation",
    "JudgeResult",
]
