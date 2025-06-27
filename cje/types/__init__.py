"""Core types for CJE - type-safe results that prevent silent failures."""

from .results import (
    LogProbResult,
    LogProbStatus,
    Result,
    BatchResult,
    SampleResult,
)

__all__ = [
    "LogProbResult",
    "LogProbStatus",
    "Result",
    "BatchResult",
    "SampleResult",
]
