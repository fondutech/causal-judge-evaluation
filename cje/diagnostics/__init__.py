"""Unified diagnostic system for CJE.

This module provides a single source of truth for all diagnostics,
consolidating previously scattered diagnostic computations.
"""

from .suite import (
    DiagnosticSuite,
    WeightMetrics,
    EstimationSummary,
    StabilityMetrics,
    DRMetrics,
    RobustInference,
)
from .runner import DiagnosticRunner, DiagnosticConfig
from .display import format_diagnostic_suite

__all__ = [
    "DiagnosticSuite",
    "WeightMetrics",
    "EstimationSummary",
    "StabilityMetrics",
    "DRMetrics",
    "RobustInference",
    "DiagnosticRunner",
    "DiagnosticConfig",
    "format_diagnostic_suite",
]
