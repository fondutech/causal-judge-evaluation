"""
CJE Research Extension

This module provides research-specific functionality for running
comprehensive experiments like the Arena CJE research pipeline.

It builds on top of existing CJE infrastructure to provide:
- Research phase coordination
- Advanced diagnostics
- Gold validation workflows
- Comprehensive result analysis
"""

from .arena_experiment import ArenaResearchExperiment
from .phase_manager import ResearchPhaseManager
from .validation import GoldValidationRunner, DiagnosticsRunner

__all__ = [
    "ArenaResearchExperiment",
    "ResearchPhaseManager",
    "GoldValidationRunner",
    "DiagnosticsRunner",
]
