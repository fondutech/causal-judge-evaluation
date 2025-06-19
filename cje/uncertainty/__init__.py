"""Uncertainty quantification for CJE.

This module provides first-class support for uncertainty in judge evaluations,
including variance calibration, weight shrinkage, and uncertainty-aware estimation.
"""

from .schemas import JudgeScore, CalibratedReward, UncertaintyAwareEstimate
from .calibration import calibrate_variance_gamma, apply_variance_scaling
from .shrinkage import apply_shrinkage, compute_optimal_shrinkage
from .diagnostics import compute_variance_decomposition, create_uncertainty_report
from .judge import UncertaintyAwareJudge, UncertaintyAPIJudge, DeterministicJudge
from .results import MultiPolicyUncertaintyResult, PolicyResult
from .estimator import UncertaintyAwareDRCPO, UncertaintyEstimatorConfig

__all__ = [
    # Schemas
    "JudgeScore",
    "CalibratedReward",
    "UncertaintyAwareEstimate",
    # Judges
    "UncertaintyAwareJudge",
    "UncertaintyAPIJudge",
    "DeterministicJudge",
    # Estimator
    "UncertaintyAwareDRCPO",
    "UncertaintyEstimatorConfig",
    # Calibration
    "calibrate_variance_gamma",
    "apply_variance_scaling",
    # Shrinkage
    "apply_shrinkage",
    "compute_optimal_shrinkage",
    # Diagnostics
    "compute_variance_decomposition",
    "create_uncertainty_report",
    # Results
    "MultiPolicyUncertaintyResult",
    "PolicyResult",
]
