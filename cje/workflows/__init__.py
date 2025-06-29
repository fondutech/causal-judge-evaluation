"""High-level workflow interfaces for common CJE use cases."""

from typing import TYPE_CHECKING

# Arena workflow imports
from ..judge import JudgeFactory, APIJudge
from ..calibration import cross_fit_calibration
from ..estimators import MultiDRCPOEstimator, BasicFeaturizer, get_estimator
from ..estimators.simplified import (
    DRCPOEstimator,
    IPSEstimator,
    estimate_value,
    EstimationResult,
)
from ..loggers import MultiTargetSampler, APIPolicyRunner
from ..data import load_dataset
from ..testing import testing_mode, MockJudge, MockPolicyRunner


# Re-export for convenience
__all__ = [
    # Core components
    "JudgeFactory",
    "APIJudge",
    "cross_fit_calibration",
    "MultiDRCPOEstimator",
    "BasicFeaturizer",
    "get_estimator",
    "MultiTargetSampler",
    "APIPolicyRunner",
    "load_dataset",
    # Simplified API
    "DRCPOEstimator",
    "IPSEstimator",
    "estimate_value",
    "EstimationResult",
    # Testing
    "testing_mode",
    "MockJudge",
    "MockPolicyRunner",
]
