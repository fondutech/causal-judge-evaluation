"""Simplified CJE (Causal Judge Evaluation) library.

A minimal implementation focused on the core CJE methodology:
- Load precomputed log probabilities and judge scores
- Calibrate judge scores to oracle KPIs
- Compute calibrated importance weights
- Get unbiased policy performance estimates

Example:
    from cje_simplified import PrecomputedSampler, CalibratedIPS

    # Load data
    sampler = PrecomputedSampler.from_jsonl("data.jsonl")

    # Run estimation
    estimator = CalibratedIPS(sampler)
    results = estimator.fit_and_estimate()

    # Analyze
    print(f"Best policy: {sampler.target_policies[results.best_policy()]}")
    print(f"95% CI: {results.confidence_interval(0.95)}")
"""

# Core classes and types
from .core import (
    # Estimators
    BaseCJEEstimator,
    CalibratedIPS,
    RawIPS,
    # Types
    LogProbResult,
    LogProbStatus,
)

# Data loading and preparation
from .data import (
    PrecomputedSampler,
    Sample,
    Dataset,
    EstimationResult,
    WeightCalibrationConfig,
    DatasetFactory,
    DatasetLoader,
    default_factory,
    add_rewards_to_existing_data,
)

from typing import Optional, List, Any, Tuple, Dict


# Convenience functions for backward compatibility
def load_dataset_from_jsonl(
    file_path: str, target_policies: Optional[List[str]] = None, **kwargs: Any
) -> Dataset:
    """Load Dataset from JSONL file.

    Convenience function using the default factory.
    """
    return default_factory.create_from_jsonl(file_path, target_policies)


# Calibration
from .calibration import (
    # Isotonic regression utilities
    cross_fit_isotonic,
    calibrate_to_target_mean,
    compute_calibration_diagnostics,
    # Judge calibration
    JudgeCalibrator,
    calibrate_judge_scores,
    CalibrationResult,
    # Dataset calibration
    calibrate_dataset,
    calibrate_from_raw_data,
)

# Utilities and diagnostics
from .utils import (
    # Weight diagnostics
    diagnose_weights,
    create_weight_summary_table,
    detect_api_nondeterminism,
    WeightDiagnostics,
    # Extreme weights analysis
    analyze_extreme_weights,
)

# Import visualization utilities if available
try:
    from .utils import (
        plot_weight_calibration_analysis,
        plot_weight_diagnostics_summary,
        plot_calibration_comparison,
    )

    _viz_available = True
except ImportError:
    _viz_available = False

# Teacher forcing
from .teacher_forcing import (
    RobustTeacherForcing,
    compute_teacher_forced_logprob,
    compute_total_logprob,
    ChatTemplateConfig,
    Llama3TemplateConfig,
    HuggingFaceTemplateConfig,
    compute_chat_logprob,
    convert_chat_to_completions,
)

__version__ = "0.1.3"

__all__ = [
    # Core functionality
    "BaseCJEEstimator",
    "CalibratedIPS",
    "RawIPS",
    "PrecomputedSampler",
    "RobustTeacherForcing",
    "compute_teacher_forced_logprob",
    "compute_total_logprob",
    # Data models
    "Sample",
    "Dataset",
    "EstimationResult",
    "WeightCalibrationConfig",
    # Data loading (SOLID-compliant)
    "DatasetFactory",
    "DatasetLoader",
    "default_factory",
    # Convenience functions
    "load_dataset_from_jsonl",
    # Types
    "LogProbResult",
    "LogProbStatus",
    # Diagnostics
    "diagnose_weights",
    "create_weight_summary_table",
    "detect_api_nondeterminism",
    "WeightDiagnostics",
    "analyze_extreme_weights",
    # Calibration - isotonic regression
    "cross_fit_isotonic",
    "calibrate_to_target_mean",
    "compute_calibration_diagnostics",
    # Judge calibration
    "JudgeCalibrator",
    "calibrate_judge_scores",
    "CalibrationResult",
    # Dataset calibration
    "calibrate_dataset",
    "calibrate_from_raw_data",
    # Reward utilities
    "add_rewards_to_existing_data",
    # Chat support
    "ChatTemplateConfig",
    "Llama3TemplateConfig",
    "HuggingFaceTemplateConfig",
    "compute_chat_logprob",
    "convert_chat_to_completions",
]

# Add visualization exports if available
if _viz_available:
    __all__.extend(
        [
            "plot_weight_calibration_analysis",
            "plot_weight_diagnostics_summary",
            "plot_calibration_comparison",
        ]
    )
