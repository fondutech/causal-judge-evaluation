"""Simplified CJE (Causal Judge Evaluation) library.

A minimal implementation focused on the core CJE methodology:
- Load precomputed log probabilities and judge scores
- Calibrate judge scores to oracle KPIs
- Compute calibrated importance weights
- Get unbiased policy performance estimates

Example:
    from cje import PrecomputedSampler, CalibratedIPS

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

# Import DR estimators if available
try:
    from .core.dr_base import DREstimator, DRCPOEstimator
    from .core.mrdr import MRDREstimator
    from .core.tmle import TMLEEstimator
    from .core.outcome_models import (
        BaseOutcomeModel,
        IsotonicOutcomeModel,
        LinearOutcomeModel,
    )
    from .data.fresh_draws import FreshDrawSample, FreshDrawDataset
    from .utils.fresh_draws import (
        load_fresh_draws_from_jsonl,
        validate_fresh_draws,
        create_synthetic_fresh_draws,
        save_fresh_draws_to_jsonl,
        load_fresh_draws_auto,
    )

    _dr_available = True
except ImportError:
    _dr_available = False

# Data loading and preparation
from .data import (
    PrecomputedSampler,
    Sample,
    Dataset,
    EstimationResult,
    DatasetFactory,
    DatasetLoader,
    default_factory,
    add_rewards_to_existing_data,
    validate_cje_data,
    validate_for_precomputed_sampler,
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
    calibrate_to_target_mean,
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
    WeightDiagnostics,
    # Extreme weights analysis
    analyze_extreme_weights,
)

# Import visualization utilities if available
try:
    from .utils import (
        plot_weight_dashboard,
        plot_calibration_comparison,
        plot_policy_estimates,
    )

    _viz_available = True
except ImportError:
    _viz_available = False

# Teacher forcing
from .teacher_forcing import (
    compute_teacher_forced_logprob,
    ChatTemplateConfig,
    Llama3TemplateConfig,
    HuggingFaceTemplateConfig,
    compute_chat_logprob,
    convert_chat_to_completions,
)

# High-level analysis API
from .analysis import analyze_dataset

# Export utilities
from .utils.export import export_results_json, export_results_csv

__version__ = "0.1.3"

__all__ = [
    # Core functionality
    "BaseCJEEstimator",
    "CalibratedIPS",
    "RawIPS",
    "PrecomputedSampler",
    "compute_teacher_forced_logprob",
    # High-level API
    "analyze_dataset",
    # Data models
    "Sample",
    "Dataset",
    "EstimationResult",
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
    "WeightDiagnostics",
    "analyze_extreme_weights",
    # Calibration - isotonic regression
    "calibrate_to_target_mean",
    # Judge calibration
    "JudgeCalibrator",
    "calibrate_judge_scores",
    "CalibrationResult",
    # Dataset calibration
    "calibrate_dataset",
    "calibrate_from_raw_data",
    # Reward utilities
    "add_rewards_to_existing_data",
    # Export utilities
    "export_results_json",
    "export_results_csv",
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
            "plot_weight_dashboard",
            "plot_calibration_comparison",
            "plot_policy_estimates",
        ]
    )

# Add DR estimators if available
if _dr_available:
    __all__.extend(
        [
            # DR estimators
            "DREstimator",
            "DRCPOEstimator",
            "MRDREstimator",
            "TMLEEstimator",
            # Outcome models
            "BaseOutcomeModel",
            "IsotonicOutcomeModel",
            "LinearOutcomeModel",
            # Fresh draw data models
            "FreshDrawSample",
            "FreshDrawDataset",
            # Fresh draw utilities
            "load_fresh_draws_from_jsonl",
            "validate_fresh_draws",
            "create_synthetic_fresh_draws",
            "save_fresh_draws_to_jsonl",
            "load_fresh_draws_auto",
        ]
    )
