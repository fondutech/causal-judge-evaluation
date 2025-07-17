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


def load_dataset_with_calibration(
    file_path: str,
    judge_score_field: str = "judge_score",
    oracle_label_field: str = "oracle_label",
    k_folds: int = 5,
    target_policies: Optional[List[str]] = None,
) -> Tuple[Dataset, Dict[str, float]]:
    """Load Dataset from JSONL with judge calibration.

    Convenience function using the default factory.
    """
    return default_factory.create_from_jsonl_with_calibration(
        file_path, judge_score_field, oracle_label_field, k_folds, target_policies
    )


# Utilities and diagnostics
from .utils import (
    # Judge calibration
    JudgeCalibrator,
    calibrate_judge_scores,
    CalibrationResult,
    # Weight diagnostics
    diagnose_weights,
    create_weight_summary_table,
    detect_api_nondeterminism,
    WeightDiagnostics,
)

# Teacher forcing
from .teacher_forcing import (
    RobustTeacherForcing,
    compute_teacher_forced_logprob,
    compute_total_logprob,
    ChatToCompletionsConverter,
    ChatTemplateConfig,
    Llama3TemplateConfig,
    Llama4TemplateConfig,
    HuggingFaceTemplateConfig,
    convert_chat_for_teacher_forcing,
    compute_chat_logprob,
)

__version__ = "0.1.3"

__all__ = [
    # Core functionality
    "BaseCJEEstimator",
    "CalibratedIPS",
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
    "load_dataset_with_calibration",
    # Types
    "LogProbResult",
    "LogProbStatus",
    # Diagnostics
    "diagnose_weights",
    "create_weight_summary_table",
    "detect_api_nondeterminism",
    "WeightDiagnostics",
    # Judge calibration
    "JudgeCalibrator",
    "calibrate_judge_scores",
    "CalibrationResult",
    # Reward utilities
    "add_rewards_to_existing_data",
    # Chat support
    "ChatToCompletionsConverter",
    "ChatTemplateConfig",
    "Llama3TemplateConfig",
    "Llama4TemplateConfig",
    "HuggingFaceTemplateConfig",
    "convert_chat_for_teacher_forcing",
    "compute_chat_logprob",
]
