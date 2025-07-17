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
    create_calibrated_rewards,
    prepare_cje_data,
    add_rewards_to_existing_data,
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
    "create_calibrated_rewards",
    "prepare_cje_data",
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
