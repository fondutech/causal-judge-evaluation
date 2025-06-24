"""CJE-Core: Counterfactual Judge Evaluation toolkit."""

__version__ = "0.1.0"

# Automatically load .env file if present
import os
from pathlib import Path

# Try to load .env file if python-dotenv is available
from .utils.imports import optional_import

dotenv, HAS_DOTENV = optional_import(
    "python-dotenv",
    "Automatic .env file loading",
    warn=False,  # Don't warn at import time
)

if HAS_DOTENV:
    from dotenv import load_dotenv

    # Look for .env file in project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

# Export reference implementation for quick experimentation
from .reference import FixedSampler, ReferenceDRCPO

# Pipeline functionality is now in cje.pipeline module
from .pipeline import CJEPipeline, PipelineConfig

# Export configuration API for Python-first usage
from .config import ConfigurationBuilder, CJEConfig
from .config.unified import simple_config, multi_policy_config

# Export data loading for convenience
from .data import load_dataset

# Export PrecomputedMultiTargetSampler
from .loggers.precomputed_sampler import PrecomputedMultiTargetSampler

# Export calibration tools
from .calibration import cross_fit_calibration

# Export new modules
from .estimators import get_estimator
from .estimators.featurizer import RichFeaturizer
from .oracle_labeling import add_oracle_labels

__all__ = [
    # Pipeline
    "CJEPipeline",
    "PipelineConfig",
    # Configuration
    "ConfigurationBuilder",
    "CJEConfig",
    "simple_config",
    "multi_policy_config",
    # Data
    "load_dataset",
    # Reference implementation
    "FixedSampler",
    "ReferenceDRCPO",
    # PrecomputedMultiTargetSampler
    "PrecomputedMultiTargetSampler",
    # Calibration
    "cross_fit_calibration",
    # New additions
    "RichFeaturizer",
    "add_oracle_labels",
    # Estimators
    "get_estimator",
]
