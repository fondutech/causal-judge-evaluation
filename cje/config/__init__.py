"""
Unified Configuration System for CJE.

This module provides a single, clean interface for all CJE configuration needs.
"""

from .unified import (
    PathsConfig,
    DatasetConfig,
    PolicyConfig,
    TargetPolicyConfig,
    JudgeConfig,
    EstimatorConfig,
    CJEConfig,
    ConfigurationBuilder,
    from_dict,
    to_dict,
    simple_config,
    multi_policy_config,
    get_example_configs,
    validate_configuration,
)

__all__ = [
    "PathsConfig",
    "DatasetConfig",
    "PolicyConfig",
    "TargetPolicyConfig",
    "JudgeConfig",
    "EstimatorConfig",
    "CJEConfig",
    "ConfigurationBuilder",
    "from_dict",
    "to_dict",
    "simple_config",
    "multi_policy_config",
    "get_example_configs",
    "validate_configuration",
]
