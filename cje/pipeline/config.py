"""
Pipeline configuration - Configuration structure for the CJE pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for the CJE pipeline."""

    # Work directory
    work_dir: Path

    # Stage configurations
    dataset_config: Dict[str, Any]
    logging_policy_config: Dict[str, Any]
    judge_config: Dict[str, Any]
    calibration_config: Dict[str, Any]
    target_policies_config: List[Dict[str, Any]]
    estimator_configs: List[Dict[str, Any]]

    # Optional configurations
    oracle_config: Optional[Dict[str, Any]] = None

    # Pipeline options
    save_intermediate: bool = True
    verbose: bool = True

    @classmethod
    def from_hydra_config(cls, cfg: Any) -> "PipelineConfig":
        """Create PipelineConfig from Hydra configuration."""
        return cls(
            work_dir=Path(cfg.paths.work_dir),
            dataset_config={
                "name": cfg.dataset.name,
                "split": cfg.dataset.split,
                "sample_limit": getattr(cfg.dataset, "sample_limit", None),
            },
            logging_policy_config={
                "provider": getattr(cfg.logging_policy, "provider", "hf"),
                "model_name": cfg.logging_policy.model_name,
                "temperature": getattr(cfg.logging_policy, "temperature", 1.0),
                "max_new_tokens": getattr(cfg.logging_policy, "max_new_tokens", 512),
                "top_p": getattr(cfg.logging_policy, "top_p", 1.0),
                "batch_size": getattr(cfg.logging_policy, "batch_size", 1),
                "checkpoint_interval": getattr(
                    cfg.logging_policy, "checkpoint_interval", 100
                ),
            },
            judge_config={
                "provider": cfg.judge.provider,
                "model_name": cfg.judge.model_name,
                "template": getattr(cfg.judge, "template", "default_template"),
                "uncertainty_method": getattr(
                    cfg.judge, "uncertainty_method", "deterministic"
                ),
                "num_samples": getattr(cfg.judge, "num_samples", 1),
                "structured_output_mode": getattr(
                    cfg.judge, "structured_output_mode", False
                ),
                "batch_size": getattr(cfg.judge, "batch_size", 10),
                "cache_ttl_hours": getattr(cfg.judge, "cache_ttl_hours", 24 * 7),
                "api_key": getattr(cfg.judge, "api_key", None),
                "base_url": getattr(cfg.judge, "base_url", None),
            },
            calibration_config={
                "n_folds": getattr(cfg.calibration, "n_folds", 5),
                "min_samples": getattr(cfg.calibration, "min_samples", 10),
                "save_plots": getattr(cfg.calibration, "save_plots", True),
            },
            target_policies_config=[
                {
                    "provider": getattr(policy, "provider", "hf"),
                    "model_name": policy.model_name,
                    "temperature": getattr(policy, "temperature", 1.0),
                    "max_new_tokens": getattr(policy, "max_new_tokens", 512),
                    "top_p": getattr(policy, "top_p", 1.0),
                    "batch_size": getattr(policy, "batch_size", 1),
                }
                for policy in cfg.target_policies
            ],
            estimator_configs=[
                {"name": est.name, "params": getattr(est, "params", {})}
                for est in cfg.estimators
            ],
            oracle_config=(
                {
                    "enabled": cfg.oracle.enabled,
                    "provider": cfg.oracle.provider,
                    "model_name": cfg.oracle.model_name,
                    "template": getattr(cfg.oracle, "template", "quick_judge"),
                    "temperature": getattr(cfg.oracle, "temperature", 0.0),
                    "max_tokens": getattr(cfg.oracle, "max_tokens", 50),
                    "logging_policy_oracle_fraction": cfg.oracle.logging_policy_oracle_fraction,
                    "seed": getattr(cfg.oracle, "seed", 42),
                    "api_key": getattr(cfg.oracle, "api_key", None),
                    "base_url": getattr(cfg.oracle, "base_url", None),
                }
                if hasattr(cfg, "oracle") and cfg.oracle.enabled
                else None
            ),
            save_intermediate=True,
            verbose=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "work_dir": str(self.work_dir),
            "dataset_config": self.dataset_config,
            "logging_policy_config": self.logging_policy_config,
            "judge_config": self.judge_config,
            "calibration_config": self.calibration_config,
            "target_policies_config": self.target_policies_config,
            "estimator_configs": self.estimator_configs,
            "oracle_config": self.oracle_config,
            "save_intermediate": self.save_intermediate,
            "verbose": self.verbose,
        }
