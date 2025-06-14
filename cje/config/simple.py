"""
Simplified configuration system for CJE.

Replace complex Hydra configs with simple dataclasses and sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration with sensible defaults."""

    name: str = "ChatbotArena"
    split: str = "train"
    sample_limit: Optional[int] = None
    path: Optional[str] = None


@dataclass
class PolicyConfig:
    """Single policy configuration."""

    name: str
    provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 2048
    system_prompt: Optional[str] = None


@dataclass
class JudgeConfig:
    """Judge configuration."""

    provider: str = "openai"
    model_name: str = "gpt-4"
    template: str = "default"
    temperature: float = 0.0
    max_tokens: int = 100


@dataclass
class EstimatorConfig:
    """Estimator configuration."""

    name: Literal["DRCPO", "MRDR", "IPS"] = "DRCPO"
    k_folds: int = 5
    clip: float = 20.0
    calibrate_weights: bool = True
    calibrate_outcome: bool = True
    samples_per_policy: int = 2


@dataclass
class CJEConfig:
    """Complete CJE experiment configuration."""

    # Required
    logging_policy: PolicyConfig
    target_policies: List[PolicyConfig]

    # Optional with defaults
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)

    # Paths
    work_dir: str = "outputs"
    cache_dir: Optional[str] = None

    # Oracle mode
    oracle_enabled: bool = False
    oracle_fraction: float = 0.25

    @classmethod
    def from_yaml(cls, path: Path) -> "CJEConfig":
        """Load config from simplified YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Convert nested dicts to dataclasses
        if "dataset" in data:
            data["dataset"] = DatasetConfig(**data["dataset"])

        if "judge" in data:
            data["judge"] = JudgeConfig(**data["judge"])

        if "estimator" in data:
            data["estimator"] = EstimatorConfig(**data["estimator"])

        # Handle policies
        logging_policy = PolicyConfig(**data["logging_policy"])
        target_policies = [PolicyConfig(**p) for p in data["target_policies"]]

        return cls(
            logging_policy=logging_policy,
            target_policies=target_policies,
            **{
                k: v
                for k, v in data.items()
                if k not in ["logging_policy", "target_policies"]
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dataset": self.dataset.__dict__,
            "logging_policy": self.logging_policy.__dict__,
            "target_policies": [p.__dict__ for p in self.target_policies],
            "judge": self.judge.__dict__,
            "estimator": self.estimator.__dict__,
            "work_dir": self.work_dir,
            "cache_dir": self.cache_dir,
            "oracle_enabled": self.oracle_enabled,
            "oracle_fraction": self.oracle_fraction,
        }


def create_example_config() -> CJEConfig:
    """Create an example configuration for quick starts."""
    return CJEConfig(
        logging_policy=PolicyConfig(
            name="baseline",
            provider="openai",
            model_name="gpt-3.5-turbo",
        ),
        target_policies=[
            PolicyConfig(
                name="improved_prompt",
                provider="openai",
                model_name="gpt-3.5-turbo",
                system_prompt="Think step by step before answering.",
            ),
            PolicyConfig(
                name="stronger_model",
                provider="openai",
                model_name="gpt-4",
            ),
        ],
        dataset=DatasetConfig(sample_limit=100),
        estimator=EstimatorConfig(name="DRCPO", k_folds=5),
    )
