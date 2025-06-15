"""
Unified Configuration System for CJE.

This module provides a single, clean interface for all CJE configuration needs.
It uses ConfigurationBuilder as the primary API with dataclass-based validation.

Key design principles:
- ConfigurationBuilder is the single source of truth for configuration creation
- YAML configs are parsed through ConfigurationBuilder (via from_dict) for consistency
- All validation logic is centralized in the dataclass __post_init__ methods
- The builder pattern provides a fluent, user-friendly API

Usage patterns:
1. Programmatic configuration:
   config = simple_config(logging_model="gpt2", logging_provider="hf", ...)

2. YAML configuration:
   config = from_dict(yaml.safe_load(config_file))

Both paths use the same validation logic via ConfigurationBuilder.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path
import warnings

from ..utils.error_handling import ConfigurationError
from ..constants import ALL_PROVIDERS

logger = logging.getLogger(__name__)


@dataclass
class PathsConfig:
    """Configuration for file paths."""

    work_dir: str = "./outputs/experiment"

    def __post_init__(self) -> None:
        # Ensure work_dir is a Path object for easier manipulation
        self.work_dir = str(Path(self.work_dir).resolve())


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    name: str
    split: str = "test"
    sample_limit: Optional[int] = None

    def __post_init__(self) -> None:
        if self.sample_limit is not None and self.sample_limit <= 0:
            raise ConfigurationError("sample_limit must be positive")


@dataclass
class PolicyConfig:
    """Configuration for policy runners."""

    model_name: str
    provider: str  # No default - must be explicitly specified
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.7
    max_new_tokens: int = 150
    top_p: float = 1.0

    # API-specific fields
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # Local model fields
    device: str = "auto"
    torch_dtype: str = "auto"

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ConfigurationError("model_name is required")

        if not self.provider:
            raise ConfigurationError("provider is required")

        valid_providers = ALL_PROVIDERS
        if self.provider not in valid_providers:
            raise ConfigurationError(
                f"Invalid provider: {self.provider}. Valid: {list(valid_providers)}"
            )

        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigurationError("temperature must be between 0.0 and 2.0")

        if not (0.0 <= self.top_p <= 1.0):
            raise ConfigurationError("top_p must be between 0.0 and 1.0")

        # Warn about potential overlap issues when top_p < 1.0
        if self.top_p < 1.0:
            warnings.warn(
                f"top_p = {self.top_p} < 1.0 may cause overlap violations in off-policy evaluation. "
                "When the logging policy uses nucleus sampling (top_p < 1.0), target policies may assign "
                "probability to tokens that were excluded from the logging policy's truncated distribution, "
                "violating the overlap assumption π'(s|x) > 0 ⇒ π₀(s|x) > 0. This can lead to infinite "
                "importance weights and biased estimates. Consider using top_p = 1.0 for robust evaluation.",
                UserWarning,
                stacklevel=2,
            )

        if self.max_new_tokens < 1:
            raise ConfigurationError("max_new_tokens must be at least 1")


@dataclass
class TargetPolicyConfig(PolicyConfig):
    """Configuration for target policies."""

    name: str = "default_target_policy"
    mc_samples: int = 5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mc_samples <= 0:
            raise ConfigurationError("mc_samples must be at least 1")


@dataclass
class JudgeConfig:
    """Configuration for judges."""

    provider: str  # Required: no inference allowed
    model_name: str  # Required: no inference allowed
    template: str = "quick_judge"
    temperature: float = 0.0
    max_retries: int = 3
    timeout: int = 30
    max_tokens: int = 100

    # API-specific fields
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # Local model fields
    device: str = "auto"
    torch_dtype: str = "auto"
    batch_size: int = 8

    # Template customization
    custom_template: Optional[str] = None
    template_variables: Dict[str, Any] = field(default_factory=dict)

    # Skip judging (use ground truth)
    skip: bool = False

    def __post_init__(self) -> None:
        if not self.provider:
            raise ConfigurationError(
                "provider is required and must be explicitly specified"
            )
        if not self.model_name:
            raise ConfigurationError(
                "model_name is required and must be explicitly specified"
            )

        valid_providers = ALL_PROVIDERS
        if self.provider not in valid_providers:
            raise ConfigurationError(
                f"Invalid provider: {self.provider}. Valid providers: {list(valid_providers)}"
            )

        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigurationError("temperature must be between 0.0 and 2.0")

        if self.max_retries < 1:
            raise ConfigurationError("max_retries must be between 1 and 10")

        if self.timeout <= 0:
            raise ConfigurationError("timeout must be between 1 and 300 seconds")


@dataclass
class OracleConfig:
    """Configuration for oracle labeling in arena analysis."""

    enabled: bool = False
    provider: str = "fireworks"
    model_name: str = "accounts/fireworks/models/llama-v3p1-70b-instruct"
    template: str = "quick_judge"
    temperature: float = 0.0
    max_tokens: int = 50

    # Oracle fraction configuration for ablation studies
    logging_policy_oracle_fraction: float = 0.25
    seed: int = 42

    # API configuration
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def __post_init__(self) -> None:
        if self.enabled:
            if not self.provider:
                raise ConfigurationError("oracle provider is required when enabled")
            if not self.model_name:
                raise ConfigurationError("oracle model_name is required when enabled")

            valid_providers = ALL_PROVIDERS
            if self.provider not in valid_providers:
                raise ConfigurationError(
                    f"Invalid oracle provider: {self.provider}. Valid: {list(valid_providers)}"
                )

            if not (0.0 < self.logging_policy_oracle_fraction <= 1.0):
                raise ConfigurationError(
                    "logging_policy_oracle_fraction must be between 0.0 and 1.0"
                )

            if not (0.0 <= self.temperature <= 2.0):
                raise ConfigurationError(
                    "oracle temperature must be between 0.0 and 2.0"
                )


@dataclass
class EstimatorConfig:
    """Configuration for estimators."""

    name: str
    k: int = 5
    seed: int = 0

    # Estimator-specific parameters
    outcome_model: str = "xgboost"
    max_iter_epsilon: int = 100
    tol_epsilon: float = 1e-6
    target_policy_strategy: str = "require_sampler"
    n_jobs: int = (
        -1
    )  # Number of parallel jobs for cross-validation (-1 for all processors)
    samples_per_policy: int = (
        2  # Number of samples per policy for DR-CPO/MRDR (2 for proper variance estimation)
    )
    score_target_policy_sampled_completions: bool = (
        True  # Whether to score target policy samples with judge (ignored if samples_per_policy=0)
    )
    stabilize_weights: bool = (
        True  # Whether to apply numerical stabilization for extreme log differences
    )
    calibrate_weights: bool = (
        True  # Whether to apply isotonic calibration to importance weights
    )
    calibrate_outcome: bool = (
        True  # Whether to apply isotonic calibration to outcome model predictions
    )

    def __post_init__(self) -> None:
        valid_estimators = {"IPS", "SNIPS", "DRCPO", "MRDR"}
        if self.name not in valid_estimators:
            raise ConfigurationError(
                f"Invalid estimator: {self.name}. Valid: {list(valid_estimators)}"
            )

        if self.k <= 0:
            raise ConfigurationError("k must be at least 1")

        if self.n_jobs == 0:
            raise ConfigurationError("n_jobs cannot be 0")

        if self.n_jobs is not None and self.n_jobs < -1:
            raise ConfigurationError(
                "n_jobs must be -1 (all cores) or positive integer"
            )


@dataclass
class CJEConfig:
    """Main configuration class for CJE experiments."""

    paths: PathsConfig
    dataset: DatasetConfig
    logging_policy: PolicyConfig
    target_policies: List[TargetPolicyConfig]
    judge: JudgeConfig
    estimator: EstimatorConfig
    oracle: Optional[OracleConfig] = None

    def __post_init__(self) -> None:
        if not self.target_policies:
            raise ConfigurationError("target_policies cannot be empty")

        # Check for duplicate target policy names
        policy_names = [policy.name for policy in self.target_policies]
        if len(policy_names) != len(set(policy_names)):
            raise ConfigurationError("target_policies must have unique names")

        # Default oracle config if not provided
        if self.oracle is None:
            self.oracle = OracleConfig()

    def run(self) -> Dict[str, Any]:
        """Run the experiment defined by this configuration and return results.

        This is a convenience method that allows running experiments directly
        from Python without going through YAML files.

        Returns:
            Dictionary with experiment results

        Example:
            config = simple_config(target_changes={"temperature": 0.3})
            results = config.run()
            print(f"Improvement: {results['target']['v_hat']:.3f}")
        """
        # Avoid circular import
        from ..pipeline import run_pipeline
        import tempfile
        import yaml  # type: ignore[import-untyped]

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(to_dict(self), f)
            temp_path = f.name

        try:
            # Extract directory and filename
            import os

            cfg_dir = os.path.dirname(temp_path)
            cfg_name = os.path.splitext(os.path.basename(temp_path))[0]

            # Run pipeline
            return run_pipeline(cfg_path=cfg_dir, cfg_name=cfg_name)
        finally:
            # Clean up temp file
            import os

            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file

        Example:
            config = simple_config(target_changes={"temperature": 0.3})
            config.to_yaml("experiment.yaml")
        """
        import yaml  # type: ignore[import-untyped]

        with open(path, "w") as f:
            yaml.dump(to_dict(self), f)

    @classmethod
    def from_yaml(cls, path: str) -> "CJEConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            CJEConfig instance

        Example:
            config = CJEConfig.from_yaml("experiment.yaml")
            results = config.run()
        """
        import yaml  # type: ignore[import-untyped]

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return from_dict(config_dict)


class ConfigurationBuilder:
    """Builder pattern for creating CJE configurations.

    This is the PRIMARY API for configuration creation. All other configuration
    paths (simple_config, multi_policy_config, from_dict) use this builder
    internally to ensure consistent validation and behavior.

    The builder pattern provides:
    - Fluent, chainable API for ease of use
    - Centralized validation through dataclass __post_init__ methods
    - Clear error messages with context
    - Flexibility to add configurations in any order

    Example:
        config = (ConfigurationBuilder()
                  .paths("./outputs")
                  .dataset("SummEval")
                  .logging_policy("gpt2", provider="hf")
                  .add_target_policy("target", "gpt2", provider="hf", temperature=0.1)
                  .judge("openai")
                  .estimator("DRCPO")
                  .build())
    """

    def __init__(self) -> None:
        self._paths: Optional[PathsConfig] = None
        self._dataset: Optional[DatasetConfig] = None
        self._logging_policy: Optional[PolicyConfig] = None
        self._target_policies: List[TargetPolicyConfig] = []
        self._judge: Optional[JudgeConfig] = None
        self._estimator: Optional[EstimatorConfig] = None
        self._oracle: Optional[OracleConfig] = None

    def paths(self, work_dir: str) -> "ConfigurationBuilder":
        """Set paths configuration."""
        self._paths = PathsConfig(work_dir=work_dir)
        return self

    def dataset(
        self, name: str, split: str = "test", sample_limit: Optional[int] = None
    ) -> "ConfigurationBuilder":
        """Set dataset configuration."""
        self._dataset = DatasetConfig(name=name, split=split, sample_limit=sample_limit)
        return self

    def logging_policy(self, model_name: str, **kwargs: Any) -> "ConfigurationBuilder":
        """Set logging policy configuration."""
        self._logging_policy = PolicyConfig(model_name=model_name, **kwargs)
        return self

    def add_target_policy(
        self, name: str, model_name: str, **kwargs: Any
    ) -> "ConfigurationBuilder":
        """Add a target policy configuration."""
        target_policy = TargetPolicyConfig(name=name, model_name=model_name, **kwargs)
        self._target_policies.append(target_policy)
        return self

    def judge(
        self, provider: str, model_name: str, **kwargs: Any
    ) -> "ConfigurationBuilder":
        """Set judge configuration with explicit provider and model."""
        self._judge = JudgeConfig(provider=provider, model_name=model_name, **kwargs)
        return self

    def estimator(self, name: str, **kwargs: Any) -> "ConfigurationBuilder":
        """Set estimator configuration."""
        self._estimator = EstimatorConfig(name=name, **kwargs)
        return self

    def oracle(
        self,
        enabled: bool = True,
        provider: str = "fireworks",
        model_name: str = "accounts/fireworks/models/llama-v3p1-70b-instruct",
        **kwargs: Any,
    ) -> "ConfigurationBuilder":
        """Set oracle configuration for arena analysis."""
        self._oracle = OracleConfig(
            enabled=enabled, provider=provider, model_name=model_name, **kwargs
        )
        return self

    def build(self) -> CJEConfig:
        """Build the final configuration."""
        if self._paths is None:
            raise ConfigurationError("paths configuration is required")
        if self._dataset is None:
            raise ConfigurationError("dataset configuration is required")
        if self._logging_policy is None:
            raise ConfigurationError("logging_policy configuration is required")
        if not self._target_policies:
            raise ConfigurationError("at least one target policy is required")
        if self._judge is None:
            raise ConfigurationError("judge configuration is required")
        if self._estimator is None:
            raise ConfigurationError("estimator configuration is required")

        return CJEConfig(
            paths=self._paths,
            dataset=self._dataset,
            logging_policy=self._logging_policy,
            target_policies=self._target_policies,
            judge=self._judge,
            estimator=self._estimator,
            oracle=self._oracle,
        )


def simple_config(
    work_dir: str = "./outputs/experiment",
    dataset_name: str = "./data/test.jsonl",
    logging_model: str = "gpt-4o-mini",
    logging_provider: str = "fireworks",
    target_model: str = "gpt-4o-mini",
    target_provider: str = "fireworks",
    target_changes: Optional[Dict[str, Any]] = None,
    judge_provider: str = "fireworks",
    judge_model: str = "gpt-4o-mini",
    estimator_name: str = "DRCPO",
) -> CJEConfig:
    """Create a simple configuration with sensible defaults.

    Args:
        work_dir: Working directory for outputs
        dataset_name: Dataset name or path to JSONL file
        logging_model: Model name for logging policy
        logging_provider: Provider for logging policy (hf, openai, anthropic, google)
        target_model: Model name for target policy
        target_provider: Provider for target policy (hf, openai, anthropic, google)
        target_changes: Additional changes to apply to target policy
        judge_provider: Provider for judge (openai, anthropic, google, hf, mock)
        judge_model: Model name for judge
        estimator_name: Estimator name (IPS, SNIPS, DRCPO, MRDR)
    """
    if target_changes is None:
        target_changes = {}

    # Create target policy with changes applied
    target_kwargs = {
        "model_name": target_model,
        "provider": target_provider,
        "mc_samples": 5,
        **target_changes,
    }

    return (
        ConfigurationBuilder()
        .paths(work_dir)
        .dataset(dataset_name)
        .logging_policy(logging_model, provider=logging_provider)
        .add_target_policy("target", **target_kwargs)
        .judge(judge_provider, judge_model)
        .estimator(estimator_name)
        .build()
    )


def multi_policy_config(
    work_dir: str = "./outputs/multi_policy",
    dataset_name: str = "./data/test.jsonl",
    logging_model: str = "gpt-4o-mini",
    logging_provider: str = "fireworks",
    target_policies: Optional[List[Dict[str, Any]]] = None,
    judge_provider: str = "fireworks",
    judge_model: str = "gpt-4o-mini",
    estimator_name: str = "DRCPO",
) -> CJEConfig:
    """Create a multi-policy configuration.

    Args:
        work_dir: Working directory for outputs
        dataset_name: Dataset name or path to JSONL file
        logging_model: Model name for logging policy
        logging_provider: Provider for logging policy (hf, openai, anthropic, google)
        target_policies: List of target policy configurations
        judge_provider: Provider for judge (openai, anthropic, google, hf, mock)
        judge_model: Model name for judge
        estimator_name: Estimator name (IPS, SNIPS, DRCPO, MRDR)
    """
    if target_policies is None:
        target_policies = [
            {"name": "conservative", "temperature": 0.1},
            {"name": "balanced", "temperature": 0.7},
            {"name": "creative", "temperature": 1.2},
        ]

    builder = (
        ConfigurationBuilder()
        .paths(work_dir)
        .dataset(dataset_name)
        .logging_policy(logging_model, provider=logging_provider)
        .judge(judge_provider, judge_model)
        .estimator(estimator_name)
    )

    for policy_config in target_policies:
        name = policy_config.pop("name", f"policy_{len(builder._target_policies)}")
        model_name = policy_config.pop("model_name", logging_model)
        provider = policy_config.pop(
            "provider", logging_provider
        )  # Default to same as logging
        mc_samples = policy_config.pop("mc_samples", 5)

        builder.add_target_policy(
            name=name,
            model_name=model_name,
            provider=provider,
            mc_samples=mc_samples,
            **policy_config,
        )

    return builder.build()


def get_example_configs() -> Dict[str, CJEConfig]:
    """Get pre-built example configurations."""
    return {
        "simple": simple_config(),
        "prompt_comparison": simple_config(
            target_changes={"system_prompt": "You are a friendly, helpful assistant."}
        ),
        "model_upgrade": simple_config(
            logging_model="gpt-3.5-turbo",
            target_model="gpt-4o",
            judge_provider="fireworks",
            judge_model="gpt-4o",
        ),
        "temperature_tuning": simple_config(target_changes={"temperature": 0.1}),
        "multi_policy": multi_policy_config(),
    }


def to_dict(config: CJEConfig) -> Dict[str, Any]:
    """Convert CJEConfig to dictionary representation."""
    from dataclasses import asdict

    return asdict(config)


def from_dict(config_dict: Dict[str, Any]) -> CJEConfig:
    """Create CJEConfig from dictionary representation using ConfigurationBuilder.

    This ensures that YAML configs go through the same validation logic as
    the ConfigurationBuilder API, maintaining a single source of truth.

    Args:
        config_dict: Dictionary containing configuration sections

    Returns:
        CJEConfig object

    Raises:
        ConfigurationError: If configuration is invalid or missing required sections
    """
    try:
        # Start with a new builder
        builder = ConfigurationBuilder()

        # 1. Paths configuration
        if "paths" not in config_dict:
            raise ConfigurationError("Missing required configuration section: 'paths'")

        paths_config = config_dict["paths"]
        work_dir = paths_config.get("work_dir", "./outputs/experiment")
        builder.paths(work_dir)

        # 2. Dataset configuration
        if "dataset" not in config_dict:
            raise ConfigurationError(
                "Missing required configuration section: 'dataset'"
            )

        dataset_config = config_dict["dataset"]
        if "name" not in dataset_config:
            raise ConfigurationError(
                "Dataset configuration missing required field: 'name'"
            )

        builder.dataset(
            name=dataset_config["name"],
            split=dataset_config.get("split", "test"),
            sample_limit=dataset_config.get("sample_limit"),
        )

        # 3. Logging policy configuration
        if "logging_policy" not in config_dict:
            raise ConfigurationError(
                "Missing required configuration section: 'logging_policy'"
            )

        logging_config = config_dict["logging_policy"]
        if "model_name" not in logging_config:
            raise ConfigurationError(
                "Logging policy missing required field: 'model_name'"
            )
        if "provider" not in logging_config:
            raise ConfigurationError(
                "Logging policy missing required field: 'provider'"
            )

        # Extract logging policy fields
        logging_kwargs = {
            k: v for k, v in logging_config.items() if k not in ["model_name"]
        }
        builder.logging_policy(logging_config["model_name"], **logging_kwargs)

        # 4. Target policies configuration
        if "target_policies" not in config_dict:
            raise ConfigurationError(
                "Missing required configuration section: 'target_policies'"
            )

        target_policies_config = config_dict["target_policies"]
        if not target_policies_config:
            raise ConfigurationError("target_policies cannot be empty")

        for i, target_config in enumerate(target_policies_config):
            if "model_name" not in target_config:
                raise ConfigurationError(
                    f"Target policy {i} missing required field: 'model_name'"
                )
            if "provider" not in target_config:
                raise ConfigurationError(
                    f"Target policy {i} missing required field: 'provider'"
                )

            # Extract name and other fields
            name = target_config.get("name", f"target_policy_{i}")
            target_kwargs = {
                k: v
                for k, v in target_config.items()
                if k not in ["name", "model_name"]
            }

            builder.add_target_policy(
                name=name, model_name=target_config["model_name"], **target_kwargs
            )

        # 5. Judge configuration
        if "judge" not in config_dict:
            raise ConfigurationError("Missing required configuration section: 'judge'")

        judge_config = config_dict["judge"]
        if "provider" not in judge_config:
            raise ConfigurationError(
                "Judge configuration missing required field: 'provider'"
            )
        if "model_name" not in judge_config:
            raise ConfigurationError(
                "Judge configuration missing required field: 'model_name'"
            )

        # Extract judge fields
        judge_kwargs = {
            k: v for k, v in judge_config.items() if k not in ["provider", "model_name"]
        }
        builder.judge(
            judge_config["provider"], judge_config["model_name"], **judge_kwargs
        )

        # 6. Estimator configuration
        if "estimator" not in config_dict:
            raise ConfigurationError(
                "Missing required configuration section: 'estimator'"
            )

        estimator_config = config_dict["estimator"]
        if "name" not in estimator_config:
            raise ConfigurationError(
                "Estimator configuration missing required field: 'name'"
            )

        # Extract estimator fields
        estimator_kwargs = {
            k: v for k, v in estimator_config.items() if k not in ["name"]
        }
        builder.estimator(estimator_config["name"], **estimator_kwargs)

        # 7. Oracle configuration (optional)
        if "oracle" in config_dict:
            oracle_config = config_dict["oracle"]
            builder.oracle(**oracle_config)

        # Build the final configuration - this will run all the same validation logic
        return builder.build()

    except KeyError as e:
        raise ConfigurationError(f"Missing required configuration field: {e}")
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(f"Configuration parsing error: {e}")


def validate_configuration(config_dict: Dict[str, Any]) -> List[str]:
    """Validate configuration and return list of errors.

    Uses the unified ConfigurationBuilder path to ensure consistent validation.

    Args:
        config_dict: Dictionary containing configuration sections

    Returns:
        List of error messages. Empty list means configuration is valid.
    """
    try:
        from_dict(config_dict)
        return []
    except ConfigurationError as e:
        return [str(e)]
