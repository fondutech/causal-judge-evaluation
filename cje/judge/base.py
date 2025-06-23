from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Protocol, Set
import jinja2
from cje.utils.progress import track
from .judges import Judge  # Import Judge base class


class JudgeProtocol(Protocol):
    """Protocol for judge implementations."""

    def score(self, context: str, response: str) -> float:
        """Score a single context-response pair."""
        ...

    def score_batch(self, samples: List[Dict[str, str]]) -> List[float]:
        """Score a batch of context-response pairs."""
        ...


@dataclass
class JudgeConfig:
    """Base configuration for judges."""

    name: str
    template: str = "quick_judge"
    temperature: float = 0.0
    max_retries: int = 3
    timeout: int = 30
    max_tokens: int = 100
    custom_template: Optional[str] = None
    template_variables: Dict[str, Any] = field(default_factory=dict)
    skip: bool = False

    # Structured output settings
    use_structured_output: bool = True
    structured_output_schema: str = (
        "JudgeEvaluation"  # JudgeScore, JudgeEvaluation, DetailedJudgeEvaluation
    )
    structured_output_method: str = (
        "auto"  # auto, function_calling, json_schema, json_mode
    )

    # Valid template names (will be populated from templates module)
    _VALID_TEMPLATES: Set[str] = field(default_factory=set, init=False, repr=False)

    # Valid structured output schemas
    _VALID_SCHEMAS: Set[str] = field(
        default_factory=lambda: {
            "JudgeScore",
            "JudgeScoreWithCI",
            "JudgeEvaluation",
            "DetailedJudgeEvaluation",
        },
        init=False,
        repr=False,
    )

    # Valid structured output methods
    _VALID_METHODS: Set[str] = field(
        default_factory=lambda: {
            "auto",
            "function_calling",
            "json_schema",
            "json_mode",
        },
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._populate_valid_templates()
        self.validate()

    def _populate_valid_templates(self) -> None:
        """Populate valid templates from the templates module."""
        try:
            from ..prompts import JUDGE_TEMPLATES

            # Get all judge templates from the new clean templates
            self._VALID_TEMPLATES = set(JUDGE_TEMPLATES.keys())
        except ImportError:
            # Fallback to basic templates if import fails
            self._VALID_TEMPLATES = {
                "deterministic",
                "confidence_interval",
                "simple",
                "comparative",
            }

    def _get_validation_errors(self) -> List[str]:
        """Get validation errors without raising an exception.

        Returns:
            List of validation error messages
        """
        errors = []

        # Validate name
        if not self.name or not isinstance(self.name, str):
            errors.append("name must be a non-empty string")

        # Validate template
        if not self.custom_template and self.template not in self._VALID_TEMPLATES:
            available = sorted(self._VALID_TEMPLATES)
            errors.append(
                f"template '{self.template}' is not valid. "
                f"Available templates: {available}"
            )

        # Validate temperature
        if not isinstance(self.temperature, (int, float)):
            errors.append("temperature must be a number")
        elif not (0.0 <= self.temperature <= 2.0):
            errors.append("temperature must be between 0.0 and 2.0")

        # Validate max_retries
        if not isinstance(self.max_retries, int):
            errors.append("max_retries must be an integer")
        elif self.max_retries < 1:
            errors.append("max_retries must be at least 1")
        elif self.max_retries > 10:
            errors.append("max_retries should not exceed 10 (too many retries)")

        # Validate timeout
        if not isinstance(self.timeout, int):
            errors.append("timeout must be an integer")
        elif self.timeout < 1:
            errors.append("timeout must be at least 1 second")
        elif self.timeout > 300:
            errors.append("timeout should not exceed 300 seconds")

        # Validate template_variables
        if not isinstance(self.template_variables, dict):
            errors.append("template_variables must be a dictionary")

        # Validate structured output settings
        if not isinstance(self.use_structured_output, bool):
            errors.append("use_structured_output must be a boolean")

        if self.structured_output_schema not in self._VALID_SCHEMAS:
            available = sorted(self._VALID_SCHEMAS)
            errors.append(
                f"structured_output_schema '{self.structured_output_schema}' is not valid. "
                f"Available schemas: {available}"
            )

        if self.structured_output_method not in self._VALID_METHODS:
            available = sorted(self._VALID_METHODS)
            errors.append(
                f"structured_output_method '{self.structured_output_method}' is not valid. "
                f"Available methods: {available}"
            )

        return errors

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        errors = self._get_validation_errors()
        if errors:
            raise ValueError(f"Invalid judge configuration: {'; '.join(errors)}")


@dataclass
class APIJudgeConfig(JudgeConfig):
    """Configuration for API-based judges."""

    provider: str = "openai"  # openai, anthropic, google, etc.
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # Valid providers - use lazy import to avoid circular dependency
    _VALID_PROVIDERS: Set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize valid providers after construction."""
        # Set up providers BEFORE calling super().__post_init__() which triggers validation
        try:
            from cje.providers import list_providers

            available_providers = list_providers()
            self._VALID_PROVIDERS = set(available_providers) | {"mock"}
        except Exception as e:
            # Fallback if provider system not available
            import warnings

            warnings.warn(
                f"Could not load provider system: {e}, using fallback providers"
            )
            self._VALID_PROVIDERS = {
                "openai",
                "anthropic",
                "google",
                "fireworks",
                "together",
                "mock",
            }

        # Now call parent's __post_init__ which will trigger validation
        super().__post_init__()

    def _get_validation_errors(self) -> List[str]:
        """Get validation errors without raising an exception."""
        # Get parent errors first
        errors = super()._get_validation_errors()

        # Validate provider
        if not isinstance(self.provider, str):
            errors.append("provider must be a string")
        elif self.provider not in self._VALID_PROVIDERS:
            available = sorted(self._VALID_PROVIDERS)
            errors.append(
                f"provider '{self.provider}' is not supported. "
                f"Available providers: {available}"
            )

        # Validate model_name
        if not self.model_name or not isinstance(self.model_name, str):
            errors.append("model_name must be a non-empty string")

        # Validate max_tokens
        if not isinstance(self.max_tokens, int):
            errors.append("max_tokens must be an integer")
        elif self.max_tokens < 1:
            errors.append("max_tokens must be at least 1")
        elif self.max_tokens > 8192:
            errors.append("max_tokens should not exceed 8192 (most models' limit)")

        # Validate base_url if provided
        if self.base_url is not None:
            if not isinstance(self.base_url, str):
                errors.append("base_url must be a string")
            elif not self.base_url.startswith(("http://", "https://")):
                errors.append("base_url must start with http:// or https://")

        # Validate api_key if provided
        if self.api_key is not None:
            if not isinstance(self.api_key, str):
                errors.append("api_key must be a string")
            elif len(self.api_key.strip()) == 0:
                errors.append("api_key cannot be empty")

        return errors

    def validate(self) -> None:
        """Validate API judge configuration."""
        errors = self._get_validation_errors()
        if errors:
            raise ValueError(f"Invalid API judge configuration: {'; '.join(errors)}")


@dataclass
class LocalJudgeConfig(JudgeConfig):
    """Configuration for local model judges."""

    model_name: str = "prometheus-eval/prometheus-13b-v1.0"
    device: str = "auto"
    torch_dtype: str = "float16"
    batch_size: int = 8

    # Valid torch dtypes
    _VALID_TORCH_DTYPES: Set[str] = field(
        default_factory=lambda: {"float16", "float32", "bfloat16"},
        init=False,
        repr=False,
    )

    # Valid devices
    _VALID_DEVICES: Set[str] = field(
        default_factory=lambda: {"auto", "cpu", "cuda", "mps"}, init=False, repr=False
    )

    def _get_validation_errors(self) -> List[str]:
        """Get validation errors without raising an exception."""
        # Get parent errors first
        errors = super()._get_validation_errors()

        # Validate model_name
        if not self.model_name or not isinstance(self.model_name, str):
            errors.append("model_name must be a non-empty string")

        # Validate device
        if not isinstance(self.device, str):
            errors.append("device must be a string")
        elif self.device not in self._VALID_DEVICES and not self.device.startswith(
            "cuda:"
        ):
            available = sorted(self._VALID_DEVICES)
            errors.append(
                f"device '{self.device}' is not valid. "
                f"Available devices: {available} or 'cuda:N' for specific GPU"
            )

        # Validate torch_dtype
        if not isinstance(self.torch_dtype, str):
            errors.append("torch_dtype must be a string")
        elif self.torch_dtype not in self._VALID_TORCH_DTYPES:
            available = sorted(self._VALID_TORCH_DTYPES)
            errors.append(
                f"torch_dtype '{self.torch_dtype}' is not valid. "
                f"Available types: {available}"
            )

        # Validate batch_size
        if not isinstance(self.batch_size, int):
            errors.append("batch_size must be an integer")
        elif self.batch_size < 1:
            errors.append("batch_size must be at least 1")
        elif self.batch_size > 128:
            errors.append("batch_size should not exceed 128 (memory concerns)")

        return errors

    def validate(self) -> None:
        """Validate local judge configuration."""
        errors = self._get_validation_errors()
        if errors:
            raise ValueError(f"Invalid local judge configuration: {'; '.join(errors)}")


class BaseJudge:
    """Base class for all judge implementations."""

    def __init__(self, config: JudgeConfig):
        self.config = config
        self._template_env = jinja2.Environment(
            loader=jinja2.DictLoader(self._get_templates())
        )

    def _get_templates(self) -> Dict[str, str]:
        """Get available prompt templates."""
        from ..prompts import JUDGE_TEMPLATES

        # Return judge templates in the expected format
        return {
            name: template["template"] for name, template in JUDGE_TEMPLATES.items()
        }

    def _render_prompt(self, context: str, response: str) -> str:
        """Render the prompt template with context and response."""
        if self.config.custom_template:
            template_str = self.config.custom_template
        else:
            templates = self._get_templates()
            template_str = templates.get(
                self.config.template,
                templates.get(
                    "deterministic", list(templates.values())[0]
                ),  # Safe fallback
            )

        template = jinja2.Template(template_str)
        result = template.render(
            context=context, response=response, **self.config.template_variables
        )
        return str(result)
