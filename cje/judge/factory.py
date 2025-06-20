"""Unified factory for creating judges that return JudgeScore with uncertainty."""

from __future__ import annotations
from typing import Dict, Any, Union, Optional, List, Literal
import logging

from .base import (
    APIJudgeConfig,
    LocalJudgeConfig,
)
from .api_judge import APIJudge, DeterministicAPIJudge, MCAPIJudge
from .local_judge import LocalJudge  # Will need to update this too
from .judges import Judge

logger = logging.getLogger(__name__)


class JudgeFactory:
    """Factory for creating unified judges with uncertainty support."""

    # Available prompt templates
    AVAILABLE_TEMPLATES = [
        "quick_judge",
        "comprehensive_judge",
        "reasoning_judge",
        "domain_judge",
    ]

    # Available structured output schemas
    AVAILABLE_SCHEMAS = [
        "JudgeScore",  # Basic score with uncertainty
        "JudgeEvaluation",  # Score + reasoning + confidence
        "DetailedJudgeEvaluation",  # All evaluation metrics
    ]

    # Available uncertainty estimation methods
    UNCERTAINTY_METHODS = [
        "deterministic",  # Always returns variance=0
        "structured",  # Model estimates its own uncertainty
        "monte_carlo",  # Multiple samples to estimate variance
    ]

    @classmethod
    def create(
        cls,
        provider: str,
        model: str,
        template: Optional[str] = None,
        structured_output_schema: Optional[str] = None,
        structured_output_method: Optional[str] = None,
        uncertainty_method: Literal[
            "deterministic", "structured", "monte_carlo"
        ] = "structured",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_size: int = 1000,
        mc_samples: int = 5,
        **kwargs: Any,
    ) -> Judge:
        """Create a unified judge with uncertainty support.

        Args:
            provider: Provider name ('openai', 'anthropic', 'google', 'fireworks')
            model: Model name (e.g., 'gpt-4o-mini', 'claude-3-haiku-20240307')
            template: Prompt template name (default: provider default)
            structured_output_schema: Schema for output (default: 'JudgeScore')
            structured_output_method: Method to use (default: provider recommended)
            uncertainty_method: How to estimate uncertainty:
                - 'deterministic': Always returns variance=0
                - 'structured': Model estimates its own uncertainty (default)
                - 'monte_carlo': Sample multiple times to estimate variance
            temperature: Temperature for generation (default: 0.0)
            api_key: API key (optional, can use environment variable)
            use_cache: Whether to wrap judge with caching (default: True)
            cache_size: Size of the cache (default: 1000)
            mc_samples: Number of samples for Monte Carlo (default: 5)
            **kwargs: Additional configuration options

        Returns:
            Configured judge instance that returns JudgeScore

        Example:
            # Deterministic judge (no uncertainty)
            judge = JudgeFactory.create(
                provider="openai",
                model="gpt-4o-mini",
                uncertainty_method="deterministic"
            )

            # Judge that estimates its own uncertainty
            judge = JudgeFactory.create(
                provider="anthropic",
                model="claude-3-haiku-20240307",
                uncertainty_method="structured",
                template="comprehensive_judge"
            )

            # Monte Carlo uncertainty estimation
            judge = JudgeFactory.create(
                provider="fireworks",
                model="accounts/fireworks/models/llama-v3-70b-instruct",
                uncertainty_method="monte_carlo",
                temperature=0.3,
                mc_samples=10
            )
        """
        # Get provider info from registry
        from cje.providers import get_provider

        provider_plugin = get_provider(provider)
        if provider_plugin is None:
            from cje.providers import list_providers

            available = list_providers()
            raise ValueError(
                f"Unknown provider '{provider}'. Available providers: {available}"
            )

        # Set defaults from provider plugin
        if template is None:
            template = provider_plugin.default_template
        if structured_output_schema is None:
            # Default to basic JudgeScore for unified interface
            structured_output_schema = "JudgeScore"
        if structured_output_method is None:
            structured_output_method = provider_plugin.recommended_method

        # Validate inputs
        if template not in cls.AVAILABLE_TEMPLATES:
            raise ValueError(
                f"Unknown template '{template}'. "
                f"Available templates: {cls.AVAILABLE_TEMPLATES}"
            )

        if structured_output_schema not in cls.AVAILABLE_SCHEMAS:
            raise ValueError(
                f"Unknown schema '{structured_output_schema}'. "
                f"Available schemas: {cls.AVAILABLE_SCHEMAS}"
            )

        if uncertainty_method not in cls.UNCERTAINTY_METHODS:
            raise ValueError(
                f"Unknown uncertainty method '{uncertainty_method}'. "
                f"Available methods: {cls.UNCERTAINTY_METHODS}"
            )

        # Validate method
        supported_methods = provider_plugin.supported_methods
        if structured_output_method == "auto":
            structured_output_method = provider_plugin.recommended_method
        elif structured_output_method not in supported_methods:
            raise ValueError(
                f"Method '{structured_output_method}' not supported by {provider}. "
                f"Supported methods: {supported_methods}"
            )

        # Create configuration
        config = APIJudgeConfig(
            name=f"{provider}-{model}",
            provider=provider,
            model_name=model,
            template=template,
            temperature=temperature,
            structured_output_schema=structured_output_schema,
            structured_output_method=structured_output_method,
            api_key=api_key,
            base_url=provider_plugin.base_url,
            **kwargs,
        )

        # Create appropriate judge based on uncertainty method
        if uncertainty_method == "deterministic":
            judge: Judge = DeterministicAPIJudge(config)
        elif uncertainty_method == "monte_carlo":
            # Ensure temperature > 0 for Monte Carlo
            if config.temperature == 0:
                config.temperature = 0.3
            judge = MCAPIJudge(config, n_samples=mc_samples)
        else:  # structured
            judge = APIJudge(config)

        # Wrap with cache if requested
        if use_cache:
            from .cached_judge import CachedJudge

            return CachedJudge(base_judge=judge, cache_size=cache_size)

        return judge

    @classmethod
    def from_config(cls, config: Union[APIJudgeConfig, LocalJudgeConfig]) -> Judge:
        """Create a judge from a configuration object.

        Args:
            config: Judge configuration object

        Returns:
            Configured judge instance
        """
        if isinstance(config, APIJudgeConfig):
            return APIJudge(config)
        elif isinstance(config, LocalJudgeConfig):
            from .local_judge import LocalJudge

            return LocalJudge(config)
        else:
            raise ValueError(f"Unknown config type: {type(config)}")

    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        from cje.providers import list_providers

        return list_providers()

    @classmethod
    def list_templates(cls) -> List[str]:
        """List available prompt templates."""
        return cls.AVAILABLE_TEMPLATES.copy()

    @classmethod
    def list_schemas(cls) -> List[str]:
        """List available output schemas."""
        return cls.AVAILABLE_SCHEMAS.copy()

    @classmethod
    def list_uncertainty_methods(cls) -> List[str]:
        """List available uncertainty estimation methods."""
        return cls.UNCERTAINTY_METHODS.copy()

    @classmethod
    def get_provider_info(cls, provider: str) -> Dict[str, Any]:
        """Get information about a specific provider."""
        from cje.providers import get_provider

        provider_plugin = get_provider(provider)
        if provider_plugin is None:
            raise ValueError(f"Unknown provider: {provider}")

        return {
            "name": provider_plugin.name,
            "capabilities": provider_plugin.capabilities,
            "supported_methods": provider_plugin.supported_methods,
            "recommended_method": provider_plugin.recommended_method,
            "default_template": provider_plugin.default_template,
            "env_var": provider_plugin.env_var,
            "base_url": provider_plugin.base_url,
        }

    @classmethod
    def print_capabilities(cls) -> None:
        """Print provider capabilities table."""
        from cje.providers import print_provider_capabilities

        print_provider_capabilities()

        # Also print uncertainty methods
        print("\nUncertainty Estimation Methods:")
        print("- deterministic: Always returns variance=0 (fastest)")
        print("- structured: Model estimates its own uncertainty (recommended)")
        print("- monte_carlo: Sample multiple times to estimate variance (slowest)")
