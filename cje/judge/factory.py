from __future__ import annotations
from typing import Dict, Any, Union, Optional, List
import logging

from .base import (
    BaseJudge,
    JudgeProtocol,
    CachedJudge,
    APIJudgeConfig,
    LocalJudgeConfig,
)
from .api_judge import APIJudge
from .local_judge import LocalJudge
from .judges import Judge

logger = logging.getLogger(__name__)


class JudgeFactory:
    """Simplified factory for creating judges with the new provider system."""

    # Available prompt templates
    AVAILABLE_TEMPLATES = [
        "quick_judge",
        "comprehensive_judge",
        "reasoning_judge",
        "domain_judge",
    ]

    # Available structured output schemas
    AVAILABLE_SCHEMAS = [
        "JudgeScore",  # Basic score only
        "JudgeEvaluation",  # Score + reasoning + confidence
        "DetailedJudgeEvaluation",  # All evaluation metrics
    ]

    @classmethod
    def create(
        cls,
        provider: str,
        model: str,
        template: Optional[str] = None,
        structured_output_schema: Optional[str] = None,
        structured_output_method: Optional[str] = None,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_size: int = 1000,
        **kwargs: Any,
    ) -> Judge:
        """Create a judge with explicit configuration.

        Args:
            provider: Provider name ('openai', 'anthropic', 'google', 'fireworks')
            model: Model name (e.g., 'gpt-4o-mini', 'claude-3-haiku-20240307')
            template: Prompt template name (default: provider default)
            structured_output_schema: Schema for output (default: 'JudgeEvaluation')
            structured_output_method: Method to use (default: provider recommended)
            temperature: Temperature for generation (default: 0.0)
            api_key: API key (optional, can use environment variable)
            use_cache: Whether to wrap judge with caching (default: True)
            cache_size: Size of the cache (default: 1000)
            **kwargs: Additional configuration options

        Returns:
            Configured judge instance

        Example:
            judge = JudgeFactory.create(
                provider="fireworks",
                model="accounts/fireworks/models/deepseek-r1-0528",
                template="comprehensive_judge",
                structured_output_schema="DetailedJudgeEvaluation"
            )
        """
        # Get provider info from new registry
        from cje.providers import get_provider

        provider_plugin = get_provider(provider)
        if provider_plugin is None:
            from cje.providers import list_providers

            available = list_providers()
            raise ValueError(
                f"Unknown provider '{provider}'. " f"Available providers: {available}"
            )

        # Set defaults from provider plugin
        if template is None:
            template = provider_plugin.default_template
        if structured_output_schema is None:
            structured_output_schema = "JudgeEvaluation"
        if structured_output_method is None:
            structured_output_method = provider_plugin.recommended_method

        # Validate template
        if template not in cls.AVAILABLE_TEMPLATES:
            raise ValueError(
                f"Unknown template '{template}'. "
                f"Available templates: {cls.AVAILABLE_TEMPLATES}"
            )

        # Validate schema
        if structured_output_schema not in cls.AVAILABLE_SCHEMAS:
            raise ValueError(
                f"Unknown schema '{structured_output_schema}'. "
                f"Available schemas: {cls.AVAILABLE_SCHEMAS}"
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

        # Create configuration with provider defaults
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

        judge: Judge = APIJudge(config)

        # Wrap with cache if requested
        if use_cache:
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
