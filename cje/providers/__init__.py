"""Plugin-based provider system for CJE.

This module provides a clean, extensible provider architecture that eliminates
circular imports and simplifies adding new providers.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Type, Any, Optional, Set
from pathlib import Path
import importlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProviderCapabilities:
    """Provider capability specification."""

    supports_logprobs: bool = False
    supports_full_sequence_logprobs: bool = False
    supports_function_calling: bool = False
    supports_json_schema: bool = False
    supports_json_mode: bool = False
    max_tokens: int = 4096
    default_temperature: float = 0.0


@dataclass
class ProviderPlugin:
    """Provider plugin definition."""

    name: str
    provider_class: Type[Any]
    structured_class: Optional[Type[Any]] = None
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)
    default_template: str = "comprehensive_judge"
    env_var: Optional[str] = None
    base_url: Optional[str] = None

    @property
    def supported_methods(self) -> List[str]:
        """Get list of supported structured output methods."""
        methods = []
        if self.capabilities.supports_function_calling:
            methods.append("function_calling")
        if self.capabilities.supports_json_schema:
            methods.append("json_schema")
        if self.capabilities.supports_json_mode:
            methods.append("json_mode")
        return methods

    @property
    def recommended_method(self) -> str:
        """Get recommended structured output method."""
        if self.capabilities.supports_json_schema:
            return "json_schema"
        elif self.capabilities.supports_function_calling:
            return "function_calling"
        elif self.capabilities.supports_json_mode:
            return "json_mode"
        else:
            return "function_calling"  # fallback


class ProviderRegistry:
    """Clean provider registry with auto-discovery."""

    def __init__(self) -> None:
        self._plugins: Dict[str, ProviderPlugin] = {}
        self._discovered = False

    def register(self, plugin: ProviderPlugin) -> None:
        """Register a provider plugin."""
        self._plugins[plugin.name] = plugin
        logger.debug(f"Registered provider: {plugin.name}")

    def get(self, name: str) -> Optional[ProviderPlugin]:
        """Get a provider plugin by name."""
        self._ensure_discovered()
        return self._plugins.get(name)

    def list_providers(self) -> List[str]:
        """List all available provider names."""
        self._ensure_discovered()
        return list(self._plugins.keys())

    def list_plugins(self) -> Dict[str, ProviderPlugin]:
        """Get all provider plugins."""
        self._ensure_discovered()
        return self._plugins.copy()

    def get_providers_with_capability(self, capability: str) -> List[str]:
        """Get providers that support a specific capability."""
        self._ensure_discovered()
        result = []
        for name, plugin in self._plugins.items():
            if getattr(plugin.capabilities, capability, False):
                result.append(name)
        return result

    def get_policy_providers(self) -> List[str]:
        """Get providers that can be used as target policies."""
        return self.get_providers_with_capability("supports_full_sequence_logprobs")

    def get_judge_providers(self) -> List[str]:
        """Get providers that can be used as judges (all providers)."""
        return self.list_providers()

    def _ensure_discovered(self) -> None:
        """Ensure providers have been auto-discovered."""
        if not self._discovered:
            self.discover_providers()
            self._discovered = True

    def discover_providers(self) -> None:
        """Auto-discover providers from the judge providers package."""
        providers_dir = Path(__file__).parent.parent / "judge" / "providers"

        if not providers_dir.exists():
            logger.warning(f"Providers directory not found: {providers_dir}")
            return

        # Define provider specifications
        provider_specs = {
            "openai": {
                "module": "openai_provider",
                "class": "OpenAIProvider",
                "capabilities": ProviderCapabilities(
                    supports_logprobs=True,
                    supports_function_calling=True,
                    supports_json_schema=True,
                ),
                "env_var": "OPENAI_API_KEY",
            },
            "anthropic": {
                "module": "anthropic_provider",
                "class": "AnthropicProvider",
                "capabilities": ProviderCapabilities(
                    supports_function_calling=True,
                ),
                "env_var": "ANTHROPIC_API_KEY",
            },
            "google": {
                "module": "google_provider",
                "class": "GoogleProvider",
                "capabilities": ProviderCapabilities(
                    supports_function_calling=True,
                ),
                "env_var": "GOOGLE_API_KEY",
            },
            "fireworks": {
                "module": "fireworks_provider",
                "class": "FireworksProvider",
                "capabilities": ProviderCapabilities(
                    supports_logprobs=True,
                    supports_full_sequence_logprobs=True,
                    supports_function_calling=True,
                    supports_json_schema=True,
                ),
                "env_var": "FIREWORKS_API_KEY",
                "base_url": "https://api.fireworks.ai/inference/v1",
            },
            "together": {
                "module": "together_provider",
                "class": "TogetherProvider",
                "capabilities": ProviderCapabilities(
                    supports_logprobs=True,
                    supports_full_sequence_logprobs=True,
                    supports_function_calling=True,
                    supports_json_schema=True,
                ),
                "env_var": "TOGETHER_API_KEY",
                "base_url": "https://api.together.xyz/v1",
            },
        }

        # Try to import and register each provider
        for name, spec in provider_specs.items():
            try:
                module_name = f"cje.judge.providers.{spec['module']}"
                module = importlib.import_module(module_name)
                provider_class = getattr(module, str(spec["class"]))

                # Try to get structured class
                structured_class = None
                structured_module_name = f"cje.judge.providers.structured_{name}"
                try:
                    structured_module = importlib.import_module(structured_module_name)
                    structured_class_name = f"Structured{spec['class']}"
                    structured_class = getattr(
                        structured_module, structured_class_name, None
                    )
                except (ImportError, AttributeError):
                    # Use same class for both if structured version not available
                    structured_class = provider_class

                plugin = ProviderPlugin(
                    name=name,
                    provider_class=provider_class,
                    structured_class=structured_class,
                    capabilities=spec["capabilities"],  # type: ignore[arg-type]
                    env_var=spec.get("env_var"),  # type: ignore[arg-type]
                    base_url=spec.get("base_url"),  # type: ignore[arg-type]
                )

                self.register(plugin)

            except ImportError as e:
                logger.debug(f"Provider {name} not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to register provider {name}: {e}")

    def print_capabilities(self) -> None:
        """Print a summary of provider capabilities."""
        from cje.utils.progress import console
        from rich.table import Table

        self._ensure_discovered()

        table = Table(title="🔧 Provider Capabilities")
        table.add_column("Provider", style="bold")
        table.add_column("Output Logprobs", justify="center")
        table.add_column("Full Sequence", justify="center")
        table.add_column("Target Policy", justify="center", style="bold")
        table.add_column("Judge", justify="center")
        table.add_column("Methods", style="dim")

        for name, plugin in self._plugins.items():
            caps = plugin.capabilities

            output_logprobs = "✅" if caps.supports_logprobs else "❌"
            full_logprobs = "✅" if caps.supports_full_sequence_logprobs else "❌"
            can_be_policy = "✅" if caps.supports_full_sequence_logprobs else "❌"
            can_be_judge = "✅"
            methods = ", ".join(plugin.supported_methods) or "none"

            table.add_row(
                name.title(),
                output_logprobs,
                full_logprobs,
                can_be_policy,
                can_be_judge,
                methods,
            )

        console.print(table)
        console.print("\n[bold blue]💡 Key:[/bold blue]")
        console.print(
            "• [bold green]Full Sequence[/bold green] = Can score P(input+output) for policy evaluation"
        )
        console.print(
            "• [bold yellow]Output Logprobs[/bold yellow] = Can score P(output|input) for judges"
        )


# Global registry instance
_registry = ProviderRegistry()


def get_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    return _registry


def get_provider(name: str) -> Optional[ProviderPlugin]:
    """Get a provider plugin by name."""
    return _registry.get(name)


def list_providers() -> List[str]:
    """List all available providers."""
    return _registry.list_providers()


def get_policy_providers() -> List[str]:
    """Get providers suitable for target policies."""
    return _registry.get_policy_providers()


def get_judge_providers() -> List[str]:
    """Get providers suitable for judges."""
    return _registry.get_judge_providers()


def print_provider_capabilities() -> None:
    """Print provider capabilities table."""
    _registry.print_capabilities()
