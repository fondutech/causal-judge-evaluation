from typing import List, Optional, Tuple, Dict, Any, Type
from dataclasses import dataclass
import warnings

from .utils.imports import ImportChecker


@dataclass
class ProviderInfo:
    """Information about a provider."""

    provider_cls: Type[Any]
    structured_cls: Type[Any]
    supports_logprobs: bool  # Output logprobs for generated text
    supports_full_sequence_logprobs: (
        bool  # Input + output logprobs (needed for policy evaluation)
    )
    default_template: str
    supported_methods: List[str]
    recommended_method: str
    base_url: Optional[str] = None
    env_var: Optional[str] = None
    name: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None


class ProviderRegistry:
    """Registry for managing AI providers and their capabilities."""

    def __init__(self) -> None:
        self._provider_info: Dict[str, ProviderInfo] = {}

    def add_provider_info(self, name: str, info: ProviderInfo) -> None:
        """Add detailed provider information."""
        self._provider_info[name] = info

    def get_providers_supporting_full_sequence_logprobs(self) -> List[str]:
        """Get list of provider names that support full sequence logprobs (needed for policy evaluation)."""
        return [
            name
            for name, info in self._provider_info.items()
            if info.supports_full_sequence_logprobs
        ]

    def supports_full_sequence_logprobs(self, provider_name: str) -> bool:
        """Check if a provider supports full sequence logprobs."""
        info = self._provider_info.get(provider_name)
        return info.supports_full_sequence_logprobs if info else False

    def print_provider_capabilities(self) -> None:
        """Print a helpful summary of provider capabilities."""
        from .utils.progress import console
        from rich.table import Table

        table = Table(title="🔧 Provider Capabilities Summary")
        table.add_column("Provider", style="bold")
        table.add_column("Output Logprobs", justify="center")
        table.add_column("Full Sequence Logprobs", justify="center")
        table.add_column("Can be Target Policy", justify="center", style="bold")
        table.add_column("Can be Judge", justify="center")
        table.add_column("Recommended Use", style="dim")

        for name, info in self._provider_info.items():
            output_logprobs = "✅" if info.supports_logprobs else "❌"
            full_logprobs = "✅" if info.supports_full_sequence_logprobs else "❌"
            can_be_policy = "✅" if info.supports_full_sequence_logprobs else "❌"
            can_be_judge = "✅"  # All providers can be judges

            if info.supports_full_sequence_logprobs:
                recommended = "Target policies + judges"
            elif info.supports_logprobs:
                recommended = "Judges only (no input logprobs)"
            else:
                recommended = "Judges only (no logprobs)"

            table.add_row(
                name.title(),
                output_logprobs,
                full_logprobs,
                can_be_policy,
                can_be_judge,
                recommended,
            )

        console.print(table)
        console.print("\n[bold blue]💡 Key Points:[/bold blue]")
        console.print(
            "• [bold green]Full sequence logprobs[/bold green] = log P(input+output) needed for importance weighting"
        )
        console.print(
            "• [bold yellow]Output-only logprobs[/bold yellow] = log P(output|input) sufficient for judges"
        )
        console.print(
            "• Only Together AI and Fireworks support full sequence scoring for policy evaluation"
        )
        console.print(
            "• OpenAI/Anthropic/Google can be used as judges but not as target policies in IPS/SNIPS"
        )


# Global registry instance
_registry = ProviderRegistry()


def registry() -> Dict[str, ProviderInfo]:
    """Get the provider registry as a dictionary."""
    return _registry._provider_info.copy()


def get_registry() -> ProviderRegistry:
    """Get the registry instance."""
    return _registry


# Initialize registry with default providers
def _initialize_registry() -> None:
    """Initialize the registry with default provider information."""

    # Import providers individually and report missing ones clearly
    providers_to_register = {}
    missing_providers = []

    # OpenAI Provider
    try:
        from cje.judge.providers.openai import OpenAIProvider

        providers_to_register["openai"] = ProviderInfo(
            provider_cls=OpenAIProvider,
            structured_cls=OpenAIProvider,  # Same class handles both
            supports_logprobs=True,  # Output logprobs only
            supports_full_sequence_logprobs=False,  # Cannot score input+output sequences (chat models don't support completions API)
            default_template="comprehensive_judge",
            supported_methods=["function_calling", "json_schema"],
            recommended_method="json_schema",
            base_url=None,
            env_var="OPENAI_API_KEY",
            name="openai",
            capabilities={"supports_logprobs": True, "supports_structured": True},
        )
    except ImportError as e:
        missing_providers.append(("openai", "OpenAI provider", str(e)))

    # Anthropic Provider
    try:
        from cje.judge.providers.anthropic import AnthropicProvider

        providers_to_register["anthropic"] = ProviderInfo(
            provider_cls=AnthropicProvider,
            structured_cls=AnthropicProvider,
            supports_logprobs=False,
            supports_full_sequence_logprobs=False,
            default_template="comprehensive_judge",
            supported_methods=["function_calling"],
            recommended_method="function_calling",
            base_url=None,
            env_var="ANTHROPIC_API_KEY",
            name="anthropic",
            capabilities={"supports_logprobs": False, "supports_structured": True},
        )
    except ImportError as e:
        missing_providers.append(("anthropic", "Anthropic provider", str(e)))

    # Google Provider
    try:
        from cje.judge.providers.google import GoogleProvider

        providers_to_register["google"] = ProviderInfo(
            provider_cls=GoogleProvider,
            structured_cls=GoogleProvider,
            supports_logprobs=False,
            supports_full_sequence_logprobs=False,
            default_template="comprehensive_judge",
            supported_methods=["function_calling"],
            recommended_method="function_calling",
            base_url=None,
            env_var="GOOGLE_API_KEY",
            name="google",
            capabilities={"supports_logprobs": False, "supports_structured": True},
        )
    except ImportError as e:
        missing_providers.append(("google", "Google provider", str(e)))

    # Together Provider
    try:
        from cje.judge.providers.together import TogetherProvider

        providers_to_register["together"] = ProviderInfo(
            provider_cls=TogetherProvider,
            structured_cls=TogetherProvider,
            supports_logprobs=True,
            supports_full_sequence_logprobs=True,
            default_template="comprehensive_judge",
            supported_methods=["function_calling", "json_schema"],
            recommended_method="json_schema",
            base_url="https://api.together.xyz/v1",
            env_var="TOGETHER_API_KEY",
            name="together",
            capabilities={"supports_logprobs": True, "supports_structured": True},
        )
    except ImportError as e:
        missing_providers.append(("together", "Together provider", str(e)))

    # Fireworks Provider
    try:
        from cje.judge.providers.fireworks import FireworksProvider

        providers_to_register["fireworks"] = ProviderInfo(
            provider_cls=FireworksProvider,
            structured_cls=FireworksProvider,
            supports_logprobs=True,
            supports_full_sequence_logprobs=True,
            default_template="comprehensive_judge",
            supported_methods=["function_calling", "json_schema"],
            recommended_method="json_schema",
            base_url="https://api.fireworks.ai/inference/v1",
            env_var="FIREWORKS_API_KEY",
            name="fireworks",
            capabilities={"supports_logprobs": True, "supports_structured": True},
        )
    except ImportError as e:
        missing_providers.append(("fireworks", "Fireworks provider", str(e)))

    # Add successfully imported providers to registry
    for name, info in providers_to_register.items():
        _registry.add_provider_info(name, info)

    # Report missing providers
    if missing_providers:
        warnings.warn(
            f"\nSome providers could not be imported:\n"
            + "\n".join(
                [
                    f"  - {name}: {desc} (Error: {err})"
                    for name, desc, err in missing_providers
                ]
            )
            + "\n\nThese providers will not be available. Check dependencies if needed.",
            ImportWarning,
            stacklevel=2,
        )


# Initialize on import
_initialize_registry()


def print_supported_providers() -> None:
    """Convenience function to print provider capabilities."""
    registry = get_registry()
    registry.print_provider_capabilities()


def list_providers() -> List[str]:
    """List available provider names."""
    return list(_registry._provider_info.keys())


def get_provider(name: str) -> Optional[ProviderInfo]:
    """Get provider info by name."""
    return _registry._provider_info.get(name)


def print_provider_capabilities() -> None:
    """Print provider capabilities table."""
    _registry.print_provider_capabilities()
