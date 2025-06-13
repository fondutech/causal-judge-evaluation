from typing import List, Optional, Tuple, Dict, Any, Type
from dataclasses import dataclass


@dataclass
class ProviderInfo:
    """Information about a provider."""

    provider_cls: Type[Any]
    structured_cls: Type[Any]
    supports_logprobs: bool  # For backward compatibility - output logprobs
    supports_full_sequence_logprobs: (
        bool  # Input + output logprobs (needed for policy evaluation)
    )
    default_template: str
    supported_methods: List[str]
    recommended_method: str


class ProviderRegistry:
    """Registry for managing AI providers and their capabilities."""

    def __init__(self) -> None:
        self.providers: List[Tuple[Type[Any], Type[Any], bool]] = []
        self._provider_info: Dict[str, ProviderInfo] = {}

    def add_provider(
        self,
        provider_cls: Type[Any],
        structured_cls: Type[Any],
        supports_logprobs: bool,
    ) -> None:
        """Add a provider to the registry."""
        self.providers.append((provider_cls, structured_cls, supports_logprobs))

    def add_provider_info(self, name: str, info: ProviderInfo) -> None:
        """Add detailed provider information."""
        self._provider_info[name] = info

    def get_providers(self) -> List[Tuple[Type[Any], Type[Any], bool]]:
        """Get all registered providers."""
        return self.providers

    def get_provider_by_name(
        self, name: str
    ) -> Optional[Tuple[Type[Any], Type[Any], bool]]:
        """Get a provider by name."""
        for provider_cls, structured_cls, supports_logprobs in self.providers:
            if provider_cls.__name__ == name or structured_cls.__name__ == name:
                return provider_cls, structured_cls, supports_logprobs
        return None

    def get_recommended_provider(self) -> Optional[Tuple[Type[Any], Type[Any]]]:
        """Get the recommended provider (first one that supports logprobs)."""
        for provider_cls, structured_cls, supports_logprobs in self.providers:
            if supports_logprobs:
                return provider_cls, structured_cls
        return None

    def get_supported_methods(self) -> List[str]:
        """Get all supported method names."""
        return [provider_cls.__name__ for provider_cls, _, _ in self.providers]

    def get_recommended_method(self) -> Optional[str]:
        """Get the recommended method name."""
        provider_result = self.get_recommended_provider()
        if provider_result:
            provider_cls, _ = provider_result
            return provider_cls.__name__
        return None

    def get_supports_logprobs(self) -> List[bool]:
        """Get logprobs support status for all providers."""
        return [supports_logprobs for _, _, supports_logprobs in self.providers]

    def get_structured_cls_by_name(self, name: str) -> Optional[Type[Any]]:
        """Get structured class by provider name."""
        for provider_cls, structured_cls, _ in self.providers:
            if provider_cls.__name__ == name or structured_cls.__name__ == name:
                return structured_cls
        return None

    def get_provider_cls_by_name(self, name: str) -> Optional[Type[Any]]:
        """Get provider class by name."""
        for provider_cls, _, _ in self.providers:
            if provider_cls.__name__ == name:
                return provider_cls
        return None

    def get_structured_cls_by_provider(
        self, provider_cls: Type[Any]
    ) -> Optional[Type[Any]]:
        """Get structured class by provider class."""
        for p_cls, structured_cls, _ in self.providers:
            if p_cls == provider_cls:
                return structured_cls
        return None

    def get_supports_logprobs_by_provider(
        self, provider_cls: Type[Any]
    ) -> Optional[bool]:
        """Get logprobs support by provider class."""
        for p_cls, _, supports_logprobs in self.providers:
            if p_cls == provider_cls:
                return supports_logprobs
        return None

    def get_provider_cls_by_method(self, method: str) -> Optional[Type[Any]]:
        """Get provider class by method name."""
        for provider_cls, _, _ in self.providers:
            if provider_cls.__name__ == method:
                return provider_cls
        return None

    def get_supports_logprobs_by_method(self, method: str) -> Optional[bool]:
        """Get logprobs support by method name."""
        for provider_cls, _, supports_logprobs in self.providers:
            if provider_cls.__name__ == method:
                return supports_logprobs
        return None

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

    # Import providers individually and handle missing ones gracefully
    providers_to_register = {}

    # OpenAI Provider
    try:
        from cje.judge.providers.openai_provider import OpenAIProvider

        providers_to_register["openai"] = ProviderInfo(
            provider_cls=OpenAIProvider,
            structured_cls=OpenAIProvider,  # Same class handles both
            supports_logprobs=True,  # Output logprobs only
            supports_full_sequence_logprobs=False,  # Cannot score input+output sequences (chat models don't support completions API)
            default_template="comprehensive_judge",
            supported_methods=["function_calling", "json_schema"],
            recommended_method="json_schema",
        )
    except ImportError:
        pass

    # Anthropic Provider
    try:
        from cje.judge.providers.anthropic_provider import AnthropicProvider

        providers_to_register["anthropic"] = ProviderInfo(
            provider_cls=AnthropicProvider,
            structured_cls=AnthropicProvider,
            supports_logprobs=False,
            supports_full_sequence_logprobs=False,
            default_template="comprehensive_judge",
            supported_methods=["function_calling"],
            recommended_method="function_calling",
        )
    except ImportError:
        pass

    # Google Provider
    try:
        from cje.judge.providers.google_provider import GoogleProvider

        providers_to_register["google"] = ProviderInfo(
            provider_cls=GoogleProvider,
            structured_cls=GoogleProvider,
            supports_logprobs=False,
            supports_full_sequence_logprobs=False,
            default_template="comprehensive_judge",
            supported_methods=["function_calling"],
            recommended_method="function_calling",
        )
    except ImportError:
        pass

    # Together Provider
    try:
        from cje.judge.providers.together_provider import TogetherProvider

        providers_to_register["together"] = ProviderInfo(
            provider_cls=TogetherProvider,
            structured_cls=TogetherProvider,
            supports_logprobs=True,
            supports_full_sequence_logprobs=True,
            default_template="comprehensive_judge",
            supported_methods=["function_calling", "json_schema"],
            recommended_method="json_schema",
        )
    except ImportError:
        pass

    # Fireworks Provider
    try:
        from cje.judge.providers.fireworks_provider import FireworksProvider

        providers_to_register["fireworks"] = ProviderInfo(
            provider_cls=FireworksProvider,
            structured_cls=FireworksProvider,
            supports_logprobs=True,
            supports_full_sequence_logprobs=True,
            default_template="comprehensive_judge",
            supported_methods=["function_calling", "json_schema"],
            recommended_method="json_schema",
        )
    except ImportError:
        pass

    # Add successfully imported providers to registry
    for name, info in providers_to_register.items():
        _registry.add_provider_info(name, info)
        _registry.add_provider(
            info.provider_cls, info.structured_cls, info.supports_logprobs
        )


# Initialize on import
_initialize_registry()


def print_supported_providers() -> None:
    """Convenience function to print provider capabilities."""
    registry = get_registry()
    registry.print_provider_capabilities()
