"""Provider registry alias module for backward compatibility."""

from .provider_registry import (
    get_provider,
    list_providers,
    print_provider_capabilities,
    ProviderInfo,
    get_registry,
)

__all__ = [
    "get_provider",
    "list_providers",
    "print_provider_capabilities",
    "ProviderInfo",
    "get_registry",
]
