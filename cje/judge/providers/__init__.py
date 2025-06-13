"""Provider implementations for judges."""

from .base import ProviderStrategy
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .google_provider import GoogleProvider

# New structured providers
from .structured_base import StructuredProviderStrategy
from .structured_openai import StructuredOpenAIProvider
from .structured_anthropic import StructuredAnthropicProvider
from .structured_google import StructuredGoogleProvider

from .fireworks_provider import FireworksProvider
from .together_provider import TogetherProvider
from .structured_fireworks import StructuredFireworksProvider

# Conditionally import Together providers to avoid missing dependency
try:
    from .structured_together import StructuredTogetherProvider

    _TOGETHER_AVAILABLE = True
except (ImportError, TypeError) as e:
    # Handle missing dependencies
    import warnings
    from typing import TYPE_CHECKING

    warnings.warn(f"Together provider not available: {e}", UserWarning)
    if TYPE_CHECKING:
        from .structured_together import StructuredTogetherProvider
    else:
        StructuredTogetherProvider = None  # type: ignore[misc]
    _TOGETHER_AVAILABLE = False

__all__ = [
    "ProviderStrategy",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "StructuredProviderStrategy",
    "StructuredOpenAIProvider",
    "StructuredAnthropicProvider",
    "StructuredGoogleProvider",
    "FireworksProvider",
    "TogetherProvider",
    "StructuredFireworksProvider",
]

# Only include Together provider if available
if _TOGETHER_AVAILABLE:
    __all__.append("StructuredTogetherProvider")
