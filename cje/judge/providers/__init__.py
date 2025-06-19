"""Provider implementations for judges."""

# Import base class
from .base import UnifiedProviderStrategy

# Import providers
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .google import GoogleProvider
from .fireworks import FireworksProvider

# Import Together provider with explicit error handling
from ...utils.imports import ImportChecker

# Try to check if Together SDK is available before importing provider
_, HAS_TOGETHER = ImportChecker.optional_import("together", warn=False)

if HAS_TOGETHER:
    try:
        from .together import TogetherProvider

        _TOGETHER_AVAILABLE = True
    except ImportError as e:
        # Provider import failed even though SDK is available
        import warnings

        warnings.warn(
            f"Together provider could not be imported: {e}\n"
            f"This is likely due to a code error, not missing dependencies.",
            RuntimeWarning,
        )
        TogetherProvider = None  # type: ignore[misc]
        _TOGETHER_AVAILABLE = False
else:
    # Together SDK not installed
    import warnings

    warnings.warn(
        "Together provider not available because 'together' package is not installed.\n"
        "To use Together AI, install it with: pip install together",
        ImportWarning,
    )
    TogetherProvider = None  # type: ignore[misc]
    _TOGETHER_AVAILABLE = False

__all__ = [
    # Base class
    "UnifiedProviderStrategy",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "FireworksProvider",
]

# Only include Together provider if available
if _TOGETHER_AVAILABLE:
    __all__.append("TogetherProvider")
