"""
Simplified error handling for CJE.
"""

from typing import TypeVar, Callable, Optional, Any
import logging
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")


def safe_execute(
    func: Callable[..., T],
    default: T,
    log_errors: bool = True,
    error_msg: Optional[str] = None,
) -> Callable[..., T]:
    """
    Simple decorator for safe execution with default fallback.

    Usage:
        @safe_execute(default=0.0, error_msg="Failed to compute score")
        def compute_score(x, y):
            return x / y
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                msg = error_msg or f"Error in {func.__name__}"
                logger.warning(f"{msg}: {e}")
            return default

    return wrapper


class CJEError(Exception):
    """Base exception for CJE errors."""

    pass


class ConfigError(CJEError):
    """Configuration-related errors."""

    pass


class DataError(CJEError):
    """Data loading or processing errors."""

    pass


class EstimationError(CJEError):
    """Estimation or statistical errors."""

    pass


class ProviderError(CJEError):
    """LLM provider errors."""

    pass


def handle_api_error(e: Exception, provider: str) -> None:
    """Convert provider-specific errors to CJEError."""
    error_msg = str(e)

    if "rate limit" in error_msg.lower():
        raise ProviderError(f"{provider} rate limit exceeded. Please wait and retry.")
    elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
        raise ProviderError(f"{provider} authentication failed. Check your API key.")
    elif "model not found" in error_msg.lower():
        raise ProviderError(f"Model not available on {provider}.")
    else:
        raise ProviderError(f"{provider} API error: {error_msg}")
