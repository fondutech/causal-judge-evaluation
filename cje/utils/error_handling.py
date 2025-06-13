"""
Unified error handling for CJE.

This module provides a consistent error hierarchy for all CJE components.
"""

import logging
import warnings
from typing import Any, Optional, Union, Callable, TypeVar
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorSeverity(Enum):
    """Error severity levels."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CJEError(Exception):
    """Base exception for all CJE errors."""

    pass


class ConfigurationError(CJEError):
    """Raised when configuration is invalid or incomplete."""

    pass


class ValidationError(CJEError):
    """Raised when data validation fails."""

    pass


class EstimationError(CJEError):
    """Raised when estimation fails due to data or model issues."""

    pass


class PolicyError(CJEError):
    """Raised when policy operations fail (generation, log probability computation)."""

    pass


class JudgeError(CJEError):
    """Raised when judge evaluation fails."""

    pass


class DataError(CJEError):
    """Raised when data loading or processing fails."""

    pass


def handle_error(
    error: Exception,
    context: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    fallback_value: Optional[T] = None,
    reraise: bool = True,
) -> Optional[T]:
    """
    Handle errors consistently across the codebase.

    Args:
        error: The exception that occurred
        context: Description of what was being attempted
        severity: How severe this error is
        fallback_value: Value to return if not re-raising
        reraise: Whether to re-raise the exception

    Returns:
        fallback_value if not re-raising, otherwise raises

    Raises:
        The original exception if reraise=True
    """
    error_msg = f"{context}: {error}"

    if severity == ErrorSeverity.WARNING:
        logger.warning(error_msg)
        warnings.warn(error_msg, UserWarning, stacklevel=3)
    elif severity == ErrorSeverity.ERROR:
        logger.error(error_msg)
    elif severity == ErrorSeverity.CRITICAL:
        logger.critical(error_msg)

    if reraise:
        raise error

    return fallback_value


def safe_call(
    func: Callable[..., T],
    *args: Any,
    fallback: Optional[T] = None,
    error_context: str = "",
    **kwargs: Any,
) -> T:
    """
    Safely call a function with error handling and optional fallback.

    Args:
        func: Function to call
        *args: Positional arguments for the function
        fallback: Value to return if function fails (None by default)
        error_context: Additional context for error messages
        **kwargs: Keyword arguments for the function

    Returns:
        Function result or fallback value if function fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_msg = f"Error in {func.__name__}"
        if error_context:
            error_msg += f" ({error_context})"
        error_msg += f": {e}"

        logger.error(error_msg)
        warnings.warn(error_msg, UserWarning, stacklevel=2)
        if fallback is None:
            raise
        return fallback


# Standard fallback values for common operations
FALLBACK_LOG_PROB = -100.0  # Very low log probability
FALLBACK_PROBABILITY = 1e-10  # Very low probability
FALLBACK_SCORE = 0.0  # Neutral score
FALLBACK_RESPONSE = "ERROR_FALLBACK_RESPONSE"  # Fallback text


# Convenience function for parameter validation
def require_positive(value: float, name: str) -> None:
    """Require that a parameter is positive."""
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive, got {value}")


def require_in_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """Require that a parameter is in a specific range."""
    if not (min_val <= value <= max_val):
        raise ConfigurationError(
            f"{name} must be in range [{min_val}, {max_val}], got {value}"
        )


def require_not_empty(value: Any, name: str) -> None:
    """Require that a value is not None or empty."""
    if value is None or (hasattr(value, "__len__") and len(value) == 0):
        raise ConfigurationError(f"{name} cannot be None or empty")
