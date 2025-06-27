"""
Unified error handling for CJE - WITHOUT dangerous fallback values.

This module provides safe error handling that prevents silent data corruption.
"""

import logging
import time
import functools
from typing import TypeVar, Callable, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import traceback

from ..types import Result

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorSeverity(Enum):
    """Error severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RetryStrategy(Enum):
    """Retry strategies for different error types."""

    NO_RETRY = "no_retry"  # Don't retry (e.g., auth errors)
    LINEAR = "linear"  # Linear backoff
    EXPONENTIAL = "exponential"  # Exponential backoff
    IMMEDIATE = "immediate"  # Retry immediately


# Keep the useful exception hierarchy
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


@dataclass
class ErrorContext:
    """Rich context for errors."""

    operation: str
    severity: ErrorSeverity
    error_type: type
    message: str
    traceback: str
    timestamp: float
    metadata: dict

    def log(self) -> None:
        """Log this error with appropriate level."""
        log_msg = (
            f"{self.operation} failed: {self.message}\n"
            f"Type: {self.error_type.__name__}\n"
            f"Metadata: {self.metadata}"
        )

        if self.severity == ErrorSeverity.INFO:
            logger.info(log_msg)
        elif self.severity == ErrorSeverity.WARNING:
            logger.warning(log_msg)
        elif self.severity == ErrorSeverity.ERROR:
            logger.error(log_msg)
        elif self.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_msg)


def classify_error(error: Exception) -> Tuple[ErrorSeverity, RetryStrategy]:
    """
    Classify an error to determine severity and retry strategy.

    No more generic fallbacks - each error type gets appropriate handling.
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # API errors
    if "401" in error_str or "unauthorized" in error_str:
        return ErrorSeverity.ERROR, RetryStrategy.NO_RETRY
    elif "403" in error_str or "forbidden" in error_str:
        return ErrorSeverity.ERROR, RetryStrategy.NO_RETRY
    elif "404" in error_str or "not found" in error_str:
        return ErrorSeverity.ERROR, RetryStrategy.NO_RETRY
    elif "429" in error_str or "rate limit" in error_str:
        return ErrorSeverity.WARNING, RetryStrategy.EXPONENTIAL
    elif any(x in error_str for x in ["500", "502", "503", "504"]):
        return ErrorSeverity.WARNING, RetryStrategy.LINEAR

    # Network errors
    elif "timeout" in error_str:
        return ErrorSeverity.WARNING, RetryStrategy.LINEAR
    elif "connection" in error_str:
        return ErrorSeverity.WARNING, RetryStrategy.EXPONENTIAL

    # Data errors
    elif "json" in error_str or "decode" in error_str:
        return ErrorSeverity.ERROR, RetryStrategy.NO_RETRY
    elif "empty" in error_str or "no tokens" in error_str:
        return ErrorSeverity.INFO, RetryStrategy.NO_RETRY

    # Programming errors
    elif error_type in ["TypeError", "ValueError", "KeyError", "AttributeError"]:
        return ErrorSeverity.ERROR, RetryStrategy.NO_RETRY

    # Default
    return ErrorSeverity.ERROR, RetryStrategy.LINEAR


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
) -> Callable:
    """
    Decorator for automatic retry with smart backoff.

    Unlike the old safe_call, this NEVER returns fallback values.
    Always returns a Result that must be handled explicitly.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., Result[T]]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Result[T]:
            last_error = None

            for attempt in range(max_attempts):
                try:
                    # Try to execute
                    result = func(*args, **kwargs)
                    return Result.ok(result)

                except Exception as e:
                    last_error = e
                    severity, strategy = classify_error(e)

                    # Create error context
                    error_ctx = ErrorContext(
                        operation=func.__name__,
                        severity=severity,
                        error_type=type(e),
                        message=str(e),
                        traceback=traceback.format_exc(),
                        timestamp=time.time(),
                        metadata={
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "args": str(args)[:100],
                            "kwargs": str(kwargs)[:100],
                        },
                    )

                    # Log the error
                    error_ctx.log()

                    # Check retry strategy
                    if strategy == RetryStrategy.NO_RETRY:
                        break
                    elif attempt < max_attempts - 1:
                        # Calculate delay
                        if strategy == RetryStrategy.IMMEDIATE:
                            delay: float = 0
                        elif strategy == RetryStrategy.LINEAR:
                            delay = initial_delay
                        elif strategy == RetryStrategy.EXPONENTIAL:
                            delay = min(
                                initial_delay * (backoff_factor**attempt), max_delay
                            )
                        else:
                            delay = initial_delay

                        if delay > 0:
                            logger.info(f"Retrying {func.__name__} in {delay:.1f}s...")
                            time.sleep(delay)

            # All attempts failed
            return Result.err(
                str(last_error),
                error_type=type(last_error).__name__,
                attempts=attempt + 1,
                final_error=True,
            )

        return wrapper

    return decorator


# REMOVED ALL DANGEROUS FALLBACK CONSTANTS!
# No more:
# - FALLBACK_LOG_PROB = -100.0
# - FALLBACK_PROBABILITY = 1e-10
# - FALLBACK_SCORE = 0.0
# - FALLBACK_RESPONSE = "ERROR_FALLBACK_RESPONSE"

# REMOVED safe_call function that returns fallback values!
# Use @with_retry decorator instead, which returns Result<T>


# Keep the useful validation functions
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
    if value is None:
        raise ConfigurationError(f"{name} cannot be None")
    if hasattr(value, "__len__") and len(value) == 0:
        raise ConfigurationError(f"{name} cannot be empty")


# Migration helper for old code
def deprecated_safe_call_replacement(func_name: str) -> None:
    """Helper to alert about deprecated safe_call usage."""
    raise DeprecationWarning(
        f"safe_call is deprecated and dangerous! "
        f"Use @with_retry decorator for {func_name} instead. "
        f"It returns Result<T> that must be handled explicitly."
    )
