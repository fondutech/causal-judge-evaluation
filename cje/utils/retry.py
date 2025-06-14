# This file uses a global mypy ignore for the return type error
# mypy: ignore-errors

import logging
from typing import Callable, TypeVar, Any, cast
import openai  # For openai.RateLimitError, openai.APIStatusError


from tenacity import (
    retry,
    wait_exponential_jitter,
    stop_after_attempt,
    retry_if_exception_type,
    RetryError,
)

logger = logging.getLogger(__name__)

# Define a generic type variable for the return type of the wrapped function
R = TypeVar("R")

# Define the specific exception types we want to retry on.
# This can be expanded as we implement other adapters.
RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIStatusError,  # Often used for 5xx errors like service unavailable
    Exception,  # As a fallback for other transient issues
)


def retry_api_call(func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
    """
    Wraps a function call with tenacity retry logic for common API errors.

    Args:
        func: The function to call.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The result of the function call.

    Raises:
        RetryError: If all retry attempts fail.
        Any other exception raised by `func` if it's not in RETRYABLE_EXCEPTIONS
                     (unless Exception itself is in RETRYABLE_EXCEPTIONS and it's the final one).
    """

    # Define the retry decorator dynamically or apply to a nested function
    # This approach allows us to pass func, args, kwargs easily.
    @retry(
        wait=wait_exponential_jitter(
            initial=1, jitter=1, max=60
        ),  # initial=1s, jitter=1s, max=60s
        stop=stop_after_attempt(5),  # Sensible default, can be configured
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,  # Reraise the last exception if all retries fail
    )
    def _wrapped_call() -> R:
        # Using cast to tell mypy that the result of func matches the generic type R
        return cast(R, func(*args, **kwargs))

    try:
        return _wrapped_call()
    except RetryError as e:  # This will be raised if all retries fail
        # Optionally, log the final failure or re-wrap in a custom exception
        # For now, reraising as specified by reraise=True in @retry
        logger.error(
            f"API call '{func.__name__}' failed after multiple retries. Last exception: {e.last_attempt.exception}"
        )
        raise
