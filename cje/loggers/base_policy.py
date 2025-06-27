"""
Base policy class with proper error handling.

All policies must inherit from this and never return raw floats!
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import numpy as np

from ..types import LogProbResult, LogProbStatus
from ..utils.error_handling import classify_error

logger = logging.getLogger(__name__)


class BasePolicy(ABC):
    """
    Base class for all policy implementations.

    Key principle: NEVER return raw floats for log probabilities.
    Always return LogProbResult to force explicit error handling.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize base policy."""
        self.name = name
        self.model_id = model_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Statistics tracking
        self.total_calls = 0
        self.failed_calls = 0
        self.retry_counts = {i: 0 for i in range(max_retries + 1)}
        self.error_types: Dict[str, int] = {}

    @abstractmethod
    def _compute_log_prob_impl(self, context: str, response: str) -> float:
        """
        Actual implementation - can raise exceptions.

        This is what subclasses implement. It can and should raise
        exceptions when things go wrong - they'll be handled properly.
        """
        pass

    def compute_log_prob(
        self,
        context: str,
        response: str,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> LogProbResult:
        """
        Public API - always returns LogProbResult, never raises.

        This method:
        1. Validates inputs
        2. Implements retry logic
        3. Tracks statistics
        4. Returns a proper result object

        Args:
            context: Input context/prompt
            response: Response to compute log probability for
            max_retries: Override default max retries
            retry_delay: Override default retry delay

        Returns:
            LogProbResult with status and value (if successful)
        """
        start_time = time.time()
        self.total_calls += 1

        # Use instance defaults if not specified
        max_retries = max_retries or self.max_retries
        retry_delay = retry_delay or self.retry_delay

        # Input validation
        if not context:
            self.failed_calls += 1
            self.error_types["empty_context"] = (
                self.error_types.get("empty_context", 0) + 1
            )
            return LogProbResult(
                status=LogProbStatus.INVALID_INPUT,
                error="Empty context",
                attempts=0,
                compute_time_ms=(time.time() - start_time) * 1000,
            )

        if not response:
            self.failed_calls += 1
            self.error_types["empty_response"] = (
                self.error_types.get("empty_response", 0) + 1
            )
            return LogProbResult(
                status=LogProbStatus.EMPTY_RESPONSE,
                error="Empty response",
                attempts=0,
                compute_time_ms=(time.time() - start_time) * 1000,
            )

        # Retry loop
        last_error = None
        for attempt in range(max_retries):
            try:
                # Call implementation
                value = self._compute_log_prob_impl(context, response)

                # Validate result
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Expected float, got {type(value).__name__}")
                if value > 0:
                    raise ValueError(f"Log prob must be <= 0, got {value}")
                if not np.isfinite(value):
                    raise ValueError(f"Log prob must be finite, got {value}")

                # Success!
                self.retry_counts[attempt] += 1
                return LogProbResult(
                    status=LogProbStatus.SUCCESS,
                    value=float(value),
                    attempts=attempt + 1,
                    compute_time_ms=(time.time() - start_time) * 1000,
                    metadata={
                        "model": self.model_id,
                        "context_len": len(context),
                        "response_len": len(response),
                    },
                )

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Classify error
                severity, retry_strategy = classify_error(e)

                # Track error type
                error_type = type(e).__name__
                self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

                # Determine status
                if "rate limit" in error_str or "429" in error_str:
                    status = LogProbStatus.RATE_LIMITED
                elif any(code in error_str for code in ["401", "403"]):
                    status = LogProbStatus.UNAUTHORIZED
                elif "404" in error_str or "not found" in error_str:
                    status = LogProbStatus.MODEL_NOT_FOUND
                elif "no tokens" in error_str:
                    status = LogProbStatus.TOKEN_MISMATCH
                else:
                    status = LogProbStatus.API_ERROR

                # Check if we should retry
                if retry_strategy.value == "no_retry":
                    break

                # Calculate retry delay
                if attempt < max_retries - 1:
                    if retry_strategy.value == "exponential":
                        wait_time = retry_delay * (2**attempt)
                    else:
                        wait_time = retry_delay

                    logger.info(
                        f"{self.name}: Attempt {attempt + 1}/{max_retries} failed with {error_type}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)

        # All retries failed
        self.failed_calls += 1
        self.retry_counts[attempt + 1] += 1

        return LogProbResult(
            status=status if "status" in locals() else LogProbStatus.RETRY_EXCEEDED,
            error=str(last_error),
            attempts=attempt + 1,
            compute_time_ms=(time.time() - start_time) * 1000,
            metadata={
                "model": self.model_id,
                "context_preview": (
                    context[:100] + "..." if len(context) > 100 else context
                ),
                "response_preview": (
                    response[:100] + "..." if len(response) > 100 else response
                ),
                "error_type": type(last_error).__name__ if last_error else None,
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        success_count = self.total_calls - self.failed_calls

        return {
            "name": self.name,
            "model": self.model_id,
            "total_calls": self.total_calls,
            "successful_calls": success_count,
            "failed_calls": self.failed_calls,
            "success_rate": (
                success_count / self.total_calls if self.total_calls > 0 else 0.0
            ),
            "retry_distribution": dict(self.retry_counts),
            "error_types": dict(self.error_types),
            "avg_retries": (
                sum(attempts * count for attempts, count in self.retry_counts.items())
                / sum(self.retry_counts.values())
                if sum(self.retry_counts.values()) > 0
                else 0
            ),
        }

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.total_calls = 0
        self.failed_calls = 0
        self.retry_counts = {i: 0 for i in range(self.max_retries + 1)}
        self.error_types.clear()

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"model='{self.model_id}', "
            f"success_rate={stats['success_rate']:.1%})"
        )
