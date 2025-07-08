"""
Result types that force explicit error handling.

No more silent failures or arbitrary fallback values!
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, TypeVar, Generic
from enum import Enum
import time

T = TypeVar("T")


class LogProbStatus(Enum):
    """Status of a log probability computation."""

    SUCCESS = "success"
    RETRY_EXCEEDED = "retry_exceeded"
    INVALID_INPUT = "invalid_input"
    API_ERROR = "api_error"
    RATE_LIMITED = "rate_limited"
    NOT_ATTEMPTED = "not_attempted"
    EMPTY_RESPONSE = "empty_response"
    TOKEN_MISMATCH = "token_mismatch"
    MODEL_NOT_FOUND = "model_not_found"
    UNAUTHORIZED = "unauthorized"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"  # Context truncated
    TOKEN_BOUNDARY_ERROR = "token_boundary_error"  # Prompt not a prefix


@dataclass
class LogProbResult:
    """
    Result of a log probability computation.

    This is the ONLY way log probs should be returned - never raw floats.
    Forces explicit handling of failures.
    """

    status: LogProbStatus
    value: Optional[float] = None
    error: Optional[str] = None
    attempts: int = 0
    compute_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if this result contains a valid log probability."""
        return self.status == LogProbStatus.SUCCESS and self.value is not None

    def unwrap(self) -> float:
        """
        Get the log probability value, raising if invalid.

        This forces explicit handling of failures.
        """
        if not self.is_valid:
            raise ValueError(
                f"Cannot unwrap invalid LogProbResult: {self.status} - {self.error}"
            )
        return self.value

    def unwrap_or(self, default: float) -> float:
        """Get value or default, but log a warning for failures."""
        if self.is_valid:
            return self.value

        # Log warning to ensure failures are visible
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            f"Using default value {default} for failed log prob: "
            f"{self.status} - {self.error}"
        )
        return default

    def __repr__(self) -> str:
        if self.is_valid:
            return f"LogProbResult(SUCCESS, value={self.value:.3f})"
        return f"LogProbResult({self.status.value}, error='{self.error}')"


@dataclass
class Result(Generic[T]):
    """
    Generic result type for operations that can fail.

    Like Rust's Result<T, E> - forces explicit error handling.
    """

    success: bool
    value: Optional[T] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, value: T, **metadata: Any) -> "Result[T]":
        """Create a successful result."""
        return cls(success=True, value=value, metadata=metadata)

    @classmethod
    def err(
        cls, error: str, error_type: Optional[str] = None, **metadata: Any
    ) -> "Result[T]":
        """Create an error result."""
        return cls(
            success=False,
            error=error,
            error_type=error_type or "unknown",
            metadata=metadata,
        )

    def unwrap(self) -> T:
        """Get value or raise if error."""
        if not self.success:
            raise RuntimeError(f"Unwrap called on error result: {self.error}")
        return self.value

    def unwrap_or(self, default: T) -> T:
        """Get value or default, logging if error."""
        if self.success:
            return self.value

        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Using default due to error: {self.error}")
        return default

    def map(self, func: Any) -> "Result[Any]":
        """Transform value if success, propagate error if not."""
        if self.success:
            try:
                return Result.ok(func(self.value))
            except Exception as e:
                return Result.err(str(e), type(e).__name__)
        return self


@dataclass
class SampleResult:
    """Result for a single sample across all policies."""

    sample_id: str
    context: str
    response: str
    policy_results: Dict[str, LogProbResult]
    importance_weights: Dict[str, Optional[float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_valid(self) -> bool:
        """Check if all policies succeeded."""
        return all(r.is_valid for r in self.policy_results.values())

    @property
    def any_valid(self) -> bool:
        """Check if any policy succeeded."""
        return any(r.is_valid for r in self.policy_results.values())

    def get_valid_weights(self) -> Dict[str, float]:
        """Get only the valid importance weights."""
        return {name: w for name, w in self.importance_weights.items() if w is not None}

    def get_failures(self) -> Dict[str, LogProbResult]:
        """Get all failed results."""
        return {
            name: result
            for name, result in self.policy_results.items()
            if not result.is_valid
        }


@dataclass
class BatchResult:
    """Results for a batch of samples."""

    results: List[SampleResult]
    total_time_seconds: float
    parallel_efficiency: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return len(self.results)

    @property
    def num_complete(self) -> int:
        """Samples where all policies succeeded."""
        return sum(1 for r in self.results if r.all_valid)

    @property
    def num_partial(self) -> int:
        """Samples where some policies succeeded."""
        return sum(1 for r in self.results if r.any_valid and not r.all_valid)

    @property
    def num_failed(self) -> int:
        """Samples where all policies failed."""
        return sum(1 for r in self.results if not r.any_valid)

    @property
    def completion_rate(self) -> float:
        """Fraction of samples with all policies successful."""
        return self.num_complete / self.num_samples if self.num_samples > 0 else 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get batch summary statistics."""
        return {
            "num_samples": self.num_samples,
            "num_complete": self.num_complete,
            "num_partial": self.num_partial,
            "num_failed": self.num_failed,
            "completion_rate": self.completion_rate,
            "total_time_seconds": self.total_time_seconds,
            "samples_per_second": (
                self.num_samples / self.total_time_seconds
                if self.total_time_seconds > 0
                else 0
            ),
            "parallel_efficiency": self.parallel_efficiency,
            **self.metadata,
        }

    def filter_complete(self) -> List[SampleResult]:
        """Get only samples where all policies succeeded."""
        return [r for r in self.results if r.all_valid]

    def filter_partial(self) -> List[SampleResult]:
        """Get samples where some policies succeeded."""
        return [r for r in self.results if r.any_valid and not r.all_valid]
