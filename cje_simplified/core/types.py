"""Core data types for simplified CJE."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np


class LogProbStatus(Enum):
    """Status of log probability computation."""

    SUCCESS = "success"
    API_ERROR = "api_error"
    TOKEN_BOUNDARY_ERROR = "token_boundary_error"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"
    EMPTY_RESPONSE = "empty_response"


@dataclass
class LogProbResult:
    """Result of log probability computation with explicit error handling."""

    value: Optional[float] = None
    status: LogProbStatus = LogProbStatus.API_ERROR
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_valid(self) -> bool:
        """Check if computation succeeded."""
        return self.status == LogProbStatus.SUCCESS and self.value is not None


# Note: EstimationResult has been moved to data_models.py for better organization
