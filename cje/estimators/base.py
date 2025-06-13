from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, TypeVar, Generic, List
import numpy as np
from .results import EstimationResult

T = TypeVar("T", bound=Dict[str, Any])


class Estimator(ABC, Generic[T]):
    """
    Abstract estimator interface. Two-step pattern: fit → estimate.

    This is the base class for all CJE estimators. Each estimator should:
    1. Implement fit() to process input logs and prepare for estimation
    2. Implement estimate() to compute the final estimates

    All estimators are multi-policy by default, with single-policy being the K=1 case.

    Type Parameters:
        T: The type of log entries this estimator expects. Must be a Dict[str, Any].
    """

    def __init__(self, *, verbose: bool = False) -> None:  # noqa: D401
        """Base estimator constructor.

        Args:
            verbose: If True, estimators may display detailed progress bars using
                cje.utils.progress.maybe_track. Default False keeps estimators
                silent unless explicitly enabled.
        """
        self.verbose: bool = verbose

    def __getattr__(self, name: str) -> Any:
        """Provide helpful error messages for common method name mistakes."""
        if name == "get_estimate":
            raise AttributeError(
                f"'{self.__class__.__name__}' has no method 'get_estimate'. "
                f"Use .estimate() instead to compute the estimates."
            )
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @abstractmethod
    def fit(self, logs: List[T], **kwargs: Any) -> None:
        """
        Process input logs and prepare for estimation.

        Args:
            logs: List of logged data points, each containing at least:
                - context: Input context
                - response: Generated sequence
                - logp: Log probability under behavior policy
                - reward: Observed reward
        """
        ...

    @abstractmethod
    def estimate(self) -> EstimationResult:
        """
        Compute and return the final estimates.

        Returns:
            EstimationResult containing results for all K policies
        """
        ...
