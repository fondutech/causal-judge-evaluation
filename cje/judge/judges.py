"""Unified judge base classes that return JudgeScore with uncertainty.

This replaces the legacy Judge interface that returns float scores.
All judges now return structured scores with mean and variance.
"""

from typing import List, Dict, Union, Optional, Any
from abc import ABC, abstractmethod

from .schemas import JudgeScore, JudgeResult


class Judge(ABC):
    """Base class for all judges with uncertainty-aware scoring.

    All judges must return JudgeScore objects with mean and variance.
    For deterministic judges, variance should be 0.
    """

    @abstractmethod
    def score(self, context: str, response: str) -> JudgeScore:
        """Score a single context-response pair with uncertainty.

        Args:
            context: The input context/question
            response: The generated response

        Returns:
            JudgeScore with mean and variance
        """
        pass

    def score_batch(
        self, samples: List[Dict[str, str]], disable_progress: bool = False
    ) -> List[JudgeScore]:
        """Score a batch of samples with progress monitoring.

        Args:
            samples: List of dicts with 'context' and 'response'
            disable_progress: Whether to disable progress bar

        Returns:
            List of JudgeScore objects
        """
        from cje.utils.progress import track

        scores = []
        for sample in track(
            samples,
            description=f"Scoring with {self.__class__.__name__}",
            disable=disable_progress,
        ):
            score = self.score(sample["context"], sample["response"])
            scores.append(score)
        return scores


class DeterministicJudge(Judge):
    """Base class for judges with no uncertainty (variance=0).

    Subclasses only need to implement score_value() returning a float.
    """

    @abstractmethod
    def score_value(self, context: str, response: str) -> float:
        """Score returning just the value (mean).

        Simplified interface for deterministic judges.
        """
        pass

    def score(self, context: str, response: str) -> JudgeScore:
        """Score with zero variance."""
        value = self.score_value(context, response)
        return JudgeScore(mean=value, variance=0.0)


class ProbabilisticJudge(Judge):
    """Base class for judges that estimate uncertainty.

    Provides utilities for computing variance from multiple samples
    or confidence estimates.
    """

    def score_with_samples(
        self, context: str, response: str, n_samples: int = 10
    ) -> JudgeScore:
        """Score by sampling multiple times to estimate variance.

        Args:
            context: Input context
            response: Generated response
            n_samples: Number of samples for variance estimation

        Returns:
            JudgeScore with empirical mean and variance
        """
        samples = []
        for _ in range(n_samples):
            # Subclasses should implement _sample_score
            sample = self._sample_score(context, response)
            samples.append(sample)

        import numpy as np

        mean = np.mean(samples)
        variance = np.var(samples)

        return JudgeScore(mean=float(mean), variance=float(variance))

    def _sample_score(self, context: str, response: str) -> float:
        """Sample a single score (to be implemented by subclasses).

        This method should include randomness (e.g., temperature > 0).
        """
        raise NotImplementedError("Subclasses must implement _sample_score")
