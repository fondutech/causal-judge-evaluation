from typing import List, Dict
from abc import ABC, abstractmethod


class Judge(ABC):
    @abstractmethod
    def score(self, context: str, response: str) -> float:
        """Score a single context-response pair.

        Args:
            context: The input context
            response: The generated response

        Returns:
            Score between 0 and 1
        """
        pass

    def score_batch(
        self, samples: List[Dict[str, str]], disable_progress: bool = False
    ) -> List[float]:
        """Score a batch of samples with progress monitoring.

        Args:
            samples: List of dicts with 'context' and 'response'
            disable_progress: Whether to disable progress bar

        Returns:
            List of scores
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
