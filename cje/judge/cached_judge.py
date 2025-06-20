"""Cached judge implementation for unified JudgeScore interface."""

from typing import Dict, List, Optional, Tuple
import hashlib
import json

from .judges import Judge
from .schemas import JudgeScore


class CachedJudge(Judge):
    """Wrapper that adds caching to any unified judge.

    Caches JudgeScore objects including their variance estimates.
    """

    def __init__(self, base_judge: Judge, cache_size: int = 1000):
        """Initialize cached judge.

        Args:
            base_judge: Underlying judge to cache
            cache_size: Maximum number of entries to cache
        """
        self.base_judge = base_judge
        self.cache: Dict[str, JudgeScore] = {}
        self.cache_size = cache_size
        self._validate_cache_size()

    def _validate_cache_size(self) -> None:
        """Validate cache size parameter."""
        if not isinstance(self.cache_size, int):
            raise ValueError("cache_size must be an integer")
        if self.cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        if self.cache_size > 100000:
            raise ValueError("cache_size should not exceed 100,000 (memory concerns)")

    def _cache_key(self, context: str, response: str) -> str:
        """Generate stable cache key for a context-response pair.

        Uses SHA256 for consistent hashing across sessions.
        """
        # Create a stable string representation
        data = json.dumps(
            {
                "context": context,
                "response": response,
                # Include judge type to avoid cache collisions
                "judge_type": type(self.base_judge).__name__,
            },
            sort_keys=True,
        )

        # Return hex digest
        return hashlib.sha256(data.encode()).hexdigest()

    def score(self, context: str, response: str) -> JudgeScore:
        """Score with caching."""
        key = self._cache_key(context, response)

        # Check cache
        if key in self.cache:
            return self.cache[key]

        # Score with base judge
        score = self.base_judge.score(context, response)

        # Simple LRU-like cache management
        if len(self.cache) >= self.cache_size:
            # Remove oldest item (not truly LRU, but simple)
            self.cache.pop(next(iter(self.cache)))

        # Cache the result
        self.cache[key] = score
        return score

    def score_batch(
        self, samples: List[Dict[str, str]], disable_progress: bool = False
    ) -> List[JudgeScore]:
        """Batch scoring with cache."""
        results: List[Optional[JudgeScore]] = []
        uncached_samples = []
        uncached_indices = []

        # Check cache first
        for i, sample in enumerate(samples):
            key = self._cache_key(sample["context"], sample["response"])
            if key in self.cache:
                results.append(self.cache[key])
            else:
                results.append(None)  # Placeholder
                uncached_samples.append(sample)
                uncached_indices.append(i)

        # Score uncached samples
        if uncached_samples:
            uncached_scores = self.base_judge.score_batch(
                uncached_samples, disable_progress=disable_progress
            )

            # Update cache and results
            for idx, score in zip(uncached_indices, uncached_scores):
                key = self._cache_key(samples[idx]["context"], samples[idx]["response"])
                self.cache[key] = score
                results[idx] = score

        # Filter out None placeholders
        return [r for r in results if r is not None]

    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.clear()

    def cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.cache_size,
            "utilization_pct": int((len(self.cache) / self.cache_size) * 100),
        }
