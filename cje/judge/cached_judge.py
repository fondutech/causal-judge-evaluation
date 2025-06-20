"""Cached judge implementation for unified JudgeScore interface."""

from typing import Dict, List, Optional, Any
from functools import lru_cache
import hashlib
import json

from .judges import Judge
from .schemas import JudgeScore


class CachedJudge(Judge):
    """Wrapper that adds caching to any unified judge.

    Uses functools.lru_cache for efficient LRU caching with proper eviction.
    Caches JudgeScore objects including their variance estimates.
    """

    def __init__(self, base_judge: Judge, cache_size: int = 1000):
        """Initialize cached judge.

        Args:
            base_judge: Underlying judge to cache
            cache_size: Maximum number of entries to cache
        """
        self.base_judge = base_judge
        self.cache_size = cache_size
        self._validate_cache_size()

        # Create the cached scoring function with the specified cache size
        self._cached_score = lru_cache(maxsize=cache_size)(self._score_impl)

        # Track cache hits/misses for statistics
        self._cache_hits = 0
        self._cache_misses = 0

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

    def _score_impl(self, cache_key: str, context: str, response: str) -> JudgeScore:
        """Internal implementation that gets cached.

        The cache_key parameter ensures proper caching behavior.
        """
        return self.base_judge.score(context, response)

    def score(self, context: str, response: str) -> JudgeScore:
        """Score with caching.

        Uses functools.lru_cache for efficient LRU caching.
        """
        key = self._cache_key(context, response)

        # Check if we'll have a cache hit by looking at the cache info
        cache_info = self._cached_score.cache_info()
        prev_hits = cache_info.hits

        # Call the cached function
        result = self._cached_score(key, context, response)

        # Update statistics
        cache_info_after = self._cached_score.cache_info()
        if cache_info_after.hits > prev_hits:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        return result

    def score_batch(
        self, samples: List[Dict[str, str]], disable_progress: bool = False
    ) -> List[JudgeScore]:
        """Batch scoring with cache.

        Efficiently handles batch scoring by checking cache first,
        then scoring only uncached samples.
        """
        results: List[Optional[JudgeScore]] = []
        uncached_samples = []
        uncached_indices = []

        # Check cache first
        for i, sample in enumerate(samples):
            key = self._cache_key(sample["context"], sample["response"])

            # Try to get from cache without scoring
            # We'll check if the key exists by looking at a dummy call
            try:
                # Use cache_info to check without side effects
                cache_dict = dict(self._cached_score.cache_info()._asdict())
                # For now, we'll just try to score and track hits/misses
                # A more sophisticated approach would peek into the cache
                # but lru_cache doesn't expose that directly

                # For batch processing, we'll just collect uncached items
                # This is a limitation of using lru_cache, but the benefits
                # of proper LRU eviction outweigh this
                uncached_samples.append(sample)
                uncached_indices.append(i)
                results.append(None)
            except Exception:
                # Fallback to uncached
                uncached_samples.append(sample)
                uncached_indices.append(i)
                results.append(None)

        # For now, since we can't peek into lru_cache efficiently,
        # we'll just process all samples through the cache
        # This is still beneficial as cached items will be fast
        results_final = []
        for sample in samples:
            score = self.score(sample["context"], sample["response"])
            results_final.append(score)

        return results_final

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cached_score.cache_clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns detailed statistics from functools.lru_cache.
        """
        info = self._cached_score.cache_info()
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        )

        return {
            "size": info.currsize,
            "max_size": info.maxsize,
            "hits": info.hits,
            "misses": info.misses,
            "utilization_pct": (
                int((info.currsize / info.maxsize) * 100) if info.maxsize else 0
            ),
            "hit_rate_pct": round(hit_rate, 1),
            "total_requests": total_requests,
        }
