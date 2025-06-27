"""
API-based policy runner with proper error handling.

No fallback values - all failures are explicit!
"""

import time
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from abc import abstractmethod
import numpy as np

from ..types import LogProbResult, LogProbStatus
from ..utils.error_handling import classify_error, require_positive, require_not_empty
from .base_policy import BasePolicy

logger = logging.getLogger(__name__)


class APIPolicyRunner(BasePolicy):
    """
    API-based policy runner with robust error handling.

    Key improvements:
    - Returns LogProbResult, not raw floats
    - Built-in retry logic with smart backoff
    - No fallback values ever
    - Comprehensive error tracking
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        system_prompt: Optional[str] = None,
        user_message_template: str = "{prompt}",
        assistant_message_template: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0,
        cache_enabled: bool = True,
        **kwargs: Any,
    ):
        """Initialize API policy runner with configuration."""
        super().__init__(
            name=f"{provider}:{model_name}",
            model_id=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        # Validate inputs
        require_not_empty(provider, "provider")
        require_not_empty(model_name, "model_name")
        require_positive(temperature, "temperature")
        require_positive(max_new_tokens, "max_new_tokens")

        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.system_prompt = system_prompt
        self.user_message_template = user_message_template
        self.assistant_message_template = assistant_message_template
        self.timeout = timeout
        self.cache_enabled = cache_enabled

        # Initialize API client
        self._init_api_client(api_key, api_base, **kwargs)

        # Cache for log probabilities
        self._cache: Dict[str, LogProbResult] = {}

    @abstractmethod
    def _init_api_client(
        self, api_key: Optional[str], api_base: Optional[str], **kwargs: Any
    ) -> None:
        """Initialize the API client for the specific provider."""
        pass

    @abstractmethod
    def _compute_log_prob_api(self, context: str, response: str) -> float:
        """
        Call the API to compute log probability.

        This method can raise exceptions - they will be handled by compute_log_prob.
        """
        pass

    def _compute_log_prob_impl(self, context: str, response: str) -> float:
        """
        Implementation that calls the API.

        Can raise exceptions - that's OK! The base class handles them.
        """
        # Check cache first
        if self.cache_enabled:
            cache_key = self._get_cache_key(context, response)
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                if cached.is_valid:
                    logger.debug(f"Cache hit for {self.model_name}")
                    return cached.value

        # Call API
        result = self._compute_log_prob_api(context, response)

        # Validate result
        if not isinstance(result, (int, float)):
            raise TypeError(f"API returned invalid type: {type(result)}")
        if result > 0:
            raise ValueError(f"Log probability must be <= 0, got {result}")
        if not np.isfinite(result):
            raise ValueError(f"Log probability must be finite, got {result}")

        # Sanity check for very short responses
        response_len = len(response.split())
        if response_len <= 3 and result < -50:
            logger.warning(
                f"Suspiciously low logprob {result:.3f} for {response_len}-word response: "
                f"'{response[:50]}...'"
            )

        return float(result)

    def compute_log_prob_batch(
        self,
        contexts: List[str],
        responses: List[str],
        show_progress: bool = True,
    ) -> List[LogProbResult]:
        """
        Compute log probabilities for a batch of samples.

        Returns:
            List of LogProbResult - one per sample, never raises
        """
        if len(contexts) != len(responses):
            raise ValueError(
                f"Mismatched batch sizes: {len(contexts)} contexts, {len(responses)} responses"
            )

        results = []
        for i, (context, response) in enumerate(zip(contexts, responses)):
            if show_progress and i % 10 == 0:
                logger.info(f"Processing sample {i+1}/{len(contexts)}")

            result = self.compute_log_prob(context, response)
            results.append(result)

            # Update cache
            if self.cache_enabled and result.is_valid:
                cache_key = self._get_cache_key(context, response)
                self._cache[cache_key] = result

        return results

    def _get_cache_key(self, context: str, response: str) -> str:
        """Generate cache key for a context-response pair."""
        # Use hash to keep keys reasonable size
        context_hash = hash(context)
        response_hash = hash(response)
        return f"{self.model_name}:{context_hash}:{response_hash}"

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache_enabled:
            return {"enabled": False}

        valid_entries = sum(1 for r in self._cache.values() if r.is_valid)
        failed_entries = len(self._cache) - valid_entries

        return {
            "enabled": True,
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "failed_entries": failed_entries,
            "hit_rate": self._cache_hits
            / max(self._cache_hits + self._cache_misses, 1),
        }

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info(f"Cleared cache for {self.name}")


class OpenAIPolicy(APIPolicyRunner):
    """OpenAI-specific implementation."""

    def _init_api_client(
        self, api_key: Optional[str], api_base: Optional[str], **kwargs: Any
    ) -> None:
        """Initialize OpenAI client."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required for OpenAI provider")

        # Set up client
        self.client = openai.Client(
            api_key=api_key, base_url=api_base, timeout=self.timeout, **kwargs
        )

    def _compute_log_prob_api(self, context: str, response: str) -> float:
        """Call OpenAI API for log probability."""
        # Format messages
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": context})
        messages.append({"role": "assistant", "content": response})

        # Call API with logprobs enabled
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=1,  # We just need log probs, not generation
            logprobs=True,
            temperature=0,  # For consistent results
        )

        # Extract log probability
        # This is provider-specific - adjust based on API response format
        if (
            hasattr(completion.choices[0], "logprobs")
            and completion.choices[0].logprobs
        ):
            # The structure of logprobs in the API response varies
            # This needs to be adapted based on the actual API response format
            logprobs_obj = completion.choices[0].logprobs
            if hasattr(logprobs_obj, "token_logprobs"):
                token_logprobs = logprobs_obj.token_logprobs  # type: ignore[attr-defined]
                if token_logprobs:
                    return float(sum(token_logprobs))
            # Try alternative structure
            if hasattr(logprobs_obj, "content") and logprobs_obj.content:
                # Sum log probs from content tokens
                total_logp = 0.0
                for token_data in logprobs_obj.content:
                    if hasattr(token_data, "logprob"):
                        total_logp += token_data.logprob
                return total_logp

        raise ValueError("No log probabilities in API response")


class AnthropicPolicy(APIPolicyRunner):
    """Anthropic-specific implementation."""

    def _init_api_client(
        self, api_key: Optional[str], api_base: Optional[str], **kwargs: Any
    ) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required for Anthropic provider")

        self.client = anthropic.Client(
            api_key=api_key, base_url=api_base, timeout=self.timeout, **kwargs
        )

    def _compute_log_prob_api(self, context: str, response: str) -> float:
        """Call Anthropic API for log probability."""
        # Note: Anthropic doesn't directly support log prob computation
        # This would need a custom implementation or workaround
        raise NotImplementedError(
            "Anthropic API doesn't directly support log probability computation. "
            "Consider using a different provider or implementing a workaround."
        )


# Factory function
def create_api_policy(provider: str, **kwargs: Any) -> APIPolicyRunner:
    """Create an API policy runner for the given provider."""
    providers = {
        "openai": OpenAIPolicy,
        "anthropic": AnthropicPolicy,
        # Add more providers as needed
    }

    if provider not in providers:
        raise ValueError(
            f"Unknown provider: {provider}. " f"Available: {list(providers.keys())}"
        )

    return providers[provider](**kwargs)
