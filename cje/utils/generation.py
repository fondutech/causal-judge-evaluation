"""Utilities for generating with log probabilities."""

from typing import Dict, Any, List, Optional, Tuple, Union, TYPE_CHECKING, cast
import numpy as np

if TYPE_CHECKING:
    from ..loggers import PolicyRunner, APIPolicyRunner

# ------------------------------------------------------------------
# Optional lightweight cache for generation & log-prob computations.
# ------------------------------------------------------------------
from .inference_cache import LLMCache

_CACHE = LLMCache()


def _create_runner(
    provider: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: Optional[int] = None,
) -> Union["PolicyRunner", "APIPolicyRunner"]:
    """Create appropriate runner based on provider type.

    Args:
        provider: Provider name (e.g., "fireworks", "openai", "anthropic", "google", "mock")
        model: Model identifier
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        batch_size: Batch size for API calls (API runners only)

    Returns:
        PolicyRunner or APIPolicyRunner instance
    """
    # Lazy import to avoid circular dependency
    from ..loggers import PolicyRunner, APIPolicyRunner

    # Handle mock provider for testing (if registered)
    if provider == "mock":
        try:
            from ..testing.mocks.policy_runners import MockAPIPolicyRunner

            return MockAPIPolicyRunner(  # type: ignore[return-value]
                provider="mock",
                model_name=model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except ImportError:
            raise ValueError("Mock provider requested but testing module not available")

    # Use API runner for known providers
    if provider in ["openai", "anthropic", "google", "fireworks", "together"]:
        from ..loggers.api_policy import create_api_policy

        if batch_size is not None:
            return create_api_policy(
                provider=provider,
                model_name=model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size,
            )
        else:
            return create_api_policy(
                provider=provider,
                model_name=model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
    else:
        # Use local runner for other providers (e.g., "hf")
        return PolicyRunner(
            model_name=model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )


def generate_with_logprobs(
    prompt: str,
    model: str,
    provider: str,
    temperature: float = 0.4,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    api_key: Optional[str] = None,
    return_token_logprobs: bool = True,
) -> Dict[str, Any]:
    """Generate a response with token-level log probabilities.

    Args:
        prompt: Input prompt/context
        model: Model identifier (e.g., "accounts/fireworks/models/qwen3-235b-a22b")
        provider: Provider name (e.g., "fireworks", "openai", "anthropic", "google")
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        api_key: Optional API key override (currently unused - set env vars instead)
        return_token_logprobs: Whether to return token-level logprobs

    Returns:
        Dictionary with:
        - text: Generated text
        - token_logprobs: Dict with tokens, logprobs, and sum (if return_token_logprobs=True)
        - total_logp: Total log probability of the sequence

    Note:
        API keys should be set via environment variables:
        - FIREWORKS_API_KEY for Fireworks models
        - OPENAI_API_KEY for OpenAI models
        - ANTHROPIC_API_KEY for Anthropic models
        - GOOGLE_API_KEY for Google models
    """
    runner = _create_runner(provider, model, temperature, top_p, max_tokens)

    # --------------------------------------------------------------
    # Attempt cache lookup first (keyed by all generation params)
    # --------------------------------------------------------------
    cache_key = (
        "gen",
        provider,
        model,
        prompt,
        temperature,
        top_p,
        max_tokens,
        return_token_logprobs,
    )

    cached = _CACHE.get(*cache_key)
    if cached is not None:
        return cast(Dict[str, Any], cached)

    # Generate with log probabilities
    results = runner.generate_with_logp(  # type: ignore[attr-defined,union-attr]
        prompts=[prompt], return_token_logprobs=return_token_logprobs
    )

    if not results:
        raise ValueError("No results returned from generation")

    result = results[0]  # Single prompt

    # Parse result based on whether token logprobs were requested
    if return_token_logprobs:
        # When requesting token logprobs, we expect a 3-tuple
        if isinstance(result, tuple) and len(result) == 3:
            text = cast(str, result[0])
            total_logp = cast(float, result[1])
            token_logps = cast(List[float], result[2])
            final_result = {
                "text": text,
                "token_logprobs": {
                    "tokens": text.split(),  # Simple tokenization for display
                    "logprobs": token_logps,
                    "sum": sum(token_logps) if token_logps else total_logp,
                    "mean": np.mean(token_logps) if token_logps else 0.0,
                },
                "total_logp": total_logp,
            }
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")
    else:
        # Handle cases where we only have text and total_logp
        if isinstance(result, tuple) and len(result) >= 2:
            text = cast(str, result[0])
            total_logp = cast(float, result[1])
        elif isinstance(result, tuple) and len(result) >= 1:
            # Fallback for single element tuple
            text = cast(str, result[0])
            total_logp = 0.0
        else:
            raise ValueError(f"Unexpected result type: {type(result)}")

        final_result = {"text": text, "total_logp": total_logp}

    # Store in cache
    _CACHE.set(final_result, *cache_key)

    return final_result


def compute_sequence_logp(
    text: str,
    model: str,
    provider: str,
    context: str,
    cache_dir: Optional[str] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> float:
    """Compute log probability of a sequence under a model.

    Args:
        text: The text/response to score
        model: Model identifier
        provider: Provider name (e.g., "fireworks", "openai", "anthropic", "google")
        context: The context/prompt that preceded the text
        cache_dir: Optional directory for caching (not currently used)
        temperature: Temperature for scoring
        top_p: Top-p for scoring

    Returns:
        Log probability of the sequence
    """
    runner = _create_runner(provider, model, temperature, top_p, 0)

    # --------------------------------------------------------------
    # Cache lookup
    # --------------------------------------------------------------
    cache_key = ("logp", provider, model, context, text, temperature, top_p)
    cached = _CACHE.get(*cache_key)
    if cached is not None:
        return float(cached)

    # Compute log probability
    logp_result = runner.log_prob(  # type: ignore[attr-defined,union-attr]
        context=context,
        response=text,
        temperature=temperature,
        top_p=top_p,
    )

    # Handle the case where log_prob returns a tuple (logp, token_logps)
    if isinstance(logp_result, tuple):
        logp = logp_result[0]
    else:
        logp = logp_result

    # Cache store
    _CACHE.set(float(logp), *cache_key)

    return float(logp)


# Convenience function for batch generation
def batch_generate_with_logprobs(
    prompts: List[str],
    model: str,
    provider: str,
    temperature: float = 0.4,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    api_key: Optional[str] = None,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """Batch generation with log probabilities.

    More efficient than calling generate_with_logprobs in a loop.

    Args:
        prompts: List of prompts
        model: Model identifier
        provider: Provider name (e.g., "fireworks", "openai", "anthropic", "google")
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        api_key: Optional API key override (currently unused - set env vars instead)
        batch_size: Batch size for API calls

    Returns:
        List of dictionaries with generation results
    """
    runner = _create_runner(provider, model, temperature, top_p, max_tokens, batch_size)

    # Generate all at once
    # Only PolicyRunner supports generation
    if not hasattr(runner, "generate_with_logp"):
        raise ValueError(
            f"Runner type {type(runner).__name__} doesn't support batch generation"
        )

    results = runner.generate_with_logp(prompts=prompts, return_token_logprobs=True)

    # Format results
    formatted_results = []
    for i, result in enumerate(results):
        # Check if we have a 3-tuple with token logprobs
        if isinstance(result, tuple) and len(result) == 3:
            text = cast(str, result[0])
            total_logp = cast(float, result[1])
            token_logps = cast(List[float], result[2])
            formatted_results.append(
                {
                    "text": text,
                    "token_logprobs": {
                        "tokens": text.split(),
                        "logprobs": token_logps,
                        "sum": sum(token_logps) if token_logps else total_logp,
                        "mean": np.mean(token_logps) if token_logps else 0.0,
                    },
                    "total_logp": total_logp,
                }
            )
        elif isinstance(result, tuple) and len(result) >= 2:
            # Handle 2-tuple case
            text = cast(str, result[0])
            total_logp = cast(float, result[1])
            formatted_results.append({"text": text, "total_logp": total_logp})
        elif isinstance(result, tuple) and len(result) >= 1:
            text = cast(str, result[0])
            total_logp = 0.0
            formatted_results.append({"text": text, "total_logp": total_logp})
        else:
            raise ValueError(f"Unexpected result type at index {i}: {type(result)}")

    return formatted_results
