"""
Robust teacher forcing implementation using the continuation method.

This module provides a streamlined implementation of teacher forcing that uses
only the most reliable method: computing log P(full) - log P(prompt).

The continuation method is the most reliable because:
1. It doesn't require token boundary detection
2. It works consistently across different tokenizers
3. It avoids the "prompt is not a prefix" errors

BEST PRACTICES FOR LOG PROBABILITY COMPUTATION:

1. **Parameter Parity**: Use the SAME temperature value for both behavior
   and target policy evaluations. The exact value doesn't matter (0.0, 0.7, 1.0)
   as long as it matches what the policy uses in production.

2. **Top-p = 1.0**: Disable nucleus sampling by setting top_p=1.0.

3. **Use Seed**: Set a seed value (e.g., 42) for reproducible results.

4. **Monitor Pi_Clone**: Always include a "clone" policy (identical to
   behavior policy) to detect API non-determinism.

Example usage:
    tf = RobustTeacherForcing(
        provider="fireworks",
        model="accounts/fireworks/models/llama-v3-8b",
        temperature=0.7,     # Match the policy's production temperature
        top_p=1.0,           # Disable nucleus sampling
        seed=42,             # For reproducibility
    )

    result = tf.compute_log_prob(prompt, response)
    if result.status == LogProbStatus.SUCCESS:
        print(f"Log probability: {result.value}")
"""

import os
import time
import openai
from typing import Optional, Dict, Any
import logging

from ..types import LogProbResult, LogProbStatus

logger = logging.getLogger(__name__)


class RobustTeacherForcing:
    """Teacher forcing using only the continuation method.

    This implementation computes log P(response|prompt) by:
    1. Getting log P(prompt + response)
    2. Getting log P(prompt)
    3. Returning log P(prompt + response) - log P(prompt)

    This avoids all token boundary detection issues and is the most
    reliable method for computing conditional probabilities.

    Args:
        provider: API provider (e.g., "fireworks", "openai")
        model: Model name
        api_key: Optional API key (uses environment variable if not provided)
        temperature: Model temperature (should match policy's production setting)
        system_prompt: Optional system prompt
        seed: Optional seed for deterministic results (not supported by all providers)
        force_continuation: Kept for compatibility but ignored (always uses continuation)
        top_p: Top-p sampling parameter (default 1.0 to disable nucleus sampling)
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.seed = seed
        self.top_p = top_p
        self.extra_kwargs = kwargs

        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            # Try environment variables
            env_vars = {
                "openai": "OPENAI_API_KEY",
                "fireworks": "FIREWORKS_API_KEY",
                "together": "TOGETHER_API_KEY",
            }
            env_var = env_vars.get(self.provider)
            if env_var:
                self.api_key = os.environ.get(env_var)
            else:
                raise ValueError(f"Unknown provider: {provider}")

        if not self.api_key:
            raise ValueError(f"API key required for provider: {provider}")

        # Initialize API client
        self.api_client = self._init_client()

        # No tokenizer needed for Fireworks
        if provider == "fireworks":
            self.tokenizer = None
            logger.info(
                "Tokenizer disabled for Fireworks provider to avoid validation issues"
            )
        else:
            # Could add tokenizer for other providers if needed
            self.tokenizer = None

        # Statistics
        self.stats = {
            "total_calls": 0,
            "successes": 0,
            "failures": 0,
        }

        # Log configuration
        logger.info(
            f"Initialized RobustTeacherForcing: provider={provider}, "
            f"model={model}, temperature={temperature}, top_p={top_p}"
        )
        if temperature != 1.0:
            logger.info(
                f"Using temperature={temperature}. Ensure this matches the "
                f"policy's production temperature for accurate importance weights."
            )

    def _init_client(self):
        """Initialize the API client based on provider."""
        if self.provider in ["fireworks", "together"]:
            # Use OpenAI-compatible client
            if self.provider == "fireworks":
                base_url = "https://api.fireworks.ai/inference/v1"
            else:  # together
                base_url = "https://api.together.xyz"

            return openai.OpenAI(api_key=self.api_key, base_url=base_url)
        elif self.provider == "openai":
            return openai.OpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def compute_log_prob(self, prompt: str, response: str) -> LogProbResult:
        """Compute log P(response|prompt) using the continuation method.

        Args:
            prompt: The prompt text
            response: The response text to compute probability for

        Returns:
            LogProbResult with the log probability or error information
        """
        self.stats["total_calls"] += 1

        try:
            # Note: _get_sequence_logprob will prepend system prompt if present
            # So we just pass the user content and it handles the rest

            # Step 1: Get log P(prompt + response) - includes system prompt
            full_text = prompt + response
            full_logprob = self._get_sequence_logprob(full_text)

            if full_logprob is None:
                self.stats["failures"] += 1
                return LogProbResult(
                    status=LogProbStatus.API_ERROR,
                    error="Failed to get log probability for full sequence",
                )

            # Step 2: Get log P(prompt) - includes system prompt
            prompt_logprob = self._get_sequence_logprob(prompt)

            if prompt_logprob is None:
                self.stats["failures"] += 1
                return LogProbResult(
                    status=LogProbStatus.API_ERROR,
                    error="Failed to get log probability for prompt",
                )

            # Step 3: Compute log P(response|prompt) = log P(full) - log P(prompt)
            # This correctly handles system prompts since both include it
            response_logprob = full_logprob - prompt_logprob

            # Check for suspiciously low values (possible token boundary issues)
            avg_logprob_per_char = response_logprob / max(1, len(response))
            if avg_logprob_per_char < -2.5:  # ~-10 per token assuming ~4 chars/token
                logger.warning(
                    f"Suspiciously low log probability: {response_logprob:.2f} "
                    f"for {len(response)} chars (avg: {avg_logprob_per_char:.2f}/char)"
                )

            self.stats["successes"] += 1

            return LogProbResult(
                status=LogProbStatus.SUCCESS,
                value=response_logprob,
                metadata={
                    "method": "continuation",
                    "full_logprob": full_logprob,
                    "prompt_logprob": prompt_logprob,
                    "response_length": len(response),
                    "avg_logprob_per_char": avg_logprob_per_char,
                },
            )

        except Exception as e:
            self.stats["failures"] += 1
            logger.error(f"Error in compute_log_prob: {str(e)}")
            return LogProbResult(
                status=LogProbStatus.API_ERROR,
                error=f"Continuation method failed: {str(e)}",
            )

    def _get_sequence_logprob(self, text: str) -> Optional[float]:
        """Get the total log probability for a sequence of text.

        Args:
            text: The text to compute log probability for

        Returns:
            Total log probability, or None if failed
        """
        try:
            # Build parameters for completion endpoint
            # Note: Fireworks v1/completions endpoint expects "prompt" not "messages"
            # System prompts need to be prepended to the text
            if self.system_prompt:
                # Prepend system prompt with clear separation
                full_text = f"{self.system_prompt}\n\n{text}"
            else:
                full_text = text

            params = {
                "model": self.model,
                "prompt": full_text,
                "max_tokens": 0,  # Don't generate any new tokens
                "echo": True,  # Return logprobs for the input
                "logprobs": 1,  # Request log probabilities
                "temperature": self.temperature,
                "top_p": self.top_p,
            }

            # Add optional parameters
            if self.seed is not None and self.provider != "fireworks":
                # Fireworks doesn't support seed parameter
                params["seed"] = self.seed

            # Make API call
            completion = self.api_client.completions.create(**params)

            # Extract log probabilities
            if not completion.choices:
                logger.error("No choices in completion response")
                return None

            choice = completion.choices[0]

            if not hasattr(choice, "logprobs") or choice.logprobs is None:
                logger.error("No logprobs in completion response")
                return None

            # Sum all token log probabilities
            token_logprobs = choice.logprobs.token_logprobs
            if not token_logprobs:
                logger.error("Empty token logprobs")
                return None

            # Filter out None values and sum
            total_logprob = sum(lp for lp in token_logprobs if lp is not None)

            return total_logprob

        except Exception as e:
            logger.error(f"Error getting sequence logprob: {str(e)}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about API calls."""
        return {
            "total_calls": self.stats["total_calls"],
            "method_successes": {
                "token_counting": 0,
                "echo_based": 0,
                "continuation": self.stats["successes"],
            },
            "method_failures": {
                "token_counting": 0,
                "echo_based": 0,
                "continuation": self.stats["failures"],
            },
            "zero_values": 0,
        }


def compute_log_prob(
    prompt: str,
    response: str,
    provider: str,
    model: str,
    temperature: float,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    top_p: float = 1.0,
    **kwargs: Any,
) -> LogProbResult:
    """Convenience function for one-off log probability computation.

    Args:
        prompt: The prompt text
        response: The response text
        provider: API provider
        model: Model name
        temperature: Temperature setting (MUST match production setting)
        api_key: Optional API key
        system_prompt: Optional system prompt
        top_p: Top-p parameter (default 1.0)
        **kwargs: Additional parameters

    Returns:
        LogProbResult with the log probability or error
    """
    tf = RobustTeacherForcing(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        system_prompt=system_prompt,
        top_p=top_p,
        **kwargs,
    )

    return tf.compute_log_prob(prompt, response)
