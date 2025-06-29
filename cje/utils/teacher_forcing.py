"""
Robust teacher forcing implementation with proper token tracking.

This module provides a robust implementation of teacher forcing that correctly
handles tokenization boundaries and edge cases discovered during the Arena 10K
analysis.
"""

import os
import time
import openai
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

from ..types import LogProbResult, LogProbStatus

logger = logging.getLogger(__name__)


class RobustTeacherForcing:
    """Teacher forcing with explicit error handling and validation.

    This implementation uses multiple methods to ensure accurate log probability
    extraction, handling edge cases where token boundaries don't align with
    text boundaries.

    Methods tried in order:
    1. Token counting - Count prompt tokens and skip that many
    2. Echo-based - Use model's echo feature to identify response tokens
    3. Continuation - Compute P(response|prompt) from full probabilities
    """

    def __init__(self, provider: str, model: str, api_key: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")

        if not self.api_key:
            raise ValueError("API key required")

        self.api_client = self._init_client()

        # Statistics
        self.stats: Dict[str, Any] = {
            "total_calls": 0,
            "method_successes": {
                "token_counting": 0,
                "echo_based": 0,
                "continuation": 0,
            },
            "method_failures": {
                "token_counting": 0,
                "echo_based": 0,
                "continuation": 0,
            },
            "zero_values": 0,
        }

    def _init_client(self) -> Any:
        """Initialize API client."""
        if self.provider == "fireworks":
            import openai

            client = openai.OpenAI(
                api_key=self.api_key, base_url="https://api.fireworks.ai/inference/v1"
            )
            return client
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def compute_log_prob(self, prompt: str, response: str) -> LogProbResult:
        """
        Compute log probability of response given prompt.

        Args:
            prompt: The prompt text
            response: The response text

        Returns:
            LogProbResult with value or error information
        """
        self.stats["total_calls"] += 1

        # Empty response is always 0.0
        if not response:
            return LogProbResult(
                value=0.0,
                status=LogProbStatus.SUCCESS,
                metadata={"method": "empty_response"},
            )

        # Try methods in order
        methods = [
            ("token_counting", self._token_counting_method),
            ("echo_based", self._echo_based_method),
            ("continuation", self._continuation_method),
        ]

        last_error = None
        for method_name, method_func in methods:
            try:
                result = method_func(prompt, response)
                if result.is_valid:
                    self.stats["method_successes"][method_name] += 1

                    # Track zero values
                    if result.value == 0.0:
                        self.stats["zero_values"] += 1
                        logger.warning(
                            f"Got 0.0 for non-empty response: '{response[:50]}...'"
                        )

                    return result
                else:
                    self.stats["method_failures"][method_name] += 1
                    last_error = result.error

            except Exception as e:
                self.stats["method_failures"][method_name] += 1
                last_error = str(e)
                logger.debug(f"Method {method_name} failed: {e}")

        # All methods failed
        return LogProbResult(
            status=LogProbStatus.API_ERROR,
            error=f"All methods failed. Last error: {last_error}",
            metadata={"attempts": len(methods)},
        )

    def _token_counting_method(self, prompt: str, response: str) -> LogProbResult:
        """
        Method 1: Count tokens in prompt, skip that many in full sequence.

        This is the primary method but can fail when tokenization changes
        at the prompt/response boundary.
        """
        try:
            # Get full text
            full_text = prompt + response

            # Get completion with echo
            completion = self.api_client.completions.create(
                model=self.model,
                prompt=full_text,
                max_tokens=0,  # Don't generate new tokens
                echo=True,  # Return prompt + response tokens
                logprobs=0,  # Get log probabilities
                temperature=0.0,
            )

            choice = completion.choices[0]

            # Extract tokens and logprobs
            if not hasattr(choice, "logprobs") or not choice.logprobs:
                return LogProbResult(
                    status=LogProbStatus.API_ERROR,
                    error="No logprobs in response",
                    metadata={"method": "token_counting"},
                )

            # Count prompt tokens separately
            prompt_completion = self.api_client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=0,
                echo=True,
                logprobs=0,
                temperature=0.0,
            )

            n_prompt_tokens = len(prompt_completion.choices[0].logprobs.tokens)
            n_total_tokens = len(choice.logprobs.tokens)

            # Extract response logprobs
            response_logprobs = choice.logprobs.token_logprobs[n_prompt_tokens:]

            if not response_logprobs:
                return LogProbResult(
                    status=LogProbStatus.API_ERROR,
                    error="No response tokens found",
                    metadata={
                        "method": "token_counting",
                        "prompt_tokens": n_prompt_tokens,
                        "total_tokens": n_total_tokens,
                    },
                )

            # Sum log probabilities (skip None values)
            total_logprob = sum(lp for lp in response_logprobs if lp is not None)

            # Validate result
            if total_logprob > 0:
                return LogProbResult(
                    status=LogProbStatus.API_ERROR,
                    error=f"Positive log probability: {total_logprob}",
                    metadata={"method": "token_counting"},
                )

            return LogProbResult(
                value=total_logprob,
                status=LogProbStatus.SUCCESS,
                metadata={
                    "method": "token_counting",
                    "prompt_tokens": n_prompt_tokens,
                    "response_tokens": len(response_logprobs),
                    "total_tokens": n_total_tokens,
                },
            )

        except Exception as e:
            return LogProbResult(
                status=LogProbStatus.API_ERROR,
                error=f"Token counting failed: {str(e)}",
                metadata={"method": "token_counting"},
            )

    def _echo_based_method(self, prompt: str, response: str) -> LogProbResult:
        """
        Method 2: Use echo mode with generation to identify response tokens.

        Some models support marking which tokens were generated vs echoed.
        """
        # This method requires specific API support
        # For now, return not implemented
        return LogProbResult(
            status=LogProbStatus.API_ERROR,
            error="Echo-based method not implemented for this provider",
            metadata={"method": "echo_based"},
        )

    def _continuation_method(self, prompt: str, response: str) -> LogProbResult:
        """
        Method 3: Compute P(response|prompt) using two API calls.

        This is the most expensive but most reliable method.
        """
        try:
            # Get log P(prompt + response)
            full_text = prompt + response
            full_completion = self.api_client.completions.create(
                model=self.model,
                prompt=full_text,
                max_tokens=0,
                echo=True,
                logprobs=0,
                temperature=0.0,
            )

            if not hasattr(full_completion.choices[0], "logprobs"):
                return LogProbResult(
                    status=LogProbStatus.API_ERROR,
                    error="No logprobs in full completion",
                    metadata={"method": "continuation"},
                )

            # Sum all token logprobs for full text
            full_logprobs = full_completion.choices[0].logprobs.token_logprobs
            full_logprob = sum(lp for lp in full_logprobs if lp is not None)

            # Get log P(prompt)
            prompt_completion = self.api_client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=0,
                echo=True,
                logprobs=0,
                temperature=0.0,
            )

            prompt_logprobs = prompt_completion.choices[0].logprobs.token_logprobs
            prompt_logprob = sum(lp for lp in prompt_logprobs if lp is not None)

            # P(response|prompt) = P(prompt, response) / P(prompt)
            # In log space: log P(response|prompt) = log P(prompt, response) - log P(prompt)
            response_logprob = full_logprob - prompt_logprob

            # Validate
            if response_logprob > 0:
                return LogProbResult(
                    status=LogProbStatus.API_ERROR,
                    error=f"Positive log probability: {response_logprob}",
                    metadata={
                        "method": "continuation",
                        "full_logprob": full_logprob,
                        "prompt_logprob": prompt_logprob,
                    },
                )

            return LogProbResult(
                value=response_logprob,
                status=LogProbStatus.SUCCESS,
                metadata={
                    "method": "continuation",
                    "full_logprob": full_logprob,
                    "prompt_logprob": prompt_logprob,
                    "response_logprob": response_logprob,
                },
            )

        except Exception as e:
            return LogProbResult(
                status=LogProbStatus.API_ERROR,
                error=f"Continuation method failed: {str(e)}",
                metadata={"method": "continuation"},
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about method usage and success rates."""
        return self.stats.copy()


def compute_teacher_forced_logprob(
    prompt: str, response: str, provider: str, model: str, api_key: Optional[str] = None
) -> LogProbResult:
    """
    Convenience function for one-off teacher forcing computation.

    Args:
        prompt: The prompt text
        response: The response text
        provider: API provider (e.g. "fireworks")
        model: Model name
        api_key: Optional API key (uses environment variable if not provided)

    Returns:
        LogProbResult with log probability or error
    """
    tf = RobustTeacherForcing(provider=provider, model=model, api_key=api_key)
    return tf.compute_log_prob(prompt, response)
