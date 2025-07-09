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
import tiktoken

from ..types import LogProbResult, LogProbStatus

logger = logging.getLogger(__name__)


class RobustTeacherForcing:
    """Teacher forcing with explicit error handling and validation.

    This implementation uses multiple methods to ensure accurate log probability
    extraction, handling edge cases where token boundaries don't align with
    text boundaries.

    Methods tried in order:
    1. Token counting - Count prompt tokens and skip that many (default)
    2. Echo-based - Use model's echo feature to identify response tokens
    3. Continuation - Compute P(response|prompt) from full probabilities

    Args:
        provider: API provider (e.g., "fireworks")
        model: Model name
        api_key: Optional API key
        temperature: Model temperature
        system_prompt: Optional system prompt
        seed: Optional seed for deterministic results
        force_continuation: If True, ONLY use continuation method (no fallback)
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        force_continuation: bool = False,
        **kwargs: Any,  # For any other policy-specific parameters
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.seed = seed
        self.force_continuation = force_continuation
        self.extra_params = kwargs  # Store any additional parameters
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")

        if not self.api_key:
            raise ValueError("API key required")

        self.api_client = self._init_client()

        # Initialize tokenizer - use cl100k_base as default for modern models
        # This is used for validation only, actual tokenization happens server-side
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback if tiktoken not available
            self.tokenizer = None
            logger.warning("tiktoken not available, some validations will be skipped")

        # Track server model revision to detect changes
        self.server_revision: Optional[str] = None

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

        # Apply system prompt if provided
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n{prompt}"

        # Detect edge cases that often cause boundary issues
        use_continuation_first = (
            self.force_continuation
            or self._should_use_continuation_method(prompt, response)
        )

        # Try methods in order
        if self.force_continuation:
            # Force continuation only - no fallback to less reliable methods
            methods = [
                ("continuation", self._continuation_method),
            ]
        elif use_continuation_first:
            # Continuation is most reliable for edge cases
            methods = [
                ("continuation", self._continuation_method),
                ("token_counting", self._token_counting_method),
                ("echo_based", self._echo_based_method),
            ]
        else:
            # Token counting is faster for normal cases
            methods = [
                ("token_counting", self._token_counting_method),
                ("continuation", self._continuation_method),
                ("echo_based", self._echo_based_method),
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
            completion_params = {
                "model": self.model,
                "prompt": full_text,
                "max_tokens": 0,  # Don't generate new tokens
                "echo": True,  # Return prompt + response tokens
                "logprobs": 1,  # Get log probabilities (must be >=1)
                "temperature": self.temperature,
            }
            if self.seed is not None:
                completion_params["seed"] = self.seed

            completion = self.api_client.completions.create(**completion_params)

            choice = completion.choices[0]

            # Patch 3: Check tokenizer revision
            if hasattr(choice, "model") and choice.model:
                if self.server_revision is None:
                    self.server_revision = choice.model
                elif choice.model != self.server_revision:
                    raise RuntimeError(
                        f"Tokenizer/model revision changed mid-run: "
                        f"{self.server_revision} -> {choice.model}"
                    )

            # Extract tokens and logprobs
            if not hasattr(choice, "logprobs") or not choice.logprobs:
                return LogProbResult(
                    status=LogProbStatus.API_ERROR,
                    error="No logprobs in response",
                    metadata={"method": "token_counting"},
                )

            # Patch 2: Truncation guard
            if self.tokenizer:
                expected = len(self.tokenizer.encode(full_text))
                got = len(choice.logprobs.tokens)
                # Allow larger discrepancies (up to 10% or 50 tokens) due to tokenizer differences
                # Different models use different tokenizers (e.g., Llama vs GPT)
                tolerance = max(50, int(expected * 0.10))
                if got < expected - tolerance:
                    return LogProbResult(
                        status=LogProbStatus.TOKEN_LIMIT_EXCEEDED,
                        error=f"Context truncated: expected {expected} tokens, got {got}",
                        metadata={
                            "expected": expected,
                            "received": got,
                            "method": "token_counting",
                            "tolerance": tolerance,
                        },
                    )

            # Count prompt tokens separately
            prompt_params = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
                "temperature": self.temperature,
            }
            if self.seed is not None:
                prompt_params["seed"] = self.seed

            prompt_completion = self.api_client.completions.create(**prompt_params)

            prompt_choice = prompt_completion.choices[0]

            # Check revision for prompt completion too
            if hasattr(prompt_choice, "model") and prompt_choice.model:
                if self.server_revision and prompt_choice.model != self.server_revision:
                    raise RuntimeError(
                        f"Tokenizer/model revision changed mid-run: "
                        f"{self.server_revision} -> {prompt_choice.model}"
                    )

            n_prompt_tokens = len(prompt_choice.logprobs.tokens)
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

            # Check for None values (can happen with certain tokens)
            none_count = sum(1 for lp in response_logprobs if lp is None)
            if none_count > 0:
                logger.warning(
                    f"Found {none_count} None logprobs out of {len(response_logprobs)} response tokens"
                )
                # If too many Nones, this method is unreliable
                if none_count > len(response_logprobs) * 0.1:  # More than 10% None
                    return LogProbResult(
                        status=LogProbStatus.API_ERROR,
                        error=f"Too many None logprobs: {none_count}/{len(response_logprobs)}",
                        metadata={
                            "method": "token_counting",
                            "none_count": none_count,
                            "total_response_tokens": len(response_logprobs),
                        },
                    )

            # Sum log probabilities (skip None values)
            total_logprob = sum(lp for lp in response_logprobs if lp is not None)

            # Critical validation: detect if we likely got the wrong tokens
            if len(response_logprobs) > 0 and response:
                # Estimate tokens in response (~4 chars per token)
                estimated_response_tokens = len(response) / 4
                actual_response_tokens = len(response_logprobs)

                # Check if we have way too many tokens (likely got full sequence)
                if actual_response_tokens > estimated_response_tokens * 2:
                    return LogProbResult(
                        status=LogProbStatus.API_ERROR,
                        error=(
                            f"Token count mismatch: got {actual_response_tokens} tokens "
                            f"for {len(response)}-char response (expected ~{estimated_response_tokens:.0f}). "
                            f"Likely boundary detection failure."
                        ),
                        metadata={
                            "method": "token_counting",
                            "prompt_tokens": n_prompt_tokens,
                            "response_tokens": actual_response_tokens,
                            "total_tokens": n_total_tokens,
                            "response_chars": len(response),
                        },
                    )

                # Check average log prob per token
                avg_logprob_per_token = total_logprob / len(response_logprobs)
                if avg_logprob_per_token < -10:  # Suspiciously negative
                    return LogProbResult(
                        status=LogProbStatus.API_ERROR,
                        error=(
                            f"Suspiciously negative avg log prob: {avg_logprob_per_token:.2f}/token. "
                            f"Total: {total_logprob:.2f} for {len(response_logprobs)} tokens. "
                            f"Likely got full sequence instead of response only."
                        ),
                        metadata={
                            "method": "token_counting",
                            "avg_per_token": avg_logprob_per_token,
                            "prompt_tokens": n_prompt_tokens,
                            "response_tokens": len(response_logprobs),
                            "total_tokens": n_total_tokens,
                        },
                    )

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
            # Patch 4: Prefix validation
            if self.tokenizer:
                prompt_ids = self.tokenizer.encode(prompt)
                full_ids = self.tokenizer.encode(prompt + response)

                # Check if prompt is a proper prefix of full sequence
                if (
                    len(full_ids) >= len(prompt_ids)
                    and full_ids[: len(prompt_ids)] != prompt_ids
                ):
                    return LogProbResult(
                        status=LogProbStatus.TOKEN_BOUNDARY_ERROR,
                        error="Prompt is not a prefix of full sequence",
                        metadata={
                            "prompt_len": len(prompt_ids),
                            "full_len": len(full_ids),
                            "method": "continuation",
                        },
                    )
            # Get log P(prompt + response)
            full_text = prompt + response
            full_params = {
                "model": self.model,
                "prompt": full_text,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
                "temperature": self.temperature,
            }
            if self.seed is not None:
                full_params["seed"] = self.seed

            full_completion = self.api_client.completions.create(**full_params)

            full_choice = full_completion.choices[0]

            # Check revision
            if hasattr(full_choice, "model") and full_choice.model:
                if self.server_revision is None:
                    self.server_revision = full_choice.model
                elif full_choice.model != self.server_revision:
                    raise RuntimeError(
                        f"Tokenizer/model revision changed mid-run: "
                        f"{self.server_revision} -> {full_choice.model}"
                    )

            if not hasattr(full_choice, "logprobs"):
                return LogProbResult(
                    status=LogProbStatus.API_ERROR,
                    error="No logprobs in full completion",
                    metadata={"method": "continuation"},
                )

            # Truncation guard for full text
            if self.tokenizer:
                expected = len(self.tokenizer.encode(full_text))
                got = len(full_choice.logprobs.tokens)
                # Allow larger discrepancies (up to 10% or 50 tokens) due to tokenizer differences
                # Different models use different tokenizers (e.g., Llama vs GPT)
                tolerance = max(50, int(expected * 0.10))
                if got < expected - tolerance:
                    return LogProbResult(
                        status=LogProbStatus.TOKEN_LIMIT_EXCEEDED,
                        error=f"Full text truncated: expected {expected} tokens, got {got}",
                        metadata={
                            "expected": expected,
                            "received": got,
                            "method": "continuation",
                            "tolerance": tolerance,
                        },
                    )

            # Sum all token logprobs for full text
            full_logprobs = full_choice.logprobs.token_logprobs
            full_logprob = sum(lp for lp in full_logprobs if lp is not None)

            # Get log P(prompt)
            prompt_params = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
                "temperature": self.temperature,
            }
            if self.seed is not None:
                prompt_params["seed"] = self.seed

            prompt_completion = self.api_client.completions.create(**prompt_params)

            prompt_choice = prompt_completion.choices[0]

            # Check revision for prompt
            if hasattr(prompt_choice, "model") and prompt_choice.model:
                if self.server_revision and prompt_choice.model != self.server_revision:
                    raise RuntimeError(
                        f"Tokenizer/model revision changed mid-run: "
                        f"{self.server_revision} -> {prompt_choice.model}"
                    )

            # Truncation guard for prompt
            if self.tokenizer:
                expected = len(self.tokenizer.encode(prompt))
                got = len(prompt_choice.logprobs.tokens)
                # Allow larger discrepancies (up to 10% or 50 tokens) due to tokenizer differences
                # Different models use different tokenizers (e.g., Llama vs GPT)
                tolerance = max(50, int(expected * 0.10))
                if got < expected - tolerance:
                    return LogProbResult(
                        status=LogProbStatus.TOKEN_LIMIT_EXCEEDED,
                        error=f"Prompt truncated: expected {expected} tokens, got {got}",
                        metadata={
                            "expected": expected,
                            "received": got,
                            "method": "continuation",
                            "tolerance": tolerance,
                        },
                    )

            prompt_logprobs = prompt_choice.logprobs.token_logprobs
            prompt_logprob = sum(lp for lp in prompt_logprobs if lp is not None)

            # P(response|prompt) = P(prompt, response) / P(prompt)
            # In log space: log P(response|prompt) = log P(prompt, response) - log P(prompt)
            response_logprob = full_logprob - prompt_logprob

            # Validate - positive log prob is impossible
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

            # Validate - check for suspiciously negative values
            # Rough heuristic: ~4 chars per token, expect -0.5 to -5 per token
            estimated_tokens = len(response) / 4
            if estimated_tokens > 0:
                avg_logprob_per_token = response_logprob / estimated_tokens
                if avg_logprob_per_token < -10:  # Very suspicious
                    logger.warning(
                        f"Suspiciously negative log prob: {response_logprob:.2f} "
                        f"for ~{estimated_tokens:.0f} tokens (avg: {avg_logprob_per_token:.2f}/token). "
                        f"Response length: {len(response)} chars"
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

    def _should_use_continuation_method(self, prompt: str, response: str) -> bool:
        """
        Detect cases where token counting is likely to fail.

        These are cases where tokenization boundaries often change:
        1. Prompt ends with punctuation
        2. Response starts with a capital letter
        3. Very short responses
        4. Responses that start with punctuation
        5. Prompt ends with incomplete word/number
        """
        # Very short responses are problematic
        if len(response) < 10:
            return True

        # Check prompt ending
        prompt_stripped = prompt.rstrip()
        if prompt_stripped and prompt_stripped[-1] in ".!?;:,\"'-":
            return True

        # Check response beginning
        if response and response[0].isupper() and not prompt.endswith(" "):
            return True

        if response and response[0] in ".!?;:,\"'-":
            return True

        # Check for incomplete tokens at boundary
        # e.g., "recent." + "Here" might merge into ".Here"
        if prompt_stripped and response:
            last_char = prompt_stripped[-1]
            first_char = response[0]
            # Punctuation followed by letter often merges
            if last_char in ".!?;:," and first_char.isalpha():
                return True

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about method usage and success rates."""
        return self.stats.copy()


def compute_teacher_forced_logprob(
    prompt: str,
    response: str,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    system_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    force_continuation: bool = False,
    **kwargs: Any,
) -> LogProbResult:
    """
    Convenience function for one-off teacher forcing computation.

    Args:
        prompt: The prompt text
        response: The response text
        provider: API provider (e.g. "fireworks")
        model: Model name
        api_key: Optional API key (uses environment variable if not provided)
        temperature: Model temperature
        system_prompt: Optional system prompt to prepend
        seed: Optional seed for deterministic results
        force_continuation: If True, ONLY use continuation method - no fallback (most reliable)
        **kwargs: Additional policy-specific parameters

    Returns:
        LogProbResult with log probability or error
    """
    tf = RobustTeacherForcing(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        system_prompt=system_prompt,
        seed=seed,
        force_continuation=force_continuation,
        **kwargs,
    )
    return tf.compute_log_prob(prompt, response)
