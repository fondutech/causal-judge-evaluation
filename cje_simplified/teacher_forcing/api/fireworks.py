"""Robust teacher forcing for computing log probabilities.

This module provides reliable computation of log P(response|prompt) using
the Fireworks API, handling edge cases discovered during Arena 10K analysis.
"""

import os
from typing import Optional, Dict, Any
import openai

from ...core.types import LogProbResult, LogProbStatus


class RobustTeacherForcing:
    """Teacher forcing with explicit error handling and validation.

    Uses the continuation method (most reliable) to compute log probabilities:
    log P(response|prompt) = log P(prompt + response) - log P(prompt)

    Args:
        model: Model name (e.g., "accounts/fireworks/models/llama-v3p2-3b-instruct")
        api_key: Fireworks API key (uses FIREWORKS_API_KEY env var if not provided)
        temperature: Model temperature (default 1.0 for sampling)
        system_prompt: Optional system prompt to prepend
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt

        # Get API key
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Fireworks API key required. Set FIREWORKS_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize client
        self.client = openai.OpenAI(
            api_key=self.api_key, base_url="https://api.fireworks.ai/inference/v1"
        )

    def compute_log_prob(self, prompt: str, response: str) -> LogProbResult:
        """Compute log probability of response given prompt.

        Args:
            prompt: The input prompt
            response: The generated response

        Returns:
            LogProbResult with log probability or error information
        """
        # Empty response is always 0.0
        if not response:
            return LogProbResult(
                value=0.0,
                status=LogProbStatus.SUCCESS,
                metadata={"method": "empty_response"},
            )

        # Apply system prompt if provided
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        try:
            # Use continuation method: log P(response|prompt) = log P(full) - log P(prompt)

            # Get log P(prompt + response)
            full_text = full_prompt + response
            full_completion = self.client.completions.create(
                model=self.model,
                prompt=full_text,
                max_tokens=0,  # Don't generate new tokens
                echo=True,  # Return prompt + response tokens
                logprobs=1,  # Get log probabilities
                temperature=self.temperature,
            )

            if not hasattr(full_completion.choices[0], "logprobs"):
                return LogProbResult(
                    status=LogProbStatus.API_ERROR,
                    error="No logprobs in response",
                    metadata={"method": "continuation"},
                )

            # Sum log probabilities for full text
            full_logprobs = full_completion.choices[0].logprobs.token_logprobs
            full_logprob = sum(lp for lp in full_logprobs if lp is not None)

            # Get log P(prompt)
            prompt_completion = self.client.completions.create(
                model=self.model,
                prompt=full_prompt,
                max_tokens=0,
                echo=True,
                logprobs=1,
                temperature=self.temperature,
            )

            prompt_logprobs = prompt_completion.choices[0].logprobs.token_logprobs
            prompt_logprob = sum(lp for lp in prompt_logprobs if lp is not None)

            # Compute log P(response|prompt)
            response_logprob = full_logprob - prompt_logprob

            # Validate result
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

            # Check for suspiciously negative values
            # Rough heuristic: ~4 chars per token, expect -0.5 to -5 per token
            estimated_tokens = len(response) / 4
            if estimated_tokens > 0:
                avg_logprob_per_token = response_logprob / estimated_tokens
                if avg_logprob_per_token < -10:
                    # Log warning but still return result
                    print(
                        f"Warning: Suspiciously negative log prob: {response_logprob:.2f} "
                        f"for ~{estimated_tokens:.0f} tokens (avg: {avg_logprob_per_token:.2f}/token)"
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
                error=f"Teacher forcing failed: {str(e)}",
                metadata={"method": "continuation"},
            )


def compute_total_logprob(
    text: str,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 1.0,
) -> LogProbResult:
    """Compute total log probability of a text string.

    Args:
        text: The complete text to score
        model: Fireworks model name
        api_key: Optional API key
        temperature: Model temperature

    Returns:
        LogProbResult with total log probability
    """
    tf = RobustTeacherForcing(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )

    # Get total log prob by treating entire text as "response" with empty prompt
    return tf.compute_log_prob("", text)


def compute_teacher_forced_logprob(
    prompt: str,
    response: str,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 1.0,
    system_prompt: Optional[str] = None,
) -> LogProbResult:
    """Convenience function for one-off teacher forcing computation.

    Args:
        prompt: The input prompt
        response: The generated response
        model: Fireworks model name
        api_key: Optional API key (uses FIREWORKS_API_KEY env var if not provided)
        temperature: Model temperature (default 1.0)
        system_prompt: Optional system prompt to prepend

    Returns:
        LogProbResult with log probability or error information

    Example:
        result = compute_teacher_forced_logprob(
            prompt="What is 2+2?",
            response="The answer is 4.",
            model="accounts/fireworks/models/llama-v3p2-3b-instruct",
            temperature=1.0
        )

        if result.is_valid:
            print(f"Log probability: {result.value}")
        else:
            print(f"Error: {result.error}")
    """
    tf = RobustTeacherForcing(
        model=model,
        api_key=api_key,
        temperature=temperature,
        system_prompt=system_prompt,
    )
    return tf.compute_log_prob(prompt, response)
