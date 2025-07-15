"""
Token counting implementation for teacher forcing.

This module contains the token counting method that was previously part of
RobustTeacherForcing. It's preserved here for reference and potential future use,
but is not currently used in production due to reliability issues with token
boundary detection.

The token counting method attempts to:
1. Get log probabilities for the full prompt+response with echo=True
2. Count prompt tokens to find where response begins
3. Sum only the response token log probabilities

Known issues:
- Token boundary detection can fail when tokenization changes between calls
- Different models use different tokenizers, making it unreliable
- The "prompt is not a prefix" error occurs when token boundaries shift
"""

import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from cje.types.results import LogProbResult, LogProbStatus


def token_counting_method(
    api_client,
    model: str,
    prompt: str,
    response: str,
    temperature: float,
    top_p: float,
    tokenizer: Optional[Any] = None,
    seed: Optional[int] = None,
    method_stats: Optional[Dict[str, int]] = None,
) -> LogProbResult:
    """
    Token counting method for computing log P(response|prompt).

    This method:
    1. Calls the API with prompt+response and echo=True
    2. Counts tokens to find where the response begins
    3. Sums log probabilities for response tokens only

    Args:
        api_client: The API client (OpenAI-compatible)
        model: Model name
        prompt: The prompt text
        response: The response text
        temperature: Temperature for sampling
        top_p: Top-p for nucleus sampling
        tokenizer: Optional tokenizer for validation
        seed: Optional random seed
        method_stats: Optional dict to track method statistics

    Returns:
        LogProbResult with the log probability or error
    """
    try:
        # Build full text
        full_text = prompt + response

        # API parameters
        params = {
            "model": model,
            "prompt": full_text,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
            "temperature": temperature,
            "top_p": top_p,
        }

        if seed is not None:
            params["seed"] = seed

        # Make API call
        completion = api_client.completions.create(**params)

        if not completion.choices:
            return LogProbResult(
                status=LogProbStatus.API_ERROR,
                error="No choices in completion",
                metadata={"method": "token_counting"},
            )

        choice = completion.choices[0]

        if not hasattr(choice, "logprobs") or choice.logprobs is None:
            return LogProbResult(
                status=LogProbStatus.API_ERROR,
                error="No logprobs in completion",
                metadata={"method": "token_counting"},
            )

        # Get tokens and log probabilities
        tokens = choice.logprobs.tokens
        token_logprobs = choice.logprobs.token_logprobs

        if not tokens or not token_logprobs:
            return LogProbResult(
                status=LogProbStatus.API_ERROR,
                error="Empty tokens or logprobs",
                metadata={"method": "token_counting"},
            )

        # Find where response starts by reconstructing text
        reconstructed = ""
        prompt_end_idx = None

        for i, token in enumerate(tokens):
            reconstructed += token
            # Check if we've reconstructed the prompt
            if prompt_end_idx is None and len(reconstructed) >= len(prompt):
                # Handle case where prompt might end mid-token
                if reconstructed[: len(prompt)] == prompt:
                    prompt_end_idx = i + 1
                    break

        if prompt_end_idx is None:
            return LogProbResult(
                status=LogProbStatus.TOKEN_BOUNDARY_ERROR,
                error="Could not find prompt boundary in tokens",
                metadata={
                    "method": "token_counting",
                    "prompt_len": len(prompt),
                    "reconstructed_len": len(reconstructed),
                },
            )

        # Validate we have response tokens
        if prompt_end_idx >= len(tokens):
            return LogProbResult(
                status=LogProbStatus.TOKEN_BOUNDARY_ERROR,
                error="No response tokens found",
                metadata={
                    "method": "token_counting",
                    "prompt_end_idx": prompt_end_idx,
                    "total_tokens": len(tokens),
                },
            )

        # Sum log probabilities for response tokens
        response_logprobs = token_logprobs[prompt_end_idx:]
        response_logprob = sum(lp for lp in response_logprobs if lp is not None)

        # Track success
        if method_stats is not None:
            method_stats["method_successes"]["token_counting"] += 1

        return LogProbResult(
            status=LogProbStatus.SUCCESS,
            value=response_logprob,
            metadata={
                "method": "token_counting",
                "prompt_tokens": prompt_end_idx,
                "response_tokens": len(tokens) - prompt_end_idx,
                "total_tokens": len(tokens),
            },
        )

    except Exception as e:
        if method_stats is not None:
            method_stats["method_failures"]["token_counting"] += 1

        return LogProbResult(
            status=LogProbStatus.API_ERROR,
            error=f"Token counting method failed: {str(e)}",
            metadata={"method": "token_counting"},
        )


def estimate_token_boundary(
    prompt: str,
    response: str,
    tokens: List[str],
    token_offsets: Optional[List[int]] = None,
) -> Optional[int]:
    """
    Estimate where the response begins in the token sequence.

    This is a more sophisticated version that handles edge cases better.

    Args:
        prompt: The prompt text
        response: The response text
        tokens: List of tokens from the API
        token_offsets: Optional list of character offsets for each token

    Returns:
        Index where response tokens begin, or None if not found
    """
    # Method 1: Use token offsets if available
    if token_offsets is not None:
        prompt_len = len(prompt)
        for i, offset in enumerate(token_offsets):
            if offset >= prompt_len:
                return i

    # Method 2: Reconstruct text and find boundary
    reconstructed = ""
    for i, token in enumerate(tokens):
        prev_len = len(reconstructed)
        reconstructed += token

        # Check if we just passed the prompt boundary
        if prev_len < len(prompt) <= len(reconstructed):
            # The boundary is within this token
            # If the token exactly ends the prompt, response starts at next token
            if reconstructed[: len(prompt)] == prompt:
                return i + 1
            # Otherwise, this token contains part of both prompt and response
            # This is the problematic case that causes failures
            return None

    return None
