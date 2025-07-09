"""
Llama.cpp teacher forcing implementation for local, deterministic log probability computation.

This module provides teacher forcing using llama.cpp models, offering a local alternative
to API-based services with full determinism and no token boundary issues.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from ..types import LogProbResult, LogProbStatus

logger = logging.getLogger(__name__)

# Global cache for loaded models to avoid reloading
_MODEL_CACHE: Dict[str, Any] = {}


class LlamaCppTeacherForcing:
    """Teacher forcing using llama.cpp for local, deterministic computation.

    This implementation uses the "single-call" recipe from the field guide,
    computing log P(response|prompt) = log P(prompt+response) - log P(prompt).

    Advantages over API-based methods:
    - Fully deterministic (with seed)
    - No token boundary issues
    - No API costs or rate limits
    - Works with quantized GGUF models

    Args:
        model_path: Path to the GGUF model file
        n_ctx: Context window size (default: 4096)
        n_gpu_layers: Number of layers to offload to GPU (default: -1 for all)
        seed: Random seed for determinism (default: 0)
        n_threads: Number of CPU threads (default: None for auto)
        use_mlock: Lock model in RAM (default: True)
        verbose: Print llama.cpp logs (default: False)
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        seed: int = 0,
        n_threads: Optional[int] = None,
        use_mlock: bool = True,
        verbose: bool = False,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.n_threads = n_threads
        self.use_mlock = use_mlock
        self.verbose = verbose

        # Validate model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Statistics
        self.stats = {
            "total_calls": 0,
            "successes": 0,
            "failures": 0,
            "cache_hits": 0,
        }

        # Cache for computed log probs
        self._cache: Dict[Tuple[str, str, str], float] = {}

        # Lazy load model
        self._model = None

    def _get_model(self) -> Any:
        """Get or create the Llama model instance."""
        if self._model is not None:
            return self._model

        # Check global cache first
        cache_key = str(self.model_path)
        if cache_key in _MODEL_CACHE:
            logger.info(f"Using cached model: {self.model_path.name}")
            self._model = _MODEL_CACHE[cache_key]
            return self._model

        # Import here to avoid dependency if not using llama.cpp
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            )

        logger.info(f"Loading model: {self.model_path.name}")

        # Create model with our parameters
        self._model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            seed=self.seed,
            n_threads=self.n_threads,
            use_mlock=self.use_mlock,
            verbose=self.verbose,
            logits_all=True,  # Required for log prob computation
        )

        # Cache globally
        _MODEL_CACHE[cache_key] = self._model

        return self._model

    def compute_log_prob(self, prompt: str, response: str) -> LogProbResult:
        """
        Compute log probability of response given prompt.

        Uses the single-call recipe:
        log P(response|prompt) = log P(prompt+response) - log P(prompt)

        Args:
            prompt: The prompt text
            response: The response text

        Returns:
            LogProbResult with log probability or error information
        """
        self.stats["total_calls"] += 1

        # Empty response is always 0.0
        if not response:
            return LogProbResult(
                value=0.0,
                status=LogProbStatus.SUCCESS,
                metadata={"method": "empty_response"},
            )

        # Check cache - but make sure we include the full text with potential space
        if (
            prompt
            and response
            and not prompt.endswith(" ")
            and not response.startswith(" ")
        ):
            full_text_for_cache = prompt + " " + response
        else:
            full_text_for_cache = prompt + response
        cache_key = (prompt, response, full_text_for_cache)

        if cache_key in self._cache:
            self.stats["cache_hits"] += 1
            return LogProbResult(
                value=self._cache[cache_key],
                status=LogProbStatus.SUCCESS,
                metadata={"method": "llama_cpp_cached"},
            )

        try:
            model = self._get_model()

            # 1. Compute log P(prompt + response)
            # Add space between prompt and response if needed
            if (
                prompt
                and response
                and not prompt.endswith(" ")
                and not response.startswith(" ")
            ):
                full_text = prompt + " " + response
            else:
                full_text = prompt + response
            full_result = model.create_completion(
                full_text,
                max_tokens=1,  # Set to 1 to get logprobs (0 seems to generate anyway)
                echo=True,  # Include prompt in output
                logprobs=True,  # Request log probabilities
                temperature=0.0,  # Deterministic
            )

            # 2. Compute log P(prompt)
            prompt_result = model.create_completion(
                prompt,
                max_tokens=1,
                echo=True,
                logprobs=True,
                temperature=0.0,
            )

            # Extract log probs from token_logprobs
            if (
                "choices" in full_result
                and len(full_result["choices"]) > 0
                and "choices" in prompt_result
                and len(prompt_result["choices"]) > 0
            ):

                full_choice = full_result["choices"][0]
                prompt_choice = prompt_result["choices"][0]

                # Get token log probabilities
                if (
                    "logprobs" in full_choice
                    and "token_logprobs" in full_choice["logprobs"]
                    and "logprobs" in prompt_choice
                    and "token_logprobs" in prompt_choice["logprobs"]
                ):

                    # Get tokens to determine actual input length
                    full_tokens = full_choice["logprobs"]["tokens"]
                    prompt_tokens = prompt_choice["logprobs"]["tokens"]

                    # Find where input ends (before generation starts)
                    # The model seems to generate a lot, so we need to find the actual input boundary
                    full_text_len = len(full_text)
                    prompt_len = len(prompt)

                    # Count tokens that are part of the input
                    full_input_tokens = 0
                    reconstructed_full = ""
                    for i, token in enumerate(full_tokens):
                        reconstructed_full += token
                        if len(reconstructed_full) >= full_text_len:
                            full_input_tokens = i + 1
                            break

                    prompt_input_tokens = 0
                    reconstructed_prompt = ""
                    for i, token in enumerate(prompt_tokens):
                        reconstructed_prompt += token
                        if len(reconstructed_prompt) >= prompt_len:
                            prompt_input_tokens = i + 1
                            break

                    # Sum only the input token log probs
                    full_token_logprobs = full_choice["logprobs"]["token_logprobs"][
                        :full_input_tokens
                    ]
                    prompt_token_logprobs = prompt_choice["logprobs"]["token_logprobs"][
                        :prompt_input_tokens
                    ]

                    # Sum log probabilities (skip None values)
                    full_logprob = sum(
                        lp for lp in full_token_logprobs if lp is not None
                    )
                    prompt_logprob = sum(
                        lp for lp in prompt_token_logprobs if lp is not None
                    )

                    # Debug info for long responses
                    if len(response) > 50:
                        logger.debug(
                            f"Long response debug: response_len={len(response)}, "
                            f"full_tokens={full_input_tokens}, prompt_tokens={prompt_input_tokens}, "
                            f"full_logp={full_logprob:.3f}, prompt_logp={prompt_logprob:.3f}"
                        )

                else:
                    return LogProbResult(
                        status=LogProbStatus.API_ERROR,
                        error="No token_logprobs in response",
                        metadata={"method": "llama_cpp"},
                    )
            else:
                return LogProbResult(
                    status=LogProbStatus.API_ERROR,
                    error="Invalid response structure",
                    metadata={"method": "llama_cpp"},
                )

            # 3. Compute log P(response|prompt)
            response_logprob = full_logprob - prompt_logprob

            # Validate result
            if response_logprob > 0:
                return LogProbResult(
                    status=LogProbStatus.API_ERROR,
                    error=f"Positive log probability: {response_logprob}",
                    metadata={"method": "llama_cpp"},
                )

            # Cache successful result
            self._cache[cache_key] = response_logprob
            self.stats["successes"] += 1

            return LogProbResult(
                value=response_logprob,
                status=LogProbStatus.SUCCESS,
                metadata={
                    "method": "llama_cpp",
                    "model": self.model_path.name,
                    "full_logprob": full_logprob,
                    "prompt_logprob": prompt_logprob,
                },
            )

        except Exception as e:
            self.stats["failures"] += 1
            logger.error(f"Llama.cpp computation failed: {str(e)}")
            return LogProbResult(
                status=LogProbStatus.API_ERROR,
                error=f"Llama.cpp error: {str(e)}",
                metadata={"method": "llama_cpp"},
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about computation."""
        return self.stats.copy()

    def clear_cache(self) -> None:
        """Clear the local cache."""
        self._cache.clear()
        logger.info(f"Cleared cache ({self.stats['cache_hits']} hits)")


def compute_llama_cpp_logprob(
    prompt: str,
    response: str,
    model_path: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    seed: int = 0,
    **kwargs: Any,
) -> LogProbResult:
    """
    Convenience function for one-off llama.cpp teacher forcing computation.

    Args:
        prompt: The prompt text
        response: The response text
        model_path: Path to GGUF model file
        n_ctx: Context window size
        n_gpu_layers: Number of layers to offload to GPU (-1 for all)
        seed: Random seed for determinism
        **kwargs: Additional parameters

    Returns:
        LogProbResult with log probability or error
    """
    tf = LlamaCppTeacherForcing(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        seed=seed,
        **kwargs,
    )
    return tf.compute_log_prob(prompt, response)
