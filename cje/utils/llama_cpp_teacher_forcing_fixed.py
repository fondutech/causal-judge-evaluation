"""
Fixed llama.cpp teacher forcing that works around the max_tokens=0 bug.

This implementation uses the low-level eval() API to compute log probabilities
without generating any tokens.
"""

import os
import logging
import threading
import unicodedata
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

from ..types import LogProbResult, LogProbStatus
from .llama_chat_templates import format_llama3_for_teacher_forcing

logger = logging.getLogger(__name__)

# Global cache for loaded models
_MODEL_CACHE: Dict[str, Any] = {}
# Global lock for thread safety
_MODEL_LOCKS: Dict[str, threading.Lock] = {}


class LlamaCppTeacherForcingFixed:
    """Fixed teacher forcing using llama.cpp's eval() API.

    This implementation works around the max_tokens=0 bug by using
    the lower-level eval() API to score tokens without generation.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_batch: int = 512,  # Batch size for eval
        n_gpu_layers: int = -1,
        seed: int = 0,
        n_threads: Optional[int] = None,
        use_mlock: bool = True,
        verbose: bool = False,
        use_chat_template: bool = True,  # Use proper Llama-3 chat template
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            )

        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.n_threads = n_threads
        self.use_mlock = use_mlock
        self.verbose = verbose
        self.use_chat_template = use_chat_template

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Initialize model
        self._model: Optional[Any] = None
        self._cache: Dict[str, Any] = {}

        # Get or create lock for this model
        cache_key = str(self.model_path)
        if cache_key not in _MODEL_LOCKS:
            _MODEL_LOCKS[cache_key] = threading.Lock()
        self._lock = _MODEL_LOCKS[cache_key]

    def _get_model(self) -> Any:
        """Get or create model instance."""
        if self._model is not None:
            return self._model

        # Check global cache
        cache_key = str(self.model_path)
        if cache_key in _MODEL_CACHE:
            logger.info("Using cached model")
            self._model = _MODEL_CACHE[cache_key]
            return self._model

        from llama_cpp import Llama

        logger.info(f"Loading model: {self.model_path.name}")

        self._model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_gpu_layers=self.n_gpu_layers,
            seed=self.seed,
            n_threads=self.n_threads,
            use_mlock=self.use_mlock,
            verbose=self.verbose,
            logits_all=True,  # REQUIRED for getting all token logits
        )

        # Cache globally
        _MODEL_CACHE[cache_key] = self._model

        return self._model

    def compute_log_prob(
        self,
        prompt: str,
        response: str,
        system_prompt: Optional[str] = None,
    ) -> LogProbResult:
        """Compute log probability using the eval() API with thread safety."""

        # Empty response is always 0.0
        if not response:
            return LogProbResult(
                value=0.0,
                status=LogProbStatus.SUCCESS,
                metadata={"method": "empty_response"},
            )

        # Normalize unicode to avoid boundary mismatches
        prompt = unicodedata.normalize("NFC", prompt)
        response = unicodedata.normalize("NFC", response)
        if system_prompt:
            system_prompt = unicodedata.normalize("NFC", system_prompt)

        # Use thread lock for safety
        with self._lock:
            try:
                model = self._get_model()

                # Use chat template for Llama-3 models if enabled
                if self.use_chat_template and "llama-3" in self.model_path.name.lower():
                    # Format with proper chat template
                    prompt_formatted, full_formatted = (
                        format_llama3_for_teacher_forcing(
                            prompt, response, system_prompt
                        )
                    )
                    prompt_tokens = model.tokenize(prompt_formatted.encode())

                    # Tokenize response separately for exact boundary
                    response_tokens = model.tokenize(response.encode(), add_bos=False)
                else:
                    # Legacy formatting - combine system prompt if provided
                    if system_prompt:
                        formatted_prompt = (
                            f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                        )
                    else:
                        formatted_prompt = prompt

                    # Tokenize prompt
                    prompt_tokens = model.tokenize(formatted_prompt.encode())

                    # Tokenize response separately for exact boundary
                    response_tokens = model.tokenize(response.encode(), add_bos=False)

                # Check context length
                total_len = len(prompt_tokens) + len(response_tokens)
                if total_len > model.n_ctx():
                    return LogProbResult(
                        status=LogProbStatus.API_ERROR,
                        error=f"Text too long: {total_len} tokens > {model.n_ctx()} context",
                        metadata={"method": "llama_cpp_eval"},
                    )

                if not response_tokens:
                    return LogProbResult(
                        status=LogProbStatus.API_ERROR,
                        error="No response tokens found",
                        metadata={"method": "llama_cpp_eval"},
                    )

                # Reset model state
                model.reset()

                # Process prompt in batches
                idx = 0
                while idx < len(prompt_tokens):
                    batch = prompt_tokens[idx : idx + self.n_batch]
                    model.eval(batch)
                    idx += len(batch)

                # Now score response tokens in batches
                total_logprob = 0.0
                idx = 0

                while idx < len(response_tokens):
                    # Get logits BEFORE feeding the token (predicting it)
                    if hasattr(model, "eval_logits"):
                        # Use public API (recommended)
                        logits = model.eval_logits[-1]
                    else:
                        # Fallback to protected attribute
                        logits = model._scores[-1, :]

                    # Convert logits to log probabilities using stable softmax
                    logits_max = np.max(logits)
                    exp_logits = np.exp(logits - logits_max)
                    log_probs = logits - logits_max - np.log(np.sum(exp_logits))

                    # Score the current batch of tokens
                    batch_end = min(idx + self.n_batch, len(response_tokens))
                    for i in range(idx, batch_end):
                        token_id = response_tokens[i]
                        token_logprob = float(log_probs[token_id])
                        total_logprob += token_logprob

                        # If not last token in batch, get next logits
                        if i < batch_end - 1:
                            # Feed single token and get new logits
                            model.eval([token_id])
                            if hasattr(model, "eval_logits"):
                                logits = model.eval_logits[-1]
                            else:
                                logits = model._scores[-1, :]
                            logits_max = np.max(logits)
                            exp_logits = np.exp(logits - logits_max)
                            log_probs = logits - logits_max - np.log(np.sum(exp_logits))

                    # Feed the last token of the batch if we have more tokens
                    if batch_end < len(response_tokens):
                        model.eval([response_tokens[batch_end - 1]])

                    idx = batch_end

                # Validate result (check for positive log prob > 1e-7)
                if total_logprob > 1e-7:
                    return LogProbResult(
                        status=LogProbStatus.API_ERROR,
                        error=f"Positive log probability: {total_logprob}",
                        metadata={"method": "llama_cpp_eval"},
                    )

                return LogProbResult(
                    value=total_logprob,
                    status=LogProbStatus.SUCCESS,
                    metadata={
                        "method": "llama_cpp_eval",
                        "prompt_tokens": len(prompt_tokens),
                        "response_tokens": len(response_tokens),
                        "model": self.model_path.name,
                        "n_batch": self.n_batch,
                    },
                )

            except Exception as e:
                logger.error(f"Error computing log prob: {e}")
                return LogProbResult(
                    status=LogProbStatus.API_ERROR,
                    error=str(e),
                    metadata={"method": "llama_cpp_eval"},
                )
