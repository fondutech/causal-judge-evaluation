# This file interfaces with multiple external APIs (OpenAI, Anthropic, Google) whose
# type definitions are either incomplete, inconsistent with their implementation, or
# change frequently between versions. In particular, the logprobs feature in OpenAI's
# API has structural differences between documented types and actual response shapes.
# Rather than adding numerous specific type ignores throughout the code, we use a
# global ignore directive as a temporary solution until type definitions stabilize.
# mypy: ignore-errors
from __future__ import annotations

from typing import (
    List,
    Tuple,
    Optional,
    Any,
    Union,
    Iterable,
    Dict,
    Type,
    Literal,
    overload,
)
import os
import logging

# Import type annotations for API clients
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import (
        Choice,
        ChoiceLogprobs,
        ChatCompletionTokenLogprob,
    )
    from openai.types.chat import (
        ChatCompletionUserMessageParam,
        ChatCompletionAssistantMessageParam,
    )
    from anthropic.types import Message as AnthropicMessage

from cje.loggers.adapters import (
    ProviderAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    GeminiAdapter,
    FireworksAdapter,
    TogetherAdapter,
)
from cje.loggers.conversation_utils import parse_context
from cje.loggers.completions_templates import (
    CompletionsTemplate,
    CompletionsTemplateConfig,
    get_completions_template,
)
from cje.loggers.template_validation import (
    validate_teacher_forcing,
    TemplateValidationError,
    ValidationCase,
)
from ..constants import OPENAI_COMPATIBLE_PROVIDERS
from cje.utils.logprobs import sum_response_logprobs_tail  # NEW
from ..utils.error_handling import safe_call
from ..utils.progress import track

logger = logging.getLogger(__name__)

_ADAPTERS: Dict[str, Type[ProviderAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "google": GeminiAdapter,
    "fireworks": FireworksAdapter,
    "together": TogetherAdapter,
}


class APIPolicyRunner:
    """Simple wrapper for hosted model APIs (OpenAI, Anthropic or Google)."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 1.0,
        batch_size: int = 16,
        system_prompt: Optional[str] = None,
        user_message_template: str = "{context}",
        completions_template_format: str = "llama4",
    ) -> None:
        """Initialize APIPolicyRunner with completions template configuration.

        Args:
            provider: API provider ('openai', 'anthropic', 'google', 'fireworks', 'together')
            model_name: Model identifier
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            batch_size: Batch size for API calls
            system_prompt: Optional system prompt
            user_message_template: Template for user messages
            completions_template_format: Format for completions API ('llama3' or 'llama4')
                                       IMPORTANT: You must specify the correct format for your model.
                                       - Use 'llama3' for Llama 3.x models
                                       - Use 'llama4' for Llama 4 models
                                       This is required for teacher forcing to work correctly.
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.system_prompt = system_prompt
        self.user_message_template = user_message_template
        self.completions_template_format = completions_template_format

        # Initialize the completions template for converting chat to continuous format
        self.template = get_completions_template(
            template_format=self.completions_template_format
        )

        # Simple in-memory cache for log-prob computations
        self._logprob_cache: Dict[str, float] = {}

        self.client: Any = None
        self.model: Any = None

        if self.provider == "openai":
            try:
                import openai
            except Exception as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "openai package is required for provider 'openai'"
                ) from exc
            self.client = openai.OpenAI()
        elif self.provider == "anthropic":
            try:
                import anthropic
            except Exception as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "anthropic package is required for provider 'anthropic'"
                ) from exc
            self.client = anthropic.Anthropic()
        elif self.provider == "google":
            try:
                import google.generativeai as genai
            except Exception as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "google.generativeai package is required for provider 'google'"
                ) from exc
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model = genai.GenerativeModel(model_name)
        elif self.provider == "fireworks":
            try:
                import openai
            except Exception as exc:
                raise ImportError(
                    "openai package required for provider 'fireworks'"
                ) from exc
            self.client = openai.OpenAI(
                api_key=os.getenv("FIREWORKS_API_KEY"),
                base_url="https://api.fireworks.ai/inference/v1",
            )
        elif self.provider == "together":
            try:
                import openai
            except Exception as exc:
                raise ImportError(
                    "openai package required for provider 'together'"
                ) from exc
            self.client = openai.OpenAI(
                api_key=os.getenv("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1",
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        adapter_class = _ADAPTERS.get(self.provider)
        if not adapter_class:
            raise ValueError(
                f"Unknown or unsupported provider for adapter: {self.provider}"
            )

        if self.provider in OPENAI_COMPATIBLE_PROVIDERS:
            provider_client_instance = self.client
        elif self.provider == "google":
            provider_client_instance = self.model
        else:
            raise ValueError(
                f"Internal error: Unhandled provider '{self.provider}' for adapter client assignment."
            )

        self.adapter: ProviderAdapter = adapter_class(
            client=provider_client_instance,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            user_message_template=self.user_message_template,
        )

    @overload
    def generate_with_logp(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        return_token_logprobs: Literal[True] = True,
    ) -> List[Tuple[str, float, List[float]]]: ...

    @overload
    def generate_with_logp(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        return_token_logprobs: Literal[False] = False,
    ) -> List[Tuple[str, float]]: ...

    @overload
    def generate_with_logp(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        return_token_logprobs: bool = False,
    ) -> List[Tuple[str, float] | Tuple[str, float, List[float]]]: ...

    def generate_with_logp(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        return_token_logprobs: bool = False,
    ) -> List[Tuple[str, float] | Tuple[str, float, List[float]]]:
        """Generate text and return summed log-probability.

        For providers that expose token-level log-probs (OpenAI, Anthropic,
        Google) we request them so that propensities can be logged exactly.
        """

        # Add progress tracking for larger batches
        if len(prompts) > 10:
            logger.info(
                f"Generating {len(prompts)} responses with {self.provider}:{self.model_name}"
            )

        gen_max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        )
        gen_temperature = temperature if temperature is not None else self.temperature
        gen_top_p = top_p if top_p is not None else self.top_p

        adapter_kwargs = {
            "max_new_tokens": gen_max_new_tokens,
            "temperature": gen_temperature,
            "top_p": gen_top_p,
            "batch_size": self.batch_size,
        }

        # Use safe_call for error handling
        adapter_results: List[Tuple[str, List[float]]] = safe_call(
            self.adapter.request,
            fallback=[("", [])] * len(prompts),
            error_context=f"Batch generation for {len(prompts)} prompts with {self.provider}:{self.model_name}",
            prompts=prompts,
            **adapter_kwargs,
        )

        processed_outputs: List[Tuple[str, float] | Tuple[str, float, List[float]]] = []
        for text_result, token_logprobs_list in adapter_results:
            final_text = text_result if text_result is not None else ""

            final_token_logprobs = (
                token_logprobs_list if token_logprobs_list is not None else []
            )
            # Use float64 to prevent overflow for long sequences (P3 fix)
            import numpy as np

            logp = (
                float(np.float64(sum(final_token_logprobs)))
                if final_token_logprobs
                else 0.0
            )

            if return_token_logprobs:
                processed_outputs.append((final_text, logp, final_token_logprobs))
            else:
                processed_outputs.append((final_text, logp))

        return processed_outputs

    def generate_with_consistent_logp(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Generate responses and return teacher-forcing log-probs for consistency with offline evaluation.

        This method implements the two-pass approach:
        1. Generate response text using chat completions API
        2. Score that exact text using teacher forcing (completions + echo)

        This ensures the behavior policy log-probs (π₀) use the same scoring method
        as target policy log-probs (πₖ), guaranteeing consistent importance weights.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate (overrides instance default)
            temperature: Sampling temperature (overrides instance default)
            top_p: Nucleus sampling threshold (overrides instance default)

        Returns:
            List of (generated_text, teacher_forcing_logp) tuples

        Example:
            >>> runner = APIPolicyRunner("fireworks", "llama-v3p1-70b-instruct")
            >>> results = runner.generate_with_consistent_logp(["What is AI?"])
            >>> text, behavior_logp = results[0]
            >>> # Store behavior_logp as π₀ for importance weight computation
        """
        # Step 1: Generate response text using standard generation
        gen_results = self.generate_with_logp(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            return_token_logprobs=False,  # We only need the text
        )

        # Step 2: Score each generated response with teacher forcing
        consistent_results = []
        for i, (prompt, gen_result) in enumerate(zip(prompts, gen_results)):
            generated_text = gen_result[0]  # Extract text from (text, logp) tuple

            # Re-score with teacher forcing for consistency
            teacher_forcing_logp = self.log_prob(
                prompt,
                generated_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            consistent_results.append((generated_text, teacher_forcing_logp))

        return consistent_results

    def log_prob(
        self,
        context: str,
        response: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        *,
        behavior_logp: Optional[float] = None,
    ) -> float:
        """Return log probability of ``response`` given ``context`` using teacher forcing.

        Args:
            context: Input context/prompt
            response: Response text to score
            max_new_tokens: Max tokens (unused for teacher forcing)
            temperature: Temperature (unused for teacher forcing)
            top_p: Top-p (unused for teacher forcing)
            behavior_logp: If provided and this is identical to behavior policy, returns this value

        Returns:
            Log probability of the response given context
        """
        # Simple cache key based on content + model
        cache_key = f"{self.model_name}:{hash(context)}:{hash(response)}"

        # Check cache first
        if cache_key in self._logprob_cache:
            return self._logprob_cache[cache_key]

        # For identical policies, use behavior log-prob if provided
        # This handles the scout=logging case cleanly
        if behavior_logp is not None and self._is_identical_to_behavior_policy():
            logger.debug(
                f"Using behavior log-prob for identical policy {self.model_name}"
            )
            self._logprob_cache[cache_key] = behavior_logp
            return behavior_logp

        # Use teacher forcing for non-identical policies
        result = self._teacher_forcing_logprob(context, response)

        # Cache result
        self._logprob_cache[cache_key] = result
        return result

    def _is_identical_to_behavior_policy(self) -> bool:
        """Check if this policy is identical to the behavior/logging policy."""
        # Simple heuristic: scout models are typically identical to logging policy
        return "scout" in self.model_name.lower()

    def _teacher_forcing_logprob(self, context: str, response: str) -> float:
        """Compute log probability using teacher forcing via completions API."""
        try:
            # Parse context into message format
            messages = parse_context(
                context, self.system_prompt, self.user_message_template
            )

            # Format as complete conversation including response
            full_prompt = self._format_conversation_with_response(messages, response)

            # Use completions API to get logprobs for the full sequence
            resp = self.client.completions.create(
                model=self.model_name,
                prompt=full_prompt,
                max_tokens=0,  # Don't generate - just score existing text
                temperature=0.0,  # Deterministic
                logprobs=5,
                echo=True,  # Return logprobs for input text
            )

            if not resp.choices or not resp.choices[0].logprobs:
                logger.warning(f"No logprobs returned for {self.model_name}")
                return 0.0

            logprobs_data = resp.choices[0].logprobs
            all_token_logprobs = logprobs_data.token_logprobs or []

            if not all_token_logprobs:
                logger.warning(f"Empty token logprobs for {self.model_name}")
                return 0.0

            # CRITICAL: Use divergence-based extraction due to tokenization differences
            # The tokenizer may create different tokens based on context
            result = self._extract_response_logprobs_by_divergence(
                messages, response, all_token_logprobs
            )

            # Sanity check: for very short responses, logprob shouldn't be extremely negative
            response_token_count = self._get_response_token_count(response)
            if response_token_count <= 3 and result < -20:
                logger.warning(
                    f"Suspiciously low logprob {result:.3f} for {response_token_count}-token response. "
                    f"Response: '{response[:50]}...'"
                )

            logger.debug(
                f"Teacher forcing: response '{response[:30]}...', logprob={result:.3f}"
            )
            return result

        except Exception as e:
            logger.warning(f"Teacher forcing failed for {self.model_name}: {e}")
            return 0.0

    def _get_response_token_count(self, response: str) -> int:
        """Get token count for response text."""
        try:
            import tiktoken

            try:
                enc = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(response))
        except ImportError:
            # Fallback: rough estimate
            return max(len(response) // 3, 1)

    def log_prob_batch(
        self,
        contexts: List[str],
        responses: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        show_progress: bool = True,
    ) -> List[float]:
        """Batch log-prob computation with caching and progress tracking.

        This is a new method that efficiently computes log-probs for many pairs.
        """

        if len(contexts) != len(responses):
            raise ValueError(
                f"Contexts ({len(contexts)}) and responses ({len(responses)}) must have same length"
            )

        results = []
        cache_hits = 0

        # Use progress tracking for larger batches
        pairs = list(zip(contexts, responses))
        if show_progress and len(pairs) > 5:
            pairs = track(pairs, description=f"Computing log-probs ({self.provider})")

        for context, response in pairs:
            # Check cache first
            cache_key = f"{self.model_name}:{hash(context)}:{hash(response)}"

            if cache_key in self._logprob_cache:
                results.append(self._logprob_cache[cache_key])
                cache_hits += 1
            else:
                # Compute with error handling (calls the enhanced log_prob method)
                logp = self.log_prob(
                    context,
                    response,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                results.append(logp)

        if cache_hits > 0:
            logger.info(
                f"Cache hits: {cache_hits}/{len(contexts)} ({cache_hits/len(contexts)*100:.1f}%)"
            )

        return results

    def clear_cache(self) -> None:
        """Clear the log-prob cache."""
        self._logprob_cache.clear()
        logger.info("Log-prob cache cleared")

    def cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {"cache_size": len(self._logprob_cache)}

    def validate_teacher_forcing(
        self,
        custom_cases: Optional[List[ValidationCase]] = None,
        fail_fast: bool = True,
        verbose: bool = True,
    ) -> bool:
        """Validate that teacher forcing is working correctly with this configuration.

        This method tests the template formatting and log probability computation
        with known test cases to ensure the setup is correct.

        Args:
            custom_cases: Optional custom validation cases
            fail_fast: Whether to stop on first failure
            verbose: Whether to log detailed information

        Returns:
            True if all validations pass

        Raises:
            TemplateValidationError: If validation fails and fail_fast=True

        Example:
            >>> runner = APIPolicyRunner("fireworks", "llama4-model")
            >>> try:
            ...     runner.validate_teacher_forcing()
            ...     print("✅ Teacher forcing validated")
            ... except TemplateValidationError as e:
            ...     print(f"❌ Validation failed: {e}")
        """
        return validate_teacher_forcing(
            api_runner=self,
            custom_cases=custom_cases,
            fail_fast=fail_fast,
            verbose=verbose,
        )

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """Convert message array back to text format for providers that need it."""
        text_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text_parts.append(f"System: {content}")
            elif role == "user":
                text_parts.append(f"Human: {content}")
            elif role == "assistant":
                text_parts.append(f"AI: {content}")
        return "\n".join(text_parts) + "\n"

    def _format_conversation_with_response(
        self, messages: List[Dict], response: str
    ) -> str:
        """
        Format the conversation as a single prompt string including the logged response.
        Uses the configured template system.
        """
        return self.template.format_with_response(messages, response)

    def _format_conversation_without_response(self, messages: List[Dict]) -> str:
        """
        Format the conversation as a single prompt string WITHOUT the response.
        This is used to calculate the exact token count of the response in context.
        Uses the configured template system.
        """
        return self.template.format_without_response(messages)

    def _extract_response_logprobs_by_divergence(
        self, messages: List[Dict], response: str, all_logprobs: List[Optional[float]]
    ) -> float:
        """
        Extract response logprobs by finding the actual response in the token sequence.

        This is more robust than simple token counting because tokenization
        can differ based on context (e.g., ']</s>' vs '] </s>').
        """
        try:
            import tiktoken

            try:
                enc = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")

            # Get the full prompt with response
            with_resp = self._format_conversation_with_response(messages, response)
            tokens_with = enc.encode(with_resp)

            # Try to find the response tokens directly
            # First try with space prefix (most common case)
            response_with_space = f" {response}"
            response_tokens = enc.encode(response_with_space)

            # Search for the response in the token sequence
            response_start = None
            response_end = None

            # Look for response with space
            for i in range(len(tokens_with) - len(response_tokens) + 1):
                if tokens_with[i : i + len(response_tokens)] == response_tokens:
                    response_start = i
                    response_end = i + len(response_tokens)
                    break

            # If not found, try without space
            if response_start is None:
                response_tokens = enc.encode(response)
                for i in range(len(tokens_with) - len(response_tokens) + 1):
                    if tokens_with[i : i + len(response_tokens)] == response_tokens:
                        response_start = i
                        response_end = i + len(response_tokens)
                        break

            if response_start is None:
                # Fallback: use the old method with divergence point
                logger.warning(
                    f"Could not find exact response '{response}' in tokens, using divergence method"
                )
                without_resp = self._format_conversation_without_response(messages)
                tokens_without = enc.encode(without_resp)

                # Find divergence
                divergence_point = len(tokens_without)
                for i in range(min(len(tokens_with), len(tokens_without))):
                    if tokens_with[i] != tokens_without[i]:
                        divergence_point = i
                        break

                # Skip the ] token and space if present
                if divergence_point < len(tokens_with) - 1:
                    next_token = enc.decode([tokens_with[divergence_point]])
                    if next_token == "]":
                        divergence_point += 1
                        if divergence_point < len(tokens_with):
                            next_token = enc.decode([tokens_with[divergence_point]])
                            if next_token == " ":
                                divergence_point += 1

                response_start = divergence_point

                # Find end-of-sequence
                eos_token = self.template.get_eos_token()
                eos_tokens = enc.encode(eos_token) if eos_token else []

                for i in range(
                    len(tokens_with) - len(eos_tokens), response_start - 1, -1
                ):
                    if tokens_with[i : i + len(eos_tokens)] == eos_tokens:
                        response_end = i
                        break

                if response_end is None:
                    response_end = len(tokens_with)

            # Extract response logprobs
            response_logprobs = all_logprobs[response_start:response_end]

            # Verify extraction
            extracted_tokens = tokens_with[response_start:response_end]
            decoded = enc.decode(extracted_tokens)
            logger.debug(
                f"Extracted tokens {response_start}:{response_end} = '{decoded.strip()}' "
                f"(expected '{response}')"
            )

            # Sum the logprobs
            result = sum(lp for lp in response_logprobs if lp is not None)

            return result

        except ImportError:
            # Fallback to simple counting if tiktoken not available
            logger.warning("tiktoken not available, using simple tail extraction")
            response_token_count = self._get_response_token_count(response)
            from cje.utils.logprobs import sum_response_logprobs_tail

            return sum_response_logprobs_tail(all_logprobs, response_token_count)
