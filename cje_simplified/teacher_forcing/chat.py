"""Chat conversation utilities for teacher forcing.

This module provides utilities to convert chat conversations to completions format
and compute log probabilities for chat-based models.
"""

from typing import List, Dict, Tuple, Optional
import logging

from ..data.models import LogProbResult, LogProbStatus
from .templates import (
    ChatTemplateConfig,
    Llama4TemplateConfig,
    HuggingFaceTemplateConfig,
)
from .api import RobustTeacherForcing, compute_total_logprob

logger = logging.getLogger(__name__)


class ChatToCompletionsConverter:
    """Convert chat messages to completions format for teacher forcing.

    This handles the careful conversion needed to compute log P(assistant_reply | context)
    using two continuation calls, preserving exact tokenization.

    Example:
        converter = ChatToCompletionsConverter()

        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."}
        ]

        # Get strings for two-call teacher forcing
        prompt_only, prompt_plus_reply = converter.convert_for_teacher_forcing(chat)

        # Use with your completions API:
        # lp_full = get_logprob(prompt_plus_reply)
        # lp_prefix = get_logprob(prompt_only)
        # lp_reply = lp_full - lp_prefix
    """

    def __init__(
        self,
        template_config: Optional[ChatTemplateConfig] = None,
        use_tokenizer: bool = False,
        tokenizer_name: Optional[str] = None,
    ):
        """Initialize converter.

        Args:
            template_config: Chat template configuration (defaults to Llama 4)
            use_tokenizer: Whether to use HuggingFace tokenizer for exact formatting
            tokenizer_name: HF model name for tokenizer (if use_tokenizer=True)
        """
        self.config = template_config or Llama4TemplateConfig()
        self.use_tokenizer = use_tokenizer
        self._tokenizer = None

        if use_tokenizer:
            self._init_tokenizer(tokenizer_name)

    def _init_tokenizer(self, tokenizer_name: Optional[str]):
        """Initialize HuggingFace tokenizer if requested."""
        if not tokenizer_name:
            raise ValueError("tokenizer_name required when use_tokenizer=True")

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Initialized tokenizer from {tokenizer_name}")
        except ImportError:
            raise ImportError(
                "transformers library required for tokenizer support. "
                "Install with: pip install transformers"
            )

    def convert_for_teacher_forcing(
        self, chat: List[Dict[str, str]]
    ) -> Tuple[str, str]:
        """Convert chat to two strings for teacher forcing.

        Args:
            chat: List of messages with 'role' and 'content' keys

        Returns:
            Tuple of (prompt_only, prompt_plus_reply) where:
            - prompt_only: Context + empty assistant header (for prefix call)
            - prompt_plus_reply: Context + assistant reply (for full call)

        Raises:
            ValueError: If chat format is invalid
        """
        if not chat:
            raise ValueError("Chat cannot be empty")

        if chat[-1]["role"] != "assistant":
            raise ValueError(
                "Last message must be assistant reply for teacher forcing. "
                f"Got role='{chat[-1]['role']}'"
            )

        # If we have a HuggingFaceTemplateConfig, always use tokenizer method
        if isinstance(self.config, HuggingFaceTemplateConfig):
            return self._convert_with_tokenizer(chat)
        elif self.use_tokenizer and self._tokenizer:
            # Use tokenizer's apply_chat_template for exact formatting
            return self._convert_with_tokenizer(chat)
        else:
            # Manual template construction
            return self._convert_manual(chat)

    def _convert_with_tokenizer(self, chat: List[Dict[str, str]]) -> Tuple[str, str]:
        """Convert using HuggingFace tokenizer for exact formatting."""
        # If config is HuggingFaceTemplateConfig, use its methods
        if isinstance(self.config, HuggingFaceTemplateConfig):
            # Full prompt = context + assistant reply
            prompt_plus_reply = self.config.apply_chat_template(
                chat, add_generation_prompt=False
            )

            # Prefix prompt = context only, with empty assistant header
            prompt_only = self.config.apply_chat_template(
                chat[:-1], add_generation_prompt=True
            )

            return prompt_only, prompt_plus_reply
        else:
            # Fall back to using standalone tokenizer
            # Full prompt = context + assistant reply
            prompt_plus_reply = self._tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False,  # Don't append empty assistant stub
            )

            # Prefix prompt = context only, with empty assistant header
            prompt_only = self._tokenizer.apply_chat_template(
                chat[:-1],  # Drop the assistant reply
                tokenize=False,
                add_generation_prompt=True,  # Adds assistant header stub
            )

            return prompt_only, prompt_plus_reply

    def _convert_manual(self, chat: List[Dict[str, str]]) -> Tuple[str, str]:
        """Manually construct chat template strings."""
        # Build context (all messages except last assistant reply)
        context_parts = []

        # Add begin_of_text if needed
        if self.config.should_add_bos():
            # For manual configs that have begin_of_text attribute
            if hasattr(self.config, "begin_of_text"):
                context_parts.append(self.config.begin_of_text)

        # Add all messages except the last assistant reply
        for msg in chat[:-1]:
            context_parts.append(
                self.config.format_message(msg["role"], msg["content"])
            )

        # Add empty assistant header for prefix
        assistant_header = self.config.format_message_header("assistant")
        context_with_header = "".join(context_parts) + assistant_header

        # Full prompt includes the assistant reply
        full_parts = list(context_parts)
        last_msg = chat[-1]
        full_parts.append(
            self.config.format_message(last_msg["role"], last_msg["content"])
        )
        prompt_plus_reply = "".join(full_parts)

        return context_with_header, prompt_plus_reply

    def validate_tokenization(self, prompt_only: str, prompt_plus_reply: str) -> bool:
        """Validate that tokenization aligns properly for subtraction.

        Args:
            prompt_only: Context + empty assistant header
            prompt_plus_reply: Context + assistant reply

        Returns:
            True if tokenization aligns properly

        Raises:
            RuntimeError: If tokenizer not initialized
        """
        if not self._tokenizer:
            raise RuntimeError(
                "Tokenizer required for validation. "
                "Initialize with use_tokenizer=True"
            )

        # Tokenize both strings
        ids_prefix = self._tokenizer(prompt_only).input_ids
        ids_full = self._tokenizer(prompt_plus_reply).input_ids

        # Check that prefix is actually a prefix of full
        if len(ids_full) < len(ids_prefix):
            logger.error(
                f"Full sequence ({len(ids_full)} tokens) "
                f"shorter than prefix ({len(ids_prefix)} tokens)"
            )
            return False

        # Check token-by-token alignment
        if ids_full[: len(ids_prefix)] != ids_prefix:
            logger.error("Prefix tokens don't match start of full sequence")
            return False

        logger.info(
            f"Tokenization validated: prefix={len(ids_prefix)} tokens, "
            f"full={len(ids_full)} tokens, "
            f"reply={len(ids_full) - len(ids_prefix)} tokens"
        )
        return True

    def format_for_display(self, chat: List[Dict[str, str]]) -> str:
        """Format chat for human-readable display with visible markers."""
        lines = []

        if self.config.should_add_bos() and hasattr(self.config, "begin_of_text"):
            lines.append(f"[BOS: {self.config.begin_of_text}]")

        for msg in chat:
            role = msg["role"]
            content = msg["content"]

            # Show the formatted message with visible markers
            formatted = self.config.format_message(role, content)
            lines.append(f"\n[FORMATTED MESSAGE]:\n{formatted}")

        return "\n".join(lines)


def convert_chat_for_teacher_forcing(
    chat: List[Dict[str, str]],
    template_config: Optional[ChatTemplateConfig] = None,
    tokenizer_name: Optional[str] = None,
    use_tokenizer: bool = True,
) -> Tuple[str, str]:
    """Convenience function to convert chat for teacher forcing.

    Args:
        chat: List of messages with 'role' and 'content'
        template_config: Chat template configuration (defaults to Llama 4)
        tokenizer_name: HF model name for tokenizer (if use_tokenizer=True)
        use_tokenizer: Whether to use HF tokenizer (recommended)

    Returns:
        Tuple of (prompt_only, prompt_plus_reply)

    Example:
        chat = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        # Use with explicit template config
        config = Llama3TemplateConfig()
        prompt_only, prompt_plus_reply = convert_chat_for_teacher_forcing(
            chat,
            template_config=config,
            tokenizer_name="meta-llama/Llama-3.2-3B-Instruct"
        )

        # Now use with your completions API
        lp_full = compute_logprob(prompt_plus_reply)
        lp_prefix = compute_logprob(prompt_only)
        lp_reply = lp_full - lp_prefix
    """
    # Create converter
    converter = ChatToCompletionsConverter(
        template_config=template_config,
        use_tokenizer=use_tokenizer,
        tokenizer_name=tokenizer_name,
    )

    return converter.convert_for_teacher_forcing(chat)


class ChatTeacherForcing:
    """Teacher forcing for chat conversations using Fireworks API.

    This class handles the complete workflow:
    1. Convert chat format to completions format
    2. Make two API calls (prefix and full)
    3. Compute log P(assistant_reply | context)

    Example:
        ctf = ChatTeacherForcing(
            model="accounts/fireworks/models/llama-v4-17b-instruct",
            use_tokenizer=True
        )

        chat = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."}
        ]

        result = ctf.compute_chat_logprob(chat)
        if result.is_valid:
            print(f"Log probability: {result.value}")
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        template_config: Optional[ChatTemplateConfig] = None,
        use_tokenizer: bool = True,
        tokenizer_name: Optional[str] = None,
    ):
        """Initialize chat teacher forcing.

        Args:
            model: Fireworks model name
            api_key: API key (uses env var if not provided)
            temperature: Sampling temperature
            template_config: Chat template configuration (defaults to Llama 4)
            use_tokenizer: Whether to use HF tokenizer for exact formatting
            tokenizer_name: HF model name for tokenizer (if use_tokenizer=True)
        """
        self.model = model
        self.temperature = temperature
        self.template_config = template_config
        self.use_tokenizer = use_tokenizer
        self.tokenizer_name = tokenizer_name

        # Initialize Fireworks teacher forcing
        self.tf = RobustTeacherForcing(
            model=model, api_key=api_key, temperature=temperature
        )

    def compute_chat_logprob(
        self, chat: List[Dict[str, str]], system_prompt: Optional[str] = None
    ) -> LogProbResult:
        """Compute log probability of assistant's reply given chat context.

        Args:
            chat: Chat messages (last must be assistant)
            system_prompt: Optional system prompt to prepend

        Returns:
            LogProbResult with log P(assistant_reply | context)
        """
        # Add system prompt to chat if provided
        if system_prompt and chat[0]["role"] != "system":
            chat = [{"role": "system", "content": system_prompt}] + chat

        # Validate chat format
        if not chat or chat[-1]["role"] != "assistant":
            return LogProbResult(
                status=LogProbStatus.API_ERROR,
                error="Chat must end with assistant message for teacher forcing",
            )

        # Extract the assistant reply we're scoring
        assistant_reply = chat[-1]["content"]

        # Convert to completions format
        try:
            prompt_only, prompt_plus_reply = convert_chat_for_teacher_forcing(
                chat,
                template_config=self.template_config,
                tokenizer_name=self.tokenizer_name,
                use_tokenizer=self.use_tokenizer,
            )
        except Exception as e:
            return LogProbResult(
                status=LogProbStatus.API_ERROR,
                error=f"Chat conversion failed: {str(e)}",
            )

        # Use continuation method with two calls
        try:
            # Get log P(prompt + reply)
            full_result = self._get_completion_logprob(prompt_plus_reply)
            if not full_result.is_valid:
                return full_result

            # Get log P(prompt only)
            prefix_result = self._get_completion_logprob(prompt_only)
            if not prefix_result.is_valid:
                return prefix_result

            # Compute log P(reply | prompt) = log P(full) - log P(prefix)
            lp_reply = full_result.value - prefix_result.value

            # Sanity checks
            if lp_reply > 0:
                return LogProbResult(
                    status=LogProbStatus.API_ERROR,
                    error=f"Positive log probability: {lp_reply}",
                    metadata={
                        "lp_full": full_result.value,
                        "lp_prefix": prefix_result.value,
                    },
                )

            # Check for extreme values
            tokens_estimate = len(assistant_reply) / 4  # ~4 chars per token
            if tokens_estimate > 0:
                avg_per_token = lp_reply / tokens_estimate
                if avg_per_token < -10:
                    logger.warning(
                        f"Extreme negative log prob: {lp_reply:.2f} "
                        f"for ~{tokens_estimate:.0f} tokens "
                        f"(avg: {avg_per_token:.2f}/token)"
                    )

            return LogProbResult(
                value=lp_reply,
                status=LogProbStatus.SUCCESS,
                metadata={
                    "method": "chat_continuation",
                    "lp_full": full_result.value,
                    "lp_prefix": prefix_result.value,
                    "reply_length": len(assistant_reply),
                    "used_tokenizer": self.use_tokenizer,
                },
            )

        except Exception as e:
            return LogProbResult(
                status=LogProbStatus.API_ERROR, error=f"API calls failed: {str(e)}"
            )

    def _get_completion_logprob(self, prompt: str) -> LogProbResult:
        """Get total log probability for a completion prompt."""
        return compute_total_logprob(
            text=prompt,
            model=self.model,
            api_key=self.tf.api_key,
            temperature=self.temperature,
        )


def compute_chat_logprob(
    chat: List[Dict[str, str]],
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 1.0,
    template_config: Optional[ChatTemplateConfig] = None,
    use_tokenizer: bool = True,
    tokenizer_name: Optional[str] = None,
) -> LogProbResult:
    """Convenience function to compute log probability for a chat.

    Args:
        chat: Chat messages (last must be assistant)
        model: Fireworks model name
        api_key: API key (uses env var if not provided)
        temperature: Sampling temperature
        template_config: Chat template configuration (defaults to Llama 4)
        use_tokenizer: Whether to use HF tokenizer
        tokenizer_name: HF model name for tokenizer (if use_tokenizer=True)

    Returns:
        LogProbResult with log P(assistant_reply | context)

    Example:
        # With explicit template config
        config = Llama3TemplateConfig()
        result = compute_chat_logprob(
            chat=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"}
            ],
            model="accounts/fireworks/models/llama-v3p2-3b-instruct",
            template_config=config,
            tokenizer_name="meta-llama/Llama-3.2-3B-Instruct"
        )
    """
    ctf = ChatTeacherForcing(
        model=model,
        api_key=api_key,
        temperature=temperature,
        template_config=template_config,
        use_tokenizer=use_tokenizer,
        tokenizer_name=tokenizer_name,
    )
    return ctf.compute_chat_logprob(chat)
