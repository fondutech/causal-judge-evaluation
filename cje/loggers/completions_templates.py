"""
Completions API template system for converting chat conversations to continuous strings.

This module provides templates for converting structured chat messages (with roles like
'user', 'assistant', 'system') into continuous string formats required by completions
API endpoints. Different models expect different formatting conventions (Llama 3,
Llama 4, ChatML, etc.) when using teacher forcing for log probability computation.

These templates are specifically for use with completions API calls (with echo=True)
for consistent log probability scoring, NOT for general prompt formatting.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Type
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CompletionsTemplate(ABC):
    """Base class for converting chat conversations to completions API format."""

    @abstractmethod
    def format_with_response(
        self, messages: List[Dict[str, str]], response: str
    ) -> str:
        """
        Format messages with response for completions API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response: The response to include

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def format_without_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages without response for token counting.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def get_eos_token(self) -> str:
        """Get the end-of-sequence token for this template."""
        pass


class Llama3CompletionsTemplate(CompletionsTemplate):
    """
    Llama 3 style template.

    Format: <s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n{user} [/INST] {response}</s>
    """

    def format_with_response(
        self, messages: List[Dict[str, str]], response: str
    ) -> str:
        system_prompt = ""
        user_message = ""

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                user_message = msg.get("content", "")

        if system_prompt:
            return f"<s>[INST] <<SYS>>\\n{system_prompt}\\n<</SYS>>\\n{user_message} [/INST] {response}</s>"
        else:
            return f"<s>[INST] {user_message} [/INST] {response}</s>"

    def format_without_response(self, messages: List[Dict[str, str]]) -> str:
        system_prompt = ""
        user_message = ""

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                user_message = msg.get("content", "")

        if system_prompt:
            return f"<s>[INST] <<SYS>>\\n{system_prompt}\\n<</SYS>>\\n{user_message} [/INST]</s>"
        else:
            return f"<s>[INST] {user_message} [/INST]</s>"

    def get_eos_token(self) -> str:
        return "</s>"


class Llama4CompletionsTemplate(CompletionsTemplate):
    """
    Llama 4 style template.

    Format:
    <|begin_of_text|>
    <|header_start|>user<|header_end|>

    {user_message}<|eot|>
    <|header_start|>assistant<|header_end|>

    {response}<|eot|>
    """

    def format_with_response(
        self, messages: List[Dict[str, str]], response: str
    ) -> str:
        parts = ["<|begin_of_text|>"]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(
                f"<|header_start|>{role}<|header_end|>\\n\\n" f"{content}<|eot|>"
            )

        parts.append(
            f"<|header_start|>assistant<|header_end|>\\n\\n" f"{response}<|eot|>"
        )

        return "".join(parts)

    def format_without_response(self, messages: List[Dict[str, str]]) -> str:
        parts = ["<|begin_of_text|>"]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(
                f"<|header_start|>{role}<|header_end|>\\n\\n" f"{content}<|eot|>"
            )

        parts.append(f"<|header_start|>assistant<|header_end|>\\n\\n")

        return "".join(parts)

    def get_eos_token(self) -> str:
        return "<|eot|>"


class ChatMLCompletionsTemplate(CompletionsTemplate):
    """
    ChatML style template used by some models.

    Format:
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {response}<|im_end|>
    """

    def format_with_response(
        self, messages: List[Dict[str, str]], response: str
    ) -> str:
        parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\\n{content}<|im_end|>")

        parts.append(f"<|im_start|>assistant\\n{response}<|im_end|>")

        return "\\n".join(parts)

    def format_without_response(self, messages: List[Dict[str, str]]) -> str:
        parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\\n{content}<|im_end|>")

        parts.append(f"<|im_start|>assistant\\n")

        return "\\n".join(parts)

    def get_eos_token(self) -> str:
        return "<|im_end|>"


class AlpacaCompletionsTemplate(CompletionsTemplate):
    """
    Alpaca style template.

    Format:
    ### Instruction:
    {instruction}

    ### Response:
    {response}
    """

    def format_with_response(
        self, messages: List[Dict[str, str]], response: str
    ) -> str:
        instruction = ""

        for msg in messages:
            if msg.get("role") == "user":
                instruction = msg.get("content", "")
                break

        return f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{response}"

    def format_without_response(self, messages: List[Dict[str, str]]) -> str:
        instruction = ""

        for msg in messages:
            if msg.get("role") == "user":
                instruction = msg.get("content", "")
                break

        return f"### Instruction:\\n{instruction}\\n\\n### Response:\\n"

    def get_eos_token(self) -> str:
        return ""  # Alpaca typically uses empty string or newline


# Completions Template Registry
COMPLETIONS_TEMPLATE_REGISTRY: Dict[str, CompletionsTemplate] = {
    "llama3": Llama3CompletionsTemplate(),
    "llama4": Llama4CompletionsTemplate(),
    "chatml": ChatMLCompletionsTemplate(),
    "alpaca": AlpacaCompletionsTemplate(),
}


# Model pattern to template mapping
MODEL_TEMPLATE_PATTERNS: Dict[str, str] = {
    # Llama models
    "llama-4": "llama4",
    "llama4": "llama4",
    "llama-3": "llama3",
    "llama3": "llama3",
    "llama-2": "llama3",  # Llama 2 uses similar format
    "llama2": "llama3",
    # Other models
    "gpt": "chatml",
    "claude": "llama3",  # Claude often uses Llama-style formatting
    "alpaca": "alpaca",
    "vicuna": "alpaca",  # Vicuna uses Alpaca-style
    # Default fallback
    "default": "llama3",
}


# Provider-specific defaults
PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "fireworks": {
        "llama-4": "llama4",
        "llama-3": "llama3",
        "llama4": "llama4",
        "llama3": "llama3",
        "default": "llama3",
    },
    "together": {"llama": "llama3", "alpaca": "alpaca", "default": "llama3"},
    "openai": {"gpt": "chatml", "default": "chatml"},
    "anthropic": {"claude": "llama3", "default": "llama3"},
}


@dataclass
class CompletionsTemplateConfig:
    """Configuration for completions API templates."""

    # Template format name
    template_format: Optional[str] = None

    # Custom template instance
    custom_template: Optional[CompletionsTemplate] = None

    # Override specific tokens (for future extensions)
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None
    role_start_token: Optional[str] = None
    role_end_token: Optional[str] = None


def get_completions_template_for_model(
    model_name: str,
    provider: Optional[str] = None,
    template_format: Optional[str] = None,
    custom_template: Optional[CompletionsTemplate] = None,
) -> CompletionsTemplate:
    """
    Get the appropriate completions API template for a model.

    This determines which template to use for converting chat messages
    to the continuous string format expected by a model's completions API.

    Args:
        model_name: The model name/path
        provider: The provider name (e.g., 'fireworks', 'together')
        template_format: Explicit template format to use
        custom_template: Custom template instance

    Returns:
        CompletionsTemplate instance
    """
    # 1. Use custom template if provided
    if custom_template:
        logger.debug(f"Using custom template for {model_name}")
        return custom_template

    # 2. Use explicit format if specified
    if template_format:
        if template_format in COMPLETIONS_TEMPLATE_REGISTRY:
            logger.debug(
                f"Using explicit template format '{template_format}' for {model_name}"
            )
            return COMPLETIONS_TEMPLATE_REGISTRY[template_format]
        else:
            raise ValueError(f"Unknown template format: {template_format}")

    # 3. Check provider-specific defaults
    if provider and provider.lower() in PROVIDER_DEFAULTS:
        provider_map = PROVIDER_DEFAULTS[provider.lower()]
        model_lower = model_name.lower()

        # Check provider-specific patterns
        for pattern, format_name in provider_map.items():
            if pattern != "default" and pattern in model_lower:
                logger.debug(f"Using provider default '{format_name}' for {model_name}")
                return COMPLETIONS_TEMPLATE_REGISTRY[format_name]

        # Use provider default
        if "default" in provider_map:
            format_name = provider_map["default"]
            logger.debug(f"Using provider default '{format_name}' for {model_name}")
            return COMPLETIONS_TEMPLATE_REGISTRY[format_name]

    # 4. Auto-detect based on model name patterns
    model_lower = model_name.lower()
    for pattern, format_name in MODEL_TEMPLATE_PATTERNS.items():
        if pattern != "default" and pattern in model_lower:
            logger.debug(f"Auto-detected template '{format_name}' for {model_name}")
            return COMPLETIONS_TEMPLATE_REGISTRY[format_name]

    # 5. Use global default
    default_format = MODEL_TEMPLATE_PATTERNS.get("default", "llama3")
    logger.debug(f"Using default template '{default_format}' for {model_name}")
    return COMPLETIONS_TEMPLATE_REGISTRY[default_format]


def register_completions_template(name: str, template: CompletionsTemplate) -> None:
    """
    Register a custom completions template.

    Args:
        name: Template name
        template: CompletionsTemplate instance
    """
    COMPLETIONS_TEMPLATE_REGISTRY[name] = template
    logger.info(f"Registered completions template '{name}'")


def list_available_completions_templates() -> List[str]:
    """Get list of available completions template names."""
    return list(COMPLETIONS_TEMPLATE_REGISTRY.keys())
