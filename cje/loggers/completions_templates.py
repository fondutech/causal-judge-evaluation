"""
Completions API template system for converting chat conversations to continuous strings.

This module provides templates for converting structured chat messages into the
continuous string format required by completions API endpoints for teacher forcing.

IMPORTANT: Users must explicitly specify the correct template format for their model.
There is no auto-detection. Using the wrong template will result in incorrect
log probabilities.

Available formats:
- 'llama3': For Llama 3.x models (3.0, 3.1, 3.2, 3.3)
- 'llama4': For Llama 4 models (Scout, Maverick, etc.)

Example:
    from cje.loggers.api_policy import APIPolicyRunner

    # You MUST specify the correct format
    runner = APIPolicyRunner(
        provider="fireworks",
        model_name="llama-v3p3-70b-instruct",
        completions_template_format="llama3"  # Required!
    )
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

    Format:
    <|begin_of_text|>
    <|start_header_id|>user<|end_header_id|>

    {user_message}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>

    {response}<|eot_id|>
    """

    def format_with_response(
        self, messages: List[Dict[str, str]], response: str
    ) -> str:
        parts = ["<|begin_of_text|>"]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n" f"{content}<|eot_id|>"
            )

        parts.append(
            f"<|start_header_id|>assistant<|end_header_id|>\n\n" f"{response}<|eot_id|>"
        )

        return "".join(parts)

    def format_without_response(self, messages: List[Dict[str, str]]) -> str:
        parts = ["<|begin_of_text|>"]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n" f"{content}<|eot_id|>"
            )

        parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n")

        return "".join(parts)

    def get_eos_token(self) -> str:
        return "<|eot_id|>"


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


# Completions Template Registry
COMPLETIONS_TEMPLATE_REGISTRY: Dict[str, CompletionsTemplate] = {
    "llama3": Llama3CompletionsTemplate(),
    "llama4": Llama4CompletionsTemplate(),
}


@dataclass
class CompletionsTemplateConfig:
    """Configuration for completions API templates.

    This is primarily for advanced users who need custom template implementations.
    Most users should simply specify completions_template_format='llama3' or 'llama4'.
    """

    # Custom template instance
    custom_template: Optional[CompletionsTemplate] = None


def get_completions_template(
    template_format: str,
    custom_template: Optional[CompletionsTemplate] = None,
) -> CompletionsTemplate:
    """
    Get a completions API template by format name.

    Args:
        template_format: Template format name ('llama3' or 'llama4')
        custom_template: Optional custom template instance (overrides template_format)

    Returns:
        CompletionsTemplate instance

    Raises:
        ValueError: If template_format is not recognized

    Example:
        >>> template = get_completions_template('llama3')
        >>> template = get_completions_template('llama4')
    """
    # Use custom template if provided
    if custom_template:
        logger.debug("Using custom template")
        return custom_template

    # Look up template by format name
    if template_format in COMPLETIONS_TEMPLATE_REGISTRY:
        logger.debug(f"Using template format '{template_format}'")
        return COMPLETIONS_TEMPLATE_REGISTRY[template_format]
    else:
        available = list(COMPLETIONS_TEMPLATE_REGISTRY.keys())
        raise ValueError(
            f"Unknown template format: '{template_format}'. "
            f"Available formats: {available}"
        )


def register_completions_template(name: str, template: CompletionsTemplate) -> None:
    """
    Register a custom completions template.

    Args:
        name: Template name (must not conflict with built-in names)
        template: CompletionsTemplate instance

    Example:
        >>> class MyTemplate(CompletionsTemplate):
        ...     # implement required methods
        >>> register_completions_template("myformat", MyTemplate())
        >>> runner = APIPolicyRunner(..., completions_template_format="myformat")
    """
    if name in COMPLETIONS_TEMPLATE_REGISTRY:
        logger.warning(f"Overwriting existing template '{name}'")
    COMPLETIONS_TEMPLATE_REGISTRY[name] = template
    logger.info(f"Registered completions template '{name}'")


def list_available_completions_templates() -> List[str]:
    """Get list of available completions template names."""
    return list(COMPLETIONS_TEMPLATE_REGISTRY.keys())
