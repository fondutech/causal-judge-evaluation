"""
Chat template utilities for different model families.

Provides proper formatting for teacher forcing with various models.
"""

from typing import Optional, List, Dict, Tuple


def format_llama3_instruct(
    prompt: str,
    response: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Format a prompt using Llama-3 Instruct chat template.

    This follows the official template:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_message}<|end_header_id|>
    <|start_header_id|>user<|end_header_id|>
    {user_message}<|end_header_id|>
    <|start_header_id|>assistant<|end_header_id|>
    {assistant_response}

    Args:
        prompt: User message
        response: Optional assistant response for teacher forcing
        system_prompt: Optional system message

    Returns:
        Properly formatted string for Llama-3 models
    """
    # Start with beginning of text
    formatted = "<|begin_of_text|>"

    # Add system message if provided
    if system_prompt:
        formatted += (
            f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        )

    # Add user message
    formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"

    # Add assistant header
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    # Add response if provided (for teacher forcing)
    if response is not None:
        formatted += response

    return formatted


def format_llama3_for_teacher_forcing(
    prompt: str,
    response: str,
    system_prompt: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Format prompt and full text for Llama-3 teacher forcing.

    Returns:
        (prompt_only, prompt_with_response) tuple
    """
    # Format prompt up to assistant header
    prompt_formatted = format_llama3_instruct(prompt, None, system_prompt)

    # Format full text including response
    full_formatted = format_llama3_instruct(prompt, response, system_prompt)

    return prompt_formatted, full_formatted


def get_llama3_special_tokens() -> Dict[str, int]:
    """Get Llama-3 special token IDs."""
    return {
        "<|begin_of_text|>": 128000,
        "<|start_header_id|>": 128001,
        "<|end_header_id|>": 128002,
        "<|eot_id|>": 128003,
        "<|end_of_text|>": 128001,  # Same as start_header_id in some versions
    }
