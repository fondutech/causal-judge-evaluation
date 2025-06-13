"""
Shared conversation parsing utilities for both local and API policy runners.

This module provides common functionality for:
- Detecting conversation formats
- Parsing conversations into message arrays
- Converting messages back to text format
- Applying system prompts and user message templates
"""

import re
import json
from typing import List, Dict, Tuple, Optional


def is_conversation(context: str) -> bool:
    """Detect if context contains a conversation format."""
    if not isinstance(context, str):
        return False

    # Check for common conversation patterns
    patterns = [
        r"\b(Human|User|H):\s*",  # "Human: " or "User: " or "H: "
        r"\b(AI|Assistant|A):\s*",  # "AI: " or "Assistant: " or "A: "
        r"<\|(user|human|assistant|ai)\|>",  # "<|user|>" tokens
        r"^\s*\[\s*\{.*\"role\".*\"content\".*\}",  # JSON message format
    ]

    return any(
        re.search(pattern, context, re.IGNORECASE | re.MULTILINE)
        for pattern in patterns
    )


def normalize_role(role: str) -> str:
    """Normalize role names to standard format."""
    role_lower = role.lower().strip()
    if role_lower in ["human", "user", "h", "u"]:
        return "user"
    elif role_lower in ["ai", "assistant", "a"]:
        return "assistant"
    return role_lower


def split_conversation_turns(context: str) -> List[Tuple[str, str]]:
    """Split conversation into (role, content) pairs."""
    turns: List[Tuple[str, str]] = []

    # Handle token format: <|user|>content<|assistant|>content
    token_pattern = (
        r"<\|(user|human|assistant|ai)\|>(.*?)(?=<\|(?:user|human|assistant|ai)\||$)"
    )
    token_matches = re.findall(token_pattern, context, re.IGNORECASE | re.DOTALL)
    if token_matches:
        return [(role, content) for role, content in token_matches]

    # Handle role: content format
    role_pattern = r"\b(Human|User|H|AI|Assistant|A):\s*(.*?)(?=\b(?:Human|User|H|AI|Assistant|A):|$)"
    role_matches = re.findall(role_pattern, context, re.IGNORECASE | re.DOTALL)
    if role_matches:
        return [(role, content) for role, content in role_matches]

    return turns


def parse_conversation(
    context: str, system_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    """Parse conversation text into message array."""
    messages = []

    # Add system prompt if configured
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Try JSON format first
    if context.strip().startswith("[") or context.strip().startswith("{"):
        try:
            parsed = json.loads(context.strip())
            if isinstance(parsed, list):
                for msg in parsed:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        role = normalize_role(msg["role"])
                        if role in ["user", "assistant"]:
                            messages.append(
                                {"role": role, "content": str(msg["content"])}
                            )
                return messages
        except (json.JSONDecodeError, KeyError):
            pass  # Fall through to text parsing

    # Parse text-based conversation formats
    turns = split_conversation_turns(context)
    for role, content in turns:
        normalized_role = normalize_role(role)
        if normalized_role in ["user", "assistant"] and content.strip():
            messages.append({"role": normalized_role, "content": content.strip()})

    return messages


def create_simple_message(
    context: str,
    system_prompt: Optional[str] = None,
    user_message_template: str = "{context}",
) -> List[Dict[str, str]]:
    """Create message array for simple context."""
    messages = []

    # Add system prompt if configured
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Apply user message template
    formatted_content = user_message_template.format(context=context)
    messages.append({"role": "user", "content": formatted_content})

    return messages


def parse_context(
    context: str,
    system_prompt: Optional[str] = None,
    user_message_template: str = "{context}",
) -> List[Dict[str, str]]:
    """Parse context into proper message format."""
    # Auto-detect conversation format
    if is_conversation(context):
        messages = parse_conversation(context, system_prompt)
        # Fallback to simple if parsing failed
        if not messages or all(
            msg.get("role") not in ["user", "assistant"]
            for msg in messages
            if msg.get("role") != "system"
        ):
            return create_simple_message(context, system_prompt, user_message_template)
        return messages
    else:
        return create_simple_message(context, system_prompt, user_message_template)


def messages_to_text(
    messages: List[Dict[str, str]], format_style: str = "standard"
) -> str:
    """Convert message array back to text format for local models.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        format_style: How to format the conversation
            - "standard": Human: ... AI: ... format
            - "simple": Concatenate with system prompt as prefix
            - "chat_template": Use proper chat template format (if model supports it)

    Returns:
        Formatted text suitable for local model input
    """
    if format_style == "simple":
        # Simple concatenation with system prompt as prefix
        text_parts = []
        for msg in messages:
            if msg["role"] == "system":
                text_parts.append(msg["content"])
            elif msg["role"] == "user" or msg["role"] == "assistant":
                text_parts.append(msg["content"])
        return "\n\n".join(text_parts)

    elif format_style == "standard":
        # Standard Human:/AI: format
        text_parts = []
        system_content = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_content = content
            elif role == "user":
                text_parts.append(f"Human: {content}")
            elif role == "assistant":
                text_parts.append(f"AI: {content}")

        conversation_text = "\n".join(text_parts)

        if system_content:
            return f"{system_content}\n\n{conversation_text}"
        return conversation_text

    else:
        raise ValueError(f"Unknown format_style: {format_style}")


def extract_last_user_message(messages: List[Dict[str, str]]) -> Optional[str]:
    """Extract the last user message from a conversation."""
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return None
