"""Teacher forcing utilities for computing log probabilities.

This module provides:
- Fireworks API integration for teacher forcing
- Chat to completions format conversion
- Support for various model templates
"""

# API implementations
from .api import (
    RobustTeacherForcing,
    compute_teacher_forced_logprob,
    compute_total_logprob,
)

# Template configurations
from .templates import (
    ChatTemplateConfig,
    Llama3TemplateConfig,
    Llama4TemplateConfig,
    HuggingFaceTemplateConfig,
)

# Chat utilities
from .chat import (
    ChatToCompletionsConverter,
    convert_chat_for_teacher_forcing,
    ChatTeacherForcing,
    compute_chat_logprob,
)

__all__ = [
    # Fireworks teacher forcing
    "RobustTeacherForcing",
    "compute_teacher_forced_logprob",
    "compute_total_logprob",
    # Template configurations
    "ChatTemplateConfig",
    "Llama3TemplateConfig",
    "Llama4TemplateConfig",
    "HuggingFaceTemplateConfig",
    # Chat conversion
    "ChatToCompletionsConverter",
    "convert_chat_for_teacher_forcing",
    # Integrated chat support
    "ChatTeacherForcing",
    "compute_chat_logprob",
]
