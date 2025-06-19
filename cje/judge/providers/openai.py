"""Unified OpenAI provider implementation."""

from .openai_compat import UnifiedOpenAICompatibleProvider


class OpenAIProvider(UnifiedOpenAICompatibleProvider):
    """OpenAI provider with unified interface."""

    ENV_VAR = "OPENAI_API_KEY"
    DEFAULT_BASE_URL = None
