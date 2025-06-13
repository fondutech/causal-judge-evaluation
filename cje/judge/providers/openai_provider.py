"""OpenAI provider strategy implementation."""

from .openai_compat import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    ENV_VAR = "OPENAI_API_KEY"
    DEFAULT_BASE_URL = None
