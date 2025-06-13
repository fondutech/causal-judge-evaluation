"""OpenAI provider with structured output support."""

from langchain_openai import ChatOpenAI

from .structured_openai_compat import StructuredOpenAICompatibleProvider


class StructuredOpenAIProvider(StructuredOpenAICompatibleProvider):
    ENV_VAR = "OPENAI_API_KEY"
    DEFAULT_BASE_URL = None
    CHAT_CLASS = ChatOpenAI
