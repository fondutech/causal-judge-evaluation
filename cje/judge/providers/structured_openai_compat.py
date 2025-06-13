from typing import Dict, Any, Type
import os
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI  # default fallback

from .structured_base import StructuredProviderStrategy


class StructuredOpenAICompatibleProvider(StructuredProviderStrategy):
    """Generic structured-output provider for OpenAI-compatible chat backends.

    Sub-classes specify *ENV_VAR* and *DEFAULT_BASE_URL* and otherwise inherit
    all behaviour.
    """

    ENV_VAR: str = "OPENAI_API_KEY"
    DEFAULT_BASE_URL: str | None = None  # sdk default for real OpenAI
    CHAT_CLASS: Type[BaseChatModel] = (
        ChatOpenAI  # subclasses may override if SDK differs
    )

    def get_chat_model(
        self, model_name: str, temperature: float = 0.0
    ) -> BaseChatModel:
        kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "temperature": temperature,
        }

        api_key = self.api_key or os.getenv(self.ENV_VAR)
        if api_key:
            kwargs["api_key"] = api_key

        base_url = self.base_url or self.DEFAULT_BASE_URL
        if base_url:
            kwargs["base_url"] = base_url

        return self.CHAT_CLASS(**kwargs)

    def get_recommended_method(self) -> str:  # same as OpenAI: json_schema best
        return "json_schema"
