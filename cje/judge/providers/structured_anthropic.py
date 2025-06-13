"""Anthropic provider with structured output support."""

import os
from typing import Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from .structured_base import StructuredProviderStrategy


class StructuredAnthropicProvider(StructuredProviderStrategy):
    """Anthropic provider with structured output support."""

    def get_chat_model(
        self, model_name: str, temperature: float = 0.0
    ) -> BaseChatModel:
        """Get the LangChain ChatAnthropic model."""
        kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "temperature": temperature,
        }

        # Add API key if provided
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key

        # Add base URL if provided
        if self.base_url:
            kwargs["base_url"] = self.base_url

        return ChatAnthropic(**kwargs)

    def get_recommended_method(self) -> str:
        """Anthropic works best with function_calling method."""
        return "function_calling"

    def get_structured_output_params(self, method: str) -> Dict[str, Any]:
        """No special parameters needed for Anthropic."""
        return {}
