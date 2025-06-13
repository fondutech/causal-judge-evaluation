"""Google provider with structured output support."""

import os
from typing import Dict, Any, cast
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from .structured_base import StructuredProviderStrategy


class StructuredGoogleProvider(StructuredProviderStrategy):
    """Google provider with structured output support."""

    def get_chat_model(
        self, model_name: str, temperature: float = 0.0
    ) -> BaseChatModel:
        """Get the LangChain ChatGoogleGenerativeAI model."""
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
        }

        # Add API key if provided
        api_key = self.api_key or os.getenv("GOOGLE_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key

        return cast(BaseChatModel, ChatGoogleGenerativeAI(**kwargs))

    def get_recommended_method(self) -> str:
        """Google works best with json_mode method."""
        return "json_mode"

    def get_structured_output_params(self, method: str) -> Dict[str, Any]:
        """No special parameters needed for Google."""
        return {}
