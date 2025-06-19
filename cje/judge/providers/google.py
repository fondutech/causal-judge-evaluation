"""Unified Google provider implementation."""

import os
from typing import Any, Dict, cast
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from .base import UnifiedProviderStrategy


class GoogleProvider(UnifiedProviderStrategy):
    """Google provider with unified interface."""

    # ------------------------------------------------------------------------
    # Legacy API implementation
    # ------------------------------------------------------------------------

    def setup_client(self) -> Any:
        """Setup Google Gemini client."""
        import google.generativeai as genai

        api_key = self.get_api_key()
        genai.configure(api_key=api_key)

        # Return the genai module itself as the "client"
        return genai

    async def score(
        self,
        client: Any,
        prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Score using Google Gemini API."""
        # Create model instance
        model = client.GenerativeModel(model_name)

        # Generate response
        response = await model.generate_content_async(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        if not response.text:
            raise ValueError("No response content from Google Gemini API")

        return str(response.text)

    def get_default_env_var(self) -> str:
        """Get default environment variable for Google API key."""
        return "GOOGLE_API_KEY"

    # ------------------------------------------------------------------------
    # Structured Output API implementation
    # ------------------------------------------------------------------------

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
