"""Google provider strategy implementation."""

from typing import Any
from .base import ProviderStrategy


class GoogleProvider(ProviderStrategy):
    """Provider strategy for Google Gemini API."""

    def setup_client(self) -> Any:
        """Setup Google Gemini client."""
        import google.generativeai as genai
        import os

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
