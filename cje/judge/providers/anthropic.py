"""Unified Anthropic provider implementation."""

import os
from typing import Any, Dict
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from .base import UnifiedProviderStrategy


class AnthropicProvider(UnifiedProviderStrategy):
    """Anthropic provider with unified interface."""

    # ------------------------------------------------------------------------
    # Legacy API implementation
    # ------------------------------------------------------------------------

    def setup_client(self) -> Any:
        """Setup Anthropic client."""
        import anthropic

        api_key = self.get_api_key()
        return anthropic.AsyncAnthropic(api_key=api_key)

    async def score(
        self,
        client: Any,
        prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Score using Anthropic API."""
        message = await client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        content = message.content[0].text
        if content is None:
            raise ValueError("No response content from Anthropic API")

        return str(content)

    def get_default_env_var(self) -> str:
        """Get default environment variable for Anthropic API key."""
        return "ANTHROPIC_API_KEY"

    # ------------------------------------------------------------------------
    # Structured Output API implementation
    # ------------------------------------------------------------------------

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
