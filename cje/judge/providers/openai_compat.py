"""Unified OpenAI-compatible provider for both legacy and structured output."""

from typing import Any, Dict, Optional, Type
import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .base import UnifiedProviderStrategy


class UnifiedOpenAICompatibleProvider(UnifiedProviderStrategy):
    """Generic provider for OpenAI-compatible chat endpoints with unified interface.

    Subclasses only need to override ``ENV_VAR``, ``DEFAULT_BASE_URL``, and optionally ``CHAT_CLASS``.
    """

    # Override these in subclasses -------------------------------------------
    ENV_VAR: str = "OPENAI_API_KEY"
    DEFAULT_BASE_URL: Optional[str] = None  # OpenAI uses the official url via SDK
    CHAT_CLASS: Type[BaseChatModel] = (
        ChatOpenAI  # subclasses may override if SDK differs
    )

    # ------------------------------------------------------------------------
    # Legacy API implementation
    # ------------------------------------------------------------------------

    def setup_client(self) -> Any:
        """Return an ``openai.AsyncOpenAI`` configured for this backend."""
        import openai  # local import keeps dependency optional for users not using it

        api_key = self.api_key or os.getenv(self.ENV_VAR)
        if not api_key:
            raise ValueError(
                f"API key not found for {self.__class__.__name__}. Provide via parameter "
                f"or set the {self.ENV_VAR} environment variable."
            )

        base_url = self.base_url or self.DEFAULT_BASE_URL
        if base_url is None:
            # Plain OpenAI – use SDK default endpoint
            return openai.AsyncOpenAI(api_key=api_key)
        return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def score(
        self,
        client: Any,
        prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Single-turn chat completion call."""
        chat = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = chat.choices[0].message.content
        if content is None:
            raise ValueError("No response content from backend API")
        return str(content)

    def get_default_env_var(self) -> str:
        """Get the default environment variable name."""
        return self.ENV_VAR

    # ------------------------------------------------------------------------
    # Structured Output API implementation
    # ------------------------------------------------------------------------

    def get_chat_model(
        self, model_name: str, temperature: float = 0.0
    ) -> BaseChatModel:
        """Get the LangChain chat model for this provider."""
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

    def get_recommended_method(self) -> str:
        """Get the recommended structured output method."""
        return "json_schema"  # OpenAI supports json_schema as the best method
