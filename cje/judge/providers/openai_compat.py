from typing import Any, Optional
import os
from .base import ProviderStrategy


class OpenAICompatibleProvider(ProviderStrategy):
    """Generic provider for OpenAI-compatible chat endpoints.

    Subclasses only need to override ``ENV_VAR`` and ``DEFAULT_BASE_URL``.
    """

    # Override these in subclasses -------------------------------------------
    ENV_VAR: str = "OPENAI_API_KEY"
    DEFAULT_BASE_URL: Optional[str] = None  # OpenAI uses the official url via SDK

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

    # Re-expose env-var getter via base helper
    def get_default_env_var(self) -> str:  # noqa: D401
        return self.ENV_VAR
