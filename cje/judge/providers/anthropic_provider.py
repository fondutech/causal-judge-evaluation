"""Anthropic provider strategy implementation."""

from typing import Any
from .base import ProviderStrategy


class AnthropicProvider(ProviderStrategy):
    """Provider strategy for Anthropic API."""

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
