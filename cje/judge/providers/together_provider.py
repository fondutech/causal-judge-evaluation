from typing import Any
from .openai_compat import OpenAICompatibleProvider


class TogetherProvider(OpenAICompatibleProvider):
    """Together AI provider (OpenAI-compatible)."""

    ENV_VAR = "TOGETHER_API_KEY"
    DEFAULT_BASE_URL = "https://api.together.xyz/v1"

    def setup_client(self) -> Any:
        """Setup Together client using OpenAI Python SDK with custom base_url."""
        import openai

        api_key = self.get_api_key()
        base_url = self.base_url or self.DEFAULT_BASE_URL
        return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def score(
        self,
        client: Any,
        prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        chat = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = chat.choices[0].message.content
        if content is None:
            raise ValueError("No response content from Together AI API")
        return str(content)

    def get_default_env_var(self) -> str:
        return self.ENV_VAR
