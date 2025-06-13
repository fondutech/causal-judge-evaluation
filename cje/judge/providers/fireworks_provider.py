from .openai_compat import OpenAICompatibleProvider


class FireworksProvider(OpenAICompatibleProvider):
    """Fireworks.ai provider (OpenAI-compatible)."""

    ENV_VAR = "FIREWORKS_API_KEY"
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
