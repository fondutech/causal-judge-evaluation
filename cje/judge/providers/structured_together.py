from langchain_together import ChatTogether

from .structured_openai_compat import StructuredOpenAICompatibleProvider


class StructuredTogetherProvider(StructuredOpenAICompatibleProvider):
    ENV_VAR = "TOGETHER_API_KEY"
    DEFAULT_BASE_URL = "https://api.together.xyz/v1"
    CHAT_CLASS = ChatTogether

    # inherits recommended method from base
