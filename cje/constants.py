from .provider_registry import registry
from cje.judge.providers.openai_compat import UnifiedOpenAICompatibleProvider

_OPENAI_LIKE = {
    name
    for name, info in registry().items()
    if issubclass(info.provider_cls, UnifiedOpenAICompatibleProvider)
}

OPENAI_COMPATIBLE_PROVIDERS = _OPENAI_LIKE
ALL_PROVIDERS = set(registry()) | {"hf", "mock"}
