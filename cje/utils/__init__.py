"""Utility functions and helper classes."""

from .generation import (
    generate_with_logprobs,
    compute_sequence_logp,
    batch_generate_with_logprobs,
)
from .checkpointing import (
    CheckpointManager,
    BatchProcessor,
    create_jsonl_checkpoint_manager,
    auto_enable_checkpointing,
    cleanup_checkpoint_file,
)

__all__ = [
    "generate_with_logprobs",
    "compute_sequence_logp",
    "batch_generate_with_logprobs",
    "CheckpointManager",
    "BatchProcessor",
    "create_jsonl_checkpoint_manager",
    "auto_enable_checkpointing",
    "cleanup_checkpoint_file",
]

# Optional AWS Secrets Manager utilities
try:
    from .aws_secrets import (
        get_api_key_from_secrets,
        setup_api_keys_from_secrets,
        get_openai_api_key,
        get_anthropic_api_key,
        get_google_api_key,
        SecretsManagerError,
    )

    __all__.extend(
        [
            "get_api_key_from_secrets",
            "setup_api_keys_from_secrets",
            "get_openai_api_key",
            "get_anthropic_api_key",
            "get_google_api_key",
            "SecretsManagerError",
        ]
    )
except ImportError:
    # boto3 not available, skip AWS Secrets Manager utilities
    pass
