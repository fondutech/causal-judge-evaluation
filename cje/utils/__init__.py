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
from .teacher_forcing import (
    RobustTeacherForcing,
    compute_log_prob,
)
from .llama_cpp_teacher_forcing import (
    LlamaCppTeacherForcing,
)
from .llama_cpp_teacher_forcing_fixed import (
    LlamaCppTeacherForcingFixed,
)
from .llama_chat_templates import (
    format_llama3_instruct,
    format_llama3_for_teacher_forcing,
)
from .logprobs import safe_sum

__all__ = [
    "generate_with_logprobs",
    "compute_sequence_logp",
    "batch_generate_with_logprobs",
    "CheckpointManager",
    "BatchProcessor",
    "create_jsonl_checkpoint_manager",
    "auto_enable_checkpointing",
    "cleanup_checkpoint_file",
    "RobustTeacherForcing",
    "compute_log_prob",
    "LlamaCppTeacherForcing",
    "LlamaCppTeacherForcingFixed",
    "format_llama3_instruct",
    "format_llama3_for_teacher_forcing",
    "safe_sum",
]
