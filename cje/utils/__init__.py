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
    compute_teacher_forced_logprob,
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
    "compute_teacher_forced_logprob",
    "safe_sum",
]
