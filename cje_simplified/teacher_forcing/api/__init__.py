"""API implementations for teacher forcing."""

from .fireworks import (
    RobustTeacherForcing,
    compute_teacher_forced_logprob,
    compute_total_logprob,
)

__all__ = [
    "RobustTeacherForcing",
    "compute_teacher_forced_logprob",
    "compute_total_logprob",
]
