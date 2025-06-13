from __future__ import annotations

from typing import List, Optional, Union, Sequence

__all__ = [
    "safe_sum",
    "sum_response_logprobs_tail",
]


def safe_sum(values: Sequence[Optional[float]]) -> float:
    """Sum a list of (possibly None) floats.

    Args:
        values: Sequence containing floats or ``None`` values as returned by some
            provider SDKs for *unavailable* token log-probs.

    Returns
    -------
    float
        The sum of the *finite* log-probabilities. ``None`` values are skipped
        and treated as 0.0, mirroring the existing behaviour in runners.
    """
    return float(sum(v for v in values if v is not None))


def sum_response_logprobs_tail(
    all_token_logprobs: List[Optional[float]],
    response_token_count: int,
) -> float:
    """Return the sum of log-probs belonging to the **response** only.

    Many hosted APIs return a single flat list of token log-probs for the full
    sequence ``prompt + response`` when we set ``echo=True``.  The standard
    convention in CJE is to assume the *last* ``response_token_count`` tokens
    belong to the assistant response and to ignore everything before that.

    This helper encapsulates that slicing logic so that every policy runner
    uses the exact same rule.

    Parameters
    ----------
    all_token_logprobs : List[Optional[float]]
        The provider-returned log-probs for the *entire* echoed sequence.
    response_token_count : int
        Number of tokens in the response part.

    Returns
    -------
    float
        The summed log-probability of the response tokens (``0.0`` if the slice
        is empty or the input list is shorter than expected).

    Raises
    ------
    RuntimeError
        If all_token_logprobs is empty but response_token_count > 0, indicating
        a provider dropped log-probs (e.g., on non-ASCII tokens). This ensures
        ESS guard-rails can detect the issue instead of silently using weight=1.
    """
    if response_token_count <= 0:
        return 0.0

    # 🔧 P4 FIX: Guard against empty token-logprob arrays
    # Some providers drop log-probs on non-ASCII tokens; catch this early
    if not all_token_logprobs and response_token_count > 0:
        raise RuntimeError(
            f"Empty token logprobs array but response_token_count={response_token_count}. "
            f"This likely indicates the provider dropped log-probs (e.g., non-ASCII tokens). "
            f"Cannot compute reliable importance weights."
        )

    # Guard in case provider returns fewer tokens than promised.
    if len(all_token_logprobs) < response_token_count:
        slice_ = all_token_logprobs
    else:
        slice_ = all_token_logprobs[-response_token_count:]

    return safe_sum(slice_)
