from __future__ import annotations

from typing import List, Optional, Union, Sequence

__all__ = [
    "safe_sum",
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
