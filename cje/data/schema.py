"""Core data schemas for CJE."""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List


@dataclass
class CJESample:
    """A single sample for counterfactual judge evaluation.

    Attributes:
        uid: Unique identifier for the sample
        context: Input context/prompt
        response: Model response/completion
        y_true: Ground truth label/value (if available)
        logp: Log probability of the response (if available)
        meta: Additional metadata about the sample
    """

    uid: str
    context: str
    response: str
    y_true: Optional[Any]
    logp: Optional[float]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CJEStep:
    """A single step in an agent trajectory for CJE.

    This generalises `CJESample` to multi-turn agents. Each step records the observation
    available to the agent (``state``), the action chosen, and the log-probability of
    that action under the *logging* policy π₀.  A step-level reward can be supplied
    (e.g. tool-use success); otherwise leave ``reward=None`` and rely on the
    trajectory-level ``y_true``.
    """

    state: Any
    action: Any
    logp: float
    reward: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CJETrajectory:
    """A full interaction trajectory.

    Attributes
    ----------
    uid
        Unique identifier (conversation id, episode id, …).
    steps
        Ordered list of `CJEStep` objects.  A single-turn prompt/response reduces to
        one step; legacy pipelines can create a single-step trajectory to remain
        backward-compatible.
    y_true
        Terminal business KPI (booking completed, accepted answer, etc.). If the KPI
        is instead logged per-step, leave this as ``None`` and rely on each step's
        ``reward``.
    meta
        Free-form metadata for the whole trajectory (user cohort, locale, etc.).
    """

    uid: str
    steps: List[CJEStep]
    y_true: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)
