"""Sampler for multi-policy agent trajectories.

This mirrors `MultiTargetSampler` but operates on lists of `CJEStep` in a
`CJETrajectory`.  For now we implement only the minimal subset required by the
upcoming DRCPOMDPEstimator: computing per-trajectory importance weights and
optionally step-wise weight arrays for variance diagnostics.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Any, Dict, Protocol, Union, TYPE_CHECKING

import numpy as np

from ..data.schema import CJEStep, CJETrajectory
from ..utils.error_handling import safe_call, FALLBACK_LOG_PROB, PolicyError

if TYPE_CHECKING:
    from .policy import PolicyRunner  # pragma: no cover
    from .api_policy import APIPolicyRunner


class PolicyRunnerProtocol(Protocol):
    """Subset of PolicyRunner used for agent trajectories (token/tool level).
    For now we only rely on `log_prob` which takes a *state* and *action* and
    returns the log-probability under π.
    """

    def log_prob(
        self, state: Any, action: Any, **kwargs: Any
    ) -> float:  # noqa: D401 (simple description)
        ...


class MultiTargetTrajectorySampler:  # pylint: disable=too-few-public-methods
    """Compute importance weights for K target policies over a trajectory.

    Parameters
    ----------
    runners : list of policy runner objects implementing `log_prob(state, action)`.
    clip : float, optional
        Importance-weight clip value (applied **after** multiplying all steps), by default 20.0.
    """

    def __init__(
        self,
        runners: Sequence[
            Union["PolicyRunner", "APIPolicyRunner", PolicyRunnerProtocol]
        ],
        clip: float = 20.0,
    ) -> None:
        if not runners:
            raise ValueError("At least one policy runner must be provided")
        self.runners = list(runners)
        self.K = len(runners)
        self.clip = clip

    # ---------------------------------------------------------------------
    # Low-level helpers
    # ---------------------------------------------------------------------
    def _logp_many(self, state: Any, action: Any) -> List[float]:
        """Return log πᵏ(action | state) for every runner."""
        logps: List[float] = []
        for idx, runner in enumerate(self.runners):
            logp_result = safe_call(
                runner.log_prob,  # type: ignore[arg-type]
                state,
                action,
                error_context=f"Computing log_prob for runner {idx}",
                fallback=FALLBACK_LOG_PROB,
            )
            # Handle the case where log_prob returns a tuple (logp, token_logps)
            if isinstance(logp_result, tuple):
                logp = logp_result[0]
            else:
                logp = logp_result
            logps.append(float(logp or FALLBACK_LOG_PROB))
        return logps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def importance_weights(self, traj: CJETrajectory) -> np.ndarray:  # type: ignore[name-defined]
        """Return importance weights vector (K,) for a trajectory.

        For each target policy πᵏ we compute
        wᵏ = ∏_t  πᵏ(a_t | s_t) / π₀(a_t | s_t).

        The behaviour log-prob is taken from each step.logp.
        """
        if not traj.steps:
            raise ValueError("Trajectory has no steps")

        # Accumulate log-weights to avoid under/overflow
        logw = np.zeros(self.K, dtype=np.float64)
        for step in traj.steps:
            target_logps = self._logp_many(step.state, step.action)
            beh_logp = step.logp
            logw += np.asarray(target_logps) - beh_logp

        weights = np.exp(logw)
        if self.clip:
            weights = np.clip(weights, 0, self.clip)
        return weights  # shape: (K,)

    def sample_many(self, state: Any, n: int = 1) -> List[List[str]]:
        """Sample *n* actions from each target policy given the *state*.

        This enables Monte-Carlo computation of μ_π(x) in DR estimators.
        The method reuses the generate_with_logp/generate fallbacks when
        available.  For non-generative policies (e.g., deterministic tool
        calls) an empty list is returned.
        """
        all_samples: List[List[str]] = []
        for idx, runner in enumerate(self.runners):
            samples: List[str] = []
            if hasattr(runner, "generate_with_logp"):
                res = safe_call(
                    runner.generate_with_logp,
                    [state] * n,
                    error_context=f"Sampling actions for runner {idx}",
                    fallback=None,
                )
                if res is not None:
                    samples = [str(t[0]) for t in res]
            if not samples and hasattr(runner, "generate"):
                res = safe_call(
                    runner.generate,
                    [state] * n,
                    error_context=f"Sampling actions (generate) for runner {idx}",
                    fallback=None,
                )
                if res is not None:
                    samples = [str(r) for r in res]
            all_samples.append(samples)
        return all_samples


# Convenience factory mirroring `make_multi_sampler`


def make_multi_trajectory_sampler(
    target_policies_cfg: Sequence[Dict[str, Any]],
    clip: float = 20.0,
) -> MultiTargetTrajectorySampler:
    """Create a trajectory sampler from list of target-policy configs (dicts)."""

    from .multi_target_sampler import make_multi_sampler

    base_sampler = make_multi_sampler(target_policies_cfg)
    return MultiTargetTrajectorySampler(base_sampler.runners, clip=clip)  # type: ignore[arg-type]
