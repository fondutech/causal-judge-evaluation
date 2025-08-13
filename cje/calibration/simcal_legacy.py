"""Score-Indexed Monotone Calibration (SIMCal) for importance weights.

This module implements SIMCal, which projects weights onto monotone curves
indexed by a score (e.g., judge score). It supports two direction-selection
strategies:
  • L2: choose the monotone direction that minimizes L2 distance to raw weights
  • IF-based: choose the direction that minimizes the empirical variance of an
    influence function (IPS or DR-correction), aligning selection with
    downstream estimator risk.

After projection, we blend toward uniform to satisfy ESS/variance caps.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class SimcalConfig:
    """Configuration for SIMCal calibration.

    SIMCal projects weights onto monotone curves indexed by judge scores,
    automatically selecting the direction (increasing/decreasing) that minimizes
    L2 distance to the original weights. When L2 distances tie, the tie_break
    parameter determines whether to prefer the direction with higher ESS or
    lower variance.

    Both ess_floor and var_cap constraints are enforced via a single convex
    blend toward uniform weights: w ↦ 1 + (1-γ)(w-1), where γ ∈ [0,1] is
    chosen to satisfy the tightest constraint.

    Args:
        ess_floor: Minimum effective sample size as fraction of n (e.g., 0.2 => ESS >= 0.2 * n)
                  Note: This implies var_cap <= 1/ess_floor - 1
        var_cap: Maximum allowed variance of calibrated weights
                Warning: If tighter than ESS-implied cap, the ESS constraint takes precedence
        epsilon: Small constant for numerical stability
        direction: "auto" (choose by L2), "increasing", or "decreasing"
        tie_break: How to break ties when L2 distances are equal ("ess" or "var")
                  "ess": prefer direction with higher effective sample size
                  "var": prefer direction with lower variance
        select_direction_by: How to pick the monotone direction when direction="auto":
                  "l2": (default) minimize L2 distance to the raw weights
                  "ips_if": minimize empirical var of IPS IF (needs rewards)
                  "dr_if": minimize empirical var of DR IF correction (needs residuals = R - g_oof(S))
    """

    ess_floor: Optional[float] = None
    var_cap: Optional[float] = None
    epsilon: float = 1e-9
    direction: str = "auto"
    tie_break: str = "ess"
    select_direction_by: str = "l2"

    def __post_init__(self) -> None:
        if self.direction not in {"auto", "increasing", "decreasing"}:
            raise ValueError(
                f"direction must be 'auto', 'increasing', or 'decreasing', got {self.direction}"
            )
        if self.tie_break not in {"ess", "var"}:
            raise ValueError(f"tie_break must be 'ess' or 'var', got {self.tie_break}")
        if self.ess_floor is not None and not (0 < self.ess_floor <= 1):
            raise ValueError(f"ess_floor must be in (0, 1], got {self.ess_floor}")
        if self.var_cap is not None and self.var_cap <= 0:
            raise ValueError(f"var_cap must be positive, got {self.var_cap}")
        if self.select_direction_by not in {"l2", "ips_if", "dr_if"}:
            raise ValueError(
                f"select_direction_by must be one of 'l2','ips_if','dr_if', got {self.select_direction_by}"
            )

        # Validate consistency between ess_floor and var_cap
        if self.ess_floor is not None and self.var_cap is not None:
            # ESS = n/(1 + Var) implies Var <= 1/ess_floor - 1
            implied_var_cap = (1.0 / self.ess_floor) - 1.0
            if self.var_cap > implied_var_cap:
                import warnings

                warnings.warn(
                    f"var_cap={self.var_cap:.3f} is looser than ESS-implied cap "
                    f"{implied_var_cap:.3f} from ess_floor={self.ess_floor}. "
                    f"The ESS constraint will dominate.",
                    UserWarning,
                )


class SIMCalibrator:
    """Score-Indexed Monotone Calibrator for importance weights.

    Takes raw mean-one weights and a score index (e.g., judge scores),
    projects onto monotone curves, chooses the closer one in L2,
    then blends toward uniform to meet variance/ESS constraints.
    """

    def __init__(self, config: SimcalConfig):
        """Initialize SIMCalibrator with configuration.

        Args:
            config: SimcalConfig with calibration parameters
        """
        self.cfg = config

    @staticmethod
    def implied_var_cap(ess_floor: float) -> float:
        """Compute the implied variance cap from an ESS floor constraint.

        Since ESS = n/(1 + Var), requiring ESS >= ess_floor * n
        implies Var <= 1/ess_floor - 1.

        Args:
            ess_floor: Minimum ESS as fraction of n (must be in (0, 1])

        Returns:
            Maximum allowed variance to satisfy the ESS constraint
        """
        if not (0 < ess_floor <= 1):
            raise ValueError(f"ess_floor must be in (0, 1], got {ess_floor}")
        return (1.0 / ess_floor) - 1.0

    def transform(
        self,
        w: np.ndarray,
        s: np.ndarray,
        *,
        rewards: Optional[np.ndarray] = None,
        residuals: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calibrate weights using score-indexed monotone projection.

        Algorithm:
        1. Project weights onto both increasing and decreasing monotone curves
           indexed by the score s (using isotonic regression)
        2. Select the direction with smaller L2 distance to original weights
           (or use tie_break criterion if distances are equal)
        3. Blend the projected weights toward uniform to satisfy constraints:
           w_cal = 1 + (1-γ)(w_proj - 1), where γ ∈ [0,1]
        4. Choose γ to satisfy the tightest constraint (ESS floor or variance cap)

        The blending preserves the mean-one property since both w_proj and
        uniform weights have mean 1.

        Args:
            w: Raw importance weights (must be positive, will be normalized to mean 1)
            s: Score index (e.g., judge scores) for ordering
            rewards: Required if select_direction_by == "ips_if" (IPS IF risk)
            residuals: Required if select_direction_by == "dr_if"
                       (cross-fitted residuals R - g_oof(S))

        Returns:
            Tuple of (calibrated_weights, info_dict) where info_dict contains:
                - direction: chosen monotone direction ("increasing" or "decreasing")
                - gamma: blending parameter (0=no blend, 1=uniform)
                - var_before: variance of input weights
                - var_after_proj: variance after projection
                - var_after_blend: variance after blending (final)
                - ess_before: ESS of input weights
                - ess_after_proj: ESS after projection
                - ess_after_blend: ESS after blending (final)
                - l2_distance_increasing: L2 distance for increasing projection (if auto)
                - l2_distance_decreasing: L2 distance for decreasing projection (if auto)
                - if_var_increasing / if_var_decreasing: IF variances if IF-based selection
                - selection_method: "l2", "ips_if" or "dr_if"

        Raises:
            ValueError: If weights contain non-positive, NaN, or infinite values
        """
        # Input validation
        w = np.asarray(w, dtype=float)
        s = np.asarray(s, dtype=float)

        if len(w) != len(s):
            raise ValueError(f"Length mismatch: weights={len(w)}, scores={len(s)}")

        if not np.all(np.isfinite(w)) or not np.all(np.isfinite(s)):
            raise ValueError("SIMCal: NaNs or infinities in inputs")

        if np.any(w <= 0):
            raise ValueError("SIMCal: weights must be positive")

        # Ensure mean-one normalization
        w = w / w.mean()

        def _isotonic_projection(increasing: bool) -> np.ndarray:
            """Project weights onto monotone curve."""
            reg = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
            # Fit isotonic regression: s -> w
            z = reg.fit(s, w).predict(s)
            # Ensure positivity and mean-one
            z = np.maximum(z, self.cfg.epsilon)
            z = z / z.mean()
            return np.asarray(z)

        # Compute candidate projections
        candidates = {}
        if self.cfg.direction == "auto":
            dirs = ["increasing", "decreasing"]
        else:
            dirs = [self.cfg.direction]

        for d in dirs:
            candidates[d] = _isotonic_projection(increasing=(d == "increasing"))

        # Helper: blend toward uniform to satisfy constraints and compute stats
        def _blend_and_stats(z: np.ndarray) -> Dict[str, Any]:
            v_proj = float(np.var(z))
            gamma = 0.0
            if v_proj > 0:
                if self.cfg.ess_floor is not None:
                    v_max_ess = (1.0 / self.cfg.ess_floor) - 1.0
                    if v_proj > v_max_ess:
                        gamma = max(gamma, 1.0 - np.sqrt(v_max_ess / v_proj))
                if self.cfg.var_cap is not None and v_proj > self.cfg.var_cap:
                    gamma = max(gamma, 1.0 - np.sqrt(self.cfg.var_cap / v_proj))
            gamma = float(np.clip(gamma, 0.0, 1.0))
            w_blend = 1.0 + (1.0 - gamma) * (z - 1.0)
            w_blend = np.maximum(w_blend, self.cfg.epsilon)
            w_blend = w_blend / w_blend.mean()
            v_final = float(np.var(w_blend))
            ess_proj = len(z) / (1.0 + v_proj)
            ess_final = len(z) / (1.0 + v_final)
            return dict(
                v_proj=v_proj,
                gamma=gamma,
                w_final=w_blend,
                v_final=v_final,
                ess_proj=ess_proj,
                ess_final=ess_final,
            )

        # Precompute per-direction stats after blending, to allow either selector
        per_dir: Dict[str, Dict[str, Any]] = {}
        for d, z in candidates.items():
            per_dir[d] = _blend_and_stats(z)
            per_dir[d]["sse"] = float(np.sum((z - w) ** 2))

        # Choose direction
        selection_method = self.cfg.select_direction_by
        if len(candidates) == 1:
            chosen = next(iter(candidates))
        else:
            if selection_method == "l2":
                sse_inc = per_dir["increasing"]["sse"]
                sse_dec = per_dir["decreasing"]["sse"]
                if abs(sse_inc - sse_dec) <= 1e-12:
                    # Tie-break with post-blend ESS/Var
                    if self.cfg.tie_break == "ess":
                        chosen = (
                            "increasing"
                            if per_dir["increasing"]["ess_final"]
                            >= per_dir["decreasing"]["ess_final"]
                            else "decreasing"
                        )
                    else:  # "var"
                        chosen = (
                            "increasing"
                            if per_dir["increasing"]["v_final"]
                            <= per_dir["decreasing"]["v_final"]
                            else "decreasing"
                        )
                else:
                    chosen = "increasing" if sse_inc <= sse_dec else "decreasing"
            else:
                # IF-based selection: need rewards/residuals
                if selection_method == "ips_if":
                    if rewards is None:
                        raise ValueError(
                            "SIMCal(select_direction_by='ips_if') requires rewards."
                        )
                    a_inc = per_dir["increasing"]["w_final"] * rewards
                    a_dec = per_dir["decreasing"]["w_final"] * rewards
                    if_var_inc = float(np.var(a_inc - a_inc.mean()))
                    if_var_dec = float(np.var(a_dec - a_dec.mean()))
                elif selection_method == "dr_if":
                    if residuals is None:
                        raise ValueError(
                            "SIMCal(select_direction_by='dr_if') requires residuals = rewards - g_oof(scores)."
                        )
                    b_inc = per_dir["increasing"]["w_final"] * residuals
                    b_dec = per_dir["decreasing"]["w_final"] * residuals
                    # Centered variance of the correction term
                    if_var_inc = float(np.var(b_inc - b_inc.mean()))
                    if_var_dec = float(np.var(b_dec - b_dec.mean()))
                else:
                    raise AssertionError("Unknown selection method")  # defensive

                per_dir["increasing"]["if_var"] = if_var_inc
                per_dir["decreasing"]["if_var"] = if_var_dec

                if abs(if_var_inc - if_var_dec) <= 1e-12:
                    if self.cfg.tie_break == "ess":
                        chosen = (
                            "increasing"
                            if per_dir["increasing"]["ess_final"]
                            >= per_dir["decreasing"]["ess_final"]
                            else "decreasing"
                        )
                    else:
                        chosen = (
                            "increasing"
                            if per_dir["increasing"]["v_final"]
                            <= per_dir["decreasing"]["v_final"]
                            else "decreasing"
                        )
                else:
                    chosen = "increasing" if if_var_inc <= if_var_dec else "decreasing"

        # Assemble output for chosen direction
        stats = per_dir[chosen]
        w_cal = stats["w_final"]
        v_before = float(np.var(w))
        info = {
            "direction": chosen,
            "gamma": stats["gamma"],
            "var_before": v_before,
            "var_after_proj": stats["v_proj"],
            "var_after_blend": stats["v_final"],
            "ess_before": len(w) / (1.0 + v_before),
            "ess_after_proj": stats["ess_proj"],
            "ess_after_blend": stats["ess_final"],
            "selection_method": selection_method,
        }
        if len(candidates) > 1:
            info["l2_distance_increasing"] = per_dir["increasing"]["sse"]
            info["l2_distance_decreasing"] = per_dir["decreasing"]["sse"]
            if "if_var" in per_dir["increasing"]:
                info["if_var_increasing"] = per_dir["increasing"]["if_var"]
                info["if_var_decreasing"] = per_dir["decreasing"]["if_var"]

        return w_cal, info
