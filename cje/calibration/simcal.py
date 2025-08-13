"""Score-Indexed Monotone Calibration (SIMCal) for importance weights.

This module implements SIMCal, which projects weights onto monotone curves
indexed by a score (e.g., judge score), choosing the direction that minimizes
L2 distance, then blending toward uniform to hit variance/ESS targets.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class SimcalConfig:
    """Configuration for SIMCal calibration.

    Args:
        ess_floor: Minimum effective sample size as fraction of n (e.g., 0.2 => ESS >= 0.2 * n)
        var_cap: Maximum allowed variance of calibrated weights
        epsilon: Small constant for numerical stability
        direction: "auto" (choose by L2), "increasing", or "decreasing"
        tie_break: How to break ties when L2 distances are equal ("ess" or "var")
    """

    ess_floor: Optional[float] = None
    var_cap: Optional[float] = None
    epsilon: float = 1e-9
    direction: str = "auto"
    tie_break: str = "ess"

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

    def transform(
        self, w: np.ndarray, s: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calibrate weights using score-indexed monotone projection.

        Args:
            w: Raw importance weights (must be positive)
            s: Score index (e.g., judge scores) for ordering

        Returns:
            Tuple of (calibrated_weights, info_dict) where info_dict contains:
                - direction: chosen monotone direction
                - gamma: blending parameter (0=no blend, 1=uniform)
                - var_before: variance of input weights
                - var_after_proj: variance after projection
                - var_after_blend: variance after blending
                - ess_before: ESS of input weights
                - ess_after_proj: ESS after projection
                - ess_after_blend: ESS after blending

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

        # Choose direction (if auto)
        if len(candidates) == 1:
            chosen = next(iter(candidates))
        else:
            # Compute L2 distances for each direction
            sse = {d: float(np.sum((z - w) ** 2)) for d, z in candidates.items()}

            # Check for tie
            if abs(sse["increasing"] - sse["decreasing"]) <= 1e-12:
                # Tie-break based on ESS or variance
                def compute_stats(z: np.ndarray) -> Tuple[float, float]:
                    v = float(np.var(z))
                    ess = len(z) / (1.0 + v) if v >= 0 else len(z)
                    return v, ess

                v_inc, ess_inc = compute_stats(candidates["increasing"])
                v_dec, ess_dec = compute_stats(candidates["decreasing"])

                if self.cfg.tie_break == "ess":
                    chosen = "increasing" if ess_inc >= ess_dec else "decreasing"
                else:  # tie_break == "var"
                    chosen = "increasing" if v_inc <= v_dec else "decreasing"
            else:
                # Choose direction with smaller L2 distance
                chosen = min(sse, key=lambda k: sse[k])

        w_proj = candidates[chosen]
        v_proj = float(np.var(w_proj))

        # Compute blending parameter to meet variance/ESS constraints
        gamma = 0.0

        if v_proj > 0:
            # ESS constraint: ESS = n / (1 + Var(w)) >= ess_floor * n
            # => Var(w) <= (1/ess_floor - 1)
            if self.cfg.ess_floor is not None:
                v_max_ess = (1.0 / self.cfg.ess_floor) - 1.0
                if v_proj > v_max_ess:
                    # Blend to reduce variance: Var((1-γ)*w + γ*1) = (1-γ)²*Var(w)
                    # Want (1-γ)²*v_proj = v_max_ess
                    gamma_ess = 1.0 - np.sqrt(v_max_ess / v_proj)
                    gamma = max(gamma, gamma_ess)

            # Variance cap constraint
            if self.cfg.var_cap is not None and v_proj > self.cfg.var_cap:
                # Want (1-γ)²*v_proj = var_cap
                gamma_var = 1.0 - np.sqrt(self.cfg.var_cap / v_proj)
                gamma = max(gamma, gamma_var)

        # Clip gamma to [0, 1]
        gamma = float(np.clip(gamma, 0.0, 1.0))

        # Apply blending: w_cal = (1-γ)*w_proj + γ*1
        # Since mean(w_proj) = 1, this preserves mean-one property
        w_cal = 1.0 + (1.0 - gamma) * (w_proj - 1.0)

        # Final safety checks
        w_cal = np.maximum(w_cal, self.cfg.epsilon)
        w_cal = w_cal / w_cal.mean()

        # Compute final statistics
        v_before = float(np.var(w))
        v_after = float(np.var(w_cal))

        info = {
            "direction": chosen,
            "gamma": gamma,
            "var_before": v_before,
            "var_after_proj": v_proj,
            "var_after_blend": v_after,
            "ess_before": len(w) / (1.0 + v_before),
            "ess_after_proj": len(w) / (1.0 + v_proj),
            "ess_after_blend": len(w) / (1.0 + v_after),
        }

        # Add L2 distances if computed
        if len(candidates) > 1:
            info["l2_distance_increasing"] = float(
                np.sum((candidates["increasing"] - w) ** 2)
            )
            info["l2_distance_decreasing"] = float(
                np.sum((candidates["decreasing"] - w) ** 2)
            )

        return w_cal, info
