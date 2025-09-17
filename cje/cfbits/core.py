"""Core CF-bits computation and gating logic.

CF-bits measure information gain as the log-ratio of baseline to actual width:
bits = log₂(W₀ / W)

Each halving of width adds 1 bit of information.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal
import numpy as np
import logging

from .config import GATE_THRESHOLDS, WMAX_THRESHOLD, deep_merge

logger = logging.getLogger(__name__)


@dataclass
class CFBits:
    """CF-bits metrics and width decomposition.

    Attributes:
        bits_tot: Total bits of information
        bits_id: Bits from identification (structural)
        bits_var: Bits from sampling efficiency
        w0: Baseline width
        w_id: Identification width
        w_var: Sampling width
        w_tot: Total width (w_id + w_var)
        w_max: Maximum of w_id and w_var
    """

    bits_tot: float
    bits_id: Optional[float]
    bits_var: Optional[float]
    w0: float
    w_id: float
    w_var: float
    w_tot: float
    w_max: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "bits_tot": self.bits_tot,
            "bits_id": self.bits_id,
            "bits_var": self.bits_var,
            "w0": self.w0,
            "w_id": self.w_id,
            "w_var": self.w_var,
            "w_tot": self.w_tot,
            "w_max": self.w_max,
            "dominant": "identification" if self.w_id > self.w_var else "sampling",
        }

    def summary(self) -> str:
        """Human-readable summary."""
        dominant = "identification" if self.w_id > self.w_var else "sampling"
        return (
            f"CF-bits: {self.bits_tot:.2f} bits total "
            f"(W: {self.w_tot:.3f} from {self.w0:.1f} baseline). "
            f"Decomposition: Wid={self.w_id:.3f}, Wvar={self.w_var:.3f}. "
            f"Dominant: {dominant}."
        )


@dataclass
class GatesDecision:
    """Reliability gating decision based on diagnostic thresholds.

    Attributes:
        state: Overall reliability state
        reasons: List of specific issues or confirmations
        suggestions: Recommended actions to improve reliability
    """

    state: Literal["GOOD", "WARNING", "CRITICAL", "REFUSE"]
    reasons: List[str]
    suggestions: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "state": self.state,
            "reasons": self.reasons,
            "suggestions": self.suggestions,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        emoji = {"GOOD": "✅", "WARNING": "⚠️", "CRITICAL": "❌", "REFUSE": "🚫"}
        return (
            f"{emoji.get(self.state, '')} {self.state}: {', '.join(self.reasons[:2])}"
        )


def bits_from_width(
    w0: float, w: Optional[float], epsilon: float = 1e-10
) -> Optional[float]:
    """Compute bits of information from width reduction.

    bits = log₂(w0 / w)

    Args:
        w0: Baseline width
        w: Actual width (can be None if not computable)
        epsilon: Small constant for numerical stability

    Returns:
        Bits of information (can be negative if w > w0), or None if w is None
    """
    if w is None:
        return None

    if w0 <= epsilon:
        logger.warning(f"Baseline width {w0} too small")
        return 0.0

    if w <= epsilon:
        logger.warning(f"Width {w} too small, capping bits")
        return 10.0  # Cap at 10 bits (1024x reduction)

    return float(np.log2(w0 / w))


def compute_cfbits(
    w0: float,
    wid: Optional[float],
    wvar: Optional[float],
    ifr_main: Optional[float] = None,
    ifr_oua: Optional[float] = None,
) -> CFBits:
    """Compute CF-bits metrics from width components.

    Args:
        w0: Baseline width (typically 1.0 for KPI ∈ [0,1])
        wid: Identification width (structural uncertainty)
        wvar: Sampling width (statistical uncertainty)
        ifr_main: Information Fraction Ratio without OUA
        ifr_oua: Information Fraction Ratio with OUA (preferred)

    Returns:
        CFBits object with all metrics
    """
    # Handle None values gracefully
    if wid is None and wvar is None:
        w_tot = None
        w_max = None
    elif wid is None:
        w_tot = wvar
        w_max = wvar
    elif wvar is None:
        w_tot = wid
        w_max = wid
    else:
        # Total width (interval Minkowski sum under independence)
        w_tot = wid + wvar
        # Maximum width (identifies bottleneck)
        w_max = max(wid, wvar)

    # Total bits
    bits_tot = bits_from_width(w0, w_tot)
    # If bits_tot is None (when w_tot is None), default to 0.0
    if bits_tot is None:
        bits_tot = 0.0

    # Identification bits (if meaningful)
    bits_id = bits_from_width(w0, wid) if wid is not None and wid > 0 else None

    # Variance bits (from IFR_OUA if available, else IFR_main)
    # bits_var = 0.5 * log₂(IFR) since width scales as √variance
    bits_var = None
    if ifr_oua is not None and ifr_oua > 0:
        # Prefer IFR_OUA as it accounts for oracle uncertainty
        bits_var = 0.5 * float(np.log2(ifr_oua))
    elif ifr_main is not None and ifr_main > 0:
        # Fall back to IFR_main if OUA not available
        bits_var = 0.5 * float(np.log2(ifr_main))

    return CFBits(
        bits_tot=bits_tot,
        bits_id=bits_id,
        bits_var=bits_var,
        w0=w0,
        w_id=wid if wid is not None else 0.0,
        w_var=wvar if wvar is not None else 0.0,
        w_tot=w_tot if w_tot is not None else 0.0,
        w_max=w_max if w_max is not None else 0.0,
    )


def apply_gates(
    aessf: Optional[float] = None,
    aessf_lcb: Optional[float] = None,
    ifr: Optional[float] = None,
    ifr_lcb: Optional[float] = None,
    tail_index: Optional[float] = None,
    var_oracle_ratio: Optional[float] = None,
    wid: Optional[float] = None,
    wvar: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> GatesDecision:
    """Apply reliability gates based on diagnostic metrics.

    Uses lower confidence bounds (LCBs) when available for conservative gating.
    Includes Wmax gating for catastrophic cases where either Wid or Wvar is extreme.

    Default thresholds from config.py:
    - A-ESSF < 0.05: REFUSE (catastrophic overlap)
    - A-ESSF < 0.20: CRITICAL (poor overlap, need DR)
    - IFR < 0.2: CRITICAL (very inefficient)
    - IFR < 0.5: WARNING (inefficient)
    - Tail index < 2.0: CRITICAL (infinite variance risk)
    - Tail index < 2.5: WARNING (heavy tails)
    - Oracle variance ratio > 1.0: WARNING (OUA dominates)
    - Wid > 0.5: WARNING (large identification uncertainty)
    - Wid > 0.8: CRITICAL (identification dominates)
    - Wmax > WMAX_THRESHOLD: REFUSE (catastrophic uncertainty)

    Args:
        aessf: Adjusted ESS Fraction on σ(S) (point estimate)
        aessf_lcb: Lower confidence bound for A-ESSF (preferred)
        ifr: Information Fraction Ratio (point estimate)
        ifr_lcb: Lower confidence bound for IFR (preferred)
        tail_index: Hill tail index
        var_oracle_ratio: Ratio of oracle variance to main variance
        wid: Identification width (structural uncertainty)
        wvar: Sampling width (statistical uncertainty)
        thresholds: Optional custom thresholds

    Returns:
        GatesDecision with state, reasons, and suggestions
    """
    # Use centralized thresholds from config
    gate_thresholds = GATE_THRESHOLDS.copy()

    if thresholds:
        gate_thresholds = deep_merge(gate_thresholds, thresholds)
    thresholds = gate_thresholds

    # Collect issues
    reasons = []
    suggestions = {}
    state: Literal["GOOD", "WARNING", "CRITICAL", "REFUSE"] = "GOOD"

    # Check overlap (A-ESSF) - use LCB if available
    aessf_check = aessf_lcb if aessf_lcb is not None else aessf
    if aessf_check is not None:
        lcb_note = " (LCB)" if aessf_lcb is not None else ""
        if aessf_check < thresholds["aessf_refuse"]:
            state = "REFUSE"
            reasons.append(f"Catastrophic overlap (A-ESSF{lcb_note}={aessf_check:.1%})")
            suggestions["change_policy"] = "Use policies with better overlap"
        elif aessf_check < thresholds["aessf_critical"]:
            if state != "REFUSE":
                state = "CRITICAL"
            reasons.append(f"Poor overlap (A-ESSF{lcb_note}={aessf_check:.1%})")
            suggestions["use_dr"] = "Use DR methods with fresh draws"
            suggestions["fresh_draws"] = "100"  # Suggested number as string

    # Check efficiency (IFR) - use LCB if available
    ifr_check = ifr_lcb if ifr_lcb is not None else ifr
    if ifr_check is not None:
        lcb_note = " (LCB)" if ifr_lcb is not None else ""
        if ifr_check < thresholds["ifr_critical"]:
            if state not in ["REFUSE", "CRITICAL"]:
                state = "CRITICAL"
            reasons.append(f"Very inefficient (IFR{lcb_note}={ifr_check:.1%})")
            suggestions["improve_estimator"] = "Use more efficient estimator"
        elif ifr_check < thresholds["ifr_warning"]:
            if state == "GOOD":
                state = "WARNING"
            reasons.append(f"Inefficient (IFR{lcb_note}={ifr_check:.1%})")
            suggestions["consider_dr"] = "Consider DR methods"

    # Check tail safety
    if tail_index is not None:
        if tail_index < thresholds["tail_critical"]:
            if state not in ["REFUSE", "CRITICAL"]:
                state = "CRITICAL"
            reasons.append(f"Infinite variance risk (tail={tail_index:.1f})")
            suggestions["robust_estimator"] = "Use robust estimator or trim weights"
        elif tail_index < thresholds["tail_warning"]:
            if state == "GOOD":
                state = "WARNING"
            reasons.append(f"Heavy tails (tail={tail_index:.1f})")
            suggestions["monitor_tails"] = "Monitor tail behavior"

    # Check oracle dominance
    if var_oracle_ratio is not None:
        if var_oracle_ratio > thresholds["oracle_warning"]:
            if state == "GOOD":
                state = "WARNING"
            reasons.append(
                f"Oracle uncertainty dominates (ratio={var_oracle_ratio:.1f})"
            )
            suggestions["add_labels"] = "Add more oracle labels"

    # Check identification width (Wid)
    if wid is not None:
        if wid > thresholds.get("wid_critical", 0.80):
            if state not in ["REFUSE", "CRITICAL"]:
                state = "CRITICAL"
            reasons.append(f"Identification dominates (Wid={wid:.2f})")
            suggestions["more_labels"] = "Need many more oracle labels"
        elif wid > thresholds.get("wid_warning", 0.50):
            if state == "GOOD":
                state = "WARNING"
            reasons.append(f"Large identification uncertainty (Wid={wid:.2f})")
            suggestions["consider_labels"] = "Consider increasing oracle labels"

    # Check Wmax (catastrophic case)
    if wid is not None and wvar is not None:
        wmax = max(wid, wvar)
        if wmax > WMAX_THRESHOLD:
            state = "REFUSE"
            dominant = "identification" if wid > wvar else "sampling"
            reasons.append(f"Catastrophic {dominant} uncertainty (Wmax={wmax:.2f})")
            suggestions["fundamental_issue"] = f"Fundamental {dominant} limitation"

    # If no issues found
    if not reasons:
        reasons.append("All diagnostics within acceptable ranges")

    return GatesDecision(
        state=state,
        reasons=reasons,
        suggestions=suggestions,
    )


# Budget helper functions


def logs_for_delta_bits(
    delta_bits: float,
    current_ifr_oua: Optional[float] = None,
) -> float:
    """Compute factor to multiply number of logs by for target CF-bits improvement.

    To gain Δ bits in the sampling channel, the adjusted sample size (n × IFR)
    must scale by 2^(2Δ). If IFR remains roughly constant, multiply n by this factor.

    Args:
        delta_bits: Target improvement in CF-bits
        current_ifr_oua: Current IFR (unused, kept for backward compatibility)

    Returns:
        Factor by which to multiply sample size n

    Example:
        >>> # To gain 0.5 bits (halve width)
        >>> logs_for_delta_bits(0.5)
        2.0  # Need 2x more logs
        >>> # To gain 1 bit (quarter width)
        >>> logs_for_delta_bits(1.0)
        4.0  # Need 4x more logs
    """
    # The adjusted sample size n_adj = n × IFR must scale by 2^(2Δ)
    # If IFR stays constant, n scales by the same factor
    return 2 ** (2 * delta_bits)


def labels_for_oua_reduction(
    current_oua_share: float,
    target_oua_share: float,
    n_samples: int,
    current_labels: int,
) -> int:
    """Compute additional oracle labels needed to reduce OUA share.

    Oracle variance typically scales as c/m where m is number of labels.
    To reduce OUA share from s to s*, we solve for required m.

    Args:
        current_oua_share: Current fraction of variance from oracle uncertainty
        target_oua_share: Target fraction (should be < current)
        n_samples: Total number of samples
        current_labels: Current number of oracle labels

    Returns:
        Additional oracle labels needed

    Example:
        >>> # Reduce OUA from 40% to 10% of total variance
        >>> labels_for_oua_reduction(0.4, 0.1, 1000, 50)
        150  # Need 150 more oracle labels
    """
    if target_oua_share >= current_oua_share:
        return 0  # No additional labels needed

    if current_oua_share <= 0 or current_labels <= 0:
        logger.warning("Invalid inputs for label budget calculation")
        return 0

    # Var_oracle ~ c/m, and OUA share = Var_oracle / Var_total
    # To change share from s to s*, need m* = m * s * (1-s*) / (s* * (1-s))
    factor = (current_oua_share * (1 - target_oua_share)) / (
        target_oua_share * (1 - current_oua_share)
    )

    new_labels = int(current_labels * factor)
    additional_needed = max(0, new_labels - current_labels)

    # Cap at total sample size
    return min(additional_needed, n_samples - current_labels)


def fresh_draws_for_dr_improvement(
    current_var_main: float,
    target_improvement: float = 0.5,
    dr_efficiency_constant: float = 0.1,
) -> int:
    """Estimate fresh draws needed for DR variance improvement.

    DR variance typically follows: Var ~ c/(n + ν) where ν is fresh draws.

    Args:
        current_var_main: Current main variance component
        target_improvement: Target CF-bits improvement
        dr_efficiency_constant: Estimator-specific efficiency constant

    Returns:
        Estimated number of fresh draws needed

    Example:
        >>> # To gain 0.5 bits with current variance 0.01
        >>> fresh_draws_for_dr_improvement(0.01, 0.5)
        100  # Need ~100 fresh draws
    """
    # To gain Δ bits, need to reduce variance by factor 2^(2Δ)
    variance_reduction_factor = 2 ** (2 * target_improvement)

    # Very rough estimate: ν ~ n * (factor - 1) / c
    # This is highly estimator-dependent
    fresh_draws = int(variance_reduction_factor * 100 / dr_efficiency_constant)

    return fresh_draws


def bits_to_width(bits: float, w0: float = 1.0) -> float:
    """Convert CF-bits back to width on KPI scale.

    Args:
        bits: CF-bits value
        w0: Baseline width (default 1.0 for [0,1] KPIs)

    Returns:
        Width on original KPI scale

    Example:
        >>> bits_to_width(2.0)  # 2 bits
        0.25  # Width is 1/4 of baseline
    """
    return w0 / (2**bits)


def width_to_bits(width: float, w0: float = 1.0) -> float:
    """Convert width to CF-bits.

    Args:
        width: Width on KPI scale
        w0: Baseline width (default 1.0 for [0,1] KPIs)

    Returns:
        CF-bits value

    Example:
        >>> width_to_bits(0.25)  # Width is 1/4
        2.0  # 2 bits of information
    """
    if width <= 0:
        return float("inf")
    return float(np.log2(w0 / width))
