"""Core CF-bits computation and gating logic.

CF-bits measure information gain as the log-ratio of baseline to actual width:
bits = log₂(W₀ / W)

Each halving of width adds 1 bit of information.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal
import numpy as np
import logging

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


def bits_from_width(w0: float, w: float, epsilon: float = 1e-10) -> float:
    """Compute bits of information from width reduction.

    bits = log₂(w0 / w)

    Args:
        w0: Baseline width
        w: Actual width
        epsilon: Small constant for numerical stability

    Returns:
        Bits of information (can be negative if w > w0)
    """
    if w0 <= epsilon:
        logger.warning(f"Baseline width {w0} too small")
        return 0.0

    if w <= epsilon:
        logger.warning(f"Width {w} too small, capping bits")
        return 10.0  # Cap at 10 bits (1024x reduction)

    return float(np.log2(w0 / w))


def compute_cfbits(
    w0: float,
    wid: float,
    wvar: float,
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
    # Total width (interval Minkowski sum under independence)
    w_tot = wid + wvar

    # Maximum width (identifies bottleneck)
    w_max = max(wid, wvar)

    # Total bits
    bits_tot = bits_from_width(w0, w_tot)

    # Identification bits (if meaningful)
    bits_id = bits_from_width(w0, wid) if wid > 0 else None

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
        w_id=wid,
        w_var=wvar,
        w_tot=w_tot,
        w_max=w_max,
    )


def apply_gates(
    aessf: Optional[float] = None,
    aessf_lcb: Optional[float] = None,
    ifr: Optional[float] = None,
    ifr_lcb: Optional[float] = None,
    tail_index: Optional[float] = None,
    var_oracle_ratio: Optional[float] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> GatesDecision:
    """Apply reliability gates based on diagnostic metrics.

    Uses lower confidence bounds (LCBs) when available for conservative gating.

    Default thresholds (can be overridden):
    - A-ESSF < 0.05: REFUSE (catastrophic overlap)
    - A-ESSF < 0.20: CRITICAL (poor overlap, need DR)
    - IFR < 0.2: CRITICAL (very inefficient)
    - IFR < 0.5: WARNING (inefficient)
    - Tail index < 2.0: CRITICAL (infinite variance risk)
    - Tail index < 2.5: WARNING (heavy tails)
    - Oracle variance ratio > 1.0: WARNING (OUA dominates)

    Args:
        aessf: Adjusted ESS Fraction on σ(S) (point estimate)
        aessf_lcb: Lower confidence bound for A-ESSF (preferred)
        ifr: Information Fraction Ratio (point estimate)
        ifr_lcb: Lower confidence bound for IFR (preferred)
        tail_index: Hill tail index
        var_oracle_ratio: Ratio of oracle variance to main variance
        thresholds: Optional custom thresholds

    Returns:
        GatesDecision with state, reasons, and suggestions
    """
    # Default thresholds
    default_thresholds = {
        "aessf_refuse": 0.05,
        "aessf_critical": 0.20,
        "ifr_critical": 0.2,
        "ifr_warning": 0.5,
        "tail_critical": 2.0,
        "tail_warning": 2.5,
        "oracle_warning": 1.0,
    }

    if thresholds:
        default_thresholds.update(thresholds)
    thresholds = default_thresholds

    # Collect issues
    reasons = []
    suggestions = {}
    state = "GOOD"

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
            suggestions["fresh_draws"] = 100  # Suggested number

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

    # If no issues found
    if not reasons:
        reasons.append("All diagnostics within acceptable ranges")

    return GatesDecision(
        state=state,
        reasons=reasons,
        suggestions=suggestions,
    )
