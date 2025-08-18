"""HERA: Hellinger–ESS Raw Audit for importance sampling diagnostics.

HERA provides production-ready overlap diagnostics using two ungameable metrics:
1. Hellinger affinity (Bhattacharyya coefficient) - structural overlap
2. Raw ESS fraction - variance inflation

These metrics are computed from raw log-probabilities before any calibration,
making them immune to gaming and providing honest assessment of identification risk.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from scipy.special import logsumexp
import logging

logger = logging.getLogger(__name__)


@dataclass
class HERAMetrics:
    """HERA (Hellinger–ESS Raw Audit) metrics for policy overlap.

    The two-number audit that tells the whole story of importance sampling risk.

    Attributes:
        hellinger_affinity: Bhattacharyya coefficient ∈ (0,1], structural overlap
        ess_raw_fraction: Raw ESS/n ∈ (0,1], variance inflation measure
        hera_status: Overall status from audit ("ok", "warning", "critical")
        auto_tuned_threshold: ESS threshold for target CI width (if computed)
        recommendation: HERA's recommendation based on the audit
    """

    hellinger_affinity: float
    ess_raw_fraction: float
    hera_status: str  # "ok", "warning", "critical"
    auto_tuned_threshold: Optional[float] = None
    recommendation: str = ""

    def summary(self) -> str:
        """One-line HERA summary."""
        return (
            f"HERA: {self.hera_status.upper()} "
            f"(H={self.hellinger_affinity:.1%}, E={self.ess_raw_fraction:.1%})"
        )

    @property
    def passes_audit(self) -> bool:
        """Whether this passes HERA's audit for IPS."""
        return self.hera_status != "critical"


def hera_hellinger(delta_log: np.ndarray) -> float:
    """Compute Hellinger affinity (the H in HERA) from log-ratios.

    This is numerically stable and works with mean-1 normalized weights.

    Args:
        delta_log: Log importance ratios log(p_π'/p_π₀)

    Returns:
        Hellinger affinity A = E[√(W/E[W])] ∈ (0,1]
    """
    delta_log = np.asarray(delta_log, dtype=float)
    if len(delta_log) == 0:
        return 0.0

    # Remove any NaN/inf values
    valid_mask = np.isfinite(delta_log)
    if not np.any(valid_mask):
        logger.warning("HERA: No valid log-ratios for Hellinger computation")
        return 0.0

    delta_log = delta_log[valid_mask]

    # Normalize weights to mean 1 for proper Hellinger
    log_mean_w = logsumexp(delta_log) - np.log(len(delta_log))
    normalized_log_weights = delta_log - log_mean_w

    # Hellinger affinity = E[√W_normalized]
    affinity = float(np.mean(np.exp(0.5 * normalized_log_weights)))

    # Bound to [0, 1]
    return min(max(affinity, 0.0), 1.0)


def hera_ess(delta_log: np.ndarray) -> float:
    """Compute raw ESS fraction (the E in HERA) from log-ratios.

    Uses log-sum-exp for numerical stability.

    Args:
        delta_log: Log importance ratios log(p_π'/p_π₀)

    Returns:
        ESS_raw/n = 1/E[W²] ∈ (0,1]
    """
    delta_log = np.asarray(delta_log, dtype=float)
    n = len(delta_log)
    if n == 0:
        return 0.0

    # Remove any NaN/inf values
    valid_mask = np.isfinite(delta_log)
    if not np.any(valid_mask):
        logger.warning("HERA: No valid log-ratios for ESS computation")
        return 0.0

    delta_log = delta_log[valid_mask]
    n = len(delta_log)

    # ESS/n = (sum W)² / (n * sum W²)
    # In log space: exp(2*LSE(Δℓ) - log(n) - LSE(2*Δℓ))
    lse1 = logsumexp(delta_log)
    lse2 = logsumexp(2.0 * delta_log)

    return float(np.exp(2.0 * lse1 - np.log(n) - lse2))


def hera_threshold(n: int, delta: float) -> float:
    """Compute HERA's auto-tuned ESS threshold for target CI half-width.

    Based on worst-case bound for 0-1 rewards:
    HW ≤ 1.96/(2√(n·ESS_raw/n))

    Args:
        n: Sample size
        delta: Target CI half-width (e.g., 0.03 for ±3%)

    Returns:
        Minimum ESS fraction needed for target precision
    """
    if n <= 0 or delta <= 0:
        return 0.10  # Default fallback

    # τ_ESS(δ) = (1.96/2)²/(nδ²) = 0.9604/(nδ²)
    threshold = 0.9604 / (n * (delta**2))

    # Cap at reasonable bounds
    return min(max(threshold, 0.001), 1.0)


def hera_audit(
    delta_log: np.ndarray,
    n_samples: Optional[int] = None,
    target_ci_halfwidth: Optional[float] = None,
) -> HERAMetrics:
    """Perform HERA audit on importance weights.

    HERA Gates:
    - CRITICAL: H < 0.20 OR E < 0.10 → refuse IPS
    - WARNING:  H < 0.35 OR E < 0.20 → warn, prefer DR
    - OK:       otherwise → standard IPS adequate

    Args:
        delta_log: Log importance ratios log(p_π'/p_π₀)
        n_samples: Sample size (defaults to len(delta_log))
        target_ci_halfwidth: Target CI half-width for auto-tuning

    Returns:
        HERAMetrics with audit results and recommendations
    """
    delta_log = np.asarray(delta_log, dtype=float)
    n = n_samples or len(delta_log)

    # Compute HERA's two numbers
    hellinger = hera_hellinger(delta_log)
    ess = hera_ess(delta_log)

    # Auto-tuned threshold if requested
    auto_threshold = None
    if target_ci_halfwidth is not None and n > 0:
        auto_threshold = hera_threshold(n, target_ci_halfwidth)

    # HERA's audit decision using preregistered gates
    if hellinger < 0.20 or ess < 0.10:
        status = "critical"
        recommendation = (
            "HERA AUDIT FAILED: Refuse IPS/Cal-IPS. "
            "Structural mismatch too severe. "
            "DR/TMLE may proceed with strong warnings."
        )
    elif hellinger < 0.35 or ess < 0.20:
        status = "warning"
        recommendation = (
            "HERA WARNING: IPS allowed with caution. "
            "DR/TMLE strongly preferred due to marginal overlap."
        )
    else:
        status = "ok"
        recommendation = "HERA APPROVED: Standard IPS should work adequately."

    # Check against auto-tuned threshold if available
    if auto_threshold is not None:
        if ess < auto_threshold:
            if status == "ok":
                status = "warning"
                recommendation = (
                    f"HERA WARNING: ESS below target for ±{target_ci_halfwidth:.1%} CI. "
                    f"Need ESS>{auto_threshold:.1%}, have {ess:.1%}."
                )

    return HERAMetrics(
        hellinger_affinity=hellinger,
        ess_raw_fraction=ess,
        hera_status=status,
        auto_tuned_threshold=auto_threshold,
        recommendation=recommendation,
    )


def hera_drill_down(
    delta_log: np.ndarray,
    index: np.ndarray,
    n_bins: int = 10,
    index_name: str = "index",
) -> Dict[str, Any]:
    """HERA drill-down analysis by bins of an index.

    This diagnostic (not a gate) helps localize where overlap problems occur.

    Args:
        delta_log: Log importance ratios
        index: Scalar index to bin by (e.g., judge scores, lengths)
        n_bins: Number of bins (default 10 for deciles)
        index_name: Name for the index in output

    Returns:
        Dictionary with HERA metrics per bin
    """
    delta_log = np.asarray(delta_log, dtype=float)
    index = np.asarray(index, dtype=float)

    if len(delta_log) != len(index):
        raise ValueError("delta_log and index must have same length")

    if len(delta_log) == 0:
        return {"index_name": index_name, "bins": []}

    # Create bins
    bin_edges = np.percentile(index, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-10  # Include maximum

    results = []
    for i in range(n_bins):
        mask = (index >= bin_edges[i]) & (index < bin_edges[i + 1])
        if np.any(mask):
            bin_delta = delta_log[mask]
            hellinger = hera_hellinger(bin_delta)
            ess = hera_ess(bin_delta)
            bin_center = 0.5 * (bin_edges[i] + bin_edges[i + 1])

            # Determine bin status
            if hellinger < 0.20 or ess < 0.10:
                bin_status = "critical"
            elif hellinger < 0.35 or ess < 0.20:
                bin_status = "warning"
            else:
                bin_status = "ok"

            results.append(
                {
                    "center": bin_center,
                    "hellinger": hellinger,
                    "ess": ess,
                    "status": bin_status,
                    "n_samples": np.sum(mask),
                }
            )

    return {
        "index_name": index_name,
        "bins": results,
    }


def format_hera_drill_down(drill_down: Dict) -> str:
    """Format HERA drill-down results as a readable table.

    Args:
        drill_down: Output from hera_drill_down()

    Returns:
        Formatted table string
    """
    lines = []
    lines.append(f"\nHERA Drill-Down by {drill_down['index_name']}:")
    lines.append("=" * 60)
    lines.append(
        f"{'Bin Center':>12} | {'H (Hellinger)':>13} | {'E (ESS)':>10} | {'Status':>8}"
    )
    lines.append("-" * 60)

    for bin_data in drill_down["bins"]:
        status_symbol = {"critical": "✗", "warning": "⚠", "ok": "✓"}[bin_data["status"]]

        lines.append(
            f"{bin_data['center']:12.3f} | "
            f"{bin_data['hellinger']:13.1%} | "
            f"{bin_data['ess']:10.1%} | "
            f"{status_symbol:>8}"
        )

    lines.append("=" * 60)
    lines.append("Legend: ✓ = OK, ⚠ = Warning, ✗ = Critical")

    return "\n".join(lines)


def hera_summary_card(
    metrics_by_policy: Dict[str, HERAMetrics],
    title: str = "HERA Audit Results",
) -> str:
    """Create a summary card of HERA results across policies.

    Args:
        metrics_by_policy: Dictionary mapping policy names to HERAMetrics
        title: Title for the card

    Returns:
        Formatted summary card
    """
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)
    lines.append(
        f"{'Policy':<20} | {'H':>6} | {'E':>6} | {'Status':>10} | {'Decision'}"
    )
    lines.append("-" * 70)

    for policy, metrics in metrics_by_policy.items():
        decision = "REFUSE" if metrics.hera_status == "critical" else "PROCEED"
        status_display = {"critical": "CRITICAL", "warning": "WARNING", "ok": "OK"}[
            metrics.hera_status
        ]

        lines.append(
            f"{policy:<20} | "
            f"{metrics.hellinger_affinity:6.1%} | "
            f"{metrics.ess_raw_fraction:6.1%} | "
            f"{status_display:>10} | "
            f"{decision}"
        )

    lines.append("=" * 70)
    lines.append("HERA Gates: H<0.20 OR E<0.10 → CRITICAL | H<0.35 OR E<0.20 → WARNING")

    return "\n".join(lines)


# Convenience function for weights
def hera_audit_weights(
    weights: np.ndarray,
    n_samples: Optional[int] = None,
    target_ci_halfwidth: Optional[float] = None,
) -> HERAMetrics:
    """Perform HERA audit on importance weights (not log-ratios).

    Args:
        weights: Importance weights (will be converted to log-ratios)
        n_samples: Sample size (defaults to len(weights))
        target_ci_halfwidth: Target CI half-width for auto-tuning

    Returns:
        HERAMetrics with audit results
    """
    weights = np.asarray(weights, dtype=float)

    # Convert to log-ratios (handle zeros safely)
    epsilon = 1e-10
    delta_log = np.log(np.maximum(weights, epsilon))

    return hera_audit(delta_log, n_samples, target_ci_halfwidth)
