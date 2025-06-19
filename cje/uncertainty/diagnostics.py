"""Diagnostics for uncertainty contributions in CJE.

This module provides tools to analyze and visualize how judge uncertainty
affects the final estimates and confidence intervals.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging

from .schemas import (
    JudgeScore,
    CalibratedReward,
    VarianceDecomposition,
    UncertaintyDiagnostics,
)

logger = logging.getLogger(__name__)


def compute_variance_decomposition(
    eif_variance: float,
    judge_variance_contribution: float,
) -> VarianceDecomposition:
    """Decompose total variance into EIF and judge components.

    Args:
        eif_variance: Variance from efficient influence function
        judge_variance_contribution: E[w²v] term from judge uncertainty

    Returns:
        VarianceDecomposition with percentages
    """
    total_variance = eif_variance + judge_variance_contribution

    if total_variance < 1e-10:
        # Handle degenerate case
        return VarianceDecomposition(
            total=0.0,
            eif=0.0,
            judge=0.0,
            eif_pct=100.0,
            judge_pct=0.0,
        )

    eif_pct = (eif_variance / total_variance) * 100
    judge_pct = (judge_variance_contribution / total_variance) * 100

    return VarianceDecomposition(
        total=total_variance,
        eif=eif_variance,
        judge=judge_variance_contribution,
        eif_pct=eif_pct,
        judge_pct=judge_pct,
    )


def analyze_variance_contributions(
    weights: np.ndarray,
    variances: np.ndarray,
    concentration_threshold: float = 0.5,
) -> UncertaintyDiagnostics:
    """Analyze per-sample variance contributions and concentration.

    Args:
        weights: Importance weights
        variances: Judge variances
        concentration_threshold: Threshold for high concentration warning

    Returns:
        UncertaintyDiagnostics with detailed analysis
    """
    # Per-sample contributions
    contributions = weights**2 * variances
    total_contribution = np.sum(contributions)

    # Find high-contribution samples
    threshold = np.percentile(contributions, 90)
    high_var_indices = np.where(contributions > threshold)[0]

    # Concentration analysis
    sorted_contributions = np.sort(contributions)[::-1]
    n_top = max(1, len(weights) // 10)
    top_10_pct_contribution = np.sum(sorted_contributions[:n_top]) / total_contribution

    # Compute gamma if available (placeholder - would come from calibration)
    gamma = 1.0  # Default if not calibrated

    # Generate warnings
    warnings = []

    if top_10_pct_contribution > concentration_threshold:
        warnings.append(
            f"High variance concentration: top 10% of samples contribute "
            f"{top_10_pct_contribution*100:.1f}% of judge uncertainty"
        )

    if np.max(contributions) > 0.1 * total_contribution:
        max_idx = np.argmax(contributions)
        warnings.append(
            f"Sample {max_idx} contributes {contributions[max_idx]/total_contribution*100:.1f}% "
            f"of total judge uncertainty"
        )

    if np.mean(variances) > 0.1:
        warnings.append(f"High average judge uncertainty: {np.mean(variances):.3f}")

    return UncertaintyDiagnostics(
        per_sample_contributions=contributions,
        high_variance_samples=high_var_indices.tolist(),
        concentration_ratio=top_10_pct_contribution,
        gamma_calibration=gamma,
        warnings=warnings,
    )


def create_uncertainty_report(
    weights: np.ndarray,
    rewards: np.ndarray,
    variances: np.ndarray,
    estimate: float,
    se_with_uncertainty: float,
    se_without_uncertainty: float,
    gamma: float = 1.0,
    shrinkage_applied: bool = False,
    shrinkage_lambda: Optional[float] = None,
) -> Dict[str, Any]:
    """Create comprehensive uncertainty analysis report.

    Args:
        weights: Importance weights
        rewards: Reward values
        variances: Judge variances
        estimate: Point estimate
        se_with_uncertainty: SE including judge uncertainty
        se_without_uncertainty: SE without judge uncertainty
        gamma: Variance calibration factor
        shrinkage_applied: Whether shrinkage was used
        shrinkage_lambda: Shrinkage parameter if applied

    Returns:
        Dictionary with full uncertainty analysis
    """
    n = len(weights)

    # Variance decomposition
    var_with = se_with_uncertainty**2 * n
    var_without = se_without_uncertainty**2 * n
    judge_var_contrib = var_with - var_without

    decomp = compute_variance_decomposition(var_without, judge_var_contrib)

    # Diagnostics
    diag = analyze_variance_contributions(weights, variances)

    # Impact metrics
    se_increase_pct = (
        (se_with_uncertainty - se_without_uncertainty) / se_without_uncertainty * 100
    )
    ci_width_increase = 2 * 1.96 * (se_with_uncertainty - se_without_uncertainty)

    # Summary statistics
    variance_stats = {
        "mean": float(np.mean(variances)),
        "median": float(np.median(variances)),
        "p90": float(np.percentile(variances, 90)),
        "max": float(np.max(variances)),
        "n_zero": int(np.sum(variances == 0)),
        "n_high": int(np.sum(variances > 0.1)),
    }

    # Build report
    report = {
        "summary": {
            "estimate": float(estimate),
            "se_with_uncertainty": float(se_with_uncertainty),
            "se_without_uncertainty": float(se_without_uncertainty),
            "se_increase_pct": float(se_increase_pct),
            "ci_width_increase": float(ci_width_increase),
        },
        "variance_decomposition": {
            "total": float(decomp.total),
            "eif": float(decomp.eif),
            "judge": float(decomp.judge),
            "eif_pct": float(decomp.eif_pct),
            "judge_pct": float(decomp.judge_pct),
        },
        "variance_statistics": variance_stats,
        "calibration": {
            "gamma": float(gamma),
            "interpretation": _interpret_gamma(gamma),
        },
        "concentration": {
            "top_10pct_contribution": float(diag.concentration_ratio),
            "high_variance_samples": diag.high_variance_samples,
            "n_high_variance": len(diag.high_variance_samples),
        },
        "warnings": diag.warnings,
    }

    # Add shrinkage info if applicable
    if shrinkage_applied:
        report["shrinkage"] = {
            "applied": True,
            "lambda": float(shrinkage_lambda) if shrinkage_lambda else 0.0,
            "description": "Variance-based weight shrinkage applied to improve ESS",
        }
    else:
        report["shrinkage"] = {"applied": False}

    # Add recommendations
    report["recommendations"] = _generate_recommendations(
        se_increase_pct, diag, gamma, variance_stats
    )

    return report


def _interpret_gamma(gamma: float) -> str:
    """Interpret gamma calibration value."""
    if gamma < 0.5:
        return "Judge is overconfident (overestimates certainty)"
    elif gamma > 2.0:
        return "Judge is underconfident (underestimates certainty)"
    elif gamma > 1.2:
        return "Judge slightly underestimates uncertainty"
    elif gamma < 0.8:
        return "Judge slightly overestimates uncertainty"
    else:
        return "Judge uncertainty is well-calibrated"


def _generate_recommendations(
    se_increase_pct: float,
    diag: UncertaintyDiagnostics,
    gamma: float,
    variance_stats: Dict[str, Any],
) -> List[str]:
    """Generate actionable recommendations based on diagnostics."""
    recommendations = []

    # SE increase recommendations
    if se_increase_pct > 50:
        recommendations.append(
            "Judge uncertainty contributes >50% to SE. Consider using a more "
            "confident judge or collecting more samples."
        )

    # Concentration recommendations
    if diag.concentration_ratio > 0.5:
        recommendations.append(
            "Variance is highly concentrated. Consider variance-based shrinkage "
            "to improve effective sample size."
        )

    # Calibration recommendations
    if gamma > 3.0:
        recommendations.append(
            "Judge is severely underconfident (γ > 3). The judge may need "
            "better calibration or training."
        )
    elif gamma < 0.33:
        recommendations.append(
            "Judge is severely overconfident (γ < 0.33). Be cautious about "
            "confidence intervals being too narrow."
        )

    # High uncertainty samples
    if variance_stats["n_high"] > variance_stats["mean"] * 0.2:
        recommendations.append(
            f"{variance_stats['n_high']} samples have high uncertainty (>0.1). "
            "Review these cases for ambiguous or difficult judgments."
        )

    # Zero variance warning
    if variance_stats["n_zero"] == len(diag.per_sample_contributions):
        recommendations.append(
            "All variances are zero - judge is not providing uncertainty estimates. "
            "Results are equivalent to standard CJE."
        )

    return recommendations


def plot_variance_contributions(
    contributions: np.ndarray,
    title: str = "Per-Sample Variance Contributions",
    log_scale: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Plot per-sample variance contributions.

    Args:
        contributions: Array of variance contributions
        title: Plot title
        log_scale: Whether to use log scale for y-axis
        save_path: Optional path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib not available for plotting")
        return

    plt.figure(figsize=(10, 6))

    # Sort contributions
    sorted_contrib = np.sort(contributions)[::-1]

    # Plot
    plt.plot(sorted_contrib, linewidth=2)

    # Add percentile markers
    n = len(sorted_contrib)
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        idx = int(n * p / 100)
        if idx < n:
            plt.axvline(idx, color="gray", linestyle="--", alpha=0.5)
            plt.text(idx, plt.ylim()[1], f"{p}%", ha="center", va="bottom")

    plt.xlabel("Sample Rank")
    plt.ylabel("Variance Contribution")
    plt.title(title)

    if log_scale and np.min(sorted_contrib) > 0:
        plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved variance contribution plot to {save_path}")
    else:
        plt.show()


def compare_with_without_uncertainty(
    results_with: Dict[str, Any],
    results_without: Dict[str, Any],
) -> str:
    """Generate comparison report between uncertainty-aware and standard results.

    Args:
        results_with: Results with uncertainty
        results_without: Standard results

    Returns:
        Formatted comparison report
    """
    report_lines = [
        "=" * 60,
        "UNCERTAINTY IMPACT COMPARISON",
        "=" * 60,
        "",
    ]

    # Extract estimates and SEs
    est_with = results_with.get("estimate", 0)
    se_with = results_with.get("se", 0)
    est_without = results_without.get("estimate", 0)
    se_without = results_without.get("se", 0)

    # Point estimates (should be similar)
    report_lines.extend(
        [
            "Point Estimates:",
            f"  Standard:           {est_without:.4f}",
            f"  Uncertainty-aware:  {est_with:.4f}",
            f"  Difference:         {abs(est_with - est_without):.4f}",
            "",
        ]
    )

    # Standard errors
    se_increase = (se_with - se_without) / se_without * 100 if se_without > 0 else 0
    report_lines.extend(
        [
            "Standard Errors:",
            f"  Standard:           {se_without:.4f}",
            f"  Uncertainty-aware:  {se_with:.4f}",
            f"  Increase:           {se_increase:.1f}%",
            "",
        ]
    )

    # Confidence intervals
    ci_with = (est_with - 1.96 * se_with, est_with + 1.96 * se_with)
    ci_without = (est_without - 1.96 * se_without, est_without + 1.96 * se_without)

    report_lines.extend(
        [
            "95% Confidence Intervals:",
            f"  Standard:           [{ci_without[0]:.4f}, {ci_without[1]:.4f}]",
            f"  Uncertainty-aware:  [{ci_with[0]:.4f}, {ci_with[1]:.4f}]",
            f"  Width increase:     {(ci_with[1]-ci_with[0])-(ci_without[1]-ci_without[0]):.4f}",
            "",
        ]
    )

    # Interpretation
    if se_increase > 20:
        interpretation = "Judge uncertainty significantly widens confidence intervals"
    elif se_increase > 5:
        interpretation = "Judge uncertainty moderately affects confidence intervals"
    else:
        interpretation = "Judge uncertainty has minimal impact on confidence intervals"

    report_lines.extend(
        [
            "Interpretation:",
            f"  {interpretation}",
            "",
        ]
    )

    return "\n".join(report_lines)
