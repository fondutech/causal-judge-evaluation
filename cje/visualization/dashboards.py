"""Multi-panel dashboard visualizations for CJE framework.

Contains complex multi-panel dashboards for comprehensive diagnostics:
- Weight diagnostics dashboard
- DR diagnostics dashboard
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path

# Import shared utilities
from ..utils.diagnostics import compute_ess


# Shared visualization utilities
def _quantile_bins(x: np.ndarray, n_bins: int = 25) -> np.ndarray:
    """Compute quantile bins for x."""
    x = np.asarray(x)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, qs)
    # ensure strictly increasing (can happen with ties)
    edges = np.unique(edges)
    # if too many duplicates, fall back to equal-width bins
    if len(edges) < max(5, n_bins // 2):
        edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
    return edges


def _bin_stats(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    q: tuple = (0.1, 0.5, 0.9),
    min_bin: int = 10,
) -> tuple:
    """Compute binned statistics."""
    x = np.asarray(x)
    y = np.asarray(y)
    xc = 0.5 * (edges[:-1] + edges[1:])
    q1 = np.full_like(xc, np.nan, dtype=float)
    q2 = np.full_like(xc, np.nan, dtype=float)
    q3 = np.full_like(xc, np.nan, dtype=float)
    for i in range(len(edges) - 1):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if np.sum(mask) >= min_bin:
            yy = y[mask]
            q1[i], q2[i], q3[i] = np.quantile(yy, q)
    return xc, q1, q2, q3


def plot_weight_dashboard_per_policy(
    raw_weights_dict: Dict[str, np.ndarray],
    calibrated_weights_dict: Optional[Dict[str, np.ndarray]] = None,
    n_samples: Optional[int] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (16, 10),
    random_seed: int = 42,
    diagnostics: Optional[Any] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """Create per-policy weight dashboards with judge score visualization.

    Creates a grid of subplots, one dashboard per policy, each showing:
    - Weight smoothing by judge score (the new visualization)
    - ESS and tail diagnostics
    - Clear per-policy view

    Args:
        raw_weights_dict: Dict mapping policy names to raw weight arrays
        calibrated_weights_dict: Optional dict of calibrated weights
        n_samples: Total number of samples
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        random_seed: Random seed for reproducibility
        diagnostics: Optional IPSDiagnostics or DRDiagnostics object
        **kwargs: Must include either 'judge_scores' dict or 'sampler'

    Returns:
        Tuple of (matplotlib Figure, metrics dict)
    """
    np.random.seed(random_seed)

    policies = list(raw_weights_dict.keys())
    n_policies = len(policies)

    # Get judge scores
    judge_scores_dict = kwargs.get("judge_scores", {})
    sampler = kwargs.get("sampler")

    # Extract judge scores from sampler if not provided directly
    if not judge_scores_dict and sampler is not None:
        judge_scores_dict = {}
        for policy in policies:
            data = sampler.get_data_for_policy(policy)
            if data:
                scores = np.array([d.get("judge_score", np.nan) for d in data])
                valid = ~np.isnan(scores)
                if valid.sum() > 0:
                    judge_scores_dict[policy] = scores[valid]

    # Determine grid layout
    if n_policies <= 2:
        rows, cols = 1, n_policies
    elif n_policies <= 4:
        rows, cols = 2, 2
    elif n_policies <= 6:
        rows, cols = 2, 3
    else:
        rows = int(np.ceil(n_policies / 3))
        cols = 3

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    # Hide extra subplots
    for i in range(n_policies, len(axes_flat)):
        axes_flat[i].axis("off")

    # Metrics storage
    all_metrics = {}

    # Plot each policy
    for idx, policy in enumerate(policies):
        ax = axes_flat[idx]

        # Get data for this policy
        raw_w = raw_weights_dict[policy]
        cal_w = (
            calibrated_weights_dict.get(policy, raw_w)
            if calibrated_weights_dict
            else raw_w
        )
        judge_scores = judge_scores_dict.get(policy, None)

        # Compute metrics for this policy
        ess_raw = compute_ess(raw_w)
        ess_cal = compute_ess(cal_w)
        uplift = ess_cal / max(ess_raw, 1e-12)

        # Top 1% mass
        w_sorted = np.sort(raw_w)[::-1]
        k = max(1, int(len(w_sorted) * 0.01))
        top1_raw = w_sorted[:k].sum() / max(w_sorted.sum(), 1e-12)

        w_sorted = np.sort(cal_w)[::-1]
        k = max(1, int(len(w_sorted) * 0.01))
        top1_cal = w_sorted[:k].sum() / max(w_sorted.sum(), 1e-12)

        # Store metrics
        all_metrics[policy] = {
            "ess_raw": ess_raw,
            "ess_cal": ess_cal,
            "ess_improvement": uplift,
            "top1_raw": top1_raw,
            "top1_cal": top1_cal,
            "n_samples": len(raw_w),
        }

        if judge_scores is not None and len(judge_scores) == len(raw_w):
            # Plot weight smoothing by score
            _plot_single_policy_weight_smoothing(
                ax,
                judge_scores,
                raw_w,
                cal_w,
                policy,
                ess_raw,
                ess_cal,
                uplift,
                top1_raw,
                top1_cal,
            )
        else:
            # Fallback: simple histogram comparison
            _plot_single_policy_weight_histogram(
                ax, raw_w, cal_w, policy, ess_raw, ess_cal, uplift, top1_raw, top1_cal
            )

    # Main title
    fig.suptitle(
        f"Weight Diagnostics by Policy (n={n_samples or len(next(iter(raw_weights_dict.values())))})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")

    return fig, all_metrics


def _plot_single_policy_weight_smoothing(
    ax: Any,
    judge_scores: np.ndarray,
    raw_w: np.ndarray,
    cal_w: np.ndarray,
    policy: str,
    ess_raw: float,
    ess_cal: float,
    uplift: float,
    top1_raw: float,
    top1_cal: float,
) -> None:
    """Plot weights vs judge scores with judge-score-based isotonic calibration."""

    # Import isotonic regression here to avoid circular imports
    from sklearn.isotonic import IsotonicRegression

    # Filter to valid values
    mask = (
        np.isfinite(judge_scores)
        & np.isfinite(raw_w)
        & np.isfinite(cal_w)
        & (raw_w > 0)
        & (cal_w > 0)
    )
    S = judge_scores[mask]
    W_raw = raw_w[mask]
    W_cal_actual = cal_w[mask]  # The actual calibrated weights from the system

    n = len(S)

    # Sort by judge scores
    sort_idx = np.argsort(S)
    S_sorted = S[sort_idx]
    W_raw_sorted = W_raw[sort_idx]
    W_cal_actual_sorted = W_cal_actual[sort_idx]

    # Note: We now show the ACTUAL calibrated weights from the estimator
    # The previous visualization was incorrectly computing its own isotonic regression
    # from judge scores to weights, which is NOT what the calibration does.

    # Plot raw weights vs judge scores
    if n > 2000:
        # Subsample for visibility
        step = max(1, n // 1000)
        indices = np.arange(0, n, step)
        ax.scatter(
            S_sorted[indices],
            W_raw_sorted[indices],
            s=2,
            alpha=0.2,
            color="C0",
            label="raw weights",
            rasterized=True,
        )
    else:
        ax.scatter(
            S,
            W_raw,
            s=3,
            alpha=0.3,
            color="C0",
            label="raw weights",
        )

    # Plot actual calibrated weights as thick line
    ax.plot(
        S_sorted,
        W_cal_actual_sorted,
        color="C2",
        linewidth=2.5,
        label="actual calibrated",
        zorder=10,
    )

    # Add horizontal line at y=1 (target mean)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, linewidth=1)

    # Set log scale for y-axis only
    ax.set_yscale("log")

    # Compute variance ratio for annotation
    var_ratio_actual = np.var(W_cal_actual) / np.var(W_raw) if np.var(W_raw) > 0 else 0

    # Title with comprehensive diagnostics
    ax.set_title(
        f"{policy}\n"
        f"ESS: {ess_raw:.0f}→{ess_cal:.0f} ({uplift:.1f}×), "
        f"Var ratio: {var_ratio_actual:.2f}",
        fontsize=10,
    )
    ax.set_xlabel("Judge Score", fontsize=9)
    ax.set_ylabel("Weight (log scale)", fontsize=9)
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3, which="both", linestyle=":")
    ax.tick_params(labelsize=8)


def _plot_single_policy_weight_histogram(
    ax: Any,
    raw_w: np.ndarray,
    cal_w: np.ndarray,
    policy: str,
    ess_raw: float,
    ess_cal: float,
    uplift: float,
    top1_raw: float,
    top1_cal: float,
) -> None:
    """Fallback: histogram comparison when judge scores unavailable."""

    # Create log-spaced bins
    raw_positive = raw_w[raw_w > 0]
    cal_positive = cal_w[cal_w > 0]

    if len(raw_positive) > 0 and len(cal_positive) > 0:
        min_val = min(raw_positive.min(), cal_positive.min())
        max_val = max(raw_positive.max(), cal_positive.max())
        bins = np.logspace(np.log10(max(min_val, 1e-6)), np.log10(max_val), 40)

        # Plot histograms
        ax.hist(
            raw_positive, bins=bins, alpha=0.4, color="C0", label="raw", density=True
        )
        ax.hist(
            cal_positive,
            bins=bins,
            alpha=0.6,
            color="C1",
            label="calibrated",
            density=True,
            histtype="step",
            linewidth=2,
        )

        ax.set_xscale("log")
        ax.set_xlabel("Weight", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)

    # Title with diagnostics
    ax.set_title(
        f"{policy} (no judge scores)\nESS: {ess_raw:.0f}→{ess_cal:.0f} ({uplift:.1f}×), "
        f"Top1%: {100*top1_raw:.1f}%→{100*top1_cal:.1f}%",
        fontsize=10,
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


def plot_weight_dashboard(
    raw_weights_dict: Dict[str, np.ndarray],
    calibrated_weights_dict: Optional[Dict[str, np.ndarray]] = None,
    n_samples: Optional[int] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 12),
    random_seed: int = 42,
    diagnostics: Optional[Any] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """Create production-ready weight diagnostics dashboard.

    Single figure with 6 panels showing all essential information for
    quick go/no-go decisions and debugging.

    Can work with either:
    1. Raw weight dictionaries (backward compatible)
    2. IPSDiagnostics or DRDiagnostics objects (preferred)

    Args:
        raw_weights_dict: Dict mapping policy names to raw weight arrays
        calibrated_weights_dict: Optional dict of calibrated weights
        n_samples: Total number of samples (for effective sample calculation)
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        random_seed: Random seed for reproducibility
        diagnostics: Optional IPSDiagnostics or DRDiagnostics object
        **kwargs: Additional arguments (e.g., sampler for judge scores)

    Returns:
        Tuple of (matplotlib Figure, metrics dict)
    """
    # Use the per-policy dashboard implementation
    return plot_weight_dashboard_per_policy(
        raw_weights_dict,
        calibrated_weights_dict,
        n_samples=n_samples,
        save_path=save_path,
        figsize=figsize,
        random_seed=random_seed,
        diagnostics=diagnostics,
        **kwargs,
    )


def plot_dr_dashboard(
    estimation_result: Any, figsize: Tuple[float, float] = (15, 5)
) -> Tuple[Figure, Dict[str, Any]]:
    """Create a compact 3-panel DR diagnostic dashboard.

    Panel A: DM vs IPS contributions per policy
    Panel B: Orthogonality check (score mean ± 2SE)
    Panel C: EIF tail behavior (CCDF)

    Args:
        estimation_result: Result from DR estimator with diagnostics
        figsize: Figure size (width, height)

    Returns:
        (fig, summary_metrics) tuple
    """
    # Check for DRDiagnostics object first (new way)
    from ..data.diagnostics import DRDiagnostics

    if isinstance(estimation_result.diagnostics, DRDiagnostics):
        # Use the new diagnostic object
        dr_diags = estimation_result.diagnostics.dr_diagnostics_per_policy
    elif "dr_diagnostics" in estimation_result.metadata:
        # Fallback to old way
        dr_diags = estimation_result.metadata["dr_diagnostics"]
    else:
        raise ValueError("No DR diagnostics found in estimation result")
    policies = list(dr_diags.keys())
    n_policies = len(policies)

    # Set up figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("DR Diagnostics Dashboard", fontsize=14, fontweight="bold")

    # Color palette
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, n_policies))

    # Panel A: DM vs IPS contributions
    ax = axes[0]
    x = np.arange(n_policies)
    width = 0.35

    dm_means = [dr_diags[p]["dm_mean"] for p in policies]
    ips_corrs = [dr_diags[p]["ips_corr_mean"] for p in policies]
    dr_estimates = [dr_diags[p]["dr_estimate"] for p in policies]

    bars1 = ax.bar(
        x - width / 2, dm_means, width, label="DM", color="steelblue", alpha=0.7
    )
    bars2 = ax.bar(
        x + width / 2,
        ips_corrs,
        width,
        label="IPS Correction",
        color="coral",
        alpha=0.7,
    )

    # Add DR estimate markers
    ax.scatter(
        x, dr_estimates, color="black", s=50, zorder=5, label="DR Estimate", marker="D"
    )

    ax.set_xlabel("Policy")
    ax.set_ylabel("Value")
    ax.set_title("A: Contributions (DM vs IPS)")
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Panel B: Orthogonality check
    ax = axes[1]

    # Check if we have score information (requires influence functions)
    has_scores = any("score_mean" in dr_diags[p] for p in policies)

    if not has_scores:
        # No scores available - show informative message
        ax.text(
            0.5,
            0.5,
            "Score test unavailable\n(influence functions not stored)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
            style="italic",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("B: Orthogonality Check (unavailable)")
    else:
        for i, policy in enumerate(policies):
            diag = dr_diags[policy]
            score_mean = diag.get("score_mean", 0.0)
            score_se = diag.get("score_se", 0.0)

            # Plot point with error bars (2 SE)
            ax.errorbar(
                i,
                score_mean,
                yerr=2 * score_se,
                fmt="o",
                color=colors[i],
                markersize=8,
                capsize=5,
                capthick=2,
                label=policy,
            )

            # Add p-value annotation
            p_val = diag.get("score_p", 1.0)
            if p_val < 0.05:
                ax.text(
                    i,
                    score_mean + 2.5 * score_se,
                    f"p={p_val:.3f}",
                    ha="center",
                    fontsize=8,
                    color="red",
                )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax.set_xlabel("Policy")
        ax.set_ylabel("Score Mean")
        ax.set_title("B: Orthogonality Check (mean ± 2SE)")
        ax.set_xticks(range(n_policies))
        ax.set_xticklabels(policies, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

    # Add note for TMLE
    if estimation_result.method == "tmle":
        ax.text(
            0.5,
            0.95,
            "TMLE: bars should straddle 0",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            color="gray",
        )

    # Panel C: EIF tail behavior (CCDF)
    ax = axes[2]

    # Check if actual influence functions are available
    has_empirical_ifs = False
    ifs_data = None

    # First check the first-class location (new API)
    if estimation_result.influence_functions is not None:
        ifs_data = estimation_result.influence_functions
        if ifs_data and all(policy in ifs_data for policy in policies):
            has_empirical_ifs = True
    # Fallback to legacy location in metadata
    elif "dr_influence" in estimation_result.metadata:
        ifs_data = estimation_result.metadata["dr_influence"]
        if ifs_data and all(policy in ifs_data for policy in policies):
            has_empirical_ifs = True

    if has_empirical_ifs:
        # Use empirical influence functions for exact CCDF
        for i, policy in enumerate(policies):
            ifs = ifs_data[policy]
            if isinstance(ifs, np.ndarray) and len(ifs) > 0:
                # Compute empirical CCDF
                abs_ifs = np.abs(ifs)
                sorted_ifs = np.sort(abs_ifs)[::-1]  # Descending
                ccdf = np.arange(1, len(sorted_ifs) + 1) / len(sorted_ifs)

                # Plot with appropriate sampling for large n
                if len(sorted_ifs) > 10000:
                    # Downsample for plotting efficiency
                    indices = np.logspace(
                        0, np.log10(len(sorted_ifs) - 1), 1000, dtype=int
                    )
                    ax.loglog(
                        sorted_ifs[indices],
                        ccdf[indices],
                        label=policy,
                        color=colors[i],
                        linewidth=2,
                    )
                else:
                    ax.loglog(
                        sorted_ifs, ccdf, label=policy, color=colors[i], linewidth=2
                    )

        # Add reference lines at p95 and p99
        ax.axhline(y=0.05, color="gray", linestyle=":", alpha=0.5, label="p95")
        ax.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5, label="p99")
    else:
        # No influence functions available - show message
        ax.text(
            0.5,
            0.5,
            "Influence functions not available\n(set store_influence=True)",
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_xlabel("|IF| (log scale)")
    ax.set_ylabel("CCDF (log scale)")
    ax.set_title("C: EIF Tail Behavior")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    # Compute summary metrics (handle missing fields gracefully)
    summary_metrics = {}

    # IF tail ratios
    if_tail_ratios = [d.get("if_tail_ratio_99_5", 0.0) for d in dr_diags.values()]
    if if_tail_ratios:
        summary_metrics["worst_if_tail_ratio"] = max(if_tail_ratios)

    # R² values
    r2_values = [d.get("r2_oof", np.nan) for d in dr_diags.values() if "r2_oof" in d]
    if r2_values and not all(np.isnan(r2_values)):
        valid_r2 = [r for r in r2_values if not np.isnan(r)]
        if valid_r2:
            summary_metrics["best_r2_oof"] = max(valid_r2)
            summary_metrics["worst_r2_oof"] = min(valid_r2)

    # RMSE values
    rmse_values = [
        d.get("residual_rmse", np.nan)
        for d in dr_diags.values()
        if "residual_rmse" in d
    ]
    if rmse_values and not all(np.isnan(rmse_values)):
        valid_rmse = [r for r in rmse_values if not np.isnan(r)]
        if valid_rmse:
            summary_metrics["avg_residual_rmse"] = np.mean(valid_rmse)

    if estimation_result.method == "tmle":
        score_means = [abs(d.get("score_mean", 0.0)) for d in dr_diags.values()]
        if score_means:
            summary_metrics["tmle_max_abs_score"] = max(score_means)

    return fig, summary_metrics
