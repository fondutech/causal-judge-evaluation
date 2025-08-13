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
from ..utils.diagnostics import compute_ess, diagnose_weights


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
    """Plot weight smoothing by judge score for a single policy - direct curves."""

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
    W_cal = cal_w[mask]

    n = len(S)

    # Sort by judge score for plotting curves
    sort_idx = np.argsort(S)
    S_sorted = S[sort_idx]
    W_raw_sorted = W_raw[sort_idx]
    W_cal_sorted = W_cal[sort_idx]

    # Plot raw weights as thin line or scatter depending on size
    if n > 5000:
        # For large datasets, subsample for raw weights visualization
        subsample_idx = np.random.RandomState(42).choice(n, min(2000, n), replace=False)
        subsample_idx.sort()
        ax.plot(
            S_sorted[subsample_idx],
            W_raw_sorted[subsample_idx],
            alpha=0.3,
            color="C0",
            linewidth=0.5,
            label="raw weights",
        )
    else:
        # For smaller datasets, plot all points as a line
        ax.plot(
            S_sorted,
            W_raw_sorted,
            alpha=0.4,
            color="C0",
            linewidth=1,
            label="raw weights",
        )

    # Plot calibrated weights as thicker solid line
    # This will show the staircase pattern from isotonic regression
    ax.plot(
        S_sorted,
        W_cal_sorted,
        color="C1",
        linewidth=2.5,
        label="calibrated weights",
        zorder=10,
    )

    # Optional: Add faint scatter for raw weights to show density
    if n <= 10000:
        ax.scatter(S, W_raw, s=1, alpha=0.05, color="gray", rasterized=True)

    # Set log scale for y-axis
    ax.set_yscale("log")

    # Set sensible limits
    y_cap_q = 0.999
    ymax = np.quantile(W_raw[W_raw > 0], y_cap_q)
    ymin = max(1e-3, np.quantile(W_raw[W_raw > 0], 0.001))
    ax.set_ylim(ymin, ymax * 1.5)  # Add some headroom

    # Reference line at y=1 (mean-one reference)
    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        alpha=0.5,
        linewidth=1,
        label="weight=1",
        zorder=5,
    )

    # Compute variance ratio for annotation
    var_ratio = np.var(W_cal) / np.var(W_raw) if np.var(W_raw) > 0 else 0

    # Title with comprehensive diagnostics
    ax.set_title(
        f"{policy}\n"
        f"ESS: {ess_raw:.0f}→{ess_cal:.0f} ({uplift:.1f}×), "
        f"Top1%: {100*top1_raw:.1f}%→{100*top1_cal:.1f}%, "
        f"Var ratio: {var_ratio:.2f}",
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
    # Check if we should use the new per-policy dashboard
    sampler = kwargs.get("sampler")
    judge_scores_dict = kwargs.get("judge_scores")

    # If we have judge scores, use the new per-policy dashboard
    if judge_scores_dict is not None or (
        sampler is not None and hasattr(sampler, "get_data_for_policy")
    ):
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

    # Otherwise use the legacy multi-policy dashboard
    return _plot_weight_dashboard_legacy(
        raw_weights_dict,
        calibrated_weights_dict,
        n_samples=n_samples,
        save_path=save_path,
        figsize=figsize,
        random_seed=random_seed,
        diagnostics=diagnostics,
    )


def _plot_weight_dashboard_legacy(
    raw_weights_dict: Dict[str, np.ndarray],
    calibrated_weights_dict: Optional[Dict[str, np.ndarray]] = None,
    n_samples: Optional[int] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 12),
    random_seed: int = 42,
    diagnostics: Optional[Any] = None,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """Legacy multi-policy weight dashboard (original implementation)."""
    # Check if we have a diagnostics object to use
    from ..data.diagnostics import IPSDiagnostics, DRDiagnostics

    if diagnostics is not None and isinstance(
        diagnostics, (IPSDiagnostics, DRDiagnostics)
    ):
        # Extract weight info from diagnostics - we still need the actual weight arrays
        # which aren't stored in diagnostics, so this is just for metadata enhancement
        if n_samples is None:
            n_samples = diagnostics.n_samples_valid
    policies = list(raw_weights_dict.keys())
    n_policies = len(policies)
    metrics = {}

    # Use calibrated weights if provided, otherwise use raw
    use_calibrated = calibrated_weights_dict is not None
    weights_to_plot = calibrated_weights_dict if use_calibrated else raw_weights_dict

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Infer n_samples if not provided - but track per-policy
    if n_samples is None:
        n_samples = len(next(iter(raw_weights_dict.values())))

    # Compute metrics for all policies
    for policy in policies:
        raw_w = raw_weights_dict[policy]
        cal_w = (
            calibrated_weights_dict.get(policy, raw_w)
            if calibrated_weights_dict
            else raw_w
        )

        # Track actual sample size per policy
        policy_n_samples = len(raw_w)

        # ESS metrics
        ess_raw = compute_ess(raw_w)
        ess_cal = compute_ess(cal_w)

        # Sample efficiency: how many samples contribute X% of weight
        sorted_w = np.sort(cal_w)[::-1]
        cumsum_w = np.cumsum(sorted_w)
        total_w = cumsum_w[-1]

        n_for_50 = np.searchsorted(cumsum_w, 0.5 * total_w) + 1
        n_for_90 = np.searchsorted(cumsum_w, 0.9 * total_w) + 1

        # Count extreme weights
        n_above_10 = np.sum(cal_w > 10)
        n_above_100 = np.sum(cal_w > 100)

        metrics[policy] = {
            "ess_raw": ess_raw,
            "ess_cal": ess_cal,
            "ess_raw_frac": ess_raw / policy_n_samples,
            "ess_cal_frac": ess_cal / policy_n_samples,
            "ess_improvement": ess_cal / max(ess_raw, 1e-10),
            "max_weight_raw": np.max(raw_w),
            "max_weight_cal": np.max(cal_w),
            "n_for_50pct": n_for_50,
            "n_for_90pct": n_for_90,
            "n_samples": policy_n_samples,
            "n_above_10": n_above_10,
            "n_above_100": n_above_100,
        }

    # Create figure with 3x2 grid for better breathing room
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        3, 2, hspace=0.35, wspace=0.35, left=0.08, right=0.95, top=0.94, bottom=0.06
    )

    # Row 1: Core metrics
    # Panel A: ESS Comparison (effective samples, not percentage)
    ax_ess = fig.add_subplot(gs[0, 0])
    _plot_ess_comparison_dashboard(ax_ess, metrics, policies)

    # Panel B: Maximum Weight
    ax_max = fig.add_subplot(gs[0, 1])
    _plot_max_weight_comparison(ax_max, metrics, policies)

    # Row 2: Distribution analysis
    # Panel C: Weight Histograms (simple comparison)
    ax_transform = fig.add_subplot(gs[1, 0])
    _plot_weight_histograms(
        ax_transform, raw_weights_dict, calibrated_weights_dict, policies
    )

    # Panel D: Tail Behavior (all policies on one CCDF)
    ax_tail = fig.add_subplot(gs[1, 1])
    if weights_to_plot is not None:
        _plot_tail_ccdf_combined(ax_tail, weights_to_plot, policies)

    # Row 3: Efficiency and summary
    # Panel E: Sample Efficiency
    ax_eff = fig.add_subplot(gs[2, 0])
    _plot_sample_efficiency(ax_eff, metrics, policies)

    # Panel F: Summary Table with recommendations
    ax_table = fig.add_subplot(gs[2, 1])
    _plot_summary_table(ax_table, metrics, policies, use_calibrated)

    # Main title
    title = "Weight Diagnostics Dashboard"
    if use_calibrated:
        title += " (Calibrated Weights)"
    plt.suptitle(title, fontsize=14, fontweight="bold")

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")

    return fig, metrics


def _plot_ess_comparison_dashboard(ax: Any, metrics: Dict, policies: List[str]) -> None:
    """Plot ESS as effective samples (not percentage)."""
    n_policies = len(policies)
    x = np.arange(n_policies)
    width = 0.35

    # Get values
    raw_ess = [metrics[p]["ess_raw"] for p in policies]
    cal_ess = [metrics[p]["ess_cal"] for p in policies]
    improvements = [metrics[p]["ess_improvement"] for p in policies]

    # Use consistent tab10 colormap
    colors = plt.cm.get_cmap("tab10")

    # Plot bars
    bars1 = ax.bar(
        x - width / 2, raw_ess, width, label="Raw", color=colors(0), alpha=0.7
    )
    bars2 = ax.bar(
        x + width / 2, cal_ess, width, label="Calibrated", color=colors(1), alpha=0.7
    )

    # Labels on bars with improvement factor
    for i, (r, c, imp) in enumerate(zip(raw_ess, cal_ess, improvements)):
        ax.text(i - width / 2, r + 5, f"{r:.0f}", ha="center", fontsize=8)
        # Show calibrated value and improvement
        ax.text(i + width / 2, c + 5, f"{c:.0f}", ha="center", fontsize=8)
        if imp > 1.5:  # Only show significant improvements
            ax.text(
                i + width / 2,
                c / 2,
                f"+{imp:.1f}×",
                ha="center",
                fontsize=7,
                style="italic",
                color="darkgreen",
            )

    ax.set_xticks(x)
    # Always rotate labels for consistency and to prevent overlaps
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("Effective Samples")
    ax.set_title("A. Effective Sample Size")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Add reference lines - compute based on actual per-policy n_samples
    if policies:
        # Use the minimum n_samples across policies for conservative thresholds
        min_n_samples = min(metrics[p]["n_samples"] for p in policies)
        max_n_samples = max(metrics[p]["n_samples"] for p in policies)

        # If sample sizes vary significantly, note it
        if max_n_samples > min_n_samples * 1.1:
            ax.text(
                0.02,
                0.98,
                f"n varies: {min_n_samples}-{max_n_samples}",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
            )

        # 50% line - good threshold
        ax.axhline(
            min_n_samples * 0.5, color="green", linestyle="--", alpha=0.3, linewidth=1
        )
        ax.text(
            n_policies - 0.5,
            min_n_samples * 0.5 + 10,
            "50% (good)",
            fontsize=7,
            color="green",
            alpha=0.7,
            ha="right",
        )

        # 10% line - warning threshold
        ax.axhline(
            min_n_samples * 0.1, color="orange", linestyle="--", alpha=0.3, linewidth=1
        )
        ax.text(
            n_policies - 0.5,
            min_n_samples * 0.1 + 10,
            "10% (marginal)",
            fontsize=7,
            color="orange",
            alpha=0.7,
            ha="right",
        )


def _plot_max_weight_comparison(ax: Any, metrics: Dict, policies: List[str]) -> None:
    """Plot maximum weights showing weight concentration risk."""
    n_policies = len(policies)
    x = np.arange(n_policies)
    width = 0.35

    raw_max = [metrics[p]["max_weight_raw"] for p in policies]
    cal_max = [metrics[p]["max_weight_cal"] for p in policies]

    # Use log scale if any weight > 10
    if max(raw_max + cal_max) > 10:
        ax.set_yscale("log")

    # Use consistent tab10 colormap - same as panel A
    colors = plt.cm.get_cmap("tab10")

    bars1 = ax.bar(
        x - width / 2, raw_max, width, label="Raw", color=colors(0), alpha=0.7
    )
    bars2 = ax.bar(
        x + width / 2, cal_max, width, label="Calibrated", color=colors(1), alpha=0.7
    )

    # Add value labels for all weights
    for i, (r, c) in enumerate(zip(raw_max, cal_max)):
        # Always show the values
        ax.text(
            i - width / 2,
            r * 1.1 if r > 0 else 0.1,
            f"{r:.0f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            i + width / 2,
            c * 1.1 if c > 0 else 0.1,
            f"{c:.0f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    # Reference lines with better labels
    ax.axhline(1, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(10, color="orange", linestyle="--", alpha=0.5)
    ax.axhline(100, color="red", linestyle="--", alpha=0.5)

    # Add text annotations for thresholds
    ax.text(n_policies - 0.5, 1.2, "Target", fontsize=7, color="gray", alpha=0.7)
    ax.text(n_policies - 0.5, 12, "High", fontsize=7, color="orange", alpha=0.7)
    ax.text(n_policies - 0.5, 120, "Extreme", fontsize=7, color="red", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(
        policies,
        rotation=45 if n_policies > 3 else 0,
        ha="right" if n_policies > 3 else "center",
    )
    ax.set_ylabel("Weight of Most Important Sample")
    ax.set_title("B. Weight Concentration: Single Sample Dominance")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")


def _plot_sample_efficiency(ax: Any, metrics: Dict, policies: List[str]) -> None:
    """Plot how many samples contribute 50% and 90% of weight."""
    n_policies = len(policies)

    # Prepare data
    data = []
    for p in policies:
        n_50 = metrics[p]["n_for_50pct"]
        n_90 = metrics[p]["n_for_90pct"]
        n_total = metrics[p]["n_samples"]
        n_rest = n_total - n_90

        # Percentages for stacking
        pct_50 = 100 * n_50 / n_total
        pct_90_50 = 100 * (n_90 - n_50) / n_total
        pct_rest = 100 * n_rest / n_total

        data.append((pct_50, pct_90_50, pct_rest, n_50, n_90))

    # Create stacked bars
    x = np.arange(n_policies)

    # Use consistent colors from tab10
    colors = plt.cm.get_cmap("tab10")

    # Bottom segment: samples contributing 50% weight (most important)
    bars1 = ax.bar(
        x,
        [d[0] for d in data],
        label=f"Samples carrying 50% of weight",
        color=colors(3),  # Red-ish from tab10
        alpha=0.8,
    )

    # Middle segment: samples contributing 50-90% weight
    bars2 = ax.bar(
        x,
        [d[1] for d in data],
        bottom=[d[0] for d in data],
        label="Additional samples for 90% weight",
        color=colors(1),  # Orange from tab10
        alpha=0.6,
    )

    # Top segment: remaining samples
    bottom_sum = [d[0] + d[1] for d in data]
    bars3 = ax.bar(
        x,
        [d[2] for d in data],
        bottom=bottom_sum,
        label="Samples with minimal weight (<10%)",
        color="lightgray",
        alpha=0.4,
    )

    # Add text annotations - simplified
    for i, (p50, p90_50, prest, n50, n90) in enumerate(data):
        policy = policies[i]
        n_total = metrics[policy]["n_samples"]

        # Only show the count and percentage for 50% weight segment
        # This is the most critical information
        if p50 > 3:  # Only show if segment is large enough to fit text
            label_50 = f"{n50}\n({n50/n_total*100:.0f}%)"
            ax.text(
                i,
                p50 / 2,
                label_50,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

        # Don't show the 90% total - it's too crowded

    ax.set_xticks(x)
    ax.set_xticklabels(
        policies,
        rotation=45 if n_policies > 3 else 0,
        ha="right" if n_policies > 3 else "center",
    )
    ax.set_ylabel("% of Total Samples")
    ax.set_ylim(0, 100)

    # More descriptive title
    ax.set_title("E. Sample Efficiency: How Many Samples Actually Matter?")

    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


# These functions were moved up to the top of the file


def _plot_weight_histograms(
    ax: Any, raw_weights_dict: Dict, calibrated_weights_dict: Dict, policies: List[str]
) -> None:
    """Simple histogram comparison for legacy dashboard."""
    for i, policy in enumerate(policies[:3]):  # Limit to 3 policies for clarity
        raw_w = raw_weights_dict[policy]
        cal_w = (
            calibrated_weights_dict.get(policy, raw_w)
            if calibrated_weights_dict
            else raw_w
        )

        # Create log-spaced bins
        raw_positive = raw_w[raw_w > 0]
        cal_positive = cal_w[cal_w > 0]

        if len(raw_positive) > 0 and len(cal_positive) > 0:
            min_val = min(raw_positive.min(), cal_positive.min())
            max_val = max(raw_positive.max(), cal_positive.max())
            bins = np.logspace(np.log10(max(min_val, 1e-6)), np.log10(max_val), 40)

            ax.hist(
                raw_positive, bins=bins, alpha=0.3, label=f"{policy} raw", density=True
            )
            ax.hist(
                cal_positive,
                bins=bins,
                alpha=0.5,
                label=f"{policy} cal",
                density=True,
                histtype="step",
                linewidth=2,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Density")
    ax.set_title("C. Weight Distributions")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_tail_ccdf_combined(ax: Any, weights_dict: Dict, policies: List[str]) -> None:
    """CCDF on log-log scale, all policies overlaid."""
    colors = plt.cm.get_cmap("Set2")(np.linspace(0, 1, len(policies)))

    for policy, color in zip(policies, colors):
        weights = weights_dict[policy]

        # Sort weights and compute CCDF
        w_sorted = np.sort(weights[weights > 0])
        if len(w_sorted) == 0:
            continue

        # CCDF: fraction of weights >= x
        ccdf = 1.0 - np.arange(len(w_sorted)) / len(w_sorted)

        ax.loglog(w_sorted, ccdf, label=policy, linewidth=2, alpha=0.7, color=color)

    ax.set_xlabel("Weight")
    ax.set_ylabel("P(W ≥ x)")
    ax.set_title("D. Tail Behavior (CCDF)")
    # Move legend to bottom left where there's more space
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # Add reference lines
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(10.0, color="orange", linestyle="--", alpha=0.5)
    ax.axvline(100.0, color="red", linestyle="--", alpha=0.5)


def _plot_summary_table(
    ax: Any, metrics: Dict, policies: List[str], use_calibrated: bool
) -> None:
    """Summary table with status and recommendations."""
    ax.axis("off")

    # Prepare table data
    headers = ["Policy", "ESS", "Status", "Recommendation"]
    rows = []

    for policy in policies:
        m = metrics[policy]
        ess_frac = m["ess_cal_frac"] if use_calibrated else m["ess_raw_frac"]
        ess_val = m["ess_cal"] if use_calibrated else m["ess_raw"]

        # Status based on ESS (avoid emoji for compatibility)
        if ess_frac > 0.5:
            status = "Excellent"
            rec = "Ready for production"
        elif ess_frac > 0.2:
            status = "Good"
            rec = "Usable with caution"
        elif ess_frac > 0.1:
            status = "Marginal"
            rec = "Consider more data"
        else:
            status = "Poor"
            rec = "Insufficient overlap"

        # Add calibration recommendation if relevant
        if not use_calibrated and m["ess_improvement"] > 2.0:
            rec = f"Use calibration ({m['ess_improvement']:.1f}× gain)"

        rows.append(
            [
                policy[:12],  # Truncate long names
                f"{ess_val:.0f} ({100*ess_frac:.0f}%)",
                status,
                rec[:20],  # Truncate long recommendations
            ]
        )

    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="left",
        loc="center",
        colWidths=[0.2, 0.25, 0.2, 0.35],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#E8E8E8")
        table[(0, i)].set_text_props(weight="bold")

    # Color code by status
    for i, policy in enumerate(policies):
        ess_frac = (
            metrics[policy]["ess_cal_frac"]
            if use_calibrated
            else metrics[policy]["ess_raw_frac"]
        )
        if ess_frac > 0.5:
            color = "#90EE90"  # Light green
        elif ess_frac > 0.2:
            color = "#FFFACD"  # Light yellow
        elif ess_frac > 0.1:
            color = "#FFE4B5"  # Light orange
        else:
            color = "#FFB6C1"  # Light red
        table[(i + 1, 2)].set_facecolor(color)


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
    if "dr_influence" in estimation_result.metadata:
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
    else:
        # Fallback: plot quantile markers only (no synthetic curves)
        for i, policy in enumerate(policies):
            diag = dr_diags[policy]
            if_p95 = diag["if_p95"]
            if_p99 = diag["if_p99"]

            # Draw vertical lines at quantiles
            ax.axvline(if_p95, color=colors[i], linestyle="--", alpha=0.5, linewidth=1)
            ax.axvline(if_p99, color=colors[i], linestyle=":", alpha=0.5, linewidth=1)

            # Add text labels
            ax.text(
                if_p95,
                0.05,
                f"{policy}\np95",
                rotation=45,
                fontsize=7,
                color=colors[i],
                alpha=0.7,
            )
            ax.text(
                if_p99,
                0.01,
                f"p99",
                rotation=45,
                fontsize=7,
                color=colors[i],
                alpha=0.7,
            )

    # Add reference lines at p95 and p99
    ax.axhline(y=0.05, color="gray", linestyle=":", alpha=0.5, label="p95")
    ax.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5, label="p99")

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
