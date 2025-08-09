"""Consolidated visualization module for weight diagnostics.

Primary: Dashboard (plot_weight_dashboard) - Production/debugging decisions
Secondary: Research plots - Available but not emphasized
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path

# Import shared utilities
from .diagnostics import compute_ess, diagnose_weights


def plot_weight_dashboard(
    raw_weights_dict: Dict[str, np.ndarray],
    calibrated_weights_dict: Optional[Dict[str, np.ndarray]] = None,
    n_samples: Optional[int] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 12),
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """Create production-ready weight diagnostics dashboard.

    Single figure with 6 panels showing all essential information for
    quick go/no-go decisions and debugging.

    Args:
        raw_weights_dict: Dict mapping policy names to raw weight arrays
        calibrated_weights_dict: Optional dict of calibrated weights
        n_samples: Total number of samples (for effective sample calculation)
        save_path: Optional path to save figure
        figsize: Figure size (width, height)

    Returns:
        Tuple of (matplotlib Figure, metrics dict)
    """
    policies = list(raw_weights_dict.keys())
    n_policies = len(policies)
    metrics = {}

    # Use calibrated weights if provided, otherwise use raw
    use_calibrated = calibrated_weights_dict is not None
    weights_to_plot = calibrated_weights_dict if use_calibrated else raw_weights_dict

    # Infer n_samples if not provided
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

        # ESS metrics
        ess_raw = compute_ess(raw_w)
        ess_cal = compute_ess(cal_w)

        # Sample efficiency: how many samples contribute X% of weight
        sorted_w = np.sort(cal_w)[::-1]
        cumsum_w = np.cumsum(sorted_w)
        total_w = cumsum_w[-1]

        n_for_50 = np.searchsorted(cumsum_w, 0.5 * total_w) + 1
        n_for_90 = np.searchsorted(cumsum_w, 0.9 * total_w) + 1

        metrics[policy] = {
            "ess_raw": ess_raw,
            "ess_cal": ess_cal,
            "ess_raw_frac": ess_raw / n_samples,
            "ess_cal_frac": ess_cal / n_samples,
            "ess_improvement": ess_cal / max(ess_raw, 1e-10),
            "max_weight_raw": np.max(raw_w),
            "max_weight_cal": np.max(cal_w),
            "n_for_50pct": n_for_50,
            "n_for_90pct": n_for_90,
            "n_samples": n_samples,
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
    # Panel C: Weight Transformation (shows calibration effect)
    ax_transform = fig.add_subplot(gs[1, 0])
    if calibrated_weights_dict:
        _plot_weight_transformation(
            ax_transform, raw_weights_dict, calibrated_weights_dict, policies
        )
    else:
        # If no calibration, show raw weights only
        _plot_weight_transformation(
            ax_transform, raw_weights_dict, raw_weights_dict, policies
        )

    # Panel D: Tail Behavior (all policies on one CCDF)
    ax_tail = fig.add_subplot(gs[1, 1])
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

    # Plot bars - use same colors as weight concentration plot
    # Raw bars in coral (same as Panel B)
    bars1 = ax.bar(x - width / 2, raw_ess, width, label="Raw", color="coral", alpha=0.7)

    # Calibrated bars in light green (same as Panel B)
    bars2 = ax.bar(
        x + width / 2, cal_ess, width, label="Calibrated", color="lightgreen", alpha=0.7
    )

    # Remove improvement text - just show the values on the bars

    # Labels on bars
    for i, (r, c) in enumerate(zip(raw_ess, cal_ess)):
        ax.text(i - width / 2, r + 5, f"{r:.0f}", ha="center", fontsize=8)
        ax.text(i + width / 2, c + 5, f"{c:.0f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    # Always rotate labels for consistency and to prevent overlaps
    ax.set_xticklabels(policies, rotation=45, ha="right")
    ax.set_ylabel("Effective Samples")
    ax.set_title("A. Effective Sample Size")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Add reference lines with labels
    if policies:
        n_samples = metrics[policies[0]]["n_samples"]

        # 50% line - good threshold
        ax.axhline(
            n_samples * 0.5, color="green", linestyle="--", alpha=0.3, linewidth=1
        )
        ax.text(
            n_policies - 0.5,
            n_samples * 0.5 + 10,
            "50% (good)",
            fontsize=7,
            color="green",
            alpha=0.7,
            ha="right",
        )

        # 10% line - warning threshold
        ax.axhline(
            n_samples * 0.1, color="orange", linestyle="--", alpha=0.3, linewidth=1
        )
        ax.text(
            n_policies - 0.5,
            n_samples * 0.1 + 10,
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

    bars1 = ax.bar(x - width / 2, raw_max, width, label="Raw", color="coral", alpha=0.7)
    bars2 = ax.bar(
        x + width / 2, cal_max, width, label="Calibrated", color="lightgreen", alpha=0.7
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

    # Bottom segment: samples contributing 50% weight (most important)
    bars1 = ax.bar(
        x,
        [d[0] for d in data],
        label=f"Samples carrying 50% of weight",
        color="darkred",
        alpha=0.8,
    )

    # Middle segment: samples contributing 50-90% weight
    bars2 = ax.bar(
        x,
        [d[1] for d in data],
        bottom=[d[0] for d in data],
        label="Additional samples for 90% weight",
        color="orange",
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


def _plot_weight_transformation(
    ax: Any, raw_weights_dict: Dict, calibrated_weights_dict: Dict, policies: List[str]
) -> None:
    """Plot weight transformation showing calibration effect for all policies."""
    colors = plt.cm.Set2(np.linspace(0, 1, len(policies)))
    markers = ["o", "s", "^", "D"]  # Different markers for each policy

    for i, (policy, color) in enumerate(zip(policies, colors)):
        raw_w = raw_weights_dict[policy]
        cal_w = calibrated_weights_dict[policy]

        # Clip to small positive value to avoid log(0)
        cal_w_clipped = np.maximum(cal_w, 1e-10)

        # Use scatter plot to show density of points
        # Smaller points with some transparency to see overlaps
        ax.scatter(
            raw_w,
            cal_w_clipped,
            label=policy,
            color=color,
            alpha=0.5,
            s=10,
            marker=markers[i % len(markers)],
            edgecolors="none",
        )

    # Use log scale for both axes - shows full range better
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Set reasonable axis limits
    ax.set_xlim(1e-40, 1e4)  # Raw weights range
    ax.set_ylim(1e-10, 1e2)  # Calibrated weights range

    # Add reference lines
    ax.axhline(
        1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="y=1 (target)"
    )

    # Add diagonal for reference (y=x means no transformation)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Plot y=x line across the full range
    diag_min = max(xlim[0], ylim[0])
    diag_max = min(xlim[1], ylim[1])
    ax.plot(
        [diag_min, diag_max],
        [diag_min, diag_max],
        "k--",
        alpha=0.3,
        linewidth=1,
        label="y=x (no change)",
    )

    ax.set_xlabel("Raw Weight (log scale)")
    ax.set_ylabel("Calibrated Weight (log scale)")
    ax.set_title("C. Weight Transformation (log-log)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")


def _plot_tail_ccdf_combined(ax: Any, weights_dict: Dict, policies: List[str]) -> None:
    """CCDF on log-log scale, all policies overlaid."""
    colors = plt.cm.Set2(np.linspace(0, 1, len(policies)))

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

    ax.set_title("F. Summary & Recommendations", fontsize=10, fontweight="bold", pad=10)


def plot_calibration_comparison(
    judge_scores: np.ndarray,
    oracle_labels: np.ndarray,
    calibrated_scores: Optional[np.ndarray] = None,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot calibration comparison (reliability diagram) for judge scores.

    Shows both raw and calibrated judge scores against oracle labels,
    with quantitative metrics for calibration quality.

    Args:
        judge_scores: Raw judge scores
        oracle_labels: True oracle labels
        calibrated_scores: Calibrated judge scores (optional)
        n_bins: Number of bins for grouping
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute calibration metrics
    def compute_calibration_error(
        predictions: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, float]:
        """Compute Expected Calibration Error (ECE) and RMSE."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1

        ece = 0.0
        total_samples = 0
        squared_errors = []

        for i in range(n_bins):
            mask = bin_indices == i
            n_in_bin = mask.sum()
            if n_in_bin > 0:
                pred_in_bin = predictions[mask].mean()
                true_in_bin = labels[mask].mean()

                # ECE: weighted average of bin-wise calibration errors
                ece += n_in_bin * abs(pred_in_bin - true_in_bin)
                total_samples += n_in_bin

                # For RMSE
                squared_errors.extend((predictions[mask] - labels[mask]) ** 2)

        ece = ece / total_samples if total_samples > 0 else 0.0
        rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else 0.0

        return ece, rmse

    # Bin the scores
    bins = np.linspace(0, 1, n_bins + 1)

    # Plot raw scores
    bin_indices = np.digitize(judge_scores, bins) - 1
    mean_pred_raw = []
    mean_true_raw = []
    counts_raw = []

    for i in range(n_bins):
        mask = bin_indices == i
        n_in_bin = mask.sum()
        if n_in_bin > 0:
            mean_pred_raw.append(judge_scores[mask].mean())
            mean_true_raw.append(oracle_labels[mask].mean())
            counts_raw.append(n_in_bin)

    # Size points by number of samples in bin
    sizes_raw = [min(200, 20 + 180 * c / len(judge_scores)) for c in counts_raw]

    ax.scatter(
        mean_pred_raw,
        mean_true_raw,
        label="Raw Judge",
        s=sizes_raw,
        alpha=0.7,
        color="coral",
        edgecolors="darkred",
        linewidth=1,
    )
    ax.plot(mean_pred_raw, mean_true_raw, "-", alpha=0.5, color="coral")

    # Compute raw metrics
    ece_raw, rmse_raw = compute_calibration_error(judge_scores, oracle_labels)

    # Plot calibrated scores if provided
    if calibrated_scores is not None:
        bin_indices_cal = np.digitize(calibrated_scores, bins) - 1
        mean_pred_cal = []
        mean_true_cal = []
        counts_cal = []

        for i in range(n_bins):
            mask = bin_indices_cal == i
            n_in_bin = mask.sum()
            if n_in_bin > 0:
                mean_pred_cal.append(calibrated_scores[mask].mean())
                mean_true_cal.append(oracle_labels[mask].mean())
                counts_cal.append(n_in_bin)

        # Size points by number of samples in bin
        sizes_cal = [
            min(200, 20 + 180 * c / len(calibrated_scores)) for c in counts_cal
        ]

        ax.scatter(
            mean_pred_cal,
            mean_true_cal,
            label="Calibrated Judge",
            s=sizes_cal,
            alpha=0.7,
            color="lightgreen",
            edgecolors="darkgreen",
            linewidth=1,
        )
        ax.plot(mean_pred_cal, mean_true_cal, "-", alpha=0.5, color="lightgreen")

        # Compute calibrated metrics
        ece_cal, rmse_cal = compute_calibration_error(calibrated_scores, oracle_labels)

        # Add improvement metrics to plot
        improvement_text = (
            f"Calibration Improvement:\n"
            f"ECE: {ece_raw:.3f} → {ece_cal:.3f} ({100*(ece_raw-ece_cal)/ece_raw:.0f}% ↓)\n"
            f"RMSE: {rmse_raw:.3f} → {rmse_cal:.3f} ({100*(rmse_raw-rmse_cal)/rmse_raw:.0f}% ↓)"
        )
    else:
        # Only raw metrics
        improvement_text = (
            f"Raw Judge Metrics:\n" f"ECE: {ece_raw:.3f}\n" f"RMSE: {rmse_raw:.3f}"
        )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Perfect calibration")

    # Add shaded region for ±0.1 calibration error
    x_perfect = np.linspace(0, 1, 100)
    ax.fill_between(
        x_perfect,
        x_perfect - 0.1,
        x_perfect + 0.1,
        alpha=0.1,
        color="gray",
        label="±0.1 tolerance",
    )

    # Add metrics text box
    ax.text(
        0.05,
        0.95,
        improvement_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Labels and formatting
    ax.set_xlabel("Mean Predicted Score")
    ax.set_ylabel("Mean Oracle Score")
    ax.set_title("Judge Calibration Comparison")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Add note about point sizes
    ax.text(
        0.95,
        0.05,
        "Point size ∝ samples in bin",
        transform=ax.transAxes,
        fontsize=8,
        horizontalalignment="right",
        alpha=0.6,
    )

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")

    return fig
