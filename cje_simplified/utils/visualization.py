"""Improved visualization functions with higher information density."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# Import shared utilities
from .weight_diagnostics import compute_ess, diagnose_weights


def plot_weight_calibration_analysis(
    raw_weights_dict: Dict[str, np.ndarray],
    calibrated_weights_dict: Dict[str, np.ndarray],
    sample_data: Optional[Dict[str, Any]] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (18, 12),
) -> plt.Figure:
    """Create comprehensive weight calibration analysis with 6 panels per policy.

    Args:
        raw_weights_dict: Dict mapping policy names to raw weight arrays
        calibrated_weights_dict: Dict mapping policy names to calibrated weight arrays
        sample_data: Optional dict with sample details for diagnostics
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_policies = len(raw_weights_dict)

    # Create figure with custom layout - further increased height for better spacing
    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_policies / 2.2))

    # Add spacing for title
    fig.subplots_adjust(top=0.96)

    # Create grid for each policy (2 rows x 3 cols per policy)
    for policy_idx, policy_name in enumerate(raw_weights_dict.keys()):
        raw_weights = raw_weights_dict[policy_name]
        cal_weights = calibrated_weights_dict.get(
            policy_name, np.ones_like(raw_weights)
        )

        # Create subplot grid for this policy
        # Calculate vertical position for this policy section
        section_height = 0.22  # Height per policy section (further reduced)
        section_top = 0.88 - policy_idx * (
            section_height + 0.12
        )  # Much larger gap between policies
        section_bottom = section_top - section_height

        gs = gridspec.GridSpec(
            2,
            3,
            left=0.05,
            right=0.95,
            top=section_top,
            bottom=section_bottom,
            wspace=0.4,
            hspace=0.55,
        )  # Increased spacing between rows

        # Add policy title
        fig.text(
            0.5,
            section_top + 0.02,
            f"Policy: {policy_name}",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )

        # Panel A: Transformation scatter
        ax_transform = fig.add_subplot(gs[0, 0])
        _plot_transformation_scatter(ax_transform, raw_weights, cal_weights)

        # Panel B: Distribution overlay
        ax_dist = fig.add_subplot(gs[0, 1])
        _plot_distribution_overlay(ax_dist, raw_weights, cal_weights)

        # Panel C: ESS & Variance comparison
        ax_metrics = fig.add_subplot(gs[0, 2])
        _plot_metrics_comparison(ax_metrics, raw_weights, cal_weights)

        # Panel D: Percentile analysis
        ax_percentile = fig.add_subplot(gs[1, 0])
        _plot_percentile_analysis(ax_percentile, raw_weights, cal_weights)

        # Panel E: Sample diagnostics
        ax_samples = fig.add_subplot(gs[1, 1])
        _plot_sample_diagnostics(
            ax_samples, raw_weights, cal_weights, policy_name, sample_data
        )

        # Panel F: Statistics table
        ax_table = fig.add_subplot(gs[1, 2])
        _plot_statistics_table(ax_table, raw_weights, cal_weights)

    plt.suptitle("Weight Calibration Analysis", fontsize=16, y=0.97)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _plot_transformation_scatter(ax, raw_weights, cal_weights):
    """Panel A: Raw to calibrated transformation scatter plot."""
    # Filter valid weights
    finite_mask = np.isfinite(raw_weights) & (raw_weights > 0)
    raw_finite = raw_weights[finite_mask]
    cal_finite = cal_weights[finite_mask]

    if len(raw_finite) > 0:
        # Sort for better visualization
        sort_idx = np.argsort(raw_finite)
        raw_sorted = raw_finite[sort_idx]
        cal_sorted = cal_finite[sort_idx]

        # Color by density
        colors = plt.cm.viridis(np.linspace(0, 1, len(raw_sorted)))
        scatter = ax.scatter(
            raw_sorted,
            cal_sorted,
            c=colors,
            alpha=0.6,
            s=20,
            edgecolors="black",
            linewidth=0.5,
        )

        # Reference lines
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="Mean=1")

        # Monotonicity check - highlight violations
        violations = []
        for i in range(len(raw_sorted) - 1):
            if cal_sorted[i] > cal_sorted[i + 1] + 1e-10:  # Allow small numerical error
                violations.append(i)

        if violations:
            ax.scatter(
                raw_sorted[violations],
                cal_sorted[violations],
                color="red",
                s=100,
                marker="x",
                label=f"{len(violations)} violations",
            )

        ax.set_xscale("log")
        ax.set_xlabel("Raw Weight (log scale)", fontsize=9)
        ax.set_ylabel("Calibrated Weight", fontsize=9)
        ax.set_title("A. Transformation", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No valid weights", ha="center", va="center")
        ax.set_title("A. Transformation", fontsize=10)


def _plot_distribution_overlay(ax, raw_weights, cal_weights):
    """Panel B: Distribution overlay comparing raw and calibrated."""
    # Create two y-axes for different scales
    ax2 = ax.twinx()

    # Raw weights histogram (log scale)
    finite_raw = raw_weights[np.isfinite(raw_weights) & (raw_weights > 0)]
    if len(finite_raw) > 0:
        log_raw = np.log10(finite_raw + 1e-10)
        n1, bins1, _ = ax.hist(
            log_raw, bins=30, alpha=0.5, color="blue", label="Raw (log₁₀)", density=True
        )
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Density\n(Raw)", color="blue", fontsize=9)
        ax.tick_params(axis="y", labelcolor="blue")

    # Calibrated weights histogram (linear scale)
    n2, bins2, _ = ax2.hist(
        cal_weights, bins=30, alpha=0.5, color="green", label="Calibrated", density=True
    )
    ax2.set_ylabel("Density\n(Calib.)", color="green", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="green")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax.set_title("B. Distribution Overlay", fontsize=10)
    ax.grid(True, alpha=0.3)


def _plot_metrics_comparison(ax, raw_weights, cal_weights):
    """Panel C: ESS and variance before/after comparison."""
    # Calculate metrics
    raw_ess = compute_ess(raw_weights)
    cal_ess = compute_ess(cal_weights)
    raw_ess_pct = raw_ess / len(raw_weights) * 100
    cal_ess_pct = cal_ess / len(cal_weights) * 100

    raw_var = np.var(raw_weights)
    cal_var = np.var(cal_weights)

    # Create grouped bar chart
    metrics = ["ESS %", "Log₁₀ Var"]
    raw_values = [raw_ess_pct, np.log10(raw_var + 1e-10)]
    cal_values = [cal_ess_pct, np.log10(cal_var + 1e-10)]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, raw_values, width, label="Raw", alpha=0.7, color="coral"
    )
    bars2 = ax.bar(
        x + width / 2,
        cal_values,
        width,
        label="Calibrated",
        alpha=0.7,
        color="lightgreen",
    )

    # Add value labels
    for bar, val in zip(bars1, raw_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, val in zip(bars2, cal_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add improvement annotations
    ess_improve = cal_ess_pct / raw_ess_pct if raw_ess_pct > 0 else 0
    var_reduce = (1 - cal_var / raw_var) * 100 if raw_var > 0 else 0

    ax.text(
        0,
        max(raw_values[0], cal_values[0]) * 1.1,
        f"{ess_improve:.1f}x",
        ha="center",
        fontsize=10,
        color="green",
    )
    ax.text(
        1,
        min(raw_values[1], cal_values[1]) - 0.5,
        f"-{var_reduce:.0f}%",
        ha="center",
        fontsize=10,
        color="green",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("C. ESS & Variance", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")


def _plot_percentile_analysis(ax, raw_weights, cal_weights):
    """Panel D: Weight by percentile rank analysis."""
    # Sort weights and get percentiles
    n = len(raw_weights)
    percentiles = np.linspace(0, 100, n)

    raw_sorted = np.sort(raw_weights)
    cal_sorted = np.sort(cal_weights)

    # Plot weight curves
    ax.plot(percentiles, raw_sorted, "b-", alpha=0.7, label="Raw", linewidth=2)
    ax.plot(percentiles, cal_sorted, "g-", alpha=0.7, label="Calibrated", linewidth=2)

    # Shade regions
    ax.fill_between(
        percentiles[: int(n * 0.1)],
        0,
        max(cal_sorted.max(), 10),
        alpha=0.2,
        color="red",
        label="Bottom 10%",
    )
    ax.fill_between(
        percentiles[int(n * 0.9) :],
        0,
        max(cal_sorted.max(), 10),
        alpha=0.2,
        color="blue",
        label="Top 10%",
    )

    # Reference line
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Percentile", fontsize=9)
    ax.set_ylabel("Weight", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-10)
    ax.set_title("D. Percentile Analysis", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_sample_diagnostics(ax, raw_weights, cal_weights, policy_name, sample_data):
    """Panel E: Extreme sample diagnostics."""
    ax.axis("off")

    # Find extreme samples
    n_show = min(5, len(raw_weights))
    highest_idx = np.argsort(raw_weights)[-n_show:][::-1]
    lowest_idx = np.argsort(raw_weights)[:n_show]

    # Create text summary
    text_lines = ["E. Sample Diagnostics\n", "-" * 25 + "\n"]

    text_lines.append(f"Top {n_show} weights:\n")
    for i, idx in enumerate(highest_idx, 1):
        text_lines.append(
            f"{i}. Sample {idx}: {raw_weights[idx]:.2e} → {cal_weights[idx]:.2f}\n"
        )

    text_lines.append(f"\nBottom {n_show} weights:\n")
    for i, idx in enumerate(lowest_idx, 1):
        text_lines.append(
            f"{i}. Sample {idx}: {raw_weights[idx]:.2e} → {cal_weights[idx]:.2f}\n"
        )

    # Display text
    ax.text(
        0.05,
        0.95,
        "".join(text_lines),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
    )


def _plot_statistics_table(ax, raw_weights, cal_weights):
    """Panel F: Comprehensive statistics table."""
    ax.axis("off")

    # Calculate all statistics
    stats = {
        "Metric": [
            "Mean",
            "Median",
            "Std Dev",
            "Min",
            "Max",
            "ESS %",
            "Zero weights",
            "Clipped",
        ],
        "Raw": [],
        "Calibrated": [],
    }

    # Raw stats
    finite_raw = raw_weights[np.isfinite(raw_weights)]
    stats["Raw"] = [
        f"{np.mean(finite_raw):.2e}",
        f"{np.median(finite_raw):.2e}",
        f"{np.std(finite_raw):.2e}",
        f"{np.min(finite_raw):.2e}",
        f"{np.max(finite_raw):.2e}",
        f"{compute_ess(raw_weights)/len(raw_weights)*100:.1f}%",
        str(np.sum(raw_weights < 1e-10)),
        str(np.sum(raw_weights >= 100)),
    ]

    # Calibrated stats
    stats["Calibrated"] = [
        f"{np.mean(cal_weights):.3f}",
        f"{np.median(cal_weights):.3f}",
        f"{np.std(cal_weights):.3f}",
        f"{np.min(cal_weights):.3f}",
        f"{np.max(cal_weights):.3f}",
        f"{compute_ess(cal_weights)/len(cal_weights)*100:.1f}%",
        str(np.sum(cal_weights < 1e-10)),
        str(np.sum(cal_weights >= 100)),
    ]

    # Create table
    table = ax.table(
        cellText=[stats["Raw"], stats["Calibrated"]],
        rowLabels=["Raw", "Calibrated"],
        colLabels=stats["Metric"],
        cellLoc="center",
        loc="center",
        colWidths=[0.12] * len(stats["Metric"]),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    # Color cells
    for i in range(len(stats["Metric"])):
        table[(0, i)].set_facecolor("#E8E8E8")  # Header
        table[(1, i)].set_facecolor("#FFE6E6")  # Raw row
        table[(2, i)].set_facecolor("#E6FFE6")  # Calibrated row

    ax.set_title("F. Statistics Table", fontsize=10, pad=20)


def plot_weight_diagnostics_summary(
    raw_weights_dict: Dict[str, np.ndarray],
    calibrated_weights_dict: Dict[str, np.ndarray],
    estimates: Optional[Dict[str, float]] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """Create cross-policy weight diagnostics summary dashboard.

    Args:
        raw_weights_dict: Dict mapping policy names to raw weight arrays
        calibrated_weights_dict: Dict mapping policy names to calibrated weight arrays
        estimates: Optional dict of policy estimates with confidence intervals
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)

    # Create grid layout with more spacing
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.5)

    # Panel A: ESS Comparison (top left, 1x1)
    ax_ess = fig.add_subplot(gs[0, 0])
    _plot_ess_comparison_grouped(ax_ess, raw_weights_dict, calibrated_weights_dict)

    # Panel B: Variance Reduction Heatmap (top middle, 1x1)
    ax_heatmap = fig.add_subplot(gs[0, 1])
    _plot_variance_reduction_heatmap(
        ax_heatmap, raw_weights_dict, calibrated_weights_dict
    )

    # Panel C: Policy Estimates (top right, 1x1)
    ax_estimates = fig.add_subplot(gs[0, 2])
    _plot_policy_estimates(ax_estimates, estimates, list(raw_weights_dict.keys()))

    # Panel D: Weight Distribution Violins (middle, full width)
    ax_violins = fig.add_subplot(gs[1, :])
    _plot_weight_distribution_violins(
        ax_violins, raw_weights_dict, calibrated_weights_dict
    )

    # Panel E: Extreme Weights Summary (bottom left, 1x1)
    ax_extremes = fig.add_subplot(gs[2, 0])
    _plot_extreme_weights_summary(
        ax_extremes, raw_weights_dict, calibrated_weights_dict
    )

    # Panel F: Weight Quality Indicators (bottom middle, 1x1)
    ax_quality = fig.add_subplot(gs[2, 1])
    _plot_weight_quality_indicators(
        ax_quality, raw_weights_dict, calibrated_weights_dict
    )

    # Panel G: Summary Statistics (bottom right, 1x1)
    ax_summary = fig.add_subplot(gs[2, 2])
    _plot_summary_statistics(ax_summary, raw_weights_dict, calibrated_weights_dict)

    plt.suptitle("Weight Diagnostics Summary", fontsize=16, fontweight="bold", y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _plot_ess_comparison_grouped(ax, raw_dict, cal_dict):
    """Panel A: Grouped bar chart comparing raw vs calibrated ESS."""
    policies = list(raw_dict.keys())
    n_policies = len(policies)

    raw_ess = [compute_ess(raw_dict[p]) / len(raw_dict[p]) * 100 for p in policies]
    cal_ess = [compute_ess(cal_dict[p]) / len(cal_dict[p]) * 100 for p in policies]

    x = np.arange(n_policies)
    width = 0.35

    bars1 = ax.bar(x - width / 2, raw_ess, width, label="Raw", color="coral", alpha=0.7)
    bars2 = ax.bar(
        x + width / 2, cal_ess, width, label="Calibrated", color="lightgreen", alpha=0.7
    )

    # Add value labels and improvement
    for i, (r, c) in enumerate(zip(raw_ess, cal_ess)):
        ax.text(i - width / 2, r + 1, f"{r:.1f}%", ha="center", fontsize=8)
        ax.text(i + width / 2, c + 1, f"{c:.1f}%", ha="center", fontsize=8)
        # Improvement arrow
        if c > r:
            ax.annotate(
                "",
                xy=(i + width / 2, c),
                xytext=(i - width / 2, r),
                arrowprops=dict(arrowstyle="->", color="green", alpha=0.5),
            )

    # Reference lines
    ax.axhline(10, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(50, color="green", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(
        policies,
        rotation=0 if len(policies) <= 3 else 45,
        ha="center" if len(policies) <= 3 else "right",
    )
    ax.set_ylabel("ESS %")
    ax.set_title("A. ESS Comparison")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")


def _plot_variance_reduction_heatmap(ax, raw_dict, cal_dict):
    """Panel B: Heatmap showing variance reduction metrics."""
    policies = list(raw_dict.keys())
    metrics = ["ESS Gain", "Var Reduce", "Range Compress"]

    # Calculate metrics
    data = []
    for policy in policies:
        raw_w = raw_dict[policy]
        cal_w = cal_dict[policy]

        ess_gain = compute_ess(cal_w) / max(compute_ess(raw_w), 1e-10)
        var_reduce = 1 - np.var(cal_w) / max(np.var(raw_w), 1e-10)

        finite_raw = raw_w[np.isfinite(raw_w) & (raw_w > 0)]
        range_compress = np.log10(max(finite_raw.max() / max(cal_w.max(), 1e-10), 1))

        data.append([ess_gain, var_reduce, range_compress])

    # Create heatmap
    im = ax.imshow(np.array(data).T, cmap="RdYlGn", aspect="auto", vmin=0)

    # Labels
    ax.set_xticks(np.arange(len(policies)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(
        policies,
        rotation=0 if len(policies) <= 3 else 45,
        ha="center" if len(policies) <= 3 else "right",
    )
    ax.set_yticklabels(metrics)

    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(policies)):
            text = ax.text(
                j,
                i,
                f"{data[j][i]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    ax.set_title("B. Improvement Metrics")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_policy_estimates(ax, estimates, policies):
    """Panel C: Policy estimates with confidence intervals."""
    if not estimates:
        ax.text(0.5, 0.5, "No estimates provided", ha="center", va="center")
        ax.set_title("C. Policy Estimates")
        return

    # Extract estimates and CIs
    y_pos = np.arange(len(policies))
    means = [estimates.get(p, {}).get("mean", 0) for p in policies]
    ci_lower = [estimates.get(p, {}).get("ci_lower", 0) for p in policies]
    ci_upper = [estimates.get(p, {}).get("ci_upper", 0) for p in policies]

    # Plot
    ax.barh(y_pos, means, alpha=0.7, color="steelblue")
    ax.errorbar(
        means,
        y_pos,
        xerr=[
            [m - l for m, l in zip(means, ci_lower)],
            [u - m for u, m in zip(ci_upper, means)],
        ],
        fmt="none",
        color="black",
        capsize=5,
    )

    # Best policy
    best_idx = np.argmax(means)
    ax.barh(
        best_idx,
        means[best_idx],
        alpha=0.9,
        color="gold",
        edgecolor="orange",
        linewidth=2,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(policies)
    ax.set_xlabel("Estimate")
    ax.set_title("C. Policy Estimates")
    ax.grid(True, alpha=0.3, axis="x")


def _plot_weight_distribution_violins(ax, raw_dict, cal_dict):
    """Panel D: Violin plots comparing weight distributions."""
    policies = list(raw_dict.keys())

    # Prepare data for violins
    raw_data = []
    cal_data = []
    positions = []

    # Increase spacing between policies
    spacing = 4  # Increased from 3

    for i, policy in enumerate(policies):
        raw_w = raw_dict[policy]
        cal_w = cal_dict[policy]

        # Log transform raw weights for better visualization
        finite_raw = raw_w[np.isfinite(raw_w) & (raw_w > 0)]
        if len(finite_raw) > 0:
            raw_data.append(np.log10(finite_raw + 1e-10))
        else:
            raw_data.append([0])

        cal_data.append(cal_w)
        positions.extend([i * spacing, i * spacing + 1])

    # Create split violins
    parts_raw = ax.violinplot(
        raw_data,
        positions=[i * spacing for i in range(len(policies))],
        widths=1.2,
        showmeans=True,
    )
    parts_cal = ax.violinplot(
        cal_data,
        positions=[i * spacing + 1.5 for i in range(len(policies))],
        widths=1.2,
        showmeans=True,
    )

    # Color violins
    for pc in parts_raw["bodies"]:
        pc.set_facecolor("coral")
        pc.set_alpha(0.7)
    for pc in parts_cal["bodies"]:
        pc.set_facecolor("lightgreen")
        pc.set_alpha(0.7)

    # Labels - adjust for new spacing
    ax.set_xticks([i * spacing + 0.75 for i in range(len(policies))])
    ax.set_xticklabels(policies, rotation=0, ha="center")
    ax.set_ylabel("Weight (Raw: log₁₀, Cal: linear)")
    ax.set_title("D. Weight Distributions")

    # Legend
    ax.plot([], [], color="coral", label="Raw (log₁₀)", linewidth=10, alpha=0.7)
    ax.plot([], [], color="lightgreen", label="Calibrated", linewidth=10, alpha=0.7)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")


def _plot_extreme_weights_summary(ax, raw_dict, cal_dict):
    """Panel E: Summary of extreme weights across policies."""
    ax.axis("off")

    # Collect extreme weight counts
    headers = ["Policy", "Near-zero", "Clipped", "Top 10%"]
    data = []

    for policy in raw_dict.keys():
        raw_w = raw_dict[policy]
        n_zero = np.sum(raw_w < 1e-10)
        n_clip = np.sum(raw_w >= 100)
        threshold_90 = np.percentile(raw_w, 90)
        n_top10 = np.sum(raw_w > threshold_90)

        data.append([policy[:10], str(n_zero), str(n_clip), str(n_top10)])

    # Create table
    table = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.3, 0.2, 0.2, 0.2],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor("#E8E8E8")

    ax.set_title("E. Extreme Weights", pad=20)


def _plot_weight_quality_indicators(ax, raw_dict, cal_dict):
    """Panel F: Weight quality indicators."""
    policies = list(raw_dict.keys())

    # Calculate quality scores (0-100)
    quality_scores = []
    for policy in policies:
        raw_w = raw_dict[policy]
        cal_w = cal_dict[policy]

        # ESS score (0-100)
        ess_score = min(compute_ess(cal_w) / len(cal_w) * 100, 100)

        # Variance score (lower is better, inverted)
        var_score = max(0, 100 - np.log10(np.var(cal_w) + 1) * 20)

        # Balance score (how close to uniform)
        balance_score = max(0, 100 - np.std(cal_w) * 50)

        overall = (ess_score + var_score + balance_score) / 3
        quality_scores.append(overall)

    # Create bar chart with color coding
    colors = [
        "red" if s < 33 else "orange" if s < 66 else "green" for s in quality_scores
    ]

    bars = ax.bar(range(len(policies)), quality_scores, color=colors, alpha=0.7)

    # Add value labels
    for bar, score in zip(bars, quality_scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{score:.0f}",
            ha="center",
            fontsize=9,
        )

    # Reference lines
    ax.axhline(33, color="red", linestyle="--", alpha=0.3)
    ax.axhline(66, color="orange", linestyle="--", alpha=0.3)

    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(
        policies,
        rotation=0 if len(policies) <= 3 else 45,
        ha="center" if len(policies) <= 3 else "right",
    )
    ax.set_ylabel("Quality Score")
    ax.set_ylim(0, 105)
    ax.set_title("F. Weight Quality")
    ax.grid(True, alpha=0.3, axis="y")


def _plot_summary_statistics(ax, raw_dict, cal_dict):
    """Panel G: Summary statistics table."""
    ax.axis("off")

    # Calculate overall statistics
    total_samples = sum(len(w) for w in raw_dict.values())
    avg_ess_raw = np.mean([compute_ess(w) / len(w) * 100 for w in raw_dict.values()])
    avg_ess_cal = np.mean([compute_ess(w) / len(w) * 100 for w in cal_dict.values()])

    avg_var_raw = np.mean([np.var(w) for w in raw_dict.values()])
    avg_var_cal = np.mean([np.var(w) for w in cal_dict.values()])

    # Create summary text
    text = [
        "G. Summary Statistics",
        "=" * 25,
        f"Total samples: {total_samples}",
        f"Policies: {len(raw_dict)}",
        "",
        "Average ESS:",
        f"  Raw: {avg_ess_raw:.1f}%",
        f"  Calibrated: {avg_ess_cal:.1f}%",
        f"  Improvement: {avg_ess_cal/avg_ess_raw:.1f}x",
        "",
        "Average Variance:",
        f"  Raw: {avg_var_raw:.2e}",
        f"  Calibrated: {avg_var_cal:.3f}",
        f"  Reduction: {(1-avg_var_cal/avg_var_raw)*100:.0f}%",
    ]

    ax.text(
        0.1,
        0.9,
        "\n".join(text),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )


def plot_calibration_comparison(
    judge_scores: np.ndarray,
    oracle_labels: np.ndarray,
    calibrated_scores: Optional[np.ndarray] = None,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot calibration comparison (reliability diagram).

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

    # Bin the scores
    bins = np.linspace(0, 1, n_bins + 1)

    # Plot raw scores
    bin_indices = np.digitize(judge_scores, bins) - 1
    mean_pred_raw = []
    mean_true_raw = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_pred_raw.append(judge_scores[mask].mean())
            mean_true_raw.append(oracle_labels[mask].mean())

    ax.scatter(
        mean_pred_raw,
        mean_true_raw,
        s=100,
        alpha=0.7,
        label="Raw Judge Scores",
        edgecolor="black",
    )

    # Plot calibrated scores if provided
    if calibrated_scores is not None:
        bin_indices_cal = np.digitize(calibrated_scores, bins) - 1
        mean_pred_cal = []
        mean_true_cal = []

        for i in range(n_bins):
            mask = bin_indices_cal == i
            if mask.sum() > 0:
                mean_pred_cal.append(calibrated_scores[mask].mean())
                mean_true_cal.append(oracle_labels[mask].mean())

        ax.scatter(
            mean_pred_cal,
            mean_true_cal,
            s=100,
            alpha=0.7,
            label="Calibrated Scores",
            marker="s",
            edgecolor="black",
        )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")

    # Labels
    ax.set_xlabel("Mean Predicted Score")
    ax.set_ylabel("Mean Oracle Label")
    ax.set_title("Calibration Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
