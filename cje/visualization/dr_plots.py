"""Visualization utilities for DR diagnostics.

Provides compact dashboards to quickly identify DR failure modes.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches


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
    if "dr_diagnostics" not in estimation_result.metadata:
        raise ValueError("No DR diagnostics found in estimation result")

    dr_diags = estimation_result.metadata["dr_diagnostics"]
    policies = list(dr_diags.keys())
    n_policies = len(policies)

    # Set up figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("DR Diagnostics Dashboard", fontsize=14, fontweight="bold")

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_policies))

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

    for i, policy in enumerate(policies):
        diag = dr_diags[policy]
        score_mean = diag["score_mean"]
        score_se = diag["score_se"]

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
        p_val = diag["score_p"]
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

    # For each policy, reconstruct approximate EIF distribution for CCDF
    for i, policy in enumerate(policies):
        diag = dr_diags[policy]

        # Create synthetic IF samples based on quantiles
        # This is approximate but good enough for visualization
        n = diag["n"]
        if_p95 = diag["if_p95"]
        if_p99 = diag["if_p99"]
        if_var = diag["if_var"]

        # Generate approximate distribution
        # Use exponential tail approximation beyond p95
        n_samples = 1000
        quantiles = np.linspace(0, 1, n_samples)

        # Simple approximation: use normal up to p95, exponential beyond
        if_values = np.zeros(n_samples)
        p95_idx = int(0.95 * n_samples)
        p99_idx = int(0.99 * n_samples)

        # Normal part (0 to p95)
        if_values[:p95_idx] = np.abs(np.random.normal(0, np.sqrt(if_var), p95_idx))
        if_values[:p95_idx] = np.sort(if_values[:p95_idx])
        if_values[:p95_idx] *= (
            if_p95 / if_values[p95_idx - 1] if if_values[p95_idx - 1] > 0 else 1
        )

        # Exponential tail (p95 to p99 and beyond)
        if if_p99 > if_p95:
            rate = np.log(if_p99 / if_p95) / (0.99 - 0.95)
            tail_quantiles = quantiles[p95_idx:] - 0.95
            if_values[p95_idx:] = if_p95 * np.exp(rate * tail_quantiles / 0.05)
        else:
            if_values[p95_idx:] = if_p95

        # Plot CCDF
        sorted_if = np.sort(if_values)[::-1]
        ccdf = np.arange(1, len(sorted_if) + 1) / len(sorted_if)

        ax.loglog(sorted_if, ccdf, label=policy, color=colors[i], linewidth=2)

    # Add reference lines at p95 and p99
    ax.axhline(y=0.05, color="gray", linestyle=":", alpha=0.5, label="p95")
    ax.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5, label="p99")

    ax.set_xlabel("|IF| (log scale)")
    ax.set_ylabel("CCDF (log scale)")
    ax.set_title("C: EIF Tail Behavior")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    # Compute summary metrics
    summary_metrics = {
        "worst_if_tail_ratio": max(d["if_tail_ratio_99_5"] for d in dr_diags.values()),
        "best_r2_oof": max(d["r2_oof"] for d in dr_diags.values()),
        "worst_r2_oof": min(d["r2_oof"] for d in dr_diags.values()),
        "avg_residual_rmse": np.mean([d["residual_rmse"] for d in dr_diags.values()]),
    }

    if estimation_result.method == "tmle":
        summary_metrics["tmle_max_abs_score"] = max(
            abs(d["score_mean"]) for d in dr_diags.values()
        )

    return fig, summary_metrics


def plot_dr_calibration(
    estimation_result: Any,
    policy: str,
    n_bins: int = 10,
    figsize: Tuple[float, float] = (8, 6),
) -> Figure:
    """Plot reliability diagram for outcome model calibration.

    Shows binned g_logged vs actual rewards to assess model fit.

    Args:
        estimation_result: Result from DR estimator
        policy: Policy name to plot
        n_bins: Number of bins for calibration plot
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if "dr_calibration_data" not in estimation_result.metadata:
        raise ValueError(
            "No calibration data found. Ensure estimator stores g_logged and rewards."
        )

    cal_data = estimation_result.metadata["dr_calibration_data"].get(policy)
    if cal_data is None:
        raise ValueError(f"No calibration data for policy {policy}")

    g_logged = cal_data["g_logged"]
    rewards = cal_data["rewards"]

    # Create bins based on g_logged
    bin_edges = np.quantile(g_logged, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-6  # Ensure last point included

    fig, ax = plt.subplots(figsize=figsize)

    # Compute bin statistics
    bin_centers = []
    bin_means = []
    bin_stds = []
    bin_counts = []

    for i in range(n_bins):
        mask = (g_logged >= bin_edges[i]) & (g_logged < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_centers.append(np.mean(g_logged[mask]))
            bin_means.append(np.mean(rewards[mask]))
            bin_stds.append(np.std(rewards[mask]))
            bin_counts.append(np.sum(mask))

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bin_counts = np.array(bin_counts)

    # Plot calibration
    ax.errorbar(
        bin_centers,
        bin_means,
        yerr=bin_stds,
        fmt="o",
        markersize=8,
        capsize=5,
        capthick=2,
        label="Binned data",
    )

    # Perfect calibration line
    lim_min = min(np.min(g_logged), np.min(rewards))
    lim_max = max(np.max(g_logged), np.max(rewards))
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        "k--",
        alpha=0.5,
        label="Perfect calibration",
    )

    # Add histogram of predictions in background
    ax2 = ax.twinx()
    ax2.hist(g_logged, bins=bin_edges, alpha=0.2, color="gray", edgecolor="none")
    ax2.set_ylabel("Count (g_logged)", color="gray", alpha=0.5)
    ax2.tick_params(axis="y", labelcolor="gray", alpha=0.5)

    # Labels and formatting
    ax.set_xlabel("Predicted (g_logged)")
    ax.set_ylabel("Actual (rewards)")
    ax.set_title(f"Outcome Model Calibration: {policy}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add statistics
    if "dr_diagnostics" in estimation_result.metadata:
        diag = estimation_result.metadata["dr_diagnostics"].get(policy, {})
        rmse = diag.get("residual_rmse", 0)
        r2 = diag.get("r2_oof", 0)
        stats_text = f"RMSE: {rmse:.3f}\nR² (OOF): {r2:.3f}"
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    return fig
