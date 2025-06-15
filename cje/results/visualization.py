"""
Visualization utilities for CJE results.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import math

from .results import EstimationResult
from ..utils.weight_diagnostics import (
    WeightDiagnostics,
    compute_importance_weights,
    analyze_arena_weights,
)


def plot_policy_comparison(
    results: Union[EstimationResult, Dict[str, EstimationResult]],
    policy_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Policy Performance Comparison",
) -> plt.Figure:
    """
    Create a forest plot comparing multiple policies with confidence intervals.

    Args:
        results: Either a single multi-policy result or dict of single-policy results
        policy_names: Names for policies (auto-generated if not provided)
        save_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure
    """
    # Convert to consistent format
    if isinstance(results, EstimationResult):
        # Multi-policy result
        v_hat = results.v_hat
        se = results.se
        n_policies = results.n_policies
        if policy_names is None:
            policy_names = [f"Policy {i+1}" for i in range(n_policies)]
    else:
        # Dict of single-policy results
        policy_names = list(results.keys())
        v_hat = np.array([r.v_hat[0] for r in results.values()])
        se = np.array([r.se[0] for r in results.values()])
        n_policies = len(policy_names)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot points and error bars
    y_pos = np.arange(n_policies)
    ci_95 = 1.96 * se

    # Sort by value for better visualization
    sort_idx = np.argsort(v_hat)
    v_hat_sorted = v_hat[sort_idx]
    se_sorted = se[sort_idx]
    ci_95_sorted = ci_95[sort_idx]
    names_sorted = [policy_names[i] for i in sort_idx]

    # Plot error bars
    ax.errorbar(
        v_hat_sorted,
        y_pos,
        xerr=ci_95_sorted,
        fmt="o",
        markersize=8,
        capsize=5,
        capthick=2,
        elinewidth=2,
    )

    # Add vertical line at 0 if it's in range
    if min(v_hat_sorted - ci_95_sorted) < 0 < max(v_hat_sorted + ci_95_sorted):
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted)
    ax.set_xlabel("Estimated Value (with 95% CI)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add value annotations
    for i, (val, err) in enumerate(zip(v_hat_sorted, se_sorted)):
        ax.text(
            val + ci_95_sorted[i] + 0.01,
            i,
            f"{val:.3f} ± {err:.3f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_weight_diagnostics(
    weight_stats: Dict[str, Any],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create diagnostic plots for importance weights.

    Args:
        weight_stats: Weight statistics from sampler
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 1. ESS by policy
    if "ess_values" in weight_stats:
        ax = axes[0]
        ess_pct = weight_stats["ess_percentage"]
        policies = [f"Policy {i+1}" for i in range(len(ess_pct))]

        bars = ax.bar(policies, ess_pct)
        # Color bars based on ESS threshold
        colors = ["red" if x < 10 else "orange" if x < 30 else "green" for x in ess_pct]
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_ylabel("ESS %")
        ax.set_title("Effective Sample Size by Policy")
        ax.axhline(y=10, color="red", linestyle="--", label="Critical (10%)")
        ax.axhline(y=30, color="orange", linestyle="--", label="Warning (30%)")
        ax.legend()

    # 2. Weight distribution
    if "weight_matrix" in weight_stats:
        ax = axes[1]
        weights = weight_stats["weight_matrix"].flatten()
        weights_log = np.log10(weights + 1e-10)

        ax.hist(weights_log, bins=50, alpha=0.7)
        ax.set_xlabel("Log10(Weight)")
        ax.set_ylabel("Count")
        ax.set_title("Weight Distribution (Log Scale)")

        # Add percentile lines
        p95 = np.percentile(weights, 95)
        p99 = np.percentile(weights, 99)
        ax.axvline(
            x=np.log10(p95), color="orange", linestyle="--", label=f"95%: {p95:.1f}"
        )
        ax.axvline(
            x=np.log10(p99), color="red", linestyle="--", label=f"99%: {p99:.1f}"
        )
        ax.legend()

    # 3. Clipping impact
    if "n_clipped" in weight_stats:
        ax = axes[2]
        n_clipped = weight_stats["n_clipped"]
        clip_frac = weight_stats["clip_fraction"]

        # Pie chart of clipped vs unclipped
        sizes = [100 * (1 - clip_frac), 100 * clip_frac]
        labels = ["Unclipped", f"Clipped ({n_clipped:,})"]
        colors = ["lightblue", "lightcoral"]

        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title("Weight Clipping Impact")

    # 4. Stabilization summary
    ax = axes[3]
    ax.axis("off")

    # Create summary text
    summary_lines = [
        f"Total samples: {weight_stats.get('n_samples', 'N/A'):,}",
        f"Number of policies: {weight_stats.get('n_policies', 'N/A')}",
        f"Clip threshold: {weight_stats.get('clip_threshold', 'N/A')}",
        f"Stabilization: {'Yes' if weight_stats.get('stabilization_applied') else 'No'}",
        "",
        "Weight Range:",
        f"  Min: {weight_stats.get('weight_range', [0, 0])[0]:.3f}",
        f"  Max: {weight_stats.get('weight_range', [0, 0])[1]:.3f}",
    ]

    ax.text(
        0.1,
        0.9,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
    )
    ax.set_title("Summary Statistics")

    plt.suptitle("Importance Weight Diagnostics", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_summary_report(
    results: EstimationResult,
    policy_names: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, plt.Figure]:
    """
    Create a comprehensive visual report of results.

    Args:
        results: Estimation results
        policy_names: Names for policies
        save_dir: Directory to save figures

    Returns:
        Dict mapping figure names to Figure objects
    """
    figures = {}

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

    # 1. Policy comparison
    fig_compare = plot_policy_comparison(
        results,
        policy_names,
        save_path=save_dir / "policy_comparison.png" if save_dir else None,
    )
    figures["policy_comparison"] = fig_compare

    # 2. Weight diagnostics (if available)
    if hasattr(results, "metadata") and "weight_stats" in results.metadata:
        fig_weights = plot_weight_diagnostics(
            results.metadata["weight_stats"],
            save_path=save_dir / "weight_diagnostics.png" if save_dir else None,
        )
        figures["weight_diagnostics"] = fig_weights

    # 3. Bootstrap distribution (if EIF components available)
    if results.eif_components is not None:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bootstrap sampling
        n_bootstrap = 1000
        bootstrap_means_list = []
        eif = results.eif_components
        n = len(eif)

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            bootstrap_means_list.append(np.mean(eif[idx], axis=0))

        bootstrap_means = np.array(bootstrap_means_list)

        # Plot distributions
        for i in range(results.n_policies):
            ax.hist(
                bootstrap_means[:, i].tolist(),  # Convert to list for mypy
                bins=50,
                alpha=0.5,
                label=f"Policy {i+1}" if not policy_names else policy_names[i],
            )

        ax.set_xlabel("Bootstrap Estimate")
        ax.set_ylabel("Frequency")
        ax.set_title("Bootstrap Distribution of Estimates")
        ax.legend()

        if save_dir:
            fig.savefig(
                save_dir / "bootstrap_distribution.png", dpi=300, bbox_inches="tight"
            )

        figures["bootstrap_distribution"] = fig

    return figures


def plot_weight_distributions(
    diagnostics: Dict[str, WeightDiagnostics],
    data: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create weight distribution plots for all policies.

    Args:
        diagnostics: Weight diagnostics by policy name
        data: Original data with log probabilities
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    if not diagnostics:
        raise ValueError("No weight diagnostics to plot")

    # Set up the plot
    n_policies = len(diagnostics)
    fig, axes = plt.subplots(n_policies, 2, figsize=(12, 4 * n_policies))

    # Handle single policy case
    if n_policies == 1:
        axes = axes.reshape(1, -1)

    behavior_logprobs = [record["logp"] for record in data]

    for i, (policy_name, diag) in enumerate(diagnostics.items()):
        # Get weights for this policy
        target_logprobs = []
        for record in data:
            logp_target = record.get("logp_target_all", {}).get(policy_name, 0.0)
            target_logprobs.append(logp_target)

        weights = compute_importance_weights(behavior_logprobs, target_logprobs)
        finite_weights = [w for w in weights if math.isfinite(w)]

        # Left plot: Weight distribution (histogram)
        ax1 = axes[i, 0]
        if finite_weights:
            log_weights = [math.log10(max(w, 1e-10)) for w in finite_weights]
            ax1.hist(log_weights, bins=30, alpha=0.7, edgecolor="black")
            ax1.axvline(0, color="red", linestyle="--", label="Expected (log₁₀=0)")
            ax1.set_xlabel("Log₁₀(Weight)")
            ax1.set_ylabel("Count")
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, "No finite weights", ha="center", va="center")

        # Color-code title by diagnostic status
        title_color = {"GOOD": "green", "WARNING": "orange", "CRITICAL": "red"}[
            diag.consistency_flag
        ]
        ax1.set_title(
            f"{policy_name} - Weight Distribution", color=title_color, fontweight="bold"
        )

        # Right plot: Weight vs sample index
        ax2 = axes[i, 1]
        if finite_weights:
            indices = list(range(len(weights)))
            clipped_weights = [min(max(w, 1e-10), 1e10) for w in weights]
            ax2.scatter(indices, clipped_weights, alpha=0.6, s=10)
            ax2.axhline(1.0, color="red", linestyle="--", label="Expected=1.0")
            ax2.set_xlabel("Sample Index")
            ax2.set_ylabel("Weight")
            ax2.set_yscale("log")
            ax2.legend()

        ax2.set_title(f"ESS: {diag.ess_fraction:.1%} | Mean: {diag.mean_weight:.4f}")

    plt.suptitle("Importance Weight Diagnostics", fontsize=16)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_ess_comparison(
    diagnostics: Dict[str, WeightDiagnostics],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create ESS comparison bar plot.

    Args:
        diagnostics: Weight diagnostics by policy name
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    if not diagnostics:
        raise ValueError("No weight diagnostics to plot")

    # Extract data
    policy_names = list(diagnostics.keys())
    ess_fractions = [diag.ess_fraction for diag in diagnostics.values()]
    flags = [diag.consistency_flag for diag in diagnostics.values()]

    # Color-code bars by status
    colors = {"GOOD": "green", "WARNING": "orange", "CRITICAL": "red"}
    bar_colors = [colors[flag] for flag in flags]

    # Create plot
    fig, ax = plt.subplots(figsize=(max(8, len(policy_names) * 1.5), 6))
    bars = ax.bar(
        policy_names, ess_fractions, color=bar_colors, alpha=0.7, edgecolor="black"
    )

    # Add reference lines
    ax.axhline(0.1, color="orange", linestyle="--", alpha=0.7, label="Warning (10%)")
    ax.axhline(0.01, color="red", linestyle="--", alpha=0.7, label="Critical (1%)")

    # Labels and formatting
    ax.set_ylabel("Effective Sample Size (ESS) Fraction")
    ax.set_title("ESS Comparison Across Policies", fontweight="bold")
    ax.set_ylim(0, max(max(ess_fractions) * 1.1, 0.2))
    plt.xticks(rotation=45, ha="right")

    # Add percentage labels on bars
    for bar, ess in zip(bars, ess_fractions):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{ess:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Legend
    legend_elements = [
        mpatches.Patch(color="green", alpha=0.7, label="Good (ESS ≥ 10%)"),
        mpatches.Patch(color="orange", alpha=0.7, label="Warning (1% ≤ ESS < 10%)"),
        mpatches.Patch(color="red", alpha=0.7, label="Critical (ESS < 1%)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
