"""
Simple Weight Diagnostic Plots

Out-of-the-box plotting utilities for visualizing importance weight issues.
Designed to be lightweight and work without heavy dependencies.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .weight_diagnostics import WeightDiagnostics, compute_importance_weights


def create_weight_distribution_plot(
    diagnostics: Dict[str, WeightDiagnostics],
    data: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> Optional[str]:
    """Create weight distribution plots for all policies."""

    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
        return None

    if not diagnostics:
        print("No weight diagnostics to plot")
        return None

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

        # Remove infinite weights for plotting
        finite_weights = [w for w in weights if math.isfinite(w)]

        # Left plot: Weight distribution (histogram)
        ax1 = axes[i, 0]
        if finite_weights:
            # Use log scale for better visualization
            log_weights = [math.log10(max(w, 1e-10)) for w in finite_weights]
            ax1.hist(log_weights, bins=30, alpha=0.7, edgecolor="black")
            ax1.axvline(
                math.log10(1.0), color="red", linestyle="--", label="Expected (log₁₀=0)"
            )
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

        # Right plot: Weight vs sample index (to detect patterns)
        ax2 = axes[i, 1]
        if finite_weights:
            indices = list(range(len(weights)))
            # Use log scale and clip extreme values for visibility
            clipped_weights = [min(max(w, 1e-10), 1e10) for w in weights]
            ax2.scatter(indices, clipped_weights, alpha=0.6, s=10)
            ax2.axhline(1.0, color="red", linestyle="--", label="Expected=1.0")
            ax2.set_xlabel("Sample Index")
            ax2.set_ylabel("Weight")
            ax2.set_yscale("log")
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "No finite weights", ha="center", va="center")

        ax2.set_title(f"ESS: {diag.ess_fraction:.1%} | Mean: {diag.mean_weight:.4f}")

        # Add diagnostic info as text
        info_text = (
            f"Range: {diag.min_weight:.1e} to {diag.max_weight:.1e}\n"
            f"Extreme: {diag.extreme_weight_count}, Zero: {diag.zero_weight_count}"
        )
        ax2.text(
            0.02,
            0.98,
            info_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.suptitle("Importance Weight Diagnostics", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        saved_path = str(Path(save_path).resolve())
        print(f"📊 Weight plots saved to: {saved_path}")
    else:
        saved_path = None

    if show_plot:
        plt.show()
    else:
        plt.close()

    return saved_path


def create_ess_comparison_plot(
    diagnostics: Dict[str, WeightDiagnostics],
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> Optional[str]:
    """Create a simple bar plot comparing ESS across policies."""

    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
        return None

    if not diagnostics:
        print("No weight diagnostics to plot")
        return None

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

    # Add horizontal lines for reference
    ax.axhline(
        0.1, color="orange", linestyle="--", alpha=0.7, label="Warning threshold (10%)"
    )
    ax.axhline(
        0.01, color="red", linestyle="--", alpha=0.7, label="Critical threshold (1%)"
    )

    # Labels and formatting
    ax.set_ylabel("Effective Sample Size (ESS) Fraction")
    ax.set_title("ESS Comparison Across Policies", fontweight="bold")
    ax.set_ylim(0, 1.1)
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

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        saved_path = str(Path(save_path).resolve())
        print(f"📊 ESS comparison saved to: {saved_path}")
    else:
        saved_path = None

    if show_plot:
        plt.show()
    else:
        plt.close()

    return saved_path


def create_weight_diagnostic_dashboard(
    data: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
    show_plots: bool = True,
) -> Dict[str, str]:
    """Create a complete diagnostic dashboard with multiple plots."""

    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
        return {}

    from .weight_diagnostics import analyze_arena_weights

    # Analyze weights
    diagnostics = analyze_arena_weights(data)

    if not diagnostics:
        print("No weight diagnostics to plot")
        return {}

    # Prepare output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path("weight_diagnostics")
        output_path.mkdir(exist_ok=True)

    saved_files = {}

    # Create distribution plot
    dist_path = output_path / "weight_distributions.png"
    saved_dist = create_weight_distribution_plot(
        diagnostics, data, str(dist_path), show_plot=show_plots
    )
    if saved_dist:
        saved_files["distributions"] = saved_dist

    # Create ESS comparison plot
    ess_path = output_path / "ess_comparison.png"
    saved_ess = create_ess_comparison_plot(
        diagnostics, str(ess_path), show_plot=show_plots
    )
    if saved_ess:
        saved_files["ess_comparison"] = saved_ess

    # Print summary
    print(f"\n📊 Weight diagnostic dashboard created in: {output_path}")
    print("Files created:")
    for plot_type, file_path in saved_files.items():
        print(f"  - {plot_type}: {Path(file_path).name}")

    return saved_files


def quick_weight_check(
    data: List[Dict[str, Any]], policy_name: Optional[str] = None
) -> None:
    """Quick visual check of weights without saving files."""

    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
        return

    from .weight_diagnostics import analyze_arena_weights

    diagnostics = analyze_arena_weights(data)

    if not diagnostics:
        print("No weight diagnostics available")
        return

    # Filter to specific policy if requested
    if policy_name:
        if policy_name in diagnostics:
            diagnostics = {policy_name: diagnostics[policy_name]}
        else:
            print(
                f"Policy '{policy_name}' not found. Available: {list(diagnostics.keys())}"
            )
            return

    # Show distribution plot only
    create_weight_distribution_plot(diagnostics, data, save_path=None, show_plot=True)
