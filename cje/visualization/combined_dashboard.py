"""Combined weight dashboard for multi-policy overview."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

from ..utils.diagnostics import compute_ess


def plot_combined_weight_dashboard(
    raw_weights_dict: Dict[str, np.ndarray],
    calibrated_weights_dict: Optional[Dict[str, np.ndarray]] = None,
    n_samples: Optional[int] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 10),
    diagnostics: Optional[Any] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """Create a combined dashboard showing all policies together.

    Creates a 6-panel dashboard with:
    1. ESS comparison across policies
    2. Weight distribution comparison
    3. Tail weight concentration
    4. Calibration improvement metrics
    5. Weight curves by judge score (if available)
    6. Summary statistics table

    Args:
        raw_weights_dict: Dict mapping policy names to raw weight arrays
        calibrated_weights_dict: Optional dict of calibrated weights
        n_samples: Total number of samples
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        diagnostics: Optional diagnostics object
        **kwargs: Additional arguments (e.g., sampler for judge scores)

    Returns:
        Tuple of (matplotlib Figure, metrics dict)
    """
    policies = list(raw_weights_dict.keys())
    n_policies = len(policies)

    # Get judge scores if available
    sampler = kwargs.get("sampler")
    judge_scores_dict = {}
    if sampler is not None:
        for policy in policies:
            data = sampler.get_data_for_policy(policy)
            if data:
                scores = np.array([d.get("judge_score", np.nan) for d in data])
                valid = ~np.isnan(scores)
                if valid.sum() > 0:
                    judge_scores_dict[policy] = scores[valid]

    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Create subplots
    ax_ess = fig.add_subplot(gs[0, :2])  # Top left - ESS comparison
    ax_summary = fig.add_subplot(gs[0, 2])  # Top right - Summary metrics
    ax_dist = fig.add_subplot(gs[1, 0])  # Middle left - Weight distributions
    ax_tail = fig.add_subplot(gs[1, 1])  # Middle center - Tail concentration
    ax_improve = fig.add_subplot(gs[1, 2])  # Middle right - Improvement metrics
    ax_curves = fig.add_subplot(gs[2, :])  # Bottom - Weight curves

    # Compute metrics for all policies
    all_metrics = {}
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

        # Tail metrics
        w_sorted = np.sort(raw_w)[::-1]
        k1 = max(1, int(len(w_sorted) * 0.01))
        k5 = max(1, int(len(w_sorted) * 0.05))
        top1_raw = w_sorted[:k1].sum() / max(w_sorted.sum(), 1e-12)
        top5_raw = w_sorted[:k5].sum() / max(w_sorted.sum(), 1e-12)

        w_sorted = np.sort(cal_w)[::-1]
        top1_cal = w_sorted[:k1].sum() / max(w_sorted.sum(), 1e-12)
        top5_cal = w_sorted[:k5].sum() / max(w_sorted.sum(), 1e-12)

        all_metrics[policy] = {
            "ess_raw": ess_raw,
            "ess_cal": ess_cal,
            "ess_improvement": ess_cal / max(ess_raw, 1e-12),
            "top1_raw": top1_raw,
            "top1_cal": top1_cal,
            "top5_raw": top5_raw,
            "top5_cal": top5_cal,
            "max_raw": raw_w.max(),
            "max_cal": cal_w.max(),
            "std_raw": raw_w.std(),
            "std_cal": cal_w.std(),
        }

    # 1. ESS Comparison
    x_pos = np.arange(n_policies)
    width = 0.35

    ess_raw_vals = [all_metrics[p]["ess_raw"] for p in policies]
    ess_cal_vals = [all_metrics[p]["ess_cal"] for p in policies]

    bars1 = ax_ess.bar(
        x_pos - width / 2,
        ess_raw_vals,
        width,
        label="Raw",
        color="lightcoral",
        alpha=0.7,
    )
    bars2 = ax_ess.bar(
        x_pos + width / 2,
        ess_cal_vals,
        width,
        label="Calibrated",
        color="lightblue",
        alpha=0.7,
    )

    ax_ess.set_xlabel("Policy")
    ax_ess.set_ylabel("Effective Sample Size")
    ax_ess.set_title("ESS Comparison")
    ax_ess.set_xticks(x_pos)
    ax_ess.set_xticklabels(policies, rotation=45, ha="right")
    ax_ess.legend()
    ax_ess.grid(True, alpha=0.3)

    # Add improvement percentages
    for i, (raw, cal) in enumerate(zip(ess_raw_vals, ess_cal_vals)):
        improvement = (cal / max(raw, 1e-12) - 1) * 100
        if improvement > 0:
            ax_ess.text(
                i,
                cal + max(ess_cal_vals) * 0.02,
                f"+{improvement:.0f}%",
                ha="center",
                fontsize=8,
                color="green",
            )

    # 2. Summary Statistics Table
    ax_summary.axis("tight")
    ax_summary.axis("off")

    # Create summary table
    headers = ["Policy", "ESS↑", "Max↓", "Std↓"]
    table_data = []
    for policy in policies[:5]:  # Show top 5 policies
        m = all_metrics[policy]
        row = [
            policy[:8],  # Truncate long names
            f"{m['ess_improvement']:.2f}x",
            f"{m['max_cal']/max(m['max_raw'], 1e-12):.2f}x",
            f"{m['std_cal']/max(m['std_raw'], 1e-12):.2f}x",
        ]
        table_data.append(row)

    table = ax_summary.table(
        cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax_summary.set_title("Improvement Factors", fontsize=10, pad=10)

    # 3. Weight Distribution Comparison
    for policy in policies[:3]:  # Show first 3 policies
        raw_w = raw_weights_dict[policy]
        cal_w = (
            calibrated_weights_dict.get(policy, raw_w)
            if calibrated_weights_dict
            else raw_w
        )

        # Log scale for better visualization
        ax_dist.hist(
            np.log10(raw_w + 1e-10),
            bins=30,
            alpha=0.3,
            label=f"{policy[:8]} (raw)",
            density=True,
        )
        ax_dist.hist(
            np.log10(cal_w + 1e-10),
            bins=30,
            alpha=0.5,
            label=f"{policy[:8]} (cal)",
            density=True,
        )

    ax_dist.set_xlabel("log₁₀(weight)")
    ax_dist.set_ylabel("Density")
    ax_dist.set_title("Weight Distributions")
    ax_dist.legend(fontsize=8)
    ax_dist.grid(True, alpha=0.3)

    # 4. Tail Concentration
    top1_raw = [all_metrics[p]["top1_raw"] * 100 for p in policies]
    top1_cal = [all_metrics[p]["top1_cal"] * 100 for p in policies]

    ax_tail.scatter(top1_raw, top1_cal, s=50, alpha=0.6)
    for i, policy in enumerate(policies):
        ax_tail.annotate(policy[:5], (top1_raw[i], top1_cal[i]), fontsize=8, alpha=0.7)

    # Add diagonal line
    lims = [0, max(max(top1_raw), max(top1_cal)) * 1.1]
    ax_tail.plot(lims, lims, "k--", alpha=0.3, label="No change")

    ax_tail.set_xlabel("Raw top 1% mass (%)")
    ax_tail.set_ylabel("Calibrated top 1% mass (%)")
    ax_tail.set_title("Tail Weight Reduction")
    ax_tail.grid(True, alpha=0.3)
    ax_tail.legend(fontsize=8)

    # 5. Improvement Metrics
    improvements = [all_metrics[p]["ess_improvement"] for p in policies]
    colors = ["green" if imp > 1 else "red" for imp in improvements]

    bars = ax_improve.barh(range(n_policies), improvements, color=colors, alpha=0.6)
    ax_improve.set_yticks(range(n_policies))
    ax_improve.set_yticklabels(policies, fontsize=9)
    ax_improve.set_xlabel("ESS Improvement Factor")
    ax_improve.set_title("Calibration Effectiveness")
    ax_improve.axvline(x=1, color="black", linestyle="--", alpha=0.3)
    ax_improve.grid(True, alpha=0.3, axis="x")

    # 6. Weight Curves by Judge Score (if available)
    if judge_scores_dict:
        for i, policy in enumerate(policies[:3]):  # Show first 3
            if policy in judge_scores_dict:
                scores = judge_scores_dict[policy]
                raw_w = raw_weights_dict[policy]
                cal_w = (
                    calibrated_weights_dict.get(policy, raw_w)
                    if calibrated_weights_dict
                    else raw_w
                )

                if len(scores) == len(raw_w):
                    # Sort by scores
                    sort_idx = np.argsort(scores)
                    scores_sorted = scores[sort_idx]
                    raw_sorted = raw_w[sort_idx]
                    cal_sorted = cal_w[sort_idx]

                    # Smooth with rolling mean
                    window = max(1, len(scores) // 50)
                    raw_smooth = np.convolve(
                        raw_sorted, np.ones(window) / window, mode="valid"
                    )
                    cal_smooth = np.convolve(
                        cal_sorted, np.ones(window) / window, mode="valid"
                    )
                    scores_smooth = scores_sorted[: len(raw_smooth)]

                    ax_curves.plot(
                        scores_smooth,
                        raw_smooth,
                        "--",
                        alpha=0.5,
                        label=f"{policy[:8]} (raw)",
                    )
                    ax_curves.plot(
                        scores_smooth,
                        cal_smooth,
                        "-",
                        alpha=0.7,
                        label=f"{policy[:8]} (cal)",
                    )

        ax_curves.set_xlabel("Judge Score")
        ax_curves.set_ylabel("Weight (smoothed)")
        ax_curves.set_title("Weight Curves by Judge Score")
        ax_curves.legend(fontsize=8, ncol=2)
        ax_curves.grid(True, alpha=0.3)
    else:
        # If no judge scores, show a message
        ax_curves.text(
            0.5,
            0.5,
            "Judge scores not available",
            ha="center",
            va="center",
            fontsize=12,
            alpha=0.5,
        )
        ax_curves.set_xticks([])
        ax_curves.set_yticks([])

    # Main title
    fig.suptitle(
        f"Combined Weight Dashboard ({n_policies} policies, n={n_samples or len(next(iter(raw_weights_dict.values())))})",
        fontsize=14,
        fontweight="bold",
    )

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")

    return fig, all_metrics
