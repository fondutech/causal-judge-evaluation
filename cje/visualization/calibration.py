"""Calibration visualization utilities."""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_calibration_comparison(
    judge_scores: np.ndarray,
    oracle_labels: np.ndarray,
    calibrated_scores: Optional[np.ndarray] = None,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plot calibration comparison showing transformation and improvement.

    Shows the calibration transformation and its effect on oracle alignment.

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
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

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

    bins = np.linspace(0, 1, n_bins + 1)

    # Panel 1: Calibration Transformation (judge -> calibrated)
    if calibrated_scores is not None:
        # Sort for smooth curve
        sorted_idx = np.argsort(judge_scores)
        judge_sorted = judge_scores[sorted_idx]
        calibrated_sorted = calibrated_scores[sorted_idx]

        # Plot transformation
        ax1.plot(
            judge_sorted,
            calibrated_sorted,
            "-",
            color="green",
            alpha=0.7,
            linewidth=2,
            label="Calibration function",
        )

        # Add scatter plot with sampling for visibility
        n_show = min(500, len(judge_scores))
        step = max(1, len(judge_scores) // n_show)
        ax1.scatter(
            judge_scores[::step],
            calibrated_scores[::step],
            alpha=0.3,
            s=10,
            color="green",
        )

        # Add diagonal reference
        ax1.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="No change (y=x)")

        # Labels and formatting
        ax1.set_xlabel("Judge Score (raw)")
        ax1.set_ylabel("Calibrated Reward")
        ax1.set_title("A. Calibration Transformation")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.legend(loc="upper left")

        # Add annotation showing compression/expansion regions
        # Find regions where slope is notably different from 1
        window = max(1, len(judge_sorted) // 20)
        for i in range(window, len(judge_sorted) - window, window):
            local_slope = (
                calibrated_sorted[i + window] - calibrated_sorted[i - window]
            ) / (judge_sorted[i + window] - judge_sorted[i - window] + 1e-10)
            if local_slope < 0.5:
                ax1.annotate(
                    "compressed",
                    xy=(judge_sorted[i], calibrated_sorted[i]),
                    fontsize=8,
                    alpha=0.6,
                    color="red",
                )
                break
        for i in range(window, len(judge_sorted) - window, window):
            local_slope = (
                calibrated_sorted[i + window] - calibrated_sorted[i - window]
            ) / (judge_sorted[i + window] - judge_sorted[i - window] + 1e-10)
            if local_slope > 1.5:
                ax1.annotate(
                    "expanded",
                    xy=(judge_sorted[i], calibrated_sorted[i]),
                    fontsize=8,
                    alpha=0.6,
                    color="blue",
                )
                break
    else:
        # If no calibration, just show message
        ax1.text(
            0.5,
            0.5,
            "No calibration applied",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            fontsize=12,
            color="gray",
        )
        ax1.set_xlabel("Judge Score (raw)")
        ax1.set_ylabel("Calibrated Reward")
        ax1.set_title("A. Calibration Transformation")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

    # Panel 2: Reliability Diagram (oracle alignment)
    # Bin the scores for reliability diagram
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

    # Plot judge vs oracle
    judge_label = "Judge (before)" if calibrated_scores is not None else "Judge"

    ax2.scatter(
        mean_pred_raw,
        mean_true_raw,
        label=judge_label,
        s=sizes_raw,
        alpha=0.7,
        color="coral",
        edgecolors="darkred",
        linewidth=1,
    )
    ax2.plot(mean_pred_raw, mean_true_raw, "-", alpha=0.5, color="coral")

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

        ax2.scatter(
            mean_pred_cal,
            mean_true_cal,
            label="Calibrated (after)",
            s=sizes_cal,
            alpha=0.7,
            color="lightgreen",
            edgecolors="darkgreen",
            linewidth=1,
        )
        ax2.plot(mean_pred_cal, mean_true_cal, "-", alpha=0.5, color="lightgreen")

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
            f"Judge Metrics:\n" f"ECE: {ece_raw:.3f}\n" f"RMSE: {rmse_raw:.3f}"
        )

    # Perfect calibration line
    ax2.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Perfect calibration")

    # Add shaded region for ±0.1 calibration error
    x_perfect = np.linspace(0, 1, 100)
    ax2.fill_between(
        x_perfect,
        x_perfect - 0.1,
        x_perfect + 0.1,
        alpha=0.1,
        color="gray",
        label="±0.1 tolerance",
    )

    # Add metrics text box
    ax2.text(
        0.05,
        0.95,
        improvement_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Labels and formatting for reliability diagram
    ax2.set_xlabel("Predicted Score")
    ax2.set_ylabel("Oracle Score")
    ax2.set_title("B. Oracle Alignment (Reliability)")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)

    # Add note about point sizes
    ax2.text(
        0.95,
        0.05,
        "Point size ∝ samples in bin",
        transform=ax2.transAxes,
        fontsize=8,
        horizontalalignment="right",
        alpha=0.6,
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), dpi=150, bbox_inches="tight")

    return fig
