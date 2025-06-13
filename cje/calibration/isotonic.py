from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from pathlib import Path


def fit_isotonic(scores: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """
    Fits monotone calibration gφ such that gφ(scores) ≈ E[y | score].
    Returns the fitted IsotonicRegression object.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores, y_true)
    return iso


def plot_reliability(
    scores: np.ndarray,
    y_true: np.ndarray,
    iso: IsotonicRegression,
    out: Path,
    n_bins: int = 10,
) -> None:
    """
    Saves a reliability curve comparing raw vs calibrated predictions.

    Args:
        scores: Raw model scores/probabilities
        y_true: True binary labels
        iso: Fitted IsotonicRegression object
        out: Path to save the plot
        n_bins: Number of bins for reliability curve
    """
    bins = np.linspace(scores.min(), scores.max(), n_bins + 1)
    idx = np.digitize(scores, bins) - 1
    mean_pred, mean_true = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(scores[mask].mean())
        mean_true.append(y_true[mask].mean())

    plt.figure(figsize=(8, 6))

    # Plot raw predictions
    plt.scatter(mean_pred, mean_true, label="Raw predictions", alpha=0.7)

    # Plot calibrated predictions
    calibrated_scores = iso.predict(scores)
    idx_cal = np.digitize(calibrated_scores, bins) - 1
    mean_pred_cal, mean_true_cal = [], []
    for b in range(n_bins):
        mask = idx_cal == b
        if mask.sum() == 0:
            continue
        mean_pred_cal.append(calibrated_scores[mask].mean())
        mean_true_cal.append(y_true[mask].mean())
    plt.scatter(mean_pred_cal, mean_true_cal, label="Calibrated predictions", alpha=0.7)

    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Mean true probability")
    plt.title("Reliability Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Ensure output directory exists
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
