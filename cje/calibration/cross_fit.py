"""Cross-fitted calibration for CJE to ensure no data leakage."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import logging

from cje.utils.progress import ProgressMonitor, console

logger = logging.getLogger(__name__)


def cross_fit_calibration(
    all_rows: List[Dict[str, Any]],
    k_folds: int = 5,
    seed: int = 42,
    score_key: str = "score_raw",
    label_key: str = "y_true",
    output_score_key: str = "score_cal",
    plot_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Cross-fitted calibration following the exact specification:
    - Oracle rows: K-fold cross-fit calibration
    - Non-oracle rows: Final model trained on all oracle data

    Args:
        all_rows: All data rows (with and without oracle labels)
        k_folds: Number of folds for cross-fitting (default: 5)
        seed: Random seed for reproducibility
        score_key: Field containing raw judge scores
        label_key: Field containing oracle labels
        output_score_key: Field to store calibrated scores
        plot_path: Optional path to save reliability plot

    Returns:
        Tuple of (calibrated_rows, diagnostics)
    """
    # Separate oracle vs non-oracle rows
    oracle_indices = []
    non_oracle_indices = []

    for i, row in enumerate(all_rows):
        if label_key in row and row[label_key] is not None:
            oracle_indices.append(i)
        else:
            non_oracle_indices.append(i)

    logger.info(
        f"Cross-fit calibration: {len(oracle_indices)} oracle rows, "
        f"{len(non_oracle_indices)} non-oracle rows"
    )

    if not oracle_indices:
        logger.warning("No oracle-labeled rows found. Returning uncalibrated data.")
        return all_rows, {"status": "no_oracle_data"}

    # Initialize result list maintaining original order
    calibrated_rows: List[Optional[Dict[str, Any]]] = [None] * len(all_rows)

    # Phase 1: K-fold cross-fit on oracle rows
    oracle_indices_array = np.array(oracle_indices)
    np.random.seed(seed)
    np.random.shuffle(oracle_indices_array)  # Shuffle for random fold assignment

    kf = KFold(n_splits=k_folds, shuffle=False)  # Already shuffled

    # Track fold assignments for diagnostics
    fold_assignments = {}
    calibration_models = []

    with ProgressMonitor() as progress:
        progress.add_task(
            "calibration", f"Cross-fit calibration ({k_folds} folds)", total=k_folds
        )

        for fold_idx, (train_idx, test_idx) in enumerate(
            kf.split(oracle_indices_array)
        ):
            logger.info(f"Processing calibration fold {fold_idx + 1}/{k_folds}")

            # Get actual row indices
            train_row_indices = oracle_indices_array[train_idx]
            test_row_indices = oracle_indices_array[test_idx]

            # Extract training data (K-1 folds)
            train_scores = []
            train_labels = []
            for idx in train_row_indices:
                row = all_rows[idx]
                if score_key in row and row[score_key] is not None:
                    train_scores.append(float(row[score_key]))
                    train_labels.append(float(row[label_key]))

            if not train_scores:
                logger.warning(f"Fold {fold_idx}: No valid training data")
                progress.update("calibration", 1)
                continue

            train_scores_array = np.array(train_scores)
            train_labels_array = np.array(train_labels)

            # Fit calibration model g^(-k) on K-1 folds
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(train_scores_array, train_labels_array)
            calibration_models.append(calibrator)

            # Apply to test fold (out-of-fold prediction)
            for idx in test_row_indices:
                row = all_rows[idx].copy()

                if score_key in row and row[score_key] is not None:
                    raw_score = float(row[score_key])
                    calibrated_score = float(calibrator.predict([raw_score])[0])

                    row[output_score_key] = calibrated_score
                    row["reward"] = calibrated_score  # For DR-CPO
                    row["calibration_fold"] = fold_idx

                    calibrated_rows[idx] = row
                    fold_assignments[idx] = fold_idx

            progress.update("calibration", 1)

    # Phase 2: Train final model on ALL oracle data for non-oracle rows
    all_oracle_scores = []
    all_oracle_labels = []

    for idx in oracle_indices:
        row = all_rows[idx]
        if score_key in row and row[score_key] is not None:
            all_oracle_scores.append(float(row[score_key]))
            all_oracle_labels.append(float(row[label_key]))

    all_oracle_scores_array = np.array(all_oracle_scores)
    all_oracle_labels_array = np.array(all_oracle_labels)

    # Fit final calibrator on entire oracle subset
    final_calibrator = IsotonicRegression(out_of_bounds="clip")
    final_calibrator.fit(all_oracle_scores_array, all_oracle_labels_array)

    # Apply to non-oracle rows (no leakage since they have no labels)
    for idx in non_oracle_indices:
        row = all_rows[idx].copy()

        if score_key in row and row[score_key] is not None:
            raw_score = float(row[score_key])
            calibrated_score = float(final_calibrator.predict([raw_score])[0])

            row[output_score_key] = calibrated_score
            row["reward"] = calibrated_score
            row["calibration_source"] = "final_model"

            calibrated_rows[idx] = row

    # Fill in any missing rows (shouldn't happen)
    final_calibrated_rows: List[Dict[str, Any]] = []
    for i, calibrated_row in enumerate(calibrated_rows):
        if calibrated_row is None:
            final_calibrated_rows.append(all_rows[i].copy())
            logger.warning(f"Row {i} was not calibrated")
        else:
            final_calibrated_rows.append(calibrated_row)

    # Compute diagnostics
    diagnostics = compute_calibration_diagnostics(
        final_calibrated_rows,
        oracle_indices_array,
        fold_assignments,
        score_key,
        label_key,
        output_score_key,
    )

    # Optional: Generate reliability plot
    if plot_path:
        plot_cross_fit_reliability(
            final_calibrated_rows,
            oracle_indices_array,
            score_key,
            label_key,
            output_score_key,
            plot_path,
        )

    return final_calibrated_rows, diagnostics


def compute_calibration_diagnostics(
    calibrated_rows: List[Dict[str, Any]],
    oracle_indices: np.ndarray,
    fold_assignments: Dict[int, int],
    score_key: str,
    label_key: str,
    output_score_key: str,
) -> Dict[str, Any]:
    """Compute diagnostics for cross-fitted calibration."""

    # Extract oracle rows with calibration
    oracle_scores_raw = []
    oracle_scores_cal = []
    oracle_labels = []
    fold_ids = []

    for idx in oracle_indices:
        row = calibrated_rows[idx]
        if output_score_key in row and row[output_score_key] is not None:
            oracle_scores_raw.append(float(row[score_key]))
            oracle_scores_cal.append(float(row[output_score_key]))
            oracle_labels.append(float(row[label_key]))
            fold_ids.append(fold_assignments.get(int(idx), -1))

    oracle_scores_raw_array = np.array(oracle_scores_raw)
    oracle_scores_cal_array = np.array(oracle_scores_cal)
    oracle_labels_array = np.array(oracle_labels)

    # Compute metrics
    rmse_raw = np.sqrt(np.mean((oracle_scores_raw_array - oracle_labels_array) ** 2))
    rmse_cal = np.sqrt(np.mean((oracle_scores_cal_array - oracle_labels_array) ** 2))

    # Calibration error (expected calibrated score should match true label)
    calibration_error = np.mean(np.abs(oracle_scores_cal_array - oracle_labels_array))

    # Coverage: fraction where true label is within ±0.1 of calibrated score
    coverage = np.mean(np.abs(oracle_scores_cal_array - oracle_labels_array) <= 0.1)

    # Per-fold statistics
    fold_stats = {}
    for fold_id in set(fold_ids):
        if fold_id >= 0:
            fold_mask = np.array(fold_ids) == fold_id
            fold_stats[f"fold_{fold_id}"] = {
                "n": int(np.sum(fold_mask)),
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                oracle_scores_cal_array[fold_mask]
                                - oracle_labels_array[fold_mask]
                            )
                            ** 2
                        )
                    )
                ),
                "coverage": float(
                    np.mean(
                        np.abs(
                            oracle_scores_cal_array[fold_mask]
                            - oracle_labels_array[fold_mask]
                        )
                        <= 0.1
                    )
                ),
            }

    return {
        "n_oracle": len(oracle_indices),
        "n_calibrated": len(oracle_scores_cal),
        "rmse_raw": float(rmse_raw),
        "rmse_calibrated": float(rmse_cal),
        "rmse_reduction": float((rmse_raw - rmse_cal) / rmse_raw * 100),
        "calibration_error": float(calibration_error),
        "coverage_at_0.1": float(coverage),
        "score_range_raw": [
            float(oracle_scores_raw_array.min()),
            float(oracle_scores_raw_array.max()),
        ],
        "score_range_cal": [
            float(oracle_scores_cal_array.min()),
            float(oracle_scores_cal_array.max()),
        ],
        "fold_stats": fold_stats,
    }


def plot_cross_fit_reliability(
    calibrated_rows: List[Dict[str, Any]],
    oracle_indices: np.ndarray,
    score_key: str,
    label_key: str,
    output_score_key: str,
    plot_path: Path,
    n_bins: int = 10,
) -> None:
    """Generate reliability plot for cross-fitted calibration."""

    # Extract data
    scores_raw = []
    scores_cal = []
    labels = []

    for idx in oracle_indices:
        row = calibrated_rows[idx]
        if output_score_key in row:
            scores_raw.append(float(row[score_key]))
            scores_cal.append(float(row[output_score_key]))
            labels.append(float(row[label_key]))

    scores_raw_array = np.array(scores_raw)
    scores_cal_array = np.array(scores_cal)
    labels_array = np.array(labels)

    plt.figure(figsize=(12, 5))

    # Left plot: Raw scores
    plt.subplot(1, 2, 1)
    bins = np.linspace(0, 1, n_bins + 1)

    # Compute binned statistics for raw scores
    bin_indices = np.digitize(scores_raw_array, bins) - 1
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            mean_score = scores_raw_array[mask].mean()
            mean_label = labels_array[mask].mean()
            plt.scatter(mean_score, mean_label, s=100, alpha=0.7, color="red")

    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean Raw Judge Score")
    plt.ylabel("Mean True Label")
    plt.title("Before Calibration")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Right plot: Calibrated scores
    plt.subplot(1, 2, 2)

    # Compute binned statistics for calibrated scores
    bin_indices = np.digitize(scores_cal_array, bins) - 1
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            mean_score = scores_cal_array[mask].mean()
            mean_label = labels_array[mask].mean()
            plt.scatter(mean_score, mean_label, s=100, alpha=0.7, color="green")

    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean Calibrated Judge Score")
    plt.ylabel("Mean True Label")
    plt.title("After Cross-Fit Calibration")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
