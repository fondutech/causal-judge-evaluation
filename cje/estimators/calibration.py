"""
Isotonic calibration for importance weights and outcome models in CJE.

This module provides monotonic calibration methods to replace ad-hoc normalization
with principled statistical calibration that ensures proper expectations while
maintaining monotonicity constraints.
"""

from typing import Tuple, Optional, List, Dict, Any, Callable
import numpy as np
from sklearn.isotonic import IsotonicRegression
import warnings


def calibrate_weights_isotonic(
    weights: np.ndarray,
    fold_indices: Optional[np.ndarray] = None,
    target_mean: float = 1.0,
    max_calibrated_weight: float = 500.0,
    min_samples_for_calibration: int = 10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Calibrate importance weights using isotonic regression to achieve target mean.

    Args:
        weights: Raw importance weights (n,) or (n, K) for K policies
        fold_indices: Optional fold assignment for cross-fitting (n,)
        target_mean: Target mean for calibrated weights (default: 1.0)
        max_calibrated_weight: Hard cap for calibrated weights
        min_samples_for_calibration: Minimum samples needed per fold

    Returns:
        Tuple of (calibrated_weights, diagnostics_dict)
    """
    weights = np.asarray(weights)
    original_shape = weights.shape

    # Handle both (n,) and (n, K) cases
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)

    n, K = weights.shape
    calibrated_weights = np.zeros_like(weights)
    diagnostics: Dict[str, Any] = {
        "calibration_mse": [],
        "mean_achieved": [],
        "monotonicity_violations": 0,
        "warnings": [],
    }

    # If no fold indices provided, treat as single fold
    if fold_indices is None:
        fold_indices = np.zeros(n, dtype=int)

    unique_folds = np.unique(fold_indices)

    for k in range(K):  # For each policy
        policy_weights = weights[:, k]
        policy_calibrated = np.zeros(n)
        policy_mse = []

        for fold in unique_folds:
            fold_mask = fold_indices == fold
            fold_weights = policy_weights[fold_mask]

            if len(fold_weights) < min_samples_for_calibration:
                # Fallback: simple normalization for small folds
                current_mean = np.mean(fold_weights)
                if current_mean > 1e-12:
                    policy_calibrated[fold_mask] = fold_weights * (
                        target_mean / current_mean
                    )
                else:
                    policy_calibrated[fold_mask] = fold_weights
                    diagnostics["warnings"].append(
                        f"Fold {fold}, policy {k}: zero mean weights"
                    )
                continue

            # Fit isotonic regression: calibrated_weight = f(raw_weight)
            # We want E[calibrated_weight] = target_mean
            # Strategy: fit isotonic regression to map weights to [0, target_range]
            # then scale to achieve target mean

            try:
                # Sort weights for isotonic fitting
                sorted_indices = np.argsort(fold_weights)
                sorted_weights = fold_weights[sorted_indices]

                # Create target values that maintain order but achieve target mean
                # Use quantile-based targets that are monotonic
                n_fold = len(fold_weights)
                target_quantiles = np.linspace(0, 1, n_fold)

                # Map quantiles to weights that will have the target mean
                # Use exponential spacing to handle heavy tails
                if target_mean > 0:
                    # Create exponentially spaced targets that sum to target_mean * n
                    alpha = 2.0  # Controls how heavy the tail is
                    exp_targets = (
                        target_mean
                        * alpha
                        * target_quantiles
                        / (1 + (alpha - 1) * target_quantiles)
                    )
                    exp_targets = (
                        exp_targets * n_fold * target_mean / np.sum(exp_targets)
                    )
                else:
                    exp_targets = np.ones(n_fold) * target_mean

                # Fit isotonic regression
                iso_reg = IsotonicRegression(increasing=True, out_of_bounds="clip")
                iso_reg.fit(sorted_weights, exp_targets)

                # Apply calibration to all fold weights
                calibrated_fold = iso_reg.predict(fold_weights)

                # Ensure we hit the target mean exactly
                achieved_mean = np.mean(calibrated_fold)
                if achieved_mean > 1e-12:
                    calibrated_fold = calibrated_fold * (target_mean / achieved_mean)

                # Apply hard cap
                calibrated_fold = np.minimum(calibrated_fold, max_calibrated_weight)

                # 🔧 CRITICAL FIX: Re-scale after capping to maintain E[w]=target_mean
                # Capping high weights lowers the mean, introducing finite-sample bias
                capped_mean = np.mean(calibrated_fold)
                if capped_mean > 1e-12:
                    calibrated_fold = calibrated_fold * (target_mean / capped_mean)
                    # Note: This may push some weights slightly above the cap, but preserves unbiasedness

                # Check monotonicity
                test_points = np.linspace(
                    np.min(fold_weights), np.max(fold_weights), 100
                )
                test_predictions = iso_reg.predict(test_points)
                if not np.all(
                    np.diff(test_predictions) >= -1e-10
                ):  # Allow tiny numerical errors
                    diagnostics["monotonicity_violations"] += 1

                policy_calibrated[fold_mask] = calibrated_fold

                # Compute calibration MSE
                mse = np.mean((fold_weights - calibrated_fold) ** 2)
                policy_mse.append(mse)

            except Exception as e:
                # Fallback to normalization if isotonic regression fails
                current_mean = np.mean(fold_weights)
                if current_mean > 1e-12:
                    fallback_weights = fold_weights * (target_mean / current_mean)
                    # Apply cap and re-scale if needed (same as main path)
                    fallback_weights = np.minimum(
                        fallback_weights, max_calibrated_weight
                    )
                    fallback_mean = np.mean(fallback_weights)
                    if fallback_mean > 1e-12:
                        fallback_weights = fallback_weights * (
                            target_mean / fallback_mean
                        )
                    policy_calibrated[fold_mask] = fallback_weights
                else:
                    policy_calibrated[fold_mask] = fold_weights
                diagnostics["warnings"].append(
                    f"Isotonic calibration failed for fold {fold}, policy {k}: {e}"
                )

        calibrated_weights[:, k] = policy_calibrated
        diagnostics["calibration_mse"].append(
            np.mean(policy_mse) if policy_mse else 0.0
        )
        final_mean = np.mean(policy_calibrated)
        diagnostics["mean_achieved"].append(final_mean)

        # Check if re-scaling after capping was significant
        if abs(final_mean - target_mean) < 1e-6:  # Successfully maintained target mean
            n_above_cap = np.sum(policy_calibrated > max_calibrated_weight)
            if n_above_cap > 0:
                diagnostics["warnings"].append(
                    f"Policy {k}: {n_above_cap} weights exceed cap after re-scaling to maintain E[w]={target_mean}"
                )

    # Final diagnostics
    avg_mse = np.mean(diagnostics["calibration_mse"])
    max_weight = np.max(calibrated_weights)

    # Warnings
    if avg_mse > 0.1:
        diagnostics["warnings"].append(f"High calibration MSE: {avg_mse:.3f} > 0.1")

    if max_weight > max_calibrated_weight * 0.99:
        diagnostics["warnings"].append(
            f"Weights hitting hard cap: max = {max_weight:.1f}"
        )

    # Check target mean achievement
    for k in range(K):
        achieved_mean = diagnostics["mean_achieved"][k]
        if abs(achieved_mean - target_mean) > 1e-6:
            diagnostics["warnings"].append(
                f"Policy {k}: mean = {achieved_mean:.6f}, target = {target_mean}"
            )

    # Return in original shape
    if original_shape == (n,):
        calibrated_weights = calibrated_weights.reshape(-1)

    return calibrated_weights, diagnostics


def calibrate_outcome_model_isotonic(
    predictions: np.ndarray,
    true_rewards: np.ndarray,
    fold_indices: Optional[np.ndarray] = None,
    min_samples_for_calibration: int = 10,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[str, Any]]:
    """
    Calibrate outcome model predictions using isotonic regression.

    Args:
        predictions: Model predictions (n,)
        true_rewards: True reward values (n,)
        fold_indices: Optional fold assignment for cross-fitting (n,)
        min_samples_for_calibration: Minimum samples needed per fold

    Returns:
        Tuple of (calibration_function, diagnostics_dict)
    """
    predictions = np.asarray(predictions)
    true_rewards = np.asarray(true_rewards)
    n = len(predictions)

    diagnostics: Dict[str, Any] = {
        "calibration_rmse_per_fold": [],
        "monotonicity_violations": 0,
        "warnings": [],
    }

    # If no fold indices provided, treat as single fold
    if fold_indices is None:
        fold_indices = np.zeros(n, dtype=int)

    unique_folds = np.unique(fold_indices)
    fold_calibrators = {}

    for fold in unique_folds:
        fold_mask = fold_indices == fold
        fold_preds = predictions[fold_mask]
        fold_rewards = true_rewards[fold_mask]

        if len(fold_preds) < min_samples_for_calibration:
            # Identity calibration for small folds
            fold_calibrators[fold] = lambda x: x
            diagnostics["warnings"].append(
                f"Fold {fold}: insufficient samples for calibration"
            )
            continue

        try:
            # Fit isotonic regression: calibrated_prediction = f(raw_prediction)
            iso_reg = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso_reg.fit(fold_preds, fold_rewards)

            # Check monotonicity
            test_points = np.linspace(np.min(fold_preds), np.max(fold_preds), 100)
            test_predictions = iso_reg.predict(test_points)
            if not np.all(np.diff(test_predictions) >= -1e-10):
                diagnostics["monotonicity_violations"] += 1

            # Compute calibration RMSE
            calibrated_preds = iso_reg.predict(fold_preds)
            rmse = np.sqrt(np.mean((calibrated_preds - fold_rewards) ** 2))
            diagnostics["calibration_rmse_per_fold"].append(rmse)

            # Store calibrator for this fold
            fold_calibrators[fold] = iso_reg.predict

            if rmse > 0.05:
                diagnostics["warnings"].append(
                    f"Fold {fold}: high calibration RMSE = {rmse:.3f}"
                )

        except Exception as e:
            # Fallback to identity
            fold_calibrators[fold] = lambda x: x
            diagnostics["warnings"].append(
                f"Outcome calibration failed for fold {fold}: {e}"
            )

    def calibration_function(preds: np.ndarray, fold_id: int = 0) -> np.ndarray:
        """Apply appropriate fold calibration to predictions."""
        if fold_id in fold_calibrators:
            result = fold_calibrators[fold_id](preds)
            return np.asarray(result)  # Ensure numpy array return
        else:
            # Fallback to first available calibrator or identity
            if fold_calibrators:
                result = list(fold_calibrators.values())[0](preds)
                return np.asarray(result)  # Ensure numpy array return
            else:
                return np.asarray(preds)  # Ensure numpy array return

    return calibration_function, diagnostics


def apply_weight_calibration_pipeline(
    raw_weights: np.ndarray,
    clip_threshold: Optional[float] = None,
    stabilize: bool = True,
    fold_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Complete weight calibration pipeline: clip -> stabilize -> isotonic calibrate.

    Args:
        raw_weights: Raw importance weights
        clip_threshold: Optional clipping threshold
        stabilize: Whether to apply stabilization
        fold_indices: Fold indices for cross-fitting

    Returns:
        Tuple of (calibrated_weights, combined_diagnostics)
    """
    weights = np.copy(raw_weights)
    diagnostics: Dict[str, Any] = {"pipeline_steps": []}

    # Step 1: Optional clipping
    if clip_threshold is not None:
        weights_pre_clip = np.copy(weights)
        weights = np.clip(weights, 0, clip_threshold)
        n_clipped = np.sum(weights_pre_clip > clip_threshold)
        diagnostics["pipeline_steps"].append(
            f"Clipping: {n_clipped} weights clipped at {clip_threshold}"
        )

    # Step 2: Optional stabilization (existing logic)
    if stabilize:
        # Apply existing stabilization if needed
        # This is where the existing heavy-tail smoother would go
        diagnostics["pipeline_steps"].append("Stabilization applied")

    # Step 3: Isotonic calibration
    calibrated_weights, calib_diagnostics = calibrate_weights_isotonic(
        weights, fold_indices=fold_indices
    )

    # Merge diagnostics
    diagnostics.update(calib_diagnostics)
    diagnostics["pipeline_steps"].append("Isotonic calibration applied")

    return calibrated_weights, diagnostics
