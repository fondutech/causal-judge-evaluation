"""
Oracle-based diagnostics for evaluating outcome model fidelity to true KPI.

This module provides diagnostics that measure how well outcome models
predict the true oracle labels (Y) rather than just the calibrated rewards (R).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.isotonic import IsotonicRegression
from dataclasses import dataclass


@dataclass
class OracleFidelityDiagnostics:
    """Diagnostics for outcome model fidelity to oracle labels."""

    # R²(S→Y): How well does the outcome model predict true Y?
    r2_to_y: float

    # R²(S→R): How well does it predict calibrated rewards?
    r2_to_r: float

    # Mean squared error to Y
    mse_to_y: float

    # Brier score (for probabilistic predictions)
    brier_score: float

    # Calibration statistics
    calibration_slope: float
    calibration_intercept: float

    # Plateau analysis
    n_distinct_steps: int
    median_step_width: float
    effective_resolution: float

    # Sufficiency test results
    sufficiency_test: Optional[Dict[str, float]] = None


def compute_oracle_fidelity(
    g_predictions: np.ndarray,
    judge_scores: np.ndarray,
    oracle_labels: np.ndarray,
    calibrated_rewards: np.ndarray,
    oracle_mask: Optional[np.ndarray] = None,
) -> OracleFidelityDiagnostics:
    """
    Compute R²(S→Y) and other oracle fidelity metrics.

    Args:
        g_predictions: Outcome model predictions g(S)
        judge_scores: Judge scores S
        oracle_labels: True oracle labels Y
        calibrated_rewards: Calibrated rewards R = f(S)
        oracle_mask: Boolean mask for oracle slice (if None, use all)

    Returns:
        OracleFidelityDiagnostics with comprehensive metrics
    """
    if oracle_mask is None:
        oracle_mask = np.ones(len(g_predictions), dtype=bool)

    # Filter to oracle slice
    g_oracle = g_predictions[oracle_mask]
    y_oracle = oracle_labels[oracle_mask]
    r_oracle = calibrated_rewards[oracle_mask]
    s_oracle = judge_scores[oracle_mask]

    # R²(S→Y): How well does g predict true Y?
    r2_to_y = r2_score(y_oracle, g_oracle)

    # R²(S→R): How well does g predict calibrated R?
    r2_to_r = r2_score(r_oracle, g_oracle)

    # MSE and Brier score
    mse_to_y = mean_squared_error(y_oracle, g_oracle)

    # Brier score (assumes predictions in [0,1])
    g_clipped = np.clip(g_oracle, 0, 1)
    brier_score = np.mean((y_oracle - g_clipped) ** 2)

    # Calibration analysis (linear regression of Y on g)
    from sklearn.linear_model import LinearRegression

    cal_model = LinearRegression()
    cal_model.fit(g_oracle.reshape(-1, 1), y_oracle)
    calibration_slope = float(cal_model.coef_[0])
    calibration_intercept = float(cal_model.intercept_)

    # Plateau analysis - analyze the step structure
    plateau_stats = analyze_plateau_structure(s_oracle, g_oracle)

    return OracleFidelityDiagnostics(
        r2_to_y=r2_to_y,
        r2_to_r=r2_to_r,
        mse_to_y=mse_to_y,
        brier_score=brier_score,
        calibration_slope=calibration_slope,
        calibration_intercept=calibration_intercept,
        n_distinct_steps=int(plateau_stats["n_steps"]),
        median_step_width=plateau_stats["median_width"],
        effective_resolution=plateau_stats["effective_resolution"],
    )


def analyze_plateau_structure(
    judge_scores: np.ndarray, predictions: np.ndarray
) -> Dict[str, float]:
    """
    Analyze the plateau/step structure of predictions.

    With limited oracle coverage and isotonic regression, predictions
    often form plateaus. This function measures the resolution.

    Args:
        judge_scores: Input scores
        predictions: Model predictions

    Returns:
        Dictionary with plateau statistics
    """
    # Sort by judge scores
    sorted_idx = np.argsort(judge_scores)
    sorted_scores = judge_scores[sorted_idx]
    sorted_preds = predictions[sorted_idx]

    # Find unique prediction levels (plateaus)
    unique_preds, change_points = np.unique(sorted_preds, return_index=True)
    n_steps = len(unique_preds)

    # Compute step widths in score space
    if n_steps > 1:
        # Get score values at change points
        change_scores = sorted_scores[change_points]
        step_widths = np.diff(change_scores)

        if len(step_widths) > 0:
            median_width = np.median(step_widths)
            min_width = np.min(step_widths)
            max_width = np.max(step_widths)
        else:
            median_width = min_width = max_width = 0.0
    else:
        median_width = min_width = max_width = sorted_scores.max() - sorted_scores.min()

    # Effective resolution
    score_range = sorted_scores.max() - sorted_scores.min()
    effective_resolution = score_range / median_width if median_width > 0 else 1.0

    return {
        "n_steps": n_steps,
        "median_width": median_width,
        "min_width": min_width,
        "max_width": max_width,
        "effective_resolution": effective_resolution,
    }


def test_conditional_sufficiency(
    judge_scores: np.ndarray,
    oracle_labels: np.ndarray,
    prompts: List[str],
    responses: List[str],
    n_bootstrap: int = 100,
) -> Dict[str, Any]:
    """
    Test whether judge score S is a sufficient statistic for Y.

    Tests if adding (X,A) improves prediction beyond S alone.
    If it does, this suggests violation of the A-J2S assumption.

    Args:
        judge_scores: Judge scores S
        oracle_labels: Oracle labels Y
        prompts: Prompt texts X
        responses: Response texts A
        n_bootstrap: Number of bootstrap samples for significance test

    Returns:
        Dictionary with test results and diagnostics
    """
    n = len(judge_scores)

    # Model 1: Y ~ iso(S)
    iso_model = IsotonicRegression(out_of_bounds="clip")
    iso_model.fit(judge_scores, oracle_labels)
    y_pred_s = iso_model.predict(judge_scores)
    mse_s_only = mean_squared_error(oracle_labels, y_pred_s)
    r2_s_only = r2_score(oracle_labels, y_pred_s)

    # Model 2: Y ~ iso(S) + features(X,A)
    # For simplicity, we'll add response length and prompt length as features
    response_lengths = np.array([len(r.split()) for r in responses])
    prompt_lengths = np.array([len(p.split()) for p in prompts])

    # Combine predictions: weighted average of isotonic and feature-based
    from sklearn.linear_model import Ridge

    features = np.column_stack(
        [
            y_pred_s,  # Isotonic predictions
            response_lengths / response_lengths.max(),  # Normalized response length
            prompt_lengths / prompt_lengths.max(),  # Normalized prompt length
        ]
    )

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(features, oracle_labels)
    y_pred_combined = ridge_model.predict(features)
    mse_combined = mean_squared_error(oracle_labels, y_pred_combined)
    r2_combined = r2_score(oracle_labels, y_pred_combined)

    # Relative improvement
    relative_improvement = (
        (mse_s_only - mse_combined) / mse_s_only if mse_s_only > 0 else 0
    )
    r2_improvement = r2_combined - r2_s_only

    # Bootstrap test for significance
    improvements = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)

        # Fit on bootstrap sample
        iso_boot = IsotonicRegression(out_of_bounds="clip")
        iso_boot.fit(judge_scores[idx], oracle_labels[idx])
        y_pred_s_boot = iso_boot.predict(judge_scores[idx])
        mse_s_boot = mean_squared_error(oracle_labels[idx], y_pred_s_boot)

        ridge_boot = Ridge(alpha=1.0)
        features_boot = np.column_stack(
            [
                y_pred_s_boot,
                response_lengths[idx] / response_lengths.max(),
                prompt_lengths[idx] / prompt_lengths.max(),
            ]
        )
        ridge_boot.fit(features_boot, oracle_labels[idx])
        y_pred_combined_boot = ridge_boot.predict(features_boot)
        mse_combined_boot = mean_squared_error(oracle_labels[idx], y_pred_combined_boot)

        imp_boot = (
            (mse_s_boot - mse_combined_boot) / mse_s_boot if mse_s_boot > 0 else 0
        )
        improvements.append(imp_boot)

    # Compute p-value: proportion of bootstrap samples with improvement > 0
    p_value = np.mean(np.array(improvements) > 0)

    return {
        "mse_s_only": mse_s_only,
        "mse_with_features": mse_combined,
        "r2_s_only": r2_s_only,
        "r2_with_features": r2_combined,
        "relative_mse_improvement": relative_improvement,
        "r2_improvement": r2_improvement,
        "p_value": p_value,
        "significant_at_05": p_value < 0.05,
        "sufficiency_risk": "HIGH" if relative_improvement > 0.05 else "LOW",
    }


def compute_surrogate_target_gap(
    g_predictions: np.ndarray,
    oracle_labels: np.ndarray,
    judge_scores: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute the surrogate-target gap δ = Y - g(S).

    This helps identify regions where the surrogate systematically
    over or under-estimates the true outcome.

    Args:
        g_predictions: Outcome model predictions
        oracle_labels: True oracle labels Y
        judge_scores: Judge scores S
        n_bins: Number of bins for binned analysis

    Returns:
        Dictionary with gap statistics
    """
    # Compute gaps
    gaps = oracle_labels - g_predictions

    # Overall statistics
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)

    # Binned analysis by judge score
    score_bins = np.percentile(judge_scores, np.linspace(0, 100, n_bins + 1))
    score_bins[0] -= 1e-6  # Ensure leftmost edge includes minimum
    score_bins[-1] += 1e-6  # Ensure rightmost edge includes maximum

    binned_gaps = []
    bin_centers = []
    bin_sizes = []

    for i in range(n_bins):
        mask = (judge_scores >= score_bins[i]) & (judge_scores < score_bins[i + 1])
        if np.any(mask):
            binned_gaps.append(np.mean(gaps[mask]))
            bin_centers.append(np.mean(judge_scores[mask]))
            bin_sizes.append(np.sum(mask))
        else:
            binned_gaps.append(np.nan)
            bin_centers.append((score_bins[i] + score_bins[i + 1]) / 2)
            bin_sizes.append(0)

    # Check for systematic bias
    # If gaps are consistently positive/negative in certain regions, that's concerning
    max_abs_gap = (
        np.nanmax(np.abs(binned_gaps)) if not np.all(np.isnan(binned_gaps)) else 0
    )

    return {
        "mean_gap": mean_gap,
        "std_gap": std_gap,
        "min_gap": np.min(gaps),
        "max_gap": np.max(gaps),
        "binned_gaps": binned_gaps,
        "bin_centers": bin_centers,
        "bin_sizes": bin_sizes,
        "max_absolute_binned_gap": max_abs_gap,
        "systematic_bias": max_abs_gap > 0.1,  # Flag if any bin has >0.1 average gap
    }
