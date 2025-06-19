"""Uncertainty-aware calibration for judge scores with variance."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)


def calibrate_beta_distribution(
    alpha: np.ndarray, beta: np.ndarray, y_true: np.ndarray
) -> Tuple[IsotonicRegression, IsotonicRegression]:
    """
    Calibrate Beta distribution parameters to match true outcomes.

    Args:
        alpha: Beta distribution alpha parameters
        beta: Beta distribution beta parameters
        y_true: True outcomes for calibration

    Returns:
        Tuple of (mean_calibrator, variance_calibrator)
    """
    # Compute uncalibrated mean and variance
    mean_uncal = alpha / (alpha + beta)
    var_uncal = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    # Fit isotonic regression for mean
    mean_calibrator = IsotonicRegression(out_of_bounds="clip")
    mean_calibrator.fit(mean_uncal, y_true)

    # Compute residual variance after mean calibration
    mean_cal = mean_calibrator.predict(mean_uncal)
    residuals_squared = (y_true - mean_cal) ** 2

    # Fit isotonic regression for variance calibration
    # Ensures calibrated variance matches empirical residual variance
    var_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0)
    var_calibrator.fit(var_uncal, residuals_squared)

    return mean_calibrator, var_calibrator


def compute_variance_scale_gamma(
    mean_scores: np.ndarray,
    raw_variances: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Compute variance scale factor γ̂ = Σ(residual²)/Σ(v_raw).

    This corrects for systematic over/under-confidence in judge uncertainty.

    Args:
        mean_scores: Judge mean scores (already calibrated if applicable)
        raw_variances: Raw judge variances
        y_true: True outcomes

    Returns:
        Scale factor γ̂ to multiply variances by
    """
    # Compute squared residuals
    residuals_squared = (y_true - mean_scores) ** 2

    # Sum of squared residuals
    sum_residuals_squared = np.sum(residuals_squared)

    # Sum of raw variances
    sum_raw_variances = np.sum(raw_variances)

    # Avoid division by zero
    if sum_raw_variances < 1e-10:
        logger.warning("Sum of raw variances too small. Using gamma=1.0")
        return 1.0

    # Compute gamma
    gamma = sum_residuals_squared / sum_raw_variances

    logger.info(f"Variance scale calibration: γ̂ = {gamma:.3f}")
    if gamma > 2.0:
        logger.warning(
            f"Large gamma ({gamma:.3f}) indicates judge is very underconfident"
        )
    elif gamma < 0.5:
        logger.warning(
            f"Small gamma ({gamma:.3f}) indicates judge is very overconfident"
        )

    return float(gamma)


def apply_variance_shrinkage(
    weights: np.ndarray,
    variances: np.ndarray,
    shrinkage_lambda: Optional[float] = None,
    rewards: Optional[np.ndarray] = None,
    outcome_preds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Apply variance-aware shrinkage to importance weights.

    w_i^star(λ) = w_i^raw / (1 + λ * v_i)

    Args:
        weights: Raw importance weights
        variances: Judge score variances
        shrinkage_lambda: Shrinkage parameter (None = auto-select)
        rewards: Calibrated rewards (for auto-selection)
        outcome_preds: Outcome model predictions (for auto-selection)

    Returns:
        Tuple of (shrunk_weights, lambda_used)
    """
    if shrinkage_lambda is None:
        # Auto-select optimal lambda using Equation 7 from spec
        if rewards is None or outcome_preds is None:
            logger.warning(
                "Cannot auto-select lambda without rewards and outcome predictions. Using lambda=0."
            )
            shrinkage_lambda = 0.0
        else:
            shrinkage_lambda = compute_optimal_lambda(
                weights, variances, rewards, outcome_preds
            )

    # Apply shrinkage: w_star = w / (1 + lambda * v)
    shrunk_weights = weights / (1 + shrinkage_lambda * variances)

    return shrunk_weights, shrinkage_lambda


def compute_optimal_lambda(
    weights: np.ndarray,
    variances: np.ndarray,
    rewards: np.ndarray,
    outcome_preds: np.ndarray,
) -> float:
    """
    Compute optimal shrinkage parameter λ* from Equation 7.

    λ* = Cov[w²v, w(r - m)] / E[w²v²]
    """
    # Compute components
    w_squared_v = weights**2 * variances
    w_times_residual = weights * (rewards - outcome_preds)

    # Compute covariance
    cov = np.cov(w_squared_v, w_times_residual)[0, 1]

    # Compute denominator
    w_squared_v_squared = weights**2 * variances**2
    denominator = np.mean(w_squared_v_squared)

    # Avoid division by zero
    if denominator < 1e-10:
        logger.warning("Denominator too small for lambda calculation. Using lambda=0.")
        return 0.0

    # Compute optimal lambda
    lambda_star = max(0.0, cov / denominator)  # Ensure non-negative

    logger.info(f"Auto-selected shrinkage lambda: {lambda_star:.4f}")

    return float(lambda_star)


def compute_uncertainty_aware_standard_error(
    eif_values: np.ndarray,
    weights: np.ndarray,
    variances: np.ndarray,
) -> float:
    """
    Compute standard error including judge uncertainty contribution.

    SE = sqrt(Var[ψ]/n + E[w²v]/n)

    Args:
        eif_values: Efficient influence function values
        weights: Calibrated importance weights
        variances: Judge score variances

    Returns:
        Standard error including uncertainty
    """
    n = len(eif_values)

    # EIF variance component
    eif_var = np.var(eif_values, ddof=1)

    # Judge variance component: E[w²v]
    judge_var_contribution = np.mean(weights**2 * variances)

    # Total variance
    total_var = eif_var + judge_var_contribution

    # Standard error
    se = np.sqrt(total_var / n)

    return float(se)


def create_variance_diagnostics(
    weights: np.ndarray,
    variances: np.ndarray,
    eif_values: np.ndarray,
) -> Dict[str, float]:
    """
    Create diagnostics for uncertainty contributions.

    Returns:
        Dictionary with diagnostic metrics
    """
    # Compute variance components
    eif_var = np.var(eif_values, ddof=1)
    judge_var_contribution = np.mean(weights**2 * variances)
    total_var = eif_var + judge_var_contribution

    # Percentage contributions
    eif_pct = (eif_var / total_var) * 100 if total_var > 0 else 0
    judge_pct = (judge_var_contribution / total_var) * 100 if total_var > 0 else 0

    # Per-sample contributions
    per_sample_judge_var = weights**2 * variances

    diagnostics = {
        "eif_variance": float(eif_var),
        "judge_variance_contribution": float(judge_var_contribution),
        "total_variance": float(total_var),
        "eif_percentage": float(eif_pct),
        "judge_percentage": float(judge_pct),
        "median_per_sample_judge_var": float(np.median(per_sample_judge_var)),
        "max_per_sample_judge_var": float(np.max(per_sample_judge_var)),
        "variance_concentration": float(
            np.sum(per_sample_judge_var[: int(0.1 * len(per_sample_judge_var))])
            / np.sum(per_sample_judge_var)
        ),  # How much of variance comes from top 10% samples
    }

    return diagnostics
