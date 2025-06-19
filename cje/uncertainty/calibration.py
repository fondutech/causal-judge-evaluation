"""Variance calibration for uncertainty-aware evaluation.

This module implements gamma calibration to correct for systematic
over/under-confidence in judge uncertainty estimates.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
from sklearn.isotonic import IsotonicRegression
import logging

from .schemas import JudgeScore, CalibratedReward

logger = logging.getLogger(__name__)


def calibrate_variance_gamma(
    scores: List[JudgeScore],
    y_true: np.ndarray,
) -> float:
    """Compute variance scale factor γ = Σ(residual²) / Σ(variance).

    This corrects for systematic mis-calibration in judge confidence.
    γ > 1 means judge underestimates uncertainty (underconfident).
    γ < 1 means judge overestimates uncertainty (overconfident).

    Args:
        scores: Judge scores with mean and variance
        y_true: True outcomes for calibration subset

    Returns:
        Gamma scale factor
    """
    if len(scores) != len(y_true):
        raise ValueError(
            f"Mismatched lengths: {len(scores)} scores, {len(y_true)} labels"
        )

    # Extract means and variances
    means = np.array([s.mean for s in scores])
    variances = np.array([s.variance for s in scores])

    # Compute squared residuals
    residuals_squared = (y_true - means) ** 2

    # Sum of squared residuals and variances
    sum_residuals = np.sum(residuals_squared)
    sum_variances = np.sum(variances)

    # Handle edge case
    if sum_variances < 1e-10:
        logger.warning("Sum of variances too small for gamma calibration. Using γ=1.0")
        return 1.0

    # Compute gamma
    gamma = sum_residuals / sum_variances

    # Log calibration result
    logger.info(f"Variance calibration: γ = {gamma:.3f}")
    if gamma > 2.0:
        logger.warning(
            f"Large γ ({gamma:.3f}) indicates judge is significantly underconfident"
        )
    elif gamma < 0.5:
        logger.warning(
            f"Small γ ({gamma:.3f}) indicates judge is significantly overconfident"
        )

    return float(gamma)


def apply_variance_scaling(
    scores: List[JudgeScore],
    gamma: float,
) -> List[JudgeScore]:
    """Apply gamma scaling to judge variances.

    Args:
        scores: Original judge scores
        gamma: Variance scale factor

    Returns:
        New scores with scaled variances
    """
    return [JudgeScore(mean=s.mean, variance=s.variance * gamma) for s in scores]


def calibrate_scores_isotonic(
    scores: List[JudgeScore],
    y_true: np.ndarray,
) -> Tuple[IsotonicRegression, float]:
    """Fit isotonic calibration for score means and compute variance gamma.

    IMPORTANT: The order matters here. We compute gamma AFTER isotonic calibration
    because gamma should measure dispersion (irreducible uncertainty) not bias.
    Computing gamma on raw scores would double-count mean-squared bias as if it
    were stochastic variance, inflating gamma and producing overly wide CIs.

    Args:
        scores: Judge scores with mean and variance
        y_true: True outcomes for calibration

    Returns:
        Tuple of (isotonic_model, gamma)
    """
    if len(scores) != len(y_true):
        raise ValueError(
            f"Mismatched lengths: {len(scores)} scores, {len(y_true)} labels"
        )

    # Extract means for isotonic regression
    means = np.array([s.mean for s in scores])

    # Fit isotonic regression for means to remove bias
    iso_model = IsotonicRegression(out_of_bounds="clip")
    iso_model.fit(means, y_true)

    # Get calibrated means (debiased)
    calibrated_means = iso_model.predict(means)

    # Create calibrated scores for gamma computation
    calibrated_scores = [
        JudgeScore(mean=m, variance=s.variance)
        for m, s in zip(calibrated_means, scores)
    ]

    # Compute variance calibration on DEBIASED residuals
    # This measures only irreducible uncertainty, not bias
    gamma = calibrate_variance_gamma(calibrated_scores, y_true)

    return iso_model, gamma


def apply_full_calibration(
    scores: List[JudgeScore],
    iso_model: IsotonicRegression,
    gamma: float,
) -> List[CalibratedReward]:
    """Apply both mean and variance calibration to scores.

    Args:
        scores: Raw judge scores
        iso_model: Fitted isotonic model for means
        gamma: Variance scale factor

    Returns:
        Fully calibrated rewards
    """
    calibrated_rewards = []

    for score in scores:
        # Calibrate mean
        calibrated_mean = float(iso_model.predict([score.mean])[0])

        # Calibrate variance
        calibrated_variance = score.variance * gamma

        # Create calibrated reward
        reward = CalibratedReward(
            value=calibrated_mean,
            variance=calibrated_variance,
            gamma=gamma,
            raw_score=score.mean,
            raw_variance=score.variance,
        )
        calibrated_rewards.append(reward)

    return calibrated_rewards


def validate_calibration(
    calibrated_rewards: List[CalibratedReward],
    min_range: float = 0.05,
) -> Dict[str, Any]:
    """Validate calibration didn't collapse to constant.

    Args:
        calibrated_rewards: Calibrated rewards to check
        min_range: Minimum acceptable range of calibrated values

    Returns:
        Dictionary with validation results
    """
    values = [r.value for r in calibrated_rewards]
    val_min = min(values)
    val_max = max(values)
    val_range = val_max - val_min

    is_valid = val_range >= min_range

    if not is_valid:
        logger.warning(
            f"Calibration may have collapsed! Range: {val_range:.4f} "
            f"(min={val_min:.4f}, max={val_max:.4f})"
        )

    return {
        "is_valid": is_valid,
        "range": val_range,
        "min": val_min,
        "max": val_max,
        "message": (
            "OK" if is_valid else f"Range {val_range:.4f} below threshold {min_range}"
        ),
    }
