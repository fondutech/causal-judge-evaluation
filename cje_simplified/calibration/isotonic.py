# isotonic.py  ·  CJE calibration helpers
# ------------------------------------------------------------
# 0) cross_fit_isotonic : vanilla K-fold isotonic regression
# 1) calibrate_to_target_mean : Calibrated-DML weight calibration
#    with a variance-increase fallback
# ------------------------------------------------------------
from __future__ import annotations
import numpy as np
import logging
from sklearn.isotonic import IsotonicRegression, isotonic_regression
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

# Tolerance constants for readability
MEAN_TOL = 1e-12
VAR_TOL = 1.001


# ---------------------------------------------------------------------
# 0.  Generic isotonic cross-fit  (unchanged)
# ---------------------------------------------------------------------
def cross_fit_isotonic(
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
    random_seed: int = 42,
    *,
    out_of_bounds: str = "clip",
) -> np.ndarray:
    n = len(X)
    k_folds = max(2, min(k_folds, n // 2))
    if n < 4:
        raise ValueError("Need ≥4 observations for cross-fit")

    calibrated = np.empty_like(X, dtype=float)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    for train, test in kf.split(X):
        iso = IsotonicRegression(out_of_bounds=out_of_bounds)
        iso.fit(X[train], y[train])
        calibrated[test] = iso.predict(X[test])

    return calibrated


# ---------------------------------------------------------------------
# 1-a.  Mean-1 PAV projection on *sorted* weights
# ---------------------------------------------------------------------
def _pav_mean1_projection(w_sorted: np.ndarray) -> np.ndarray:
    """
    Pool-Adjacent-Violators projection to monotone non-decreasing vector with mean 1.

    CRITICAL: Input MUST be sorted by its own values (ascending order).
    Do NOT pass data sorted by external criteria (e.g., by different weights).

    Parameters
    ----------
    w_sorted : np.ndarray
        Weights sorted in ascending order BY THEIR OWN VALUES.

    Returns
    -------
    np.ndarray
        Monotone non-decreasing vector with exact mean 1.0 and reduced variance.
    """
    n = len(w_sorted)
    if n == 1:
        return np.array([1.0])

    # Verify input is actually sorted (with small tolerance for numerical error)
    if not np.all(np.diff(w_sorted) >= -1e-12):
        raise ValueError(
            "_pav_mean1_projection requires input sorted in ascending order. "
            "The input appears to be unsorted or sorted by external criteria."
        )

    w_norm = w_sorted / w_sorted.mean()  # mean = 1
    z = np.cumsum(w_norm - 1.0)  # excess mass
    y = isotonic_regression(z, increasing=False)  # PAV
    v = np.diff(np.concatenate(([0.0], y))) + 1.0
    v = np.clip(v, 0.0, None)
    v *= n / v.sum()  # exact mean 1
    return v


# ---------------------------------------------------------------------
# 1-b.  Variance-safe blending utility
# ---------------------------------------------------------------------
def variance_safe_blend(
    raw_weights: np.ndarray,
    calibrated_weights: np.ndarray,
    target_mean: float = 1.0,
    max_variance_ratio: float = 1.0,
) -> np.ndarray:
    """Blend raw and calibrated weights to ensure variance constraint.

    This preserves monotonicity if calibrated_weights is monotone in raw_weights.
    The blend is: w = alpha * calibrated + (1-alpha) * raw_normalized

    Args:
        raw_weights: Original uncalibrated weights
        calibrated_weights: Isotonic calibrated weights
        target_mean: Target mean for final weights
        max_variance_ratio: Maximum allowed var(output)/var(raw)

    Returns:
        Blended weights with mean=target_mean and var <= var(raw) * max_variance_ratio
    """
    raw = raw_weights.astype(float)
    cal = calibrated_weights.astype(float)

    # Normalize both to mean=1 for clean blending
    raw_norm = raw / (raw.mean() + 1e-12)
    cal_norm = cal / (cal.mean() + 1e-12)

    raw_var = raw_norm.var()
    cal_var = cal_norm.var()
    target_var = raw_var * max_variance_ratio

    # If calibrated already satisfies constraint, use it
    if cal_var <= target_var:
        result = cal_norm
    else:
        # Binary search for optimal blend
        # alpha=1 -> calibrated, alpha=0 -> raw
        lo, hi = 0.0, 1.0
        best_alpha = 0.0

        for _ in range(30):  # More iterations for precision
            alpha = 0.5 * (lo + hi)
            blend = alpha * cal_norm + (1 - alpha) * raw_norm
            blend_var = blend.var()

            if blend_var <= target_var:
                best_alpha = alpha
                lo = alpha  # Can increase alpha
            else:
                hi = alpha  # Must decrease alpha

        result = best_alpha * cal_norm + (1 - best_alpha) * raw_norm

        # Log the blend for transparency
        logger.debug(
            f"Variance-safe blend: alpha={best_alpha:.3f}, "
            f"var_ratio={result.var()/raw_var:.3f}"
        )

    # Scale to target mean
    result *= target_mean / (result.mean() + 1e-12)

    # Verify constraints (with small tolerance for numerical precision)
    assert abs(result.mean() - target_mean) < MEAN_TOL, "Mean not preserved"
    # Note: Due to the final rescaling for mean, variance might slightly exceed target
    # This is acceptable as long as it's within numerical tolerance
    final_var_ratio = result.var() / raw_var
    if (
        final_var_ratio > max_variance_ratio * 1.01
    ):  # 1% tolerance for numerical precision
        logger.warning(
            f"Variance constraint slightly exceeded due to mean rescaling: "
            f"ratio={final_var_ratio:.4f} (target={max_variance_ratio})"
        )

    return np.asarray(result)


# ---------------------------------------------------------------------
# 1-c.  Calibrated-DML weight calibration with variance safeguard
# ---------------------------------------------------------------------
def calibrate_to_target_mean(
    weights: np.ndarray,
    target_mean: float = 1.0,
    k_folds: int = 3,
    random_seed: int = 42,
    enforce_variance_nonincrease: bool = True,  # Default: prevent variance explosion
) -> np.ndarray:
    """
    Monotone, mean-preserving calibration for importance weights.

    • Uses mean-1 PAV projection cross-fitted across folds.
    • Runs a second mean-1 PAV projection on the merged vector to heal
      fold boundaries (weights-only ⇒ no leakage).
    • **Default safeguard**: Blends with raw weights if calibration would
      increase variance, preventing pathological variance explosion.

    Guarantees
    ----------
    * output ≥ 0
    * exact sample mean == target_mean
    * variance never exceeds raw variance (when enforce_variance_nonincrease=True)
    * global monotonicity in the rank order of raw weights

    Note: enforce_variance_nonincrease defaults to True to prevent the common
    case of 10x-40x variance explosions with poor overlap. Set to False only
    if you specifically want pure isotonic calibration without variance control.
    """
    n = len(weights)
    if n < 4:
        return np.asarray(weights * (target_mean / weights.mean()))

    # If weights are essentially uniform (e.g., all 1.0), just rescale to target mean
    # This happens when base and target policies are identical
    weight_range = weights.max() - weights.min()
    if weight_range < 1e-6 or weights.var() < 1e-6:
        logger.debug(
            f"Weights are essentially uniform (range={weight_range:.2e}, var={weights.var():.2e}). "
            f"Skipping calibration, just rescaling to target_mean={target_mean}"
        )
        return np.full_like(weights, target_mean)

    logger.debug(
        f"calibrate_to_target_mean: n_samples={n}, "
        f"raw_mean={weights.mean():.3f}, raw_std={weights.std():.3f}, "
        f"raw_var={weights.var():.6f}, raw_range=[{weights.min():.3f}, {weights.max():.3f}], "
        f"target_mean={target_mean}, k_folds={k_folds}"
    )

    # ---------- per-fold projection ----------
    k_folds = max(2, min(k_folds, n // 2))
    prelim = np.zeros_like(weights, dtype=float)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

    for train, test in kf.split(weights):
        w_train = weights[train]
        order = np.argsort(w_train)
        v_train = _pav_mean1_projection(w_train[order])

        # Use IsotonicRegression for monotone-safe weight → plateau mapping
        iso_map = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso_map.fit(w_train[order], v_train)  # learn weight → plateau
        prelim[test] = iso_map.predict(weights[test])  # monotone by construction

    # ---------- global monotone fix-up (weights-only, no leakage) ----------
    # This step ensures strict monotonicity across fold boundaries
    order_all = np.argsort(weights)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(np.arange(n), prelim[order_all])  # x = indices, y = preliminary values
    final_sorted = iso.predict(np.arange(n))
    calibrated = np.empty_like(prelim)
    calibrated[order_all] = final_sorted

    # ---------- rescale to target_mean ----------
    calibrated *= target_mean / calibrated.mean()

    # ---------- enforce variance constraint if requested ----------
    if enforce_variance_nonincrease:
        # Compare variance in normalized space (both with mean=1)
        raw_normalized = weights * (target_mean / weights.mean())
        raw_norm_var = raw_normalized.var()
        cal_var = calibrated.var()

        if cal_var > raw_norm_var * VAR_TOL:  # Small tolerance for numerical precision
            logger.debug(
                f"Applying variance-safe blend: raw_norm_var={raw_norm_var:.6f}, "
                f"cal_var={cal_var:.6f}"
            )
            calibrated = variance_safe_blend(
                raw_normalized, calibrated, target_mean, max_variance_ratio=1.0
            )

    # ---------- log calibration statistics ----------
    raw_normalized = weights * (target_mean / weights.mean())
    raw_norm_var = raw_normalized.var()
    norm_variance_ratio = calibrated.var() / (raw_norm_var + 1e-12)

    logger.debug(
        f"After calibration: mean={calibrated.mean():.6f}, var={calibrated.var():.6f}, "
        f"norm_variance_ratio={norm_variance_ratio:.3f}"
    )

    # ---------- final checks ----------
    # Critical assertions
    assert calibrated.min() >= 0.0, f"Negative calibrated weight: {calibrated.min()}"
    assert (
        abs(calibrated.mean() - target_mean) < MEAN_TOL
    ), f"Mean not preserved: {calibrated.mean()} != {target_mean}"

    # Note: With enforce_variance_nonincrease=True (default), we blend with raw weights
    # to ensure variance doesn't exceed the normalized baseline. Without enforcement,
    # isotonic can increase variance to achieve L2-optimality.

    # Calculate diagnostics for logging
    ess_before = (weights.sum() ** 2) / (weights**2).sum() if weights.sum() > 0 else 0
    ess_after = (
        (calibrated.sum() ** 2) / (calibrated**2).sum() if calibrated.sum() > 0 else 0
    )

    # Log calibration diagnostics
    logger.debug(
        f"Weight calibration: var {weights.var():.3e} → {calibrated.var():.3e} "
        f"(norm_ratio: {norm_variance_ratio:.2f}), ESS {ess_before:.1f} → {ess_after:.1f}"
    )

    # Warn only for truly suspicious cases (relative to normalized baseline)
    if norm_variance_ratio > 3.0 and raw_norm_var > 1e-3:
        logger.warning(
            f"Large normalized variance after calibration: {norm_variance_ratio:.1f}x "
            f"(may indicate poor overlap between policies)."
        )

    # Check monotonicity
    sorted_idx = np.argsort(weights)
    sorted_cal = calibrated[sorted_idx]
    diffs = np.diff(sorted_cal)
    min_diff = np.min(diffs) if len(diffs) > 0 else 0
    assert (
        min_diff >= -1e-12
    ), f"Monotonicity violated: min_diff = {min_diff:.2e} < -1e-12"

    return calibrated


# ---------------------------------------------------------------------
# 2.  Diagnostics helper
# ---------------------------------------------------------------------
def compute_calibration_diagnostics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    *,
    coverage_threshold: float = 0.1,
) -> dict[str, float]:
    residuals = predictions - actuals
    return dict(
        rmse=float(np.sqrt(np.mean(residuals**2))),
        mae=float(np.mean(np.abs(residuals))),
        coverage=float(np.mean(np.abs(residuals) <= coverage_threshold)),
        correlation=float(np.corrcoef(predictions, actuals)[0, 1]),
    )
