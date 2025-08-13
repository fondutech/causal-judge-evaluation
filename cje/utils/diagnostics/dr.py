"""
Doubly robust diagnostic computations.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from scipy import stats
import logging

from .weights import tail_weight_ratio, mass_concentration

logger = logging.getLogger(__name__)


def _p_value_from_z(z: float) -> float:
    """Convert z-score to two-sided p-value."""
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def compute_dr_policy_diagnostics(
    dm_component: np.ndarray,
    ips_correction: np.ndarray,
    dr_estimate: float,
    fresh_rewards: Optional[np.ndarray] = None,
    outcome_predictions: Optional[np.ndarray] = None,
    influence_functions: Optional[np.ndarray] = None,
    unique_folds: Optional[List[int]] = None,
    policy: str = "unknown",
) -> Dict[str, Any]:
    """Compute comprehensive DR diagnostics for a single policy.

    Args:
        dm_component: Direct method (outcome model) component
        ips_correction: IPS correction component
        dr_estimate: Final DR estimate
        fresh_rewards: Fresh draw rewards (for outcome model R²)
        outcome_predictions: Outcome model predictions
        influence_functions: Per-sample influence functions
        unique_folds: Unique fold IDs used in cross-fitting
        policy: Policy name

    Returns:
        Dictionary with diagnostic metrics
    """
    n = len(dm_component)

    diagnostics = {
        "policy": policy,
        "n_samples": n,
        "dm_mean": float(dm_component.mean()),
        "ips_corr_mean": float(ips_correction.mean()),
        "dr_estimate": float(dr_estimate),
        "dm_std": float(dm_component.std()),
        "ips_corr_std": float(ips_correction.std()),
    }

    # Outcome model fit (if fresh rewards available)
    if fresh_rewards is not None and outcome_predictions is not None:
        mask = ~np.isnan(fresh_rewards) & ~np.isnan(outcome_predictions)
        if mask.sum() > 0:
            residuals = fresh_rewards[mask] - outcome_predictions[mask]
            diagnostics["residual_mean"] = float(residuals.mean())
            diagnostics["residual_std"] = float(residuals.std())
            diagnostics["residual_rmse"] = float(np.sqrt((residuals**2).mean()))

            # R² (out-of-fold if cross-fitted)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((fresh_rewards[mask] - fresh_rewards[mask].mean()) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-12)
            diagnostics["r2_oof"] = float(r2)
        else:
            diagnostics["r2_oof"] = np.nan
            diagnostics["residual_rmse"] = np.nan

    # Influence function diagnostics
    if influence_functions is not None:
        diagnostics["if_mean"] = float(influence_functions.mean())
        diagnostics["if_std"] = float(influence_functions.std())
        diagnostics["if_var"] = float(influence_functions.var())

        # Percentiles for visualization
        diagnostics["if_p1"] = float(np.percentile(influence_functions, 1))
        diagnostics["if_p5"] = float(np.percentile(influence_functions, 5))
        diagnostics["if_p25"] = float(np.percentile(influence_functions, 25))
        diagnostics["if_p50"] = float(np.percentile(influence_functions, 50))
        diagnostics["if_p75"] = float(np.percentile(influence_functions, 75))
        diagnostics["if_p95"] = float(np.percentile(influence_functions, 95))
        diagnostics["if_p99"] = float(np.percentile(influence_functions, 99))

        # Check for heavy tails
        diagnostics["if_tail_ratio_99_5"] = tail_weight_ratio(
            np.abs(influence_functions), 0.05, 0.99
        )
        diagnostics["if_top1_mass"] = mass_concentration(
            np.abs(influence_functions), 0.01
        )

        # Score function test (should be mean zero for TMLE)
        score_mean = influence_functions.mean()
        score_se = influence_functions.std() / np.sqrt(n)
        score_z = score_mean / score_se if score_se > 0 else 0
        diagnostics["score_mean"] = float(score_mean)
        diagnostics["score_se"] = float(score_se)
        diagnostics["score_z"] = float(score_z)
        diagnostics["score_p"] = _p_value_from_z(score_z)

    # Cross-fitting info
    if unique_folds is not None:
        diagnostics["cross_fitted"] = True
        diagnostics["unique_folds"] = len(unique_folds)
    else:
        diagnostics["cross_fitted"] = False
        diagnostics["unique_folds"] = 1

    # Coverage check (do we have enough fresh draws?)
    diagnostics["coverage_ok"] = True
    if fresh_rewards is not None:
        coverage = (~np.isnan(fresh_rewards)).mean()
        diagnostics["fresh_draw_coverage"] = float(coverage)
        if coverage < 0.8:
            diagnostics["coverage_ok"] = False
            logger.warning(f"Low fresh draw coverage for {policy}: {coverage:.1%}")

    # Component correlation (ideally low for orthogonality)
    corr = np.corrcoef(dm_component, ips_correction)[0, 1]
    diagnostics["component_correlation"] = float(corr) if not np.isnan(corr) else 0.0

    return diagnostics


def compute_dr_diagnostics_all(
    estimator: Any,
    influence_functions: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Compute DR diagnostics for all policies.

    Args:
        estimator: A DR estimator with _dm_component and _ips_correction
        influence_functions: Optional dict of influence functions by policy

    Returns:
        Dictionary with per-policy diagnostics and summary metrics
    """
    all_diagnostics = {}

    for policy in estimator.sampler.target_policies:
        # Get components
        dm = estimator._dm_component.get(policy)
        ips = estimator._ips_correction.get(policy)

        if dm is None or ips is None:
            logger.warning(f"Missing DR components for {policy}")
            continue

        # Get optional data
        fresh_rewards = None
        outcome_preds = None
        if hasattr(estimator, "_fresh_rewards"):
            fresh_rewards = estimator._fresh_rewards.get(policy)
        if hasattr(estimator, "_outcome_predictions"):
            outcome_preds = estimator._outcome_predictions.get(policy)

        # Get influence functions if available
        ifs = None
        if influence_functions and policy in influence_functions:
            ifs = influence_functions[policy]

        # Compute diagnostics
        all_diagnostics[policy] = compute_dr_policy_diagnostics(
            dm_component=dm,
            ips_correction=ips,
            dr_estimate=dm.mean() + ips.mean(),
            fresh_rewards=fresh_rewards,
            outcome_predictions=outcome_preds,
            influence_functions=ifs,
            policy=policy,
        )

    return all_diagnostics
