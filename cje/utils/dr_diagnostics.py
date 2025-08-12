"""Diagnostics for Doubly Robust estimators.

Provides lightweight diagnostics to catch common DR failure modes:
- Leakage and cross-fitting issues
- Poor outcome model fit
- Heavy tails in influence functions
- Orthogonality violations
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from math import erf, sqrt

from .diagnostics import tail_weight_ratio, mass_concentration


@dataclass
class DRPolicyDiagnostics:
    """Diagnostics for a single policy in DR estimation."""

    # Basic stats
    n: int
    dm_mean: float
    ips_corr_mean: float
    dr_estimate: float
    se: float

    # Orthogonality check
    score_mean: float
    score_se: float
    score_z: float
    score_p: float

    # Outcome model fit
    residual_rmse: float
    r2_oof: float

    # EIF tail behavior
    if_var: float
    if_p95: float
    if_p99: float
    if_top1_share: float
    if_tail_ratio_99_5: float

    # Fresh draw stats
    draws_per_prompt: int
    g_fresh_draw_var_mean: float
    coverage_ok: bool
    missing_prompts: int

    # Cross-fitting info
    cross_fitted: bool
    unique_folds: int


def _p_value_from_z(z: float) -> float:
    """Two-sided p-value from standard normal z-score."""
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2))))


def compute_dr_policy_diagnostics(
    weights: np.ndarray,
    rewards: np.ndarray,
    g_logged: np.ndarray,
    g_fresh: np.ndarray,
    dr_estimate: float,
    se: float,
    fresh_draw_var_per_prompt: Optional[np.ndarray] = None,
    draws_per_prompt: int = 0,
    coverage_ok: bool = True,
    missing_prompts: int = 0,
    cross_fitted: bool = True,
    n_folds: int = 5,
) -> DRPolicyDiagnostics:
    """Compute comprehensive DR diagnostics for a single policy.

    Args:
        weights: Mean-one calibrated weights
        rewards: Logged rewards
        g_logged: OOF outcome model predictions for logged data
        g_fresh: Per-prompt averages from fresh draws
        dr_estimate: Final DR estimate
        se: Standard error of DR estimate
        fresh_draw_var_per_prompt: Variance of g(draws) per prompt
        draws_per_prompt: Number of fresh draws per prompt
        coverage_ok: Whether fresh draws have good coverage
        missing_prompts: Number of prompts missing fresh draws
        cross_fitted: Whether cross-fitting was used
        n_folds: Number of CV folds

    Returns:
        DRPolicyDiagnostics with all computed metrics
    """
    n = len(rewards)

    # Residuals and scores
    resid = rewards - g_logged
    score = weights * resid

    # Orthogonality check (should be ~0 for TMLE)
    score_mean = float(np.mean(score))
    score_sd = float(np.std(score, ddof=1)) if n > 1 else 0.0
    score_se = score_sd / np.sqrt(n) if n > 1 else 0.0
    score_z = score_mean / (score_se + 1e-12)
    score_p = float(_p_value_from_z(score_z))

    # Outcome model fit
    residual_rmse = float(np.sqrt(np.mean(resid**2)))
    reward_var = np.var(rewards)
    r2_oof = (
        float(1.0 - np.var(resid) / (reward_var + 1e-12)) if reward_var > 1e-12 else 0.0
    )

    # Influence function analysis
    IF = g_fresh + score - dr_estimate
    absIF = np.abs(IF)
    if_var = float(np.var(IF, ddof=1)) if n > 1 else 0.0
    if_p95 = float(np.quantile(absIF, 0.95))
    if_p99 = float(np.quantile(absIF, 0.99))
    if_top1_share = mass_concentration(absIF, top_pct=0.01)
    if_tail = tail_weight_ratio(absIF, 0.05, 0.99)

    # Fresh draw variance
    g_fresh_draw_var_mean = 0.0
    if fresh_draw_var_per_prompt is not None and len(fresh_draw_var_per_prompt) > 0:
        g_fresh_draw_var_mean = float(np.mean(fresh_draw_var_per_prompt))

    return DRPolicyDiagnostics(
        n=n,
        dm_mean=float(np.mean(g_fresh)),
        ips_corr_mean=float(np.mean(score)),
        dr_estimate=float(dr_estimate),
        se=float(se),
        score_mean=score_mean,
        score_se=score_se,
        score_z=float(score_z),
        score_p=score_p,
        residual_rmse=residual_rmse,
        r2_oof=r2_oof,
        if_var=if_var,
        if_p95=if_p95,
        if_p99=if_p99,
        if_top1_share=if_top1_share,
        if_tail_ratio_99_5=if_tail,
        draws_per_prompt=int(draws_per_prompt),
        g_fresh_draw_var_mean=g_fresh_draw_var_mean,
        coverage_ok=bool(coverage_ok),
        missing_prompts=int(missing_prompts),
        cross_fitted=bool(cross_fitted),
        unique_folds=int(n_folds),
    )


def compute_dr_diagnostics_all(
    estimation_result: Any, per_policy: bool = True
) -> Dict[str, Any]:
    """Extract and aggregate DR diagnostics from estimation result.

    Args:
        estimation_result: Result from DR estimator with diagnostics in metadata
        per_policy: Whether to include per-policy details

    Returns:
        Dictionary with aggregated diagnostics
    """
    if "dr_diagnostics" not in estimation_result.metadata:
        return {}

    dr_diags = estimation_result.metadata["dr_diagnostics"]
    policies = list(dr_diags.keys())

    result = {
        "policies": policies,
        "n_policies": len(policies),
    }

    if per_policy:
        result["per_policy"] = dr_diags

    # Aggregate key metrics
    if policies:
        result["dm_vs_ips"] = {
            p: (d["dm_mean"], d["ips_corr_mean"]) for p, d in dr_diags.items()
        }

        result["worst_if_tail_ratio"] = max(
            d["if_tail_ratio_99_5"] for d in dr_diags.values()
        )

        result["best_r2_oof"] = max(d["r2_oof"] for d in dr_diags.values())

        result["worst_r2_oof"] = min(d["r2_oof"] for d in dr_diags.values())

        # For TMLE, check orthogonality
        if estimation_result.method == "tmle":
            result["tmle_score_abs_mean"] = {
                p: abs(d["score_mean"]) for p, d in dr_diags.items()
            }
            result["tmle_max_score_z"] = max(
                abs(d["score_z"]) for d in dr_diags.values()
            )

    return result


def format_dr_diagnostic_summary(diagnostics: Dict[str, Any]) -> str:
    """Format DR diagnostics as a readable table.

    Args:
        diagnostics: Dictionary from compute_dr_diagnostics_all

    Returns:
        Formatted string table
    """
    if not diagnostics or "per_policy" not in diagnostics:
        return "No DR diagnostics available"

    lines = []
    lines.append("=" * 100)
    lines.append("DR DIAGNOSTICS SUMMARY")
    lines.append("=" * 100)

    # Header
    lines.append(
        f"{'Policy':<15} {'DM':>8} {'IPS':>8} {'DR±SE':<20} "
        f"{'Score(mean±se, p)':<25} {'RMSE(R,g)':>10} {'|IF| tail(p99/p5)':>18}"
    )
    lines.append("-" * 100)

    # Per-policy rows
    for policy, diag in diagnostics["per_policy"].items():
        dm = diag["dm_mean"]
        ips = diag["ips_corr_mean"]
        dr = diag["dr_estimate"]
        se = diag["se"]
        score_mean = diag["score_mean"]
        score_se = diag["score_se"]
        score_p = diag["score_p"]
        rmse = diag["residual_rmse"]
        tail_ratio = diag["if_tail_ratio_99_5"]

        dr_str = f"{dr:>7.3f}±{se:<5.3f}"
        score_str = f"{score_mean:>6.3f}±{score_se:<5.3f} (p={score_p:.2f})"

        lines.append(
            f"{policy:<15} {dm:>8.3f} {ips:>8.3f} {dr_str:<20} "
            f"{score_str:<25} {rmse:>10.3f} {tail_ratio:>18.1f}"
        )

    lines.append("-" * 100)

    # Summary stats
    if "worst_if_tail_ratio" in diagnostics:
        lines.append(
            f"Worst IF tail ratio (p99/p5): {diagnostics['worst_if_tail_ratio']:.1f}"
        )
    if "best_r2_oof" in diagnostics:
        lines.append(
            f"R² OOF range: [{diagnostics['worst_r2_oof']:.3f}, {diagnostics['best_r2_oof']:.3f}]"
        )
    if "tmle_max_score_z" in diagnostics:
        lines.append(
            f"TMLE max |score z|: {diagnostics['tmle_max_score_z']:.2f} (should be ~0)"
        )

    lines.append("=" * 100)

    return "\n".join(lines)
