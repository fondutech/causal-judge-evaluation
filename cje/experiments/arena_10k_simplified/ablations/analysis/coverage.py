"""Coverage and calibration analysis for ablation experiments."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from .constants import (
    POLICIES,
    WELL_BEHAVED_POLICIES,
    DEFAULT_ALPHA,
    Z_CRITICAL_95,
    DEFAULT_N_ORACLE,
)


def compute_coverage_metrics(
    results: List[Dict[str, Any]],
    alpha: float = 0.05,
    include_oracle_adjusted: bool = True,
) -> pd.DataFrame:
    """Compute coverage metrics for confidence intervals.

    Args:
        results: List of experiment results
        alpha: Significance level for confidence intervals (default 0.05 for 95% CI)
        include_oracle_adjusted: If True, also compute oracle-adjusted coverage

    Returns:
        DataFrame with coverage metrics per configuration
    """
    rows = []
    z = Z_CRITICAL_95

    for result in results:
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})

        # Get CI sources in priority order
        ci_map = (
            result.get("robust_confidence_intervals")
            or result.get("confidence_intervals")
            or {}
        )
        robust_ses = result.get("robust_standard_errors", {}) or {}
        base_ses = result.get("standard_errors", {}) or {}
        spec = result.get("spec", {})

        # Skip if missing required data
        if not (estimates and oracle_truths):
            continue

        # Get oracle sample counts for oracle-adjusted coverage
        oracle_counts = result.get("oracle_counts", {})
        n_oracle_default = result.get("n_oracle_truth", DEFAULT_N_ORACLE)

        # Compute coverage for each policy
        policy_coverage = {}
        for policy in POLICIES:
            if policy in estimates and policy in oracle_truths:
                est = estimates[policy]
                truth = oracle_truths[policy]

                # 1) Prefer stored CI if available
                if (
                    policy in ci_map
                    and isinstance(ci_map[policy], (list, tuple))
                    and len(ci_map[policy]) == 2
                ):
                    ci_lower, ci_upper = ci_map[policy]
                    covered = ci_lower <= truth <= ci_upper
                    ci_width = ci_upper - ci_lower
                else:
                    # 2) Rebuild from SE (prefer robust over base)
                    se = robust_ses.get(policy) or base_ses.get(policy)
                    if not (isinstance(se, (int, float)) and se > 0):
                        continue
                    ci_lower = est - z * se
                    ci_upper = est + z * se
                    covered = ci_lower <= truth <= ci_upper
                    ci_width = ci_upper - ci_lower

                policy_coverage[f"{policy}_covered"] = covered
                policy_coverage[f"{policy}_ci_width"] = ci_width

                # Oracle-adjusted coverage (widening CI for oracle sampling variance)
                if include_oracle_adjusted:
                    n_oracle = oracle_counts.get(policy, n_oracle_default)
                    oracle_var = min(truth * (1 - truth), 0.25) / max(1, n_oracle)
                    oracle_se = np.sqrt(oracle_var)

                    # Widen half-width in quadrature
                    half = ci_width / 2.0
                    half_oa = np.sqrt(half**2 + (z * oracle_se) ** 2)
                    ci_lower_oa = est - half_oa
                    ci_upper_oa = est + half_oa
                    covered_oa = ci_lower_oa <= truth <= ci_upper_oa

                    policy_coverage[f"{policy}_covered_oa"] = covered_oa
                    policy_coverage[f"{policy}_ci_width_oa"] = ci_upper_oa - ci_lower_oa

        rows.append(
            {
                "estimator": spec.get("estimator"),
                "sample_size": spec.get("sample_size"),
                "oracle_coverage": spec.get("oracle_coverage"),
                "quadrant": result.get("quadrant", "Unknown"),
                "use_weight_calibration": spec.get("extra", {}).get(
                    "use_weight_calibration", False
                ),
                "use_iic": spec.get("extra", {}).get("use_iic", False),
                "seed": spec.get("seed_base", 0),
                **policy_coverage,
            }
        )

    return pd.DataFrame(rows)


def compute_interval_scores(
    results: List[Dict[str, Any]], alpha: float = 0.05
) -> pd.DataFrame:
    """Compute interval scores for calibration assessment.

    Interval Score = CI Width + (2/alpha) * penalty for non-coverage
    Lower scores indicate better calibrated and sharper intervals.

    Args:
        results: List of experiment results
        alpha: Significance level (default 0.05)

    Returns:
        DataFrame with interval scores
    """
    rows = []

    for result in results:
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})

        # Get CI sources in priority order
        ci_map = (
            result.get("robust_confidence_intervals")
            or result.get("confidence_intervals")
            or {}
        )
        robust_ses = result.get("robust_standard_errors", {}) or {}
        base_ses = result.get("standard_errors", {}) or {}
        spec = result.get("spec", {})

        if not (estimates and oracle_truths):
            continue

        # Compute interval scores for each policy
        policy_scores = {}
        for policy in POLICIES:
            if policy in estimates and policy in oracle_truths:
                est = estimates[policy]
                truth = oracle_truths[policy]

                # Get CI bounds (prefer stored, then rebuild from SE)
                if (
                    policy in ci_map
                    and isinstance(ci_map[policy], (list, tuple))
                    and len(ci_map[policy]) == 2
                ):
                    ci_lower, ci_upper = ci_map[policy]
                else:
                    se = robust_ses.get(policy) or base_ses.get(policy)
                    if not (isinstance(se, (int, float)) and se > 0):
                        continue
                    z = 1.96
                    ci_lower = est - z * se
                    ci_upper = est + z * se

                interval_width = ci_upper - ci_lower
                undercoverage_penalty = (2 / alpha) * max(0, ci_lower - truth)
                overcoverage_penalty = (2 / alpha) * max(0, truth - ci_upper)

                interval_score = (
                    interval_width + undercoverage_penalty + overcoverage_penalty
                )
                policy_scores[f"{policy}_interval_score"] = interval_score

        rows.append(
            {
                "estimator": spec.get("estimator"),
                "sample_size": spec.get("sample_size"),
                "oracle_coverage": spec.get("oracle_coverage"),
                "use_weight_calibration": spec.get("extra", {}).get(
                    "use_weight_calibration", False
                ),
                "use_iic": spec.get("extra", {}).get("use_iic", False),
                **policy_scores,
            }
        )

    return pd.DataFrame(rows)


def aggregate_coverage_by_estimator(
    df: pd.DataFrame, use_oracle_adjusted: bool = False
) -> pd.DataFrame:
    """Aggregate coverage metrics by estimator configuration.

    Args:
        df: DataFrame with coverage metrics
        use_oracle_adjusted: If True, use oracle-adjusted coverage columns

    Returns:
        DataFrame with aggregated coverage percentages and calibration scores
    """
    # Group by estimator configuration
    grouped = df.groupby(["estimator", "use_weight_calibration", "use_iic"])

    results = []
    suffix = "_oa" if use_oracle_adjusted else ""

    for name, group in grouped:
        row = {
            "estimator": name[0],
            "use_weight_calibration": name[1],
            "use_iic": name[2],
            "n_experiments": len(group),
        }

        # Compute coverage percentage for each policy
        for policy in POLICIES:
            covered_col = f"{policy}_covered{suffix}"
            if covered_col in group.columns:
                coverage_pct = group[covered_col].mean() * 100
                row[f"{policy}_coverage_pct"] = coverage_pct

        # Compute calibration score (average absolute deviation from 95%)
        well_behaved_coverages = []
        for policy in WELL_BEHAVED_POLICIES:
            covered_col = f"{policy}_covered{suffix}"
            if covered_col in group.columns:
                coverage_pct = group[covered_col].mean() * 100
                well_behaved_coverages.append(abs(coverage_pct - 95.0))

        if well_behaved_coverages:
            row["calibration_score"] = np.mean(well_behaved_coverages)

        results.append(row)

    return pd.DataFrame(results).sort_values("calibration_score")
