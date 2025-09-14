"""Diagnostic metrics for ablation experiments."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from .constants import POLICIES, WELL_BEHAVED_POLICIES


def compute_diagnostic_metrics(
    results: List[Dict[str, Any]], by_quadrant: bool = False
) -> pd.DataFrame:
    """Compute diagnostic metrics (ESS, tail index, etc.).

    Args:
        results: List of experiment results
        by_quadrant: Whether to group by data regime quadrant

    Returns:
        DataFrame with diagnostic metrics per configuration
    """
    rows = []

    # Group results appropriately
    if by_quadrant:
        grouped_data: Dict[tuple, List[Dict]] = {}
        for result in results:
            estimator = result.get("spec", {}).get("estimator")
            quadrant = result.get("quadrant", "Unknown")
            use_weight_cal = (
                result.get("spec", {})
                .get("extra", {})
                .get("use_weight_calibration", False)
            )
            use_iic = result.get("spec", {}).get("extra", {}).get("use_iic", False)

            key = (estimator, use_weight_cal, use_iic, quadrant)
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(result)
    else:
        grouped_data: Dict[tuple, List[Dict]] = {}
        for result in results:
            estimator = result.get("spec", {}).get("estimator")
            use_weight_cal = (
                result.get("spec", {})
                .get("extra", {})
                .get("use_weight_calibration", False)
            )
            use_iic = result.get("spec", {}).get("extra", {}).get("use_iic", False)

            key = (estimator, use_weight_cal, use_iic)
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(result)

    # Compute metrics for each group
    for key, group_results in grouped_data.items():
        if by_quadrant:
            estimator, use_weight_cal, use_iic, quadrant = key
            row = {
                "estimator": estimator,
                "use_weight_calibration": use_weight_cal,
                "use_iic": use_iic,
                "quadrant": quadrant,
                "n_experiments": len(group_results),
            }
        else:
            estimator, use_weight_cal, use_iic = key
            row = {
                "estimator": estimator,
                "use_weight_calibration": use_weight_cal,
                "use_iic": use_iic,
                "n_experiments": len(group_results),
            }

        # Aggregate diagnostics across experiments
        for policy in POLICIES:
            ess_values = []
            hill_values = []
            hellinger_values = []
            max_weight_values = []

            for result in group_results:
                if result.get("ess_relative", {}).get(policy) is not None:
                    ess_values.append(result["ess_relative"][policy])
                if result.get("tail_alpha", {}).get(policy) is not None:
                    hill_values.append(result["tail_alpha"][policy])
                if result.get("hellinger_affinity", {}).get(policy) is not None:
                    hellinger_values.append(result["hellinger_affinity"][policy])
                if result.get("max_weight", {}).get(policy) is not None:
                    max_weight_values.append(result["max_weight"][policy])

            # Store mean values
            row[f"{policy}_ess"] = np.mean(ess_values) if ess_values else np.nan
            row[f"{policy}_tail_index"] = (
                np.mean(hill_values) if hill_values else np.nan
            )
            row[f"{policy}_hellinger"] = (
                np.mean(hellinger_values) if hellinger_values else np.nan
            )
            row[f"{policy}_max_weight"] = (
                np.mean(max_weight_values) if max_weight_values else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def compute_boundary_analysis(
    results: List[Dict[str, Any]], calibrated_methods: Optional[List[str]] = None
) -> pd.DataFrame:
    """Analyze calibration boundary proximity.

    Args:
        results: List of experiment results
        calibrated_methods: List of methods that use calibrated rewards

    Returns:
        DataFrame with boundary proximity metrics
    """
    if calibrated_methods is None:
        calibrated_methods = ["calibrated-ips", "dr-cpo", "oc-dr-cpo", "stacked-dr"]

    rows = []

    for method_name in calibrated_methods:
        # Collect boundary diagnostics for this method
        policy_dists: Dict[str, List[float]] = {policy: [] for policy in POLICIES}
        policy_flags: Dict[str, int] = {policy: 0 for policy in POLICIES}

        total_experiments = 0

        # Get results for this estimator
        method_results = [
            r for r in results if r.get("spec", {}).get("estimator") == method_name
        ]

        for result in method_results:
            total_experiments += 1

            cal_min = result.get("calibrated_reward_min")
            cal_max = result.get("calibrated_reward_max")
            estimates = result.get("estimates", {})

            if cal_min is None or cal_max is None:
                continue

            # Adaptive threshold based on calibration range
            cal_range = cal_max - cal_min
            boundary_thr = min(0.2 * cal_range, 0.15)

            # Collect all estimates for outlier detection
            all_estimates = [estimates[p] for p in POLICIES if p in estimates]
            if all_estimates:
                median_est = np.median(all_estimates)
                mad = np.median(np.abs(all_estimates - median_est))

            # Check each policy's proximity to boundaries
            for policy in POLICIES:
                if policy in estimates:
                    reward = estimates[policy]

                    # Distance to nearest boundary
                    dist_to_min = abs(reward - cal_min)
                    dist_to_max = abs(reward - cal_max)
                    nearest_dist = min(dist_to_min, dist_to_max)

                    policy_dists[policy].append(nearest_dist)

                    # Adaptive flagging logic
                    is_near_boundary = nearest_dist < boundary_thr
                    is_outside_range = reward < cal_min or reward > cal_max

                    # Outlier detection using MAD
                    is_outlier = False
                    if all_estimates and mad > 0.05:
                        is_outlier = abs(reward - median_est) > 2 * mad

                    # Flag if problematic
                    if is_near_boundary or is_outside_range or is_outlier:
                        policy_flags[policy] += 1

        # Create summary row
        row = {"method": method_name, "n_experiments": total_experiments}

        for policy in POLICIES:
            if policy_dists[policy]:
                row[f"{policy}_mean_boundary_dist"] = np.mean(policy_dists[policy])
                row[f"{policy}_min_boundary_dist"] = np.min(policy_dists[policy])
                row[f"{policy}_pct_flagged"] = (
                    100.0 * policy_flags[policy] / total_experiments
                    if total_experiments > 0
                    else np.nan
                )
            else:
                row[f"{policy}_mean_boundary_dist"] = np.nan
                row[f"{policy}_min_boundary_dist"] = np.nan
                row[f"{policy}_pct_flagged"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def compare_ips_diagnostics(results: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """Compare diagnostics between raw IPS and calibrated IPS.

    Args:
        results: List of experiment results

    Returns:
        Dictionary with comparison DataFrames for different metrics
    """
    # Filter to just IPS methods
    ips_results = [
        r
        for r in results
        if r.get("spec", {}).get("estimator") in ["raw-ips", "calibrated-ips"]
    ]

    # Get diagnostics by quadrant
    diag_df = compute_diagnostic_metrics(ips_results, by_quadrant=True)

    # Create comparison tables for each metric
    comparisons = {}

    # ESS comparison
    ess_comparison = []
    for estimator in ["raw-ips", "calibrated-ips"]:
        est_df = diag_df[diag_df["estimator"] == estimator]
        row = {"method": estimator}

        for policy in POLICIES:
            ess_col = f"{policy}_ess"
            if ess_col in est_df.columns:
                row[f"{policy}_mean_ess"] = est_df[ess_col].mean()

        ess_comparison.append(row)

    comparisons["ess"] = pd.DataFrame(ess_comparison)

    # Tail index comparison
    tail_comparison = []
    for estimator in ["raw-ips", "calibrated-ips"]:
        est_df = diag_df[diag_df["estimator"] == estimator]
        row = {"method": estimator}

        for policy in POLICIES:
            tail_col = f"{policy}_tail_index"
            if tail_col in est_df.columns:
                row[f"{policy}_mean_tail"] = est_df[tail_col].mean()

        tail_comparison.append(row)

    comparisons["tail_index"] = pd.DataFrame(tail_comparison)

    # Hellinger affinity comparison
    hell_comparison = []
    for estimator in ["raw-ips", "calibrated-ips"]:
        est_df = diag_df[diag_df["estimator"] == estimator]
        row = {"method": estimator}

        for policy in POLICIES:
            hell_col = f"{policy}_hellinger"
            if hell_col in est_df.columns:
                row[f"{policy}_mean_hellinger"] = est_df[hell_col].mean()

        hell_comparison.append(row)

    comparisons["hellinger"] = pd.DataFrame(hell_comparison)

    return comparisons
