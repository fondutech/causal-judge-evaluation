"""Bias analysis for ablation experiments."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any


from .constants import POLICIES, WELL_BEHAVED_POLICIES
from .constants import POLICIES, WELL_BEHAVED_POLICIES


def compute_bias_analysis(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compute bias patterns across estimators.

    Args:
        results: List of experiment results

    Returns:
        DataFrame with bias metrics by estimator
    """
    # Collect errors for each estimator
    bias_by_estimator: Dict[str, Dict[str, List[float]]] = {}

    for result in results:
        estimator = result.get("spec", {}).get("estimator")
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})

        if not estimator or not estimates or not oracle_truths:
            continue

        if estimator not in bias_by_estimator:
            bias_by_estimator[estimator] = {policy: [] for policy in POLICIES}

        # Collect errors for each policy
        for policy in POLICIES:
            if policy in estimates and policy in oracle_truths:
                error = estimates[policy] - oracle_truths[policy]
                bias_by_estimator[estimator][policy].append(error)

    # Compute bias statistics
    rows = []
    for estimator, errors_dict in bias_by_estimator.items():
        row = {"estimator": estimator}

        # Per-policy bias
        for policy in POLICIES:
            errors = errors_dict[policy]
            if errors:
                row[f"{policy}_mean_bias"] = np.mean(errors)
                row[f"{policy}_se_bias"] = np.std(errors) / np.sqrt(len(errors))
                row[f"{policy}_t_stat"] = (
                    abs(row[f"{policy}_mean_bias"] / row[f"{policy}_se_bias"])
                    if row[f"{policy}_se_bias"] > 0
                    else 0
                )
                row[f"{policy}_n"] = len(errors)

        # Overall bias metrics (well-behaved policies only)
        well_behaved_errors = []
        for policy in WELL_BEHAVED_POLICIES:
            well_behaved_errors.extend(errors_dict.get(policy, []))

        if well_behaved_errors:
            row["overall_mean_bias"] = np.mean(well_behaved_errors)
            row["overall_mean_abs_bias"] = np.mean(
                [abs(e) for e in well_behaved_errors]
            )
            row["overall_max_abs_bias"] = max([abs(e) for e in well_behaved_errors])

            # Classify bias pattern
            all_biases = [
                np.mean(errors_dict[p]) for p in WELL_BEHAVED_POLICIES if errors_dict[p]
            ]
            if all(b < -0.005 for b in all_biases):
                row["bias_pattern"] = "Systematic negative"
            elif all(b > 0.005 for b in all_biases):
                row["bias_pattern"] = "Systematic positive"
            elif row["overall_max_abs_bias"] < 0.01:
                row["bias_pattern"] = "Nearly unbiased"
            elif max(all_biases) - min(all_biases) > 0.1:
                row["bias_pattern"] = "High variance"
            else:
                row["bias_pattern"] = "Mixed"

        rows.append(row)

    return pd.DataFrame(rows).sort_values("overall_mean_abs_bias")


def compute_bias_by_quadrant(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compute bias patterns by data regime quadrant.

    Args:
        results: List of experiment results with quadrant classification

    Returns:
        DataFrame with bias metrics by estimator and quadrant
    """
    rows = []

    # Group by estimator and quadrant
    grouped_data: Dict[tuple, List[Dict]] = {}
    for result in results:
        estimator = result.get("spec", {}).get("estimator")
        quadrant = result.get("quadrant", "Unknown")

        if not estimator:
            continue

        key = (estimator, quadrant)
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(result)

    # Compute bias for each group
    for (estimator, quadrant), group_results in grouped_data.items():
        errors_by_policy: Dict[str, List[float]] = {p: [] for p in POLICIES}

        for result in group_results:
            estimates = result.get("estimates", {})
            oracle_truths = result.get("oracle_truths", {})

            for policy in POLICIES:
                if policy in estimates and policy in oracle_truths:
                    error = estimates[policy] - oracle_truths[policy]
                    errors_by_policy[policy].append(error)

        row = {
            "estimator": estimator,
            "quadrant": quadrant,
            "n_experiments": len(group_results),
        }

        # Compute bias metrics for well-behaved policies
        well_behaved_errors = []
        for policy in WELL_BEHAVED_POLICIES:
            if errors_by_policy[policy]:
                mean_bias = np.mean(errors_by_policy[policy])
                row[f"{policy}_bias"] = mean_bias
                well_behaved_errors.extend(errors_by_policy[policy])

        if well_behaved_errors:
            row["overall_bias"] = np.mean(well_behaved_errors)
            row["overall_abs_bias"] = np.mean([abs(e) for e in well_behaved_errors])

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["estimator", "quadrant"])
