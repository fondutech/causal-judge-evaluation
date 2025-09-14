"""RMSE analysis for ablation experiments."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any


from .constants import POLICIES, WELL_BEHAVED_POLICIES
from .constants import POLICIES, WELL_BEHAVED_POLICIES


def compute_rmse_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compute RMSE metrics across experiments.

    Args:
        results: List of experiment results

    Returns:
        DataFrame with RMSE metrics by estimator and configuration
    """
    rows = []

    for result in results:
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})
        spec = result.get("spec", {})

        # Skip if missing data
        if not estimates or not oracle_truths:
            continue

        # Compute per-policy RMSE
        policy_rmse = {}
        for policy in POLICIES:
            if policy in estimates and policy in oracle_truths:
                error = estimates[policy] - oracle_truths[policy]
                policy_rmse[policy] = error**2  # Store squared error

        # Overall RMSE (well-behaved policies only)
        well_behaved_errors = [
            (estimates[p] - oracle_truths[p]) ** 2
            for p in WELL_BEHAVED_POLICIES
            if p in estimates and p in oracle_truths
        ]

        if well_behaved_errors:
            overall_rmse = np.sqrt(np.mean(well_behaved_errors))
        else:
            overall_rmse = np.nan

        rows.append(
            {
                "estimator": spec.get("estimator"),
                "config_string": result.get("config_string", "unknown"),
                "sample_size": spec.get("sample_size"),
                "oracle_coverage": spec.get("oracle_coverage"),
                "quadrant": result.get("quadrant", "Unknown"),
                "use_weight_calibration": result.get("use_weight_calibration", False),
                "use_iic": result.get("use_iic", False),
                "reward_calibration_mode": result.get(
                    "reward_calibration_mode", "unknown"
                ),
                "seed": spec.get("seed_base", 0),
                "overall_rmse": overall_rmse,
                **{f"{p}_mse": policy_rmse.get(p, np.nan) for p in POLICIES},
            }
        )

    return pd.DataFrame(rows)


def compute_debiased_rmse(
    results: List[Dict[str, Any]], n_oracle_truth: int = 4989
) -> pd.DataFrame:
    """Compute oracle-noise-debiased RMSE.

    Adjusts for the sampling variance in the oracle ground truth estimates.

    Args:
        results: List of experiment results
        n_oracle_truth: Number of samples used to compute oracle ground truth

    Returns:
        DataFrame with debiased RMSE metrics
    """
    rows = []

    for result in results:
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})
        spec = result.get("spec", {})

        if not estimates or not oracle_truths:
            continue

        # Compute debiased RMSE for each policy
        policy_debiased = {}
        for policy in WELL_BEHAVED_POLICIES:
            if policy in estimates and policy in oracle_truths:
                mse = (estimates[policy] - oracle_truths[policy]) ** 2

                # Oracle variance (conservative Bernoulli estimate)
                truth = oracle_truths[policy]
                oracle_var = min(truth * (1 - truth), 0.25) / n_oracle_truth

                # Debias by subtracting oracle variance
                mse_debiased = max(mse - oracle_var, 0.0)
                policy_debiased[policy] = np.sqrt(mse_debiased)

        # Overall debiased RMSE
        if policy_debiased:
            overall_debiased = np.mean(list(policy_debiased.values()))
        else:
            overall_debiased = np.nan

        rows.append(
            {
                "estimator": spec.get("estimator"),
                "sample_size": spec.get("sample_size"),
                "oracle_coverage": spec.get("oracle_coverage"),
                "use_weight_calibration": spec.get("extra", {}).get(
                    "use_weight_calibration", False
                ),
                "use_iic": spec.get("extra", {}).get("use_iic", False),
                "overall_rmse_debiased": overall_debiased,
                **{
                    f"{p}_rmse_debiased": policy_debiased.get(p, np.nan)
                    for p in WELL_BEHAVED_POLICIES
                },
            }
        )

    return pd.DataFrame(rows)


def aggregate_rmse_by_quadrant(
    df: pd.DataFrame, quadrant_col: str = "quadrant"
) -> pd.DataFrame:
    """Aggregate RMSE metrics by quadrant and estimator.

    Args:
        df: DataFrame with RMSE metrics and quadrant classification
        quadrant_col: Name of quadrant column

    Returns:
        DataFrame with aggregated metrics
    """
    # Check if quadrant column exists
    if quadrant_col not in df.columns:
        return pd.DataFrame()  # Return empty if no quadrant info

    # Group by estimator and quadrant
    grouped = df.groupby(["estimator", quadrant_col])

    # Aggregate: mean RMSE and count
    agg_dict = {
        "overall_rmse": ["mean", "min", "max", "count"],
    }

    # Add per-policy aggregations if they exist
    for policy in POLICIES:
        mse_col = f"{policy}_mse"
        if mse_col in df.columns:
            agg_dict[mse_col] = ["mean", "min", "max"]

    aggregated = grouped.agg(agg_dict).reset_index()

    # Flatten column names
    aggregated.columns = ["_".join(col).strip("_") for col in aggregated.columns]

    # Convert MSE to RMSE for display
    for policy in POLICIES:
        mse_mean_col = f"{policy}_mse_mean"
        if mse_mean_col in aggregated.columns:
            aggregated[f"{policy}_rmse"] = np.sqrt(aggregated[mse_mean_col])

    return aggregated
