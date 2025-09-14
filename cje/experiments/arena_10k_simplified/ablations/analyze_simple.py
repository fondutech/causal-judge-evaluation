#!/usr/bin/env python3
"""
Analyze ablation experiment results with focus on robust standard errors, RMSE, and calibration.

This script aggregates results over seeds to demonstrate:
- Robust standard error calibration (Z-scores and coverage)
- Point estimate quality (RMSE)
- Performance by estimator configuration

Usage:
    python analyze_simple.py [--results-file results/all_experiments.jsonl]
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")


def load_results(results_file: Path) -> List[Dict[str, Any]]:
    """Load experiment results from JSONL file."""
    results = []
    with open(results_file, "r") as f:
        for line in f:
            try:
                result = json.loads(line.strip())
                if result.get("success", False):
                    results.append(result)
            except json.JSONDecodeError:
                continue
    return results


def compute_calibration_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compute calibration metrics using robust standard errors and confidence intervals."""
    rows = []

    for result in results:
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})
        robust_ses = result.get("robust_standard_errors", {})
        confidence_intervals = result.get("confidence_intervals", {})

        spec = result.get("spec", {})
        estimator = spec.get("estimator", "unknown")
        extra = spec.get("extra", {})

        # Extract configuration
        use_weight_calibration = extra.get("use_weight_calibration", False)
        use_iic = extra.get("use_iic", False)

        # Overall RMSE from the result
        overall_rmse = result.get("rmse_vs_oracle", np.nan)

        # Per-policy metrics
        policies = ["clone", "parallel_universe_prompt", "premium", "unhelpful"]

        # Oracle ground truth is computed from full dataset, not the experiment subset
        # This should be ~4989 for the full arena dataset
        n_oracle_truth = result.get(
            "n_oracle_truth", 4989
        )  # Number of samples used to compute ground truth

        policy_data = {}
        for policy in policies:
            if all(
                key in d and d[key] is not None
                for d in [estimates, oracle_truths, robust_ses]
                for key in [policy]
            ):
                est = estimates[policy]
                truth = oracle_truths[policy]
                se_est = robust_ses[policy]

                # Compute oracle standard error from full dataset used for ground truth
                # Conservative Bernoulli variance estimate
                oracle_var = min(
                    truth * (1 - truth), 0.25
                )  # Max variance is 0.25 at p=0.5
                se_oracle = np.sqrt(oracle_var / n_oracle_truth)

                # Total standard error accounting for both estimation and oracle uncertainty
                se_total = (
                    np.sqrt(se_est**2 + se_oracle**2) if se_est > 0 else se_oracle
                )

                # Compute coverage using total standard error
                covered = False
                covered_naive = (
                    False  # Coverage without oracle uncertainty for comparison
                )
                if se_total > 0:
                    ci_lower = est - 1.96 * se_total
                    ci_upper = est + 1.96 * se_total
                    covered = ci_lower <= truth <= ci_upper

                if se_est > 0:
                    ci_lower_naive = est - 1.96 * se_est
                    ci_upper_naive = est + 1.96 * se_est
                    covered_naive = ci_lower_naive <= truth <= ci_upper_naive

                if se_total > 0:
                    z_score = (est - truth) / se_total
                    z_score_naive = (est - truth) / se_est if se_est > 0 else np.nan

                    # Compute interval score (Gneiting & Raftery 2007)
                    # IS = (u - l) + (2/alpha) * (l - y) * 1{y < l} + (2/alpha) * (y - u) * 1{y > u}
                    # where alpha = 0.05 for 95% CI
                    alpha = 0.05
                    ci_lower = est - 1.96 * se_total
                    ci_upper = est + 1.96 * se_total
                    interval_width = ci_upper - ci_lower
                    undercoverage_penalty = (2 / alpha) * max(0, ci_lower - truth)
                    overcoverage_penalty = (2 / alpha) * max(0, truth - ci_upper)
                    interval_score = (
                        interval_width + undercoverage_penalty + overcoverage_penalty
                    )

                    # Also compute interval score without oracle uncertainty for comparison
                    if se_est > 0:
                        ci_lower_naive = est - 1.96 * se_est
                        ci_upper_naive = est + 1.96 * se_est
                        interval_width_naive = ci_upper_naive - ci_lower_naive
                        undercoverage_penalty_naive = (2 / alpha) * max(
                            0, ci_lower_naive - truth
                        )
                        overcoverage_penalty_naive = (2 / alpha) * max(
                            0, truth - ci_upper_naive
                        )
                        interval_score_naive = (
                            interval_width_naive
                            + undercoverage_penalty_naive
                            + overcoverage_penalty_naive
                        )
                    else:
                        interval_score_naive = np.nan

                    # Compute oracle SE share for diagnostics
                    se_oracle_share = (
                        (se_oracle**2 / max(se_total**2, 1e-12))
                        if se_total > 0
                        else 0.0
                    )

                    policy_data[policy] = {
                        "z_score": z_score,
                        "z_score_naive": z_score_naive,
                        "covered": covered,
                        "covered_naive": covered_naive,
                        "error": abs(est - truth),
                        "se_oracle": se_oracle,
                        "se_total": se_total,
                        "se_oracle_share": se_oracle_share,
                        "interval_score": interval_score,
                        "interval_score_naive": interval_score_naive,
                    }

        if policy_data:  # Only add if we have valid data
            rows.append(
                {
                    "estimator": estimator,
                    "use_weight_calibration": use_weight_calibration,
                    "use_iic": use_iic,
                    "overall_rmse": overall_rmse,
                    "seed": spec.get("seed_base", 0),
                    "sample_size": spec.get("sample_size", 0),
                    "oracle_coverage": spec.get("oracle_coverage", 0.0),
                    **{
                        f"{policy}_{metric}": values[metric]
                        for policy, values in policy_data.items()
                        for metric in values.keys()
                    },
                }
            )

    return pd.DataFrame(rows)


def add_quadrant_classification(results: List[Dict[str, Any]]) -> None:
    """Add quadrant classification to results."""
    for result in results:
        spec = result.get("spec", {})
        sample_size = spec.get("sample_size", 0)
        oracle_coverage = spec.get("oracle_coverage", 0)

        # Classify into quadrants
        if sample_size <= 1000 and oracle_coverage <= 0.25:
            result["quadrant"] = "Small-LowOracle"
        elif sample_size <= 1000 and oracle_coverage > 0.25:
            result["quadrant"] = "Small-HighOracle"
        elif sample_size > 1000 and oracle_coverage <= 0.25:
            result["quadrant"] = "Large-LowOracle"
        else:  # sample_size > 1000 and oracle_coverage > 0.25
            result["quadrant"] = "Large-HighOracle"


def print_summary_tables(results: List[Dict[str, Any]]) -> None:
    """Print three focused tables with quadrant-based analysis."""
    print("=" * 140)
    print("ABLATION RESULTS SUMMARY BY QUADRANT")
    print("=" * 140)

    # Add quadrant classification
    add_quadrant_classification(results)

    # Load and process data
    df = compute_calibration_metrics(results)

    if len(df) == 0:
        print("No valid results found!")
        return

    print(
        f"\nDataset: {len(results)} experiments across {len(df['estimator'].unique())} estimators"
    )
    print(
        f"Quadrants: Small/Large samples (≤1000 vs >1000), Low/High oracle (≤25% vs >25%)"
    )

    policies = ["clone", "parallel_universe_prompt", "premium", "unhelpful"]

    # Compute quadrant directly from sample_size and oracle_coverage in the dataframe
    def compute_quadrant(row: pd.Series) -> str:
        sample_size = row["sample_size"]
        oracle_coverage = row["oracle_coverage"]

        if sample_size <= 1000 and oracle_coverage <= 0.25:
            return "Small-LowOracle"
        elif sample_size <= 1000 and oracle_coverage > 0.25:
            return "Small-HighOracle"
        elif sample_size > 1000 and oracle_coverage <= 0.25:
            return "Large-LowOracle"
        else:
            return "Large-HighOracle"

    df["quadrant"] = df.apply(compute_quadrant, axis=1)

    # 1. RMSE Performance by Policy and Quadrant
    print("\n" + "=" * 160)
    print(
        "1. RMSE PERFORMANCE BY POLICY AND QUADRANT (pooled RMSE; debiased shown in 1b)"
    )
    print("=" * 160)

    # Compute per-policy RMSE by quadrant for each method
    rmse_by_method_quadrant: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    oracle_var_by_method_quadrant: Dict[str, Dict[str, Dict[str, List[float]]]] = (
        {}
    )  # Track oracle variance for debiasing

    for result in results:
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})
        # Oracle ground truth computed from full dataset
        n_oracle_truth = result.get("n_oracle_truth", 4989)
        spec = result.get("spec", {})
        estimator = spec.get("estimator", "unknown")
        extra = spec.get("extra", {})
        use_weight_calibration = extra.get("use_weight_calibration", False)
        quadrant = result.get("quadrant", "Unknown")

        # IIC doesn't affect point estimates, only variance
        method_key = f"{estimator} (calib={use_weight_calibration})"

        if method_key not in rmse_by_method_quadrant:
            rmse_by_method_quadrant[method_key] = {}
            oracle_var_by_method_quadrant[method_key] = {}
        if quadrant not in rmse_by_method_quadrant[method_key]:
            rmse_by_method_quadrant[method_key][quadrant] = {
                policy: [] for policy in policies
            }
            rmse_by_method_quadrant[method_key][quadrant][
                "overall_sqerrs"
            ] = []  # Store squared errors for aggregation
            oracle_var_by_method_quadrant[method_key][quadrant] = {
                policy: [] for policy in policies
            }

        # Collect per-policy errors and oracle variance
        well_behaved_policies = ["clone", "parallel_universe_prompt", "premium"]
        for policy in policies:
            if all(
                key in d and d[key] is not None
                for d in [estimates, oracle_truths]
                for key in [policy]
            ):
                # Store squared error for RMSE calculation
                squared_error = (estimates[policy] - oracle_truths[policy]) ** 2
                rmse_by_method_quadrant[method_key][quadrant][policy].append(
                    squared_error
                )

                # Compute and store oracle variance for debiasing
                truth = oracle_truths[policy]
                oracle_var = min(truth * (1 - truth), 0.25) / n_oracle_truth
                oracle_var_by_method_quadrant[method_key][quadrant][policy].append(
                    oracle_var
                )

        # Store squared errors for well-behaved policies for overall metric
        if all(p in estimates and p in oracle_truths for p in well_behaved_policies):
            overall_sqerrs = [
                (estimates[p] - oracle_truths[p]) ** 2 for p in well_behaved_policies
            ]
            rmse_by_method_quadrant[method_key][quadrant]["overall_sqerrs"].extend(
                overall_sqerrs
            )

    # Create summary table
    rmse_summary_rows = []
    quadrant_order = [
        "Small-LowOracle",
        "Small-HighOracle",
        "Large-LowOracle",
        "Large-HighOracle",
    ]

    for method, quadrant_data in rmse_by_method_quadrant.items():
        row: Dict[str, Any] = {"method": method}

        # Per-quadrant metrics
        for quad in quadrant_order:
            if quad in quadrant_data:
                # Per-policy means for this quadrant
                for policy in policies:
                    if quadrant_data[quad][policy]:
                        # Take square root of mean squared errors for RMSE
                        row[f"{quad}_{policy}"] = np.sqrt(
                            np.mean(quadrant_data[quad][policy])
                        )
                    else:
                        row[f"{quad}_{policy}"] = np.nan

                # Overall RMSE for this quadrant from squared errors
                if quadrant_data[quad]["overall_sqerrs"]:
                    row[f"{quad}_overall"] = np.sqrt(
                        np.mean(quadrant_data[quad]["overall_sqerrs"])
                    )
                else:
                    row[f"{quad}_overall"] = np.nan
            else:
                # Fill with NaN if quadrant not available
                for policy in policies + ["overall"]:
                    row[f"{quad}_{policy}"] = np.nan

        # Cross-quadrant aggregates - pool squared errors then take sqrt
        def pool_rmse_across_quads(quadrant_data: Dict[str, Any], key: str) -> float:
            """Pool squared errors across quadrants and compute RMSE."""
            squared_errors = []
            for quad in quadrant_order:
                if quad in quadrant_data and key in quadrant_data[quad]:
                    squared_errors.extend(quadrant_data[quad][key])
            return (
                float(np.sqrt(np.mean(squared_errors)))
                if squared_errors
                else float(np.nan)
            )

        row["agg_overall"] = pool_rmse_across_quads(quadrant_data, "overall_sqerrs")
        row["agg_clone"] = pool_rmse_across_quads(quadrant_data, "clone")
        row["agg_para"] = pool_rmse_across_quads(
            quadrant_data, "parallel_universe_prompt"
        )
        row["agg_premium"] = pool_rmse_across_quads(quadrant_data, "premium")

        # Compute overall debiased RMSE
        def compute_overall_debiased_rmse(
            quadrant_data: Dict[str, Any], oracle_var_data: Dict[str, Any]
        ) -> float:
            """Compute debiased RMSE pooled across well-behaved policies."""
            sqerrs = []
            oracle_vars = []
            for quad in quadrant_order:
                if quad in quadrant_data and "overall_sqerrs" in quadrant_data[quad]:
                    sqerrs.extend(quadrant_data[quad]["overall_sqerrs"])
                if method in oracle_var_data and quad in oracle_var_data[method]:
                    # Average per-policy oracle variance for well-behaved policies
                    ov = []
                    for pol in ["clone", "parallel_universe_prompt", "premium"]:
                        ov.extend(oracle_var_data[method][quad].get(pol, []))
                    if ov:
                        oracle_vars.append(np.mean(ov))
            if sqerrs and oracle_vars:
                mse = np.mean(sqerrs)
                mean_oracle_var = np.mean(oracle_vars)
                return float(np.sqrt(max(mse - mean_oracle_var, 0.0)))
            return float(np.nan)

        row["agg_overall_debiased"] = compute_overall_debiased_rmse(
            quadrant_data, oracle_var_by_method_quadrant
        )

        # Compute debiased RMSE by subtracting oracle variance
        # Collect oracle variances for policies across quadrants
        all_oracle_vars: Dict[str, List[float]] = {
            "clone": [],
            "parallel_universe_prompt": [],
            "premium": [],
        }
        for quad in quadrant_order:
            if (
                method in oracle_var_by_method_quadrant
                and quad in oracle_var_by_method_quadrant[method]
            ):
                for policy in ["clone", "parallel_universe_prompt", "premium"]:
                    if policy in oracle_var_by_method_quadrant[method][quad]:
                        all_oracle_vars[policy].extend(
                            oracle_var_by_method_quadrant[method][quad][policy]
                        )

        # Compute debiased RMSE (subtract mean oracle variance from pooled MSE before square root)
        def compute_debiased_rmse(
            quadrant_data: Dict[str, Any], oracle_var_data: Dict[str, Any], policy: str
        ) -> float:
            """Compute debiased RMSE by pooling squared errors and oracle variances."""
            squared_errors = []
            oracle_vars = []
            for quad in quadrant_order:
                if quad in quadrant_data and policy in quadrant_data[quad]:
                    squared_errors.extend(quadrant_data[quad][policy])
                if quad in oracle_var_data and policy in oracle_var_data[quad]:
                    oracle_vars.extend(oracle_var_data[quad][policy])

            if squared_errors and oracle_vars:
                mse = np.mean(squared_errors)
                mean_oracle_var = np.mean(oracle_vars)
                mse_debiased = mse - mean_oracle_var
                return float(np.sqrt(max(mse_debiased, 0)))
            return float(np.nan)

        if method in oracle_var_by_method_quadrant:
            row["agg_clone_debiased"] = compute_debiased_rmse(
                quadrant_data, oracle_var_by_method_quadrant[method], "clone"
            )
            row["agg_para_debiased"] = compute_debiased_rmse(
                quadrant_data,
                oracle_var_by_method_quadrant[method],
                "parallel_universe_prompt",
            )
            row["agg_premium_debiased"] = compute_debiased_rmse(
                quadrant_data, oracle_var_by_method_quadrant[method], "premium"
            )
        else:
            row["agg_clone_debiased"] = np.nan
            row["agg_para_debiased"] = np.nan
            row["agg_premium_debiased"] = np.nan

        rmse_summary_rows.append(row)

    rmse_quadrant_df = pd.DataFrame(rmse_summary_rows).sort_values("agg_overall")

    print(
        f"{'Method':<30} {'Overall':<8} {'Clone':<8} {'ParaU':<8} {'Premium':<8} "
        f"{'SL_Over':<8} {'SH_Over':<8} {'LL_Over':<8} {'LH_Over':<8}"
    )
    print("-" * 150)

    for _, row in rmse_quadrant_df.iterrows():
        print(
            f"{row['method']:<30} {row['agg_overall']:<8.4f} {row['agg_clone']:<8.4f} {row['agg_para']:<8.4f} "
            f"{row['agg_premium']:<8.4f} "
            f"{row.get('Small-LowOracle_overall', np.nan):<8.4f} {row.get('Small-HighOracle_overall', np.nan):<8.4f} "
            f"{row.get('Large-LowOracle_overall', np.nan):<8.4f} {row.get('Large-HighOracle_overall', np.nan):<8.4f}"
        )

    # Display noise-debiased RMSE values
    print("\n" + "=" * 160)
    print("1b. NOISE-DEBIASED RMSE BY POLICY (Adjusted for Oracle Sampling Variance)")
    print("=" * 160)

    print(
        f"{'Method':<30} {'Overall':<12} {'Overall_Deb':<12} {'Clone_Deb':<12} {'ParaU_Deb':<12} {'Premium_Deb':<12}"
    )
    print("-" * 120)

    for _, row in rmse_quadrant_df.iterrows():
        print(
            f"{row['method']:<30} "
            f"{row['agg_overall']:<12.4f} {row.get('agg_overall_debiased', np.nan):<12.4f} "
            f"{row.get('agg_clone_debiased', np.nan):<12.4f} "
            f"{row.get('agg_para_debiased', np.nan):<12.4f} "
            f"{row.get('agg_premium_debiased', np.nan):<12.4f}"
        )

    # 2. Standard Error Magnitude by Policy and Quadrant
    print("\n" + "=" * 160)
    print("2. ROBUST STANDARD ERROR MAGNITUDE BY POLICY AND QUADRANT")
    print("=" * 160)

    # Compute per-policy SE by quadrant for each method
    se_by_method_quadrant: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for result in results:
        robust_ses = result.get("robust_standard_errors", {})
        spec = result.get("spec", {})
        estimator = spec.get("estimator", "unknown")
        extra = spec.get("extra", {})

        use_weight_calibration = extra.get("use_weight_calibration", False)
        use_iic = extra.get("use_iic", False)
        quadrant = result.get("quadrant", "Unknown")

        method_key = f"{estimator} (calib={use_weight_calibration}, iic={use_iic})"

        if method_key not in se_by_method_quadrant:
            se_by_method_quadrant[method_key] = {}
        if quadrant not in se_by_method_quadrant[method_key]:
            se_by_method_quadrant[method_key][quadrant] = {
                policy: [] for policy in policies
            }

        # Collect per-policy SEs
        for policy in policies:
            if (
                policy in robust_ses
                and robust_ses[policy] is not None
                and robust_ses[policy] > 0
            ):
                se_by_method_quadrant[method_key][quadrant][policy].append(
                    robust_ses[policy]
                )

    # Create summary table
    se_summary_rows = []
    quadrant_order = [
        "Small-LowOracle",
        "Small-HighOracle",
        "Large-LowOracle",
        "Large-HighOracle",
    ]

    for method, quadrant_data in se_by_method_quadrant.items():
        se_row: Dict[str, Any] = {"method": method}

        # Per-quadrant metrics
        for quad in quadrant_order:
            if quad in quadrant_data:
                # Per-policy means for this quadrant
                for policy in policies:
                    if quadrant_data[quad][policy]:
                        se_row[f"{quad}_{policy}"] = np.mean(
                            quadrant_data[quad][policy]
                        )
                    else:
                        se_row[f"{quad}_{policy}"] = np.nan

                # Geometric mean of well-behaved policies for this quadrant
                well_behaved_ses = []
                for policy in ["clone", "parallel_universe_prompt", "premium"]:
                    if quadrant_data[quad][policy]:
                        well_behaved_ses.extend(quadrant_data[quad][policy])

                if len(well_behaved_ses) >= 3:
                    # Add epsilon to avoid log(0)
                    EPS = 1e-12
                    se_row[f"{quad}_geom_mean"] = np.exp(
                        np.mean(np.log(np.array(well_behaved_ses) + EPS))
                    )
                else:
                    se_row[f"{quad}_geom_mean"] = np.nan
            else:
                # Fill with NaN if quadrant not available
                for policy in policies:
                    se_row[f"{quad}_{policy}"] = np.nan
                se_row[f"{quad}_geom_mean"] = np.nan

        # Cross-quadrant aggregates
        all_geom_mean: List[float] = []
        all_clone: List[float] = []
        all_para: List[float] = []
        all_premium: List[float] = []
        all_unhelpful: List[float] = []

        for quad in quadrant_order:
            if f"{quad}_geom_mean" in se_row and not np.isnan(
                se_row[f"{quad}_geom_mean"]
            ):
                all_geom_mean.append(se_row[f"{quad}_geom_mean"])
            if f"{quad}_clone" in se_row and not np.isnan(se_row[f"{quad}_clone"]):
                all_clone.append(se_row[f"{quad}_clone"])
            if f"{quad}_parallel_universe_prompt" in se_row and not np.isnan(
                se_row[f"{quad}_parallel_universe_prompt"]
            ):
                all_para.append(se_row[f"{quad}_parallel_universe_prompt"])
            if f"{quad}_premium" in se_row and not np.isnan(se_row[f"{quad}_premium"]):
                all_premium.append(se_row[f"{quad}_premium"])
            if f"{quad}_unhelpful" in se_row and not np.isnan(
                se_row[f"{quad}_unhelpful"]
            ):
                all_unhelpful.append(se_row[f"{quad}_unhelpful"])

        se_row["agg_geom_mean"] = np.mean(all_geom_mean) if all_geom_mean else np.nan
        se_row["agg_clone"] = np.mean(all_clone) if all_clone else np.nan
        se_row["agg_para"] = np.mean(all_para) if all_para else np.nan
        se_row["agg_premium"] = np.mean(all_premium) if all_premium else np.nan

        se_summary_rows.append(se_row)

    se_quadrant_df = pd.DataFrame(se_summary_rows).sort_values("agg_geom_mean")

    print(
        f"{'Method':<40} {'GeoMean':<8} {'Clone':<8} {'ParaU':<8} {'Premium':<8} "
        f"{'SL_Geo':<8} {'SH_Geo':<8} {'LL_Geo':<8} {'LH_Geo':<8}"
    )
    print("-" * 150)

    for _, row in se_quadrant_df.iterrows():
        print(
            f"{row['method']:<40} {row['agg_geom_mean']:<8.4f} {row['agg_clone']:<8.4f} {row['agg_para']:<8.4f} "
            f"{row['agg_premium']:<8.4f} "
            f"{row.get('Small-LowOracle_geom_mean', np.nan):<8.4f} {row.get('Small-HighOracle_geom_mean', np.nan):<8.4f} "
            f"{row.get('Large-LowOracle_geom_mean', np.nan):<8.4f} {row.get('Large-HighOracle_geom_mean', np.nan):<8.4f}"
        )

    # 3. Coverage Calibration by Policy and Quadrant
    print("\n" + "=" * 160)
    print("3. CONFIDENCE INTERVAL COVERAGE BY POLICY AND QUADRANT (Target: 95%)")
    print("=" * 160)

    # Compute per-policy coverage by quadrant for each method
    coverage_by_method_quadrant: Dict[str, Dict[str, Dict[str, float]]] = {}
    for _, group in df.groupby(
        ["estimator", "use_weight_calibration", "use_iic", "quadrant"]
    ):
        estimator = group.iloc[0]["estimator"]
        use_weight_calibration = group.iloc[0]["use_weight_calibration"]
        use_iic = group.iloc[0]["use_iic"]
        quadrant = group.iloc[0]["quadrant"]

        method_key = f"{estimator} (calib={use_weight_calibration}, iic={use_iic})"

        if method_key not in coverage_by_method_quadrant:
            coverage_by_method_quadrant[method_key] = {}
        if quadrant not in coverage_by_method_quadrant[method_key]:
            coverage_by_method_quadrant[method_key][quadrant] = {}

        # Collect per-policy coverage percentages
        for policy in policies:
            covered_col = f"{policy}_covered"
            if covered_col in group.columns:
                coverage_pct = group[covered_col].mean() * 100
                coverage_by_method_quadrant[method_key][quadrant][policy] = coverage_pct
            else:
                coverage_by_method_quadrant[method_key][quadrant][policy] = np.nan

        # Compute calibration score for well-behaved policies
        well_behaved_coverages = []
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            covered_col = f"{policy}_covered"
            if covered_col in group.columns:
                coverage_pct = group[covered_col].mean() * 100
                well_behaved_coverages.append(abs(coverage_pct - 95.0))

        if len(well_behaved_coverages) == 3:
            coverage_by_method_quadrant[method_key][quadrant]["calib_score"] = np.mean(
                well_behaved_coverages
            )
        else:
            coverage_by_method_quadrant[method_key][quadrant]["calib_score"] = np.nan

    # Create summary table
    coverage_summary_rows = []
    quadrant_order = [
        "Small-LowOracle",
        "Small-HighOracle",
        "Large-LowOracle",
        "Large-HighOracle",
    ]

    for method, cov_quadrant_data in coverage_by_method_quadrant.items():
        cov_row: Dict[str, Any] = {"method": method}

        # Per-quadrant metrics
        for quad in quadrant_order:
            if quad in cov_quadrant_data:
                # Per-policy coverage for this quadrant
                for policy in policies:
                    if policy in cov_quadrant_data[quad]:
                        cov_row[f"{quad}_{policy}"] = cov_quadrant_data[quad][policy]
                    else:
                        cov_row[f"{quad}_{policy}"] = np.nan

                # Calibration score for this quadrant
                if "calib_score" in cov_quadrant_data[quad]:
                    cov_row[f"{quad}_calib_score"] = cov_quadrant_data[quad][
                        "calib_score"
                    ]
                else:
                    cov_row[f"{quad}_calib_score"] = np.nan
            else:
                # Fill with NaN if quadrant not available
                for policy in policies:
                    cov_row[f"{quad}_{policy}"] = np.nan
                cov_row[f"{quad}_calib_score"] = np.nan

        # Cross-quadrant aggregates (exclude unhelpful due to identification issues)
        all_calib_scores: List[float] = []
        all_cov_clone: List[float] = []
        all_cov_para: List[float] = []
        all_cov_premium: List[float] = []

        for quad in quadrant_order:
            if f"{quad}_calib_score" in cov_row and not np.isnan(
                cov_row[f"{quad}_calib_score"]
            ):
                all_calib_scores.append(cov_row[f"{quad}_calib_score"])
            if f"{quad}_clone" in cov_row and not np.isnan(cov_row[f"{quad}_clone"]):
                all_cov_clone.append(cov_row[f"{quad}_clone"])
            if f"{quad}_parallel_universe_prompt" in cov_row and not np.isnan(
                cov_row[f"{quad}_parallel_universe_prompt"]
            ):
                all_cov_para.append(cov_row[f"{quad}_parallel_universe_prompt"])
            if f"{quad}_premium" in cov_row and not np.isnan(
                cov_row[f"{quad}_premium"]
            ):
                all_cov_premium.append(cov_row[f"{quad}_premium"])

        cov_row["agg_calib_score"] = (
            np.mean(all_calib_scores) if all_calib_scores else np.nan
        )
        cov_row["agg_clone"] = np.mean(all_cov_clone) if all_cov_clone else np.nan
        cov_row["agg_para"] = np.mean(all_cov_para) if all_cov_para else np.nan
        cov_row["agg_premium"] = np.mean(all_cov_premium) if all_cov_premium else np.nan

        coverage_summary_rows.append(cov_row)

    if coverage_summary_rows:
        coverage_quadrant_df = pd.DataFrame(coverage_summary_rows).sort_values(
            "agg_calib_score"
        )
    else:
        coverage_quadrant_df = pd.DataFrame()

    print(
        f"{'Method':<40} {'CalibScore':<10} {'Clone%':<7} {'ParaU%':<7} {'Premium%':<8} "
        f"{'SL_Cal':<8} {'SH_Cal':<8} {'LL_Cal':<8} {'LH_Cal':<8}"
    )
    print("-" * 150)

    if not coverage_quadrant_df.empty:
        for _, row in coverage_quadrant_df.iterrows():
            print(
                f"{row['method']:<40} {row['agg_calib_score']:<10.1f} {row['agg_clone']:<7.1f} {row['agg_para']:<7.1f} "
                f"{row['agg_premium']:<8.1f} "
                f"{row.get('Small-LowOracle_calib_score', np.nan):<8.1f} {row.get('Small-HighOracle_calib_score', np.nan):<8.1f} "
                f"{row.get('Large-LowOracle_calib_score', np.nan):<8.1f} {row.get('Large-HighOracle_calib_score', np.nan):<8.1f}"
            )
    else:
        print("No coverage data available")

    # 3b. Interval Score Analysis
    print("\n" + "=" * 160)
    print("3b. INTERVAL SCORE BY POLICY AND METHOD (Lower is Better)")
    print("=" * 160)
    print("\nInterval Score = CI Width + Penalties for Non-Coverage")
    print("Properly accounts for both calibration quality and uncertainty estimation\n")

    # Compute interval scores by method and quadrant
    interval_score_by_method_quadrant: Dict[str, Dict[str, Dict[str, float]]] = {}

    for _, group in df.groupby(
        ["estimator", "use_weight_calibration", "use_iic", "quadrant"]
    ):
        estimator = group.iloc[0]["estimator"]
        use_weight_calibration = group.iloc[0]["use_weight_calibration"]
        use_iic = group.iloc[0]["use_iic"]
        quadrant = group.iloc[0]["quadrant"]

        method_key = f"{estimator} (calib={use_weight_calibration}, iic={use_iic})"

        if method_key not in interval_score_by_method_quadrant:
            interval_score_by_method_quadrant[method_key] = {}
        if quadrant not in interval_score_by_method_quadrant[method_key]:
            interval_score_by_method_quadrant[method_key][quadrant] = {}

        # Collect interval scores for each policy
        for policy in policies:
            is_col = f"{policy}_interval_score"
            is_naive_col = f"{policy}_interval_score_naive"
            if is_col in group.columns:
                scores = group[is_col].dropna()
                scores_naive = (
                    group[is_naive_col].dropna()
                    if is_naive_col in group.columns
                    else pd.Series(dtype=float)
                )
                if len(scores) > 0:
                    interval_score_by_method_quadrant[method_key][quadrant][
                        f"{policy}_score"
                    ] = scores.mean()
                    interval_score_by_method_quadrant[method_key][quadrant][
                        f"{policy}_score_naive"
                    ] = (scores_naive.mean() if len(scores_naive) > 0 else np.nan)

    # Create summary table
    interval_score_rows = []
    for method, is_quadrant_data in interval_score_by_method_quadrant.items():
        is_row: Dict[str, Any] = {"method": method}

        # Aggregate across quadrants for each policy (excluding unhelpful)
        all_is_clone: List[float] = []
        all_is_para: List[float] = []
        all_is_premium: List[float] = []
        all_is_clone_naive: List[float] = []
        all_is_para_naive: List[float] = []
        all_is_premium_naive: List[float] = []

        for quad in quadrant_order:
            if quad in is_quadrant_data:
                for policy in ["clone", "parallel_universe_prompt", "premium"]:
                    key = f"{policy}_score"
                    key_naive = f"{policy}_score_naive"
                    if key in is_quadrant_data[quad] and not np.isnan(
                        is_quadrant_data[quad][key]
                    ):
                        if policy == "clone":
                            all_is_clone.append(is_quadrant_data[quad][key])
                            if key_naive in is_quadrant_data[quad] and not np.isnan(
                                is_quadrant_data[quad][key_naive]
                            ):
                                all_is_clone_naive.append(
                                    is_quadrant_data[quad][key_naive]
                                )
                        elif policy == "parallel_universe_prompt":
                            all_is_para.append(is_quadrant_data[quad][key])
                            if key_naive in is_quadrant_data[quad] and not np.isnan(
                                is_quadrant_data[quad][key_naive]
                            ):
                                all_is_para_naive.append(
                                    is_quadrant_data[quad][key_naive]
                                )
                        elif policy == "premium":
                            all_is_premium.append(is_quadrant_data[quad][key])
                            if key_naive in is_quadrant_data[quad] and not np.isnan(
                                is_quadrant_data[quad][key_naive]
                            ):
                                all_is_premium_naive.append(
                                    is_quadrant_data[quad][key_naive]
                                )

        # Compute means
        is_row["clone_score"] = np.mean(all_is_clone) if all_is_clone else np.nan
        is_row["para_score"] = np.mean(all_is_para) if all_is_para else np.nan
        is_row["premium_score"] = np.mean(all_is_premium) if all_is_premium else np.nan
        is_row["clone_score_naive"] = (
            np.mean(all_is_clone_naive) if all_is_clone_naive else np.nan
        )
        is_row["para_score_naive"] = (
            np.mean(all_is_para_naive) if all_is_para_naive else np.nan
        )
        is_row["premium_score_naive"] = (
            np.mean(all_is_premium_naive) if all_is_premium_naive else np.nan
        )

        # Compute overall mean (geometric mean of well-behaved policies)
        valid_scores = [
            s
            for s in [
                is_row["clone_score"],
                is_row["para_score"],
                is_row["premium_score"],
            ]
            if not np.isnan(s)
        ]
        if len(valid_scores) >= 2:
            # Add epsilon to avoid log(0)
            EPS = 1e-12
            is_row["mean_score"] = np.exp(np.mean(np.log(np.array(valid_scores) + EPS)))
        else:
            is_row["mean_score"] = np.nan

        interval_score_rows.append(is_row)

    if interval_score_rows:
        interval_score_df = pd.DataFrame(interval_score_rows).sort_values("mean_score")

        print(
            f"{'Method':<40} {'Mean':<8} {'Clone':<10} {'Clone_OA':<10} {'ParaU':<10} {'ParaU_OA':<10} {'Premium':<10} {'Premium_OA':<10}"
        )
        print("-" * 130)
        print("(OA = With Oracle Adjustment)\n")

        for _, row in interval_score_df.iterrows():
            # Handle potential NaN values for naive scores
            clone_naive = (
                f"{row['clone_score_naive']:.4f}"
                if not np.isnan(row["clone_score_naive"])
                else "N/A"
            )
            para_naive = (
                f"{row['para_score_naive']:.4f}"
                if not np.isnan(row["para_score_naive"])
                else "N/A"
            )
            premium_naive = (
                f"{row['premium_score_naive']:.4f}"
                if not np.isnan(row["premium_score_naive"])
                else "N/A"
            )

            print(
                f"{row['method']:<40} {row['mean_score']:<8.4f} "
                f"{clone_naive:<10} {row['clone_score']:<10.4f} "
                f"{para_naive:<10} {row['para_score']:<10.4f} "
                f"{premium_naive:<10} {row['premium_score']:<10.4f}"
            )

        print("\nInterpretation:")
        print("• Interval score penalizes both wide CIs and non-coverage")
        print("• Oracle adjustment (OA) accounts for oracle sampling uncertainty")
        print("• Lower scores are better - indicate sharp and well-calibrated CIs")
    else:
        print("No interval score data available")

    # 3c. Oracle Adjustment Share Analysis
    print("\n" + "=" * 160)
    print("3c. ORACLE ADJUSTMENT (OA) SHARE BY METHOD")
    print("=" * 160)
    print(
        "\nShows what percentage of total SE variance comes from oracle uncertainty\n"
    )

    # Compute OA share by method
    oa_share_by_method = {}

    for _, group in df.groupby(["estimator", "use_weight_calibration", "use_iic"]):
        estimator = group.iloc[0]["estimator"]
        use_weight_calibration = group.iloc[0]["use_weight_calibration"]
        use_iic = group.iloc[0]["use_iic"]

        method_key = f"{estimator} (calib={use_weight_calibration}, iic={use_iic})"

        # Collect OA shares for well-behaved policies
        oa_shares = []
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            share_col = f"{policy}_se_oracle_share"
            if share_col in group.columns:
                shares = group[share_col].dropna()
                oa_shares.extend(shares.tolist())

        if oa_shares:
            oa_share_by_method[method_key] = {
                "mean": np.mean(oa_shares),
                "median": np.median(oa_shares),
                "max": np.max(oa_shares),
                "n": len(oa_shares),
            }

    # Sort by mean OA share
    sorted_methods = sorted(oa_share_by_method.items(), key=lambda x: x[1]["mean"])

    print(f"{'Method':<40} {'Mean%':<8} {'Median%':<10} {'Max%':<8} {'N':<6}")
    print("-" * 70)

    for method, stats in sorted_methods:
        print(
            f"{method:<40} {stats['mean']*100:<8.2f} {stats['median']*100:<10.2f} {stats['max']*100:<8.2f} {stats['n']:<6}"
        )

    print("\nInterpretation:")
    print(
        "• With ~5000 oracle samples, OA share can still be large when estimator SEs are very small"
    )
    print(
        "• Large OA share means oracle uncertainty dominates total SE (most visible for stacked-DR with ~76%)"
    )
    print("• Raw IPS has tiny OA share because its own SEs are very large")

    # 4. Bias Analysis Table
    print("\n" + "=" * 160)
    print(
        "4. BIAS ANALYSIS: Mean Error (Estimate - Oracle Truth) by Estimator and Policy"
    )
    print("=" * 160)
    print("\nShows systematic bias patterns across estimators")
    print(
        "Negative values indicate underestimation, positive values indicate overestimation\n"
    )

    # Compute bias statistics for each estimator
    bias_by_estimator: Dict[str, Dict[str, List[float]]] = {}

    for result in results:
        estimator = result.get("spec", {}).get("estimator")
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})

        if estimator not in bias_by_estimator:
            bias_by_estimator[estimator] = {
                "clone": [],
                "parallel_universe_prompt": [],
                "premium": [],
                "unhelpful": [],
            }

        # Collect errors for each policy
        for policy in ["clone", "parallel_universe_prompt", "premium", "unhelpful"]:
            if policy in estimates and policy in oracle_truths:
                error = estimates[policy] - oracle_truths[policy]
                bias_by_estimator[estimator][policy].append(error)

    # Create summary table
    print(
        f"{'Estimator':<20} {'N':<6} {'Clone':<18} {'ParallelU':<18} {'Premium':<18} {'Unhelpful':<18}"
    )
    print("-" * 110)

    # Sort estimators by overall RMSE for better presentation
    estimator_order = [
        "raw-ips",
        "calibrated-ips",
        "orthogonalized-ips",
        "dr-cpo",
        "oc-dr-cpo",
        "tr-cpo",
        "tr-cpo-e",
        "stacked-dr",
        "mrdr",
        "tmle",
    ]

    for estimator in estimator_order:
        if estimator not in bias_by_estimator:
            continue

        errors = bias_by_estimator[estimator]
        n_samples = len(errors["clone"]) if errors["clone"] else 0

        row_str = f"{estimator:<20} {n_samples:<6}"

        # For each policy, compute mean bias and standard error
        for policy in ["clone", "parallel_universe_prompt", "premium", "unhelpful"]:
            if errors[policy]:
                mean_bias = np.mean(errors[policy])
                se_bias = np.std(errors[policy]) / np.sqrt(len(errors[policy]))

                # Color code based on bias magnitude and significance
                # Significant bias: |t-stat| > 2
                t_stat = abs(mean_bias / se_bias) if se_bias > 0 else 0

                if t_stat > 3:
                    # Very significant bias
                    bias_str = f"{mean_bias:+.4f}±{se_bias:.4f}*"
                elif t_stat > 2:
                    # Significant bias
                    bias_str = f"{mean_bias:+.4f}±{se_bias:.4f}"
                else:
                    # Not significant
                    bias_str = f"{mean_bias:+.4f}±{se_bias:.4f}"

                row_str += f" {bias_str:<18}"
            else:
                row_str += " " * 18

        print(row_str)

    # Add interpretation
    print("\n* indicates |t-statistic| > 3 (highly significant bias)")
    print("\nKey Insights:")
    print("• Raw-IPS: High variance bias, especially for parallel_universe (-0.17)")
    print(
        "• CalibratedIPS: Consistent negative bias across all policies (~-0.01 to -0.03)"
    )
    print("• DR methods: Nearly unbiased (< ±0.01 for most policies)")
    print(
        "• Trade-off: Calibration reduces variance but introduces systematic negative bias"
    )

    # Add a summary of bias patterns
    print("\n" + "=" * 160)
    print("4b. BIAS PATTERN SUMMARY")
    print("=" * 160)

    # Compute aggregate statistics
    print(
        f"\n{'Estimator':<20} {'Mean |Bias|':<15} {'Max |Bias|':<15} {'Bias Range':<20} {'Pattern':<30}"
    )
    print("-" * 100)

    for estimator in estimator_order:
        if estimator not in bias_by_estimator:
            continue

        errors = bias_by_estimator[estimator]

        # Compute statistics across well-behaved policies (exclude unhelpful)
        all_biases: List[float] = []
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if errors[policy]:
                all_biases.extend([float(np.mean(errors[policy]))])

        if all_biases:
            mean_abs_bias = np.mean([abs(b) for b in all_biases])
            max_abs_bias = max([abs(b) for b in all_biases])
            bias_range = f"[{min(all_biases):+.3f}, {max(all_biases):+.3f}]"

            # Classify pattern
            if all([b < -0.005 for b in all_biases]):
                pattern = "Systematic negative"
            elif all([b > 0.005 for b in all_biases]):
                pattern = "Systematic positive"
            elif max_abs_bias < 0.01:
                pattern = "Nearly unbiased"
            elif max_abs_bias - min(all_biases) > 0.1:
                pattern = "High variance"
            else:
                pattern = "Mixed"

            print(
                f"{estimator:<20} {mean_abs_bias:<15.4f} {max_abs_bias:<15.4f} {bias_range:<20} {pattern:<30}"
            )

    # 5. Diagnostics Comparison: IPS vs CalibratedIPS
    print("\n" + "=" * 160)
    print("5. DIAGNOSTICS COMPARISON: IPS vs CalibratedIPS BY POLICY AND QUADRANT")
    print("=" * 160)

    # Create comparison between raw-ips and calibrated-ips
    diagnostics_by_method_quadrant: Dict[str, Dict[str, Dict[str, float]]] = {}

    for _, group in df.groupby(
        ["estimator", "use_weight_calibration", "use_iic", "quadrant"]
    ):
        estimator = group.iloc[0]["estimator"]
        use_weight_calibration = group.iloc[0]["use_weight_calibration"]
        use_iic = group.iloc[0]["use_iic"]
        quadrant = group.iloc[0]["quadrant"]

        # Only consider IPS estimators (raw vs calibrated)
        if estimator not in ["raw-ips", "calibrated-ips"]:
            continue

        method_key = estimator

        if method_key not in diagnostics_by_method_quadrant:
            diagnostics_by_method_quadrant[method_key] = {}
        if quadrant not in diagnostics_by_method_quadrant[method_key]:
            diagnostics_by_method_quadrant[method_key][quadrant] = {}

        # Collect diagnostic metrics from original results data
        quad_results = [
            r
            for r in results
            if (
                r.get("quadrant") == quadrant
                and r.get("spec", {}).get("estimator") == estimator
                and r.get("spec", {})
                .get("extra", {})
                .get("use_weight_calibration", False)
                == use_weight_calibration
                and r.get("spec", {}).get("extra", {}).get("use_iic", False) == use_iic
            )
        ]

        if not quad_results:
            continue

        # Aggregate diagnostics across experiments in this quadrant
        policies = ["clone", "parallel_universe_prompt", "premium", "unhelpful"]

        for policy in policies:
            ess_values = []
            hill_values = []
            hellinger_values = []
            max_weight_values = []

            for result in quad_results:
                if result.get("ess_relative", {}).get(policy) is not None:
                    ess_values.append(result["ess_relative"][policy])
                if result.get("tail_alpha", {}).get(policy) is not None:
                    hill_values.append(result["tail_alpha"][policy])
                if result.get("hellinger_affinity", {}).get(policy) is not None:
                    hellinger_values.append(result["hellinger_affinity"][policy])
                if result.get("max_weight", {}).get(policy) is not None:
                    max_weight_values.append(result["max_weight"][policy])

            diagnostics_by_method_quadrant[method_key][quadrant][f"{policy}_ess"] = (
                np.mean(ess_values) if ess_values else np.nan
            )
            diagnostics_by_method_quadrant[method_key][quadrant][f"{policy}_hill"] = (
                np.mean(hill_values) if hill_values else np.nan
            )
            diagnostics_by_method_quadrant[method_key][quadrant][
                f"{policy}_hellinger"
            ] = (np.mean(hellinger_values) if hellinger_values else np.nan)
            diagnostics_by_method_quadrant[method_key][quadrant][
                f"{policy}_max_weight"
            ] = (np.mean(max_weight_values) if max_weight_values else np.nan)

    # Create summary tables for each diagnostic
    quadrant_order = [
        "Small-LowOracle",
        "Small-HighOracle",
        "Large-LowOracle",
        "Large-HighOracle",
    ]

    # ESS Comparison Table
    print(f"\nEffective Sample Size (ESS%) - Higher is Better")
    print(
        f"{'Method':<15} {'Clone_Agg':<10} {'ParaU_Agg':<10} {'Premium_Agg':<12} {'Unhelp_Agg':<12}"
    )
    print("-" * 60)

    for method in ["raw-ips", "calibrated-ips"]:
        if method in diagnostics_by_method_quadrant:
            method_data = diagnostics_by_method_quadrant[method]

            # Aggregate across quadrants (include unhelpful for diagnostics)
            ess_clone_agg: List[float] = []
            ess_para_agg: List[float] = []
            ess_premium_agg: List[float] = []
            ess_unhelpful_agg: List[float] = []

            ess_clone_by_quad: Dict[str, float] = {}

            for quad in quadrant_order:
                if quad in method_data:
                    if not np.isnan(method_data[quad].get("clone_ess", np.nan)):
                        ess_clone_agg.append(method_data[quad]["clone_ess"])
                        ess_clone_by_quad[quad] = method_data[quad]["clone_ess"]
                    if not np.isnan(
                        method_data[quad].get("parallel_universe_prompt_ess", np.nan)
                    ):
                        ess_para_agg.append(
                            method_data[quad]["parallel_universe_prompt_ess"]
                        )
                    if not np.isnan(method_data[quad].get("premium_ess", np.nan)):
                        ess_premium_agg.append(method_data[quad]["premium_ess"])
                    if not np.isnan(method_data[quad].get("unhelpful_ess", np.nan)):
                        ess_unhelpful_agg.append(method_data[quad]["unhelpful_ess"])
                else:
                    ess_clone_by_quad[quad] = np.nan

            clone_agg_val = np.mean(ess_clone_agg) if ess_clone_agg else np.nan
            para_agg_val = np.mean(ess_para_agg) if ess_para_agg else np.nan
            premium_agg_val = np.mean(ess_premium_agg) if ess_premium_agg else np.nan
            unhelpful_agg_val = (
                np.mean(ess_unhelpful_agg) if ess_unhelpful_agg else np.nan
            )

            print(
                f"{method:<15} {clone_agg_val:<10.1f} {para_agg_val:<10.1f} {premium_agg_val:<12.1f} {unhelpful_agg_val:<12.1f}"
            )

    # Hill Tail Index Comparison Table
    print(f"\nHill Tail Index - Higher is Better (>2 for finite variance)")
    print(
        f"{'Method':<15} {'Clone_Agg':<10} {'ParaU_Agg':<10} {'Premium_Agg':<12} {'Unhelp_Agg':<12}"
    )
    print("-" * 60)

    for method in ["raw-ips", "calibrated-ips"]:
        if method in diagnostics_by_method_quadrant:
            method_data = diagnostics_by_method_quadrant[method]

            # Aggregate across quadrants (include unhelpful for diagnostics)
            hill_clone_agg: List[float] = []
            hill_para_agg: List[float] = []
            hill_premium_agg: List[float] = []
            hill_unhelpful_agg: List[float] = []

            hill_clone_by_quad: Dict[str, float] = {}

            for quad in quadrant_order:
                if quad in method_data:
                    if not np.isnan(method_data[quad].get("clone_hill", np.nan)):
                        hill_clone_agg.append(method_data[quad]["clone_hill"])
                        hill_clone_by_quad[quad] = method_data[quad]["clone_hill"]
                    if not np.isnan(
                        method_data[quad].get("parallel_universe_prompt_hill", np.nan)
                    ):
                        hill_para_agg.append(
                            method_data[quad]["parallel_universe_prompt_hill"]
                        )
                    if not np.isnan(method_data[quad].get("premium_hill", np.nan)):
                        hill_premium_agg.append(method_data[quad]["premium_hill"])
                    if not np.isnan(method_data[quad].get("unhelpful_hill", np.nan)):
                        hill_unhelpful_agg.append(method_data[quad]["unhelpful_hill"])
                else:
                    hill_clone_by_quad[quad] = np.nan

            clone_agg_val = np.mean(hill_clone_agg) if hill_clone_agg else np.nan
            para_agg_val = np.mean(hill_para_agg) if hill_para_agg else np.nan
            premium_agg_val = np.mean(hill_premium_agg) if hill_premium_agg else np.nan
            unhelpful_agg_val = (
                np.mean(hill_unhelpful_agg) if hill_unhelpful_agg else np.nan
            )

            print(
                f"{method:<15} {clone_agg_val:<10.2f} {para_agg_val:<10.2f} {premium_agg_val:<12.2f} {unhelpful_agg_val:<12.2f}"
            )

    # Hellinger Affinity Comparison Table
    print(f"\nHellinger Affinity - Higher is Better (>0.5 good overlap)")
    print(
        f"{'Method':<15} {'Clone_Agg':<10} {'ParaU_Agg':<10} {'Premium_Agg':<12} {'Unhelp_Agg':<12}"
    )
    print("-" * 60)

    for method in ["raw-ips", "calibrated-ips"]:
        if method in diagnostics_by_method_quadrant:
            method_data = diagnostics_by_method_quadrant[method]

            # Aggregate across quadrants (include unhelpful for diagnostics)
            hell_clone_agg: List[float] = []
            hell_para_agg: List[float] = []
            hell_premium_agg: List[float] = []
            hell_unhelpful_agg: List[float] = []

            hell_clone_by_quad: Dict[str, float] = {}

            for quad in quadrant_order:
                if quad in method_data:
                    if not np.isnan(method_data[quad].get("clone_hellinger", np.nan)):
                        hell_clone_agg.append(method_data[quad]["clone_hellinger"])
                        hell_clone_by_quad[quad] = method_data[quad]["clone_hellinger"]
                    if not np.isnan(
                        method_data[quad].get(
                            "parallel_universe_prompt_hellinger", np.nan
                        )
                    ):
                        hell_para_agg.append(
                            method_data[quad]["parallel_universe_prompt_hellinger"]
                        )
                    if not np.isnan(method_data[quad].get("premium_hellinger", np.nan)):
                        hell_premium_agg.append(method_data[quad]["premium_hellinger"])
                    if not np.isnan(
                        method_data[quad].get("unhelpful_hellinger", np.nan)
                    ):
                        hell_unhelpful_agg.append(
                            method_data[quad]["unhelpful_hellinger"]
                        )
                else:
                    hell_clone_by_quad[quad] = np.nan

            clone_agg_val = np.mean(hell_clone_agg) if hell_clone_agg else np.nan
            para_agg_val = np.mean(hell_para_agg) if hell_para_agg else np.nan
            premium_agg_val = np.mean(hell_premium_agg) if hell_premium_agg else np.nan
            unhelpful_agg_val = (
                np.mean(hell_unhelpful_agg) if hell_unhelpful_agg else np.nan
            )

            print(
                f"{method:<15} {clone_agg_val:<10.3f} {para_agg_val:<10.3f} {premium_agg_val:<12.3f} {unhelpful_agg_val:<12.3f}"
            )

    # Reward Calibration Boundary Proximity - All calibrated methods
    print(f"\nReward Calibration Boundary Proximity (All Calibrated Methods)")
    print(
        f"Shows: Avg distance to nearest boundary | % flagged (near boundary OR outlier relative to other policies)"
    )
    print(
        f"{'Method':<20} {'Clone':<20} {'ParaU':<20} {'Premium':<20} {'Unhelpful':<20}"
    )
    print("-" * 100)

    # Check all methods that use calibrated rewards
    calibrated_methods = ["calibrated-ips", "dr-cpo", "oc-dr-cpo", "stacked-dr"]

    for method_name in calibrated_methods:
        # Collect boundary diagnostics for this method
        clone_nearest_dists = []
        para_nearest_dists = []
        premium_nearest_dists = []
        unhelpful_nearest_dists = []

        # Track flags for each policy
        clone_flags = 0
        para_flags = 0
        premium_flags = 0
        unhelpful_flags = 0

        total_experiments = 0

        # Get all results for this estimator with weight_calibration=True
        method_results = [
            r
            for r in results
            if (
                r.get("spec", {}).get("estimator") == method_name
                and r.get("spec", {})
                .get("extra", {})
                .get("use_weight_calibration", True)
                == True
            )
        ]

        for result in method_results:
            total_experiments += 1

            cal_min = result.get("calibrated_reward_min")
            cal_max = result.get("calibrated_reward_max")
            estimates = result.get("estimates", {})

            if cal_min is not None and cal_max is not None and estimates:
                # Get all policy estimates for this experiment
                policy_estimates = {}
                for policy in [
                    "clone",
                    "parallel_universe_prompt",
                    "premium",
                    "unhelpful",
                ]:
                    if policy in estimates:
                        policy_estimates[policy] = estimates[policy]

                if len(policy_estimates) >= 2:  # Need at least 2 policies to compare
                    # Calculate calibration range and determine adaptive threshold
                    cal_range = cal_max - cal_min

                    # Adaptive threshold: 20% of range or 0.15, whichever is smaller
                    # This prevents false alarms when range is narrow
                    boundary_threshold = min(0.2 * cal_range, 0.15)

                    # Calculate median of all policy estimates for outlier detection
                    all_estimates = list(policy_estimates.values())
                    median_est = np.median(all_estimates)
                    mad = np.median(
                        [abs(e - median_est) for e in all_estimates]
                    )  # Median absolute deviation

                    # Process each policy
                    for policy, est in policy_estimates.items():
                        # Distance to nearest boundary
                        min_dist = est - cal_min
                        max_dist = cal_max - est
                        nearest_dist = min(min_dist, max_dist)

                        # Flag if:
                        # 1. Too close to boundary (using adaptive threshold)
                        # 2. Outside calibration range
                        # 3. Outlier relative to other policies (>2 MAD from median if MAD > 0.05)
                        is_near_boundary = nearest_dist < boundary_threshold
                        is_outside = min_dist < 0 or max_dist < 0
                        is_outlier = mad > 0.05 and abs(est - median_est) > 2 * mad

                        flagged = is_near_boundary or is_outside or is_outlier

                        if policy == "clone":
                            clone_nearest_dists.append(nearest_dist)
                            if flagged:
                                clone_flags += 1
                        elif policy == "parallel_universe_prompt":
                            para_nearest_dists.append(nearest_dist)
                            if flagged:
                                para_flags += 1
                        elif policy == "premium":
                            premium_nearest_dists.append(nearest_dist)
                            if flagged:
                                premium_flags += 1
                        elif policy == "unhelpful":
                            unhelpful_nearest_dists.append(nearest_dist)
                            if flagged:
                                unhelpful_flags += 1

        if total_experiments > 0:
            # Compute statistics
            clone_nearest_avg = (
                np.mean(clone_nearest_dists) if clone_nearest_dists else np.nan
            )
            para_nearest_avg = (
                np.mean(para_nearest_dists) if para_nearest_dists else np.nan
            )
            premium_nearest_avg = (
                np.mean(premium_nearest_dists) if premium_nearest_dists else np.nan
            )
            unhelpful_nearest_avg = (
                np.mean(unhelpful_nearest_dists) if unhelpful_nearest_dists else np.nan
            )

            # Compute flag percentages
            clone_flag_pct = (
                (clone_flags / total_experiments * 100) if total_experiments > 0 else 0
            )
            para_flag_pct = (
                (para_flags / total_experiments * 100) if total_experiments > 0 else 0
            )
            premium_flag_pct = (
                (premium_flags / total_experiments * 100)
                if total_experiments > 0
                else 0
            )
            unhelpful_flag_pct = (
                (unhelpful_flags / total_experiments * 100)
                if total_experiments > 0
                else 0
            )

            # Format as "avg | pct%"
            clone_str = f"{clone_nearest_avg:.3f} | {clone_flag_pct:5.1f}%"
            para_str = f"{para_nearest_avg:.3f} | {para_flag_pct:5.1f}%"
            premium_str = f"{premium_nearest_avg:.3f} | {premium_flag_pct:5.1f}%"
            unhelpful_str = f"{unhelpful_nearest_avg:.3f} | {unhelpful_flag_pct:5.1f}%"

            print(
                f"{method_name:<20} {clone_str:<20} {para_str:<20} {premium_str:<20} {unhelpful_str:<20}"
            )

    print(f"\nInterpretation:")
    print(f"• Distance shown is to NEAREST boundary (min or max)")
    print(
        f"• Flags triggered when: (1) Near boundary (<20% of range), (2) Outside range, or (3) Outlier vs other policies"
    )
    print(f"• Smart threshold adapts to calibration range to avoid false alarms")
    print(
        f"• Key insight: Unhelpful policy shows systematic boundary/outlier issues in DR methods"
    )


def print_quadrant_comparison(results: List[Dict[str, Any]]) -> None:
    """Print comparison across the 4 quadrants."""
    print("=" * 120)
    print("QUADRANT COMPARISON ANALYSIS")
    print("=" * 120)

    # Group results by quadrant
    quadrant_results: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        quad = result.get("quadrant", "Unknown")
        if quad not in quadrant_results:
            quadrant_results[quad] = []
        quadrant_results[quad].append(result)

    print(f"\nQuadrant Definitions:")
    print(f"• Small-LowOracle: ≤1000 samples, ≤25% oracle coverage")
    print(f"• Small-HighOracle: ≤1000 samples, >25% oracle coverage")
    print(f"• Large-LowOracle: >1000 samples, ≤25% oracle coverage")
    print(f"• Large-HighOracle: >1000 samples, >25% oracle coverage")

    # Compute metrics for each quadrant
    quadrant_summaries = {}
    for quad_name, quad_results in quadrant_results.items():
        if not quad_results:
            continue

        quad_df = compute_calibration_metrics(quad_results)
        if len(quad_df) == 0:
            continue

        # Best RMSE method in this quadrant
        rmse_summary = (
            quad_df.groupby(["estimator", "use_weight_calibration"])
            .agg({"overall_rmse": "mean"})
            .reset_index()
            .sort_values("overall_rmse")
        )

        # Best calibration method (well-behaved policies only)
        coverage_data = []
        policies = ["clone", "parallel_universe_prompt", "premium"]
        for _, group in quad_df.groupby(
            ["estimator", "use_weight_calibration", "use_iic"]
        ):
            row_data = {
                "estimator": group.iloc[0]["estimator"],
                "use_weight_calibration": group.iloc[0]["use_weight_calibration"],
            }

            coverages = []
            for policy in policies:
                covered_col = f"{policy}_covered"
                if covered_col in group.columns:
                    coverage = group[covered_col].mean() * 100
                    coverages.append(abs(coverage - 95.0))

            if coverages:
                row_data["calib_score"] = np.mean(coverages)
                coverage_data.append(row_data)

        if coverage_data:
            best_calib = min(coverage_data, key=lambda x: x["calib_score"])
        else:
            best_calib = None

        quadrant_summaries[quad_name] = {
            "n_experiments": len(quad_results),
            "best_rmse_method": f"{rmse_summary.iloc[0]['estimator']} (calib={rmse_summary.iloc[0]['use_weight_calibration']})",
            "best_rmse_value": rmse_summary.iloc[0]["overall_rmse"],
            "best_calib_method": (
                f"{best_calib['estimator']} (calib={best_calib['use_weight_calibration']})"
                if best_calib
                else "None"
            ),
            "best_calib_score": best_calib["calib_score"] if best_calib else np.nan,
        }

    # Print summary table
    print(
        f"\n{'Quadrant':<20} {'N':<6} {'Best RMSE Method':<25} {'RMSE':<8} {'Best Calib Method':<25} {'CalibScore':<10}"
    )
    print("-" * 120)

    for quad_name, summary in quadrant_summaries.items():
        print(
            f"{quad_name:<20} {summary['n_experiments']:<6} {summary['best_rmse_method']:<25} "
            f"{summary['best_rmse_value']:<8.4f} {summary['best_calib_method']:<25} {summary['best_calib_score']:<10.1f}"
        )

    # Analysis insights
    print(f"\n{'='*60}")
    print("QUADRANT INSIGHTS")
    print("=" * 60)

    # Compare performance across data regimes
    small_rmse = [
        v["best_rmse_value"] for k, v in quadrant_summaries.items() if "Small" in k
    ]
    large_rmse = [
        v["best_rmse_value"] for k, v in quadrant_summaries.items() if "Large" in k
    ]

    if small_rmse and large_rmse:
        print(
            f"• Sample Size Effect: Small data RMSE={np.mean(small_rmse):.4f}, Large data RMSE={np.mean(large_rmse):.4f}"
        )

    low_oracle_rmse = [
        v["best_rmse_value"] for k, v in quadrant_summaries.items() if "LowOracle" in k
    ]
    high_oracle_rmse = [
        v["best_rmse_value"] for k, v in quadrant_summaries.items() if "HighOracle" in k
    ]

    if low_oracle_rmse and high_oracle_rmse:
        print(
            f"• Oracle Effect: Low oracle RMSE={np.mean(low_oracle_rmse):.4f}, High oracle RMSE={np.mean(high_oracle_rmse):.4f}"
        )

    # Check if methods are consistent across quadrants
    rmse_methods = [v["best_rmse_method"] for v in quadrant_summaries.values()]
    calib_methods = [
        v["best_calib_method"]
        for v in quadrant_summaries.values()
        if not pd.isna(v["best_calib_score"])
    ]

    from collections import Counter

    rmse_counts = Counter(rmse_methods)
    calib_counts = Counter(calib_methods)

    most_common_rmse = rmse_counts.most_common(1)[0] if rmse_counts else None
    most_common_calib = calib_counts.most_common(1)[0] if calib_counts else None

    if most_common_rmse:
        print(
            f"• Most Robust RMSE Method: {most_common_rmse[0]} (wins {most_common_rmse[1]}/4 quadrants)"
        )
    if most_common_calib:
        print(
            f"• Most Robust Calib Method: {most_common_calib[0]} (wins {most_common_calib[1]}/4 quadrants)"
        )


def compute_ranking_metrics(
    results: List[Dict[str, Any]],
    policies: List[str],
    min_oracle_gap: float = 0.0,
) -> Dict[str, pd.DataFrame]:
    """Compute ranking metrics per experiment and aggregates by estimator/quadrant.

    Metrics per experiment:
      - tau_b: Kendall's tau-b between estimate and truth vectors
      - top1_acc: 1 if argmax(est) == argmax(truth) else 0
      - pairwise_acc: fraction of policy pairs where sign(est_i - est_j) matches sign(truth_i - truth_j),
                      restricted to pairs with |truth_i - truth_j| >= min_oracle_gap
      - regret: truth[oracle_best] - truth[pred_best]
    """
    rows: List[Dict[str, Any]] = []

    for result in results:
        spec = result.get("spec", {})
        estimator = spec.get("estimator", "unknown")
        seed = spec.get("seed_base", 0)
        sample_size = spec.get("sample_size", 0)
        oracle_cov = spec.get("oracle_coverage", 0.0)
        quad = result.get("quadrant", "Unknown")
        est_map = result.get("estimates", {}) or {}
        truth_map = result.get("oracle_truths", {}) or {}

        try:
            est_vec = np.array([float(est_map[p]) for p in policies])
            truth_vec = np.array([float(truth_map[p]) for p in policies])
        except Exception:
            continue

        if not (np.isfinite(est_vec).all() and np.isfinite(truth_vec).all()):
            continue

        try:
            tau, _ = kendalltau(est_vec, truth_vec)
        except Exception:
            tau = np.nan

        try:
            top1_acc = int(int(np.argmax(est_vec)) == int(np.argmax(truth_vec)))
        except Exception:
            top1_acc = 0

        correct = 0
        total = 0
        for i in range(len(policies)):
            for j in range(i + 1, len(policies)):
                dt = truth_vec[i] - truth_vec[j]
                if abs(dt) < float(min_oracle_gap):
                    continue
                de = est_vec[i] - est_vec[j]
                if de == 0:
                    continue
                if (dt > 0 and de > 0) or (dt < 0 and de < 0):
                    correct += 1
                total += 1

        pairwise_acc = (correct / total) if total > 0 else np.nan

        try:
            regret = float(np.max(truth_vec) - truth_vec[int(np.argmax(est_vec))])
        except Exception:
            regret = np.nan

        rows.append(
            {
                "estimator": estimator,
                "seed": seed,
                "sample_size": sample_size,
                "oracle_coverage": oracle_cov,
                "quadrant": quad,
                "tau_b": float(tau) if tau is not None else np.nan,
                "top1_acc": float(top1_acc),
                "pairwise_acc": (
                    float(pairwise_acc) if pairwise_acc == pairwise_acc else np.nan
                ),
                "regret": float(regret) if regret == regret else np.nan,
                "n_pairs": int(total),
            }
        )

    per_exp = pd.DataFrame(rows)
    if per_exp.empty:
        return {
            "per_experiment": per_exp,
            "agg_quadrant": pd.DataFrame(),
            "agg_overall": pd.DataFrame(),
        }

    def _agg(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
        grp = df.groupby(by, dropna=False)
        out = grp[["tau_b", "top1_acc", "pairwise_acc", "regret"]].mean().reset_index()
        out["n_experiments"] = grp.size().values
        out["avg_n_pairs"] = grp["n_pairs"].mean().values
        return out

    agg_q = _agg(per_exp, ["estimator", "quadrant"])
    agg_o = _agg(per_exp, ["estimator"])
    return {"per_experiment": per_exp, "agg_quadrant": agg_q, "agg_overall": agg_o}


def print_ranking_summary(
    results: List[Dict[str, Any]], min_oracle_gap: float = 0.0
) -> None:
    """Print a concise ranking performance block."""
    print("\n" + "=" * 160)
    print("RANKING PERFORMANCE: Kendall τ-b, Top-1, Pairwise, Regret")
    print("=" * 160)

    policies = ["clone", "parallel_universe_prompt", "premium", "unhelpful"]
    rk = compute_ranking_metrics(
        results, policies=policies, min_oracle_gap=min_oracle_gap
    )
    agg_q = rk["agg_quadrant"]
    agg_o = rk["agg_overall"]

    if agg_q.empty and agg_o.empty:
        print("No ranking data available")
        return

    if not agg_q.empty:
        print("\nBy Quadrant (means across experiments)")
        cols_q = [
            "estimator",
            "quadrant",
            "tau_b",
            "top1_acc",
            "pairwise_acc",
            "regret",
            "n_experiments",
            "avg_n_pairs",
        ]
        print(agg_q[cols_q].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if not agg_o.empty:
        print("\nOverall (means across experiments)")
        cols_o = [
            "estimator",
            "tau_b",
            "top1_acc",
            "pairwise_acc",
            "regret",
            "n_experiments",
            "avg_n_pairs",
        ]
        print(agg_o[cols_o].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze ablation experiment results")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("results/all_experiments.jsonl"),
        help="Path to results JSONL file",
    )
    parser.add_argument(
        "--mode",
        choices=["overall", "quadrants"],
        default="overall",
        help="Analysis mode: overall summary or quadrant comparison",
    )
    parser.add_argument(
        "--ranking-min-gap",
        type=float,
        default=0.0,
        help="Minimum oracle gap for pairwise ranking (exclude near-ties)",
    )

    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"Results file not found: {args.results_file}")
        return

    # Load and analyze results
    results = load_results(args.results_file)
    if not results:
        print("No valid results found!")
        return

    if args.mode == "quadrants":
        # Add quadrant classification
        for result in results:
            spec = result.get("spec", {})
            sample_size = spec.get("sample_size", 0)
            oracle_coverage = spec.get("oracle_coverage", 0)

            # Classify into quadrants
            if sample_size <= 1000 and oracle_coverage <= 0.25:
                result["quadrant"] = "Small-LowOracle"
            elif sample_size <= 1000 and oracle_coverage > 0.25:
                result["quadrant"] = "Small-HighOracle"
            elif sample_size > 1000 and oracle_coverage <= 0.25:
                result["quadrant"] = "Large-LowOracle"
            else:
                result["quadrant"] = "Large-HighOracle"

        print_quadrant_comparison(results)
    else:
        print_summary_tables(results)
        # Also print a concise ranking block
        print_ranking_summary(results, min_oracle_gap=args.ranking_min_gap)


if __name__ == "__main__":
    main()
