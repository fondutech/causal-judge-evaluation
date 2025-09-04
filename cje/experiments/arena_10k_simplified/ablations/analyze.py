#!/usr/bin/env python3
"""
Analysis script for unified experiment results.

Generates tables and plots for the paper from unified experiment results.
Works with the actual format produced by run_unified.py.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


def load_results(
    results_path: str = "results/all_experiments.jsonl",
) -> List[Dict[str, Any]]:
    """Load experiment results from JSONL file.

    Args:
        results_path: Path to results file

    Returns:
        List of experiment result dictionaries
    """
    results = []
    path = Path(results_path)

    if not path.exists():
        logger.error(f"Results file not found: {results_path}")
        return []

    with open(path, "r") as f:
        for line in f:
            try:
                result = json.loads(line)
                results.append(result)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed result: {e}")

    logger.info(f"Loaded {len(results)} experiment results")
    return results


def create_diagnostic_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create summary table of key diagnostics across experiments.

    Args:
        results: List of experiment results

    Returns:
        DataFrame with diagnostic summary
    """
    rows = []

    for r in results:
        if not r.get("success"):
            continue

        spec = r["spec"]

        # Average diagnostics across policies
        avg_ess = (
            np.mean(list(r.get("ess_relative", {}).values()))
            if r.get("ess_relative")
            else 0
        )
        avg_tail = (
            np.mean([v for v in r.get("tail_alpha", {}).values() if np.isfinite(v)])
            if r.get("tail_alpha")
            else np.nan
        )
        avg_cv = (
            np.mean([v for v in r.get("weight_cv", {}).values() if np.isfinite(v)])
            if r.get("weight_cv")
            else np.nan
        )

        # Average overlap diagnostics
        avg_hellinger = (
            np.mean(list(r.get("hellinger_affinity", {}).values()))
            if r.get("hellinger_affinity")
            else np.nan
        )

        # Count overlap quality categories
        quality_counts: Dict[str, int] = {}
        if r.get("overlap_quality"):
            for q in r["overlap_quality"].values():
                quality_counts[q] = quality_counts.get(q, 0) + 1
        predominant_quality = (
            max(quality_counts, key=lambda k: quality_counts[k])
            if quality_counts
            else "N/A"
        )

        # DR-specific diagnostics
        avg_ortho = np.nan
        avg_mc_var = np.nan
        if r.get("orthogonality_score"):
            ortho_vals = [
                v for v in r["orthogonality_score"].values() if np.isfinite(v)
            ]
            avg_ortho = np.mean(ortho_vals) if ortho_vals else np.nan
        if r.get("mc_variance_share"):
            mc_vals = [v for v in r["mc_variance_share"].values() if np.isfinite(v)]
            avg_mc_var = np.mean(mc_vals) if mc_vals else np.nan

        # Extract parameters from spec.extra if present
        extra = spec.get("extra", {})
        use_calibration = extra.get(
            "use_calibration", spec.get("use_calibration", False)
        )
        use_iic = extra.get("use_iic", spec.get("use_iic", False))
        weight_mode = extra.get("weight_mode", "hajek")
        reward_calibration_mode = extra.get("reward_calibration_mode", "auto")

        rows.append(
            {
                "Estimator": spec["estimator"],
                "Sample Size": spec.get("sample_size", "N/A"),
                "Oracle Coverage": spec.get("oracle_coverage", "N/A"),
                "Use Calibration": use_calibration,
                "Use IIC": use_iic,
                "Weight Mode": weight_mode,
                "Reward Calib Mode": reward_calibration_mode,
                "Seed": r.get("seed", spec.get("seed_base", "N/A")),
                "Avg ESS (%)": avg_ess,
                "Avg Tail Index": avg_tail,
                "Avg Weight CV": avg_cv,
                "Avg Hellinger": avg_hellinger,
                "Overlap Quality": predominant_quality,
                "Avg Orthogonality": avg_ortho,
                "Avg MC Var Share": avg_mc_var,
                "RMSE vs Oracle": r.get("rmse_vs_oracle", np.nan),
                "Mean CI Width": r.get("mean_ci_width", np.nan),
                "Runtime (s)": r.get("runtime_s", np.nan),
            }
        )

    return pd.DataFrame(rows)


def aggregate_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by experimental scenario.

    Groups by all parameter combinations and computes mean and std across seeds.
    """
    if df.empty:
        return pd.DataFrame()

    group_cols = [
        "Estimator",
        "Sample Size",
        "Oracle Coverage",
        "Use Calibration",
        "Use IIC",
        "Weight Mode",
        "Reward Calib Mode",
    ]

    # Metrics to aggregate
    numeric_cols = [
        "Avg ESS (%)",
        "Avg Tail Index",
        "Avg Weight CV",
        "Avg Hellinger",
        "Avg Orthogonality",
        "Avg MC Var Share",
        "RMSE vs Oracle",
        "Mean CI Width",
        "Runtime (s)",
    ]

    # Group and aggregate
    aggregated = (
        df.groupby(group_cols)
        .agg(
            {
                **{col: ["mean", "std", "count"] for col in numeric_cols},
                "Overlap Quality": lambda x: x.mode().iloc[0] if not x.empty else "N/A",
            }
        )
        .reset_index()
    )

    # Flatten column names - preserve spaces in base column names
    new_columns = []
    for col in aggregated.columns.values:
        if col[1] and col[1] != "<lambda>":
            # For aggregated columns, join with underscore but keep spaces in base name
            new_columns.append(f"{col[0]}_{col[1]}".replace(" ", ""))
        else:
            # For non-aggregated columns, keep original name
            new_columns.append(col[0])
    aggregated.columns = new_columns

    return aggregated


def create_main_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create main comparison table for the paper.

    Shows key metrics for each estimator configuration.
    """
    if df.empty:
        logger.warning("No data to create table")
        return pd.DataFrame()

    # Select key columns - handle both possible naming conventions
    # After aggregation, spaces in column names are replaced with no spaces
    columns_to_select = []

    # Try to find the right column names
    for col in ["Estimator"]:
        if col in df.columns:
            columns_to_select.append(col)

    # Handle the aggregated column names (spaces removed, _mean/_std added)
    base_cols = {
        "SampleSize": "Sample Size",
        "OracleCoverage": "Oracle Coverage",
        "UseCalibration": "Use Calibration",
        "UseIIC": "Use IIC",
    }

    for target, source in base_cols.items():
        if target in df.columns:
            columns_to_select.append(target)
        elif source in df.columns:
            columns_to_select.append(source)

    # Metrics with _mean/_std suffixes
    metric_cols = [
        "RMSEvsOracle_mean",
        "RMSE vs Oracle_mean",
        "RMSEvsOracle_std",
        "RMSE vs Oracle_std",
        "MeanCIWidth_mean",
        "Mean CI Width_mean",
        "MeanCIWidth_std",
        "Mean CI Width_std",
        "AvgESS(%)_mean",
        "Avg ESS (%)_mean",
        "AvgHellinger_mean",
        "Avg Hellinger_mean",
    ]

    for col in metric_cols:
        if col in df.columns:
            columns_to_select.append(col)

    # Get overlap quality if it exists
    for col in ["OverlapQuality", "Overlap Quality"]:
        if col in df.columns:
            columns_to_select.append(col)
            break

    table_df = df[columns_to_select].copy()

    # Standardize column names for downstream processing
    table_df.columns = [col.replace(" ", "") for col in table_df.columns]

    # Format estimator name to include calibration/IIC/weight mode status
    def format_estimator(row: pd.Series) -> str:
        est = str(row["Estimator"])
        use_cal = row.get("UseCalibration", False)
        use_iic = row.get("UseIIC", False)
        weight_mode = row.get("WeightMode", "hajek")

        # For IPS, show calibration status
        if est == "ips":
            if use_cal:
                est = "ips-cal"
            else:
                est = "ips-raw"
        # For DR methods, show calibration status
        elif use_cal:
            est += "-cal"

        # Add weight mode if not default hajek
        if weight_mode == "raw":
            est += "-HT"  # Horvitz-Thompson

        # Add IIC if enabled
        if use_iic:
            est += "+IIC"

        return est

    table_df["Method"] = table_df.apply(format_estimator, axis=1)

    # Format numeric columns
    table_df["RMSE"] = table_df.apply(
        lambda r: (
            f"{r.get('RMSEvsOracle_mean', 0):.4f} ± {r.get('RMSEvsOracle_std', 0):.4f}"
            if not pd.isna(r.get("RMSEvsOracle_mean"))
            else "N/A"
        ),
        axis=1,
    )

    table_df["CI Width"] = table_df.apply(
        lambda r: (
            f"{r['MeanCIWidth_mean']:.4f} ± {r['MeanCIWidth_std']:.4f}"
            if not np.isnan(r["MeanCIWidth_mean"])
            else "N/A"
        ),
        axis=1,
    )

    table_df["ESS (%)"] = table_df["AvgESS(%)_mean"].apply(
        lambda x: f"{x:.1f}" if not np.isnan(x) else "N/A"
    )
    table_df["Hellinger"] = table_df["AvgHellinger_mean"].apply(
        lambda x: f"{x:.3f}" if not np.isnan(x) else "N/A"
    )

    # Select final columns
    final_table = table_df[
        [
            "Method",
            "SampleSize",
            "OracleCoverage",
            "RMSE",
            "CI Width",
            "ESS (%)",
            "Hellinger",
            "OverlapQuality",
        ]
    ]

    # Sort by sample size, oracle coverage, and RMSE
    final_table = final_table.sort_values(
        ["SampleSize", "OracleCoverage", "RMSEvsOracle_mean"],
        ascending=[True, True, True],
    )

    return final_table


def create_calibration_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create table comparing calibration on/off for DR methods."""
    if df.empty:
        return pd.DataFrame()

    # Focus on DR methods
    dr_methods = ["dr-cpo", "tmle", "mrdr", "stacked-dr"]
    dr_df = df[df["Estimator"].isin(dr_methods)]

    comparison_data = []

    for method in dr_methods:
        method_df = dr_df[dr_df["Estimator"] == method]

        # Compare at different oracle coverages
        for oracle_cov in [0.05, 0.10, 0.25]:
            # With calibration
            with_cal = method_df[
                (method_df["OracleCoverage"] == oracle_cov)
                & (method_df["UseCalibration"] == True)
            ]

            # Without calibration
            without_cal = method_df[
                (method_df["OracleCoverage"] == oracle_cov)
                & (method_df["UseCalibration"] == False)
            ]

            if not with_cal.empty and not without_cal.empty:
                # Average across sample sizes
                rmse_with = with_cal["RMSEvsOracle_mean"].mean()
                rmse_without = without_cal["RMSEvsOracle_mean"].mean()

                if not np.isnan(rmse_with) and not np.isnan(rmse_without):
                    improvement = (rmse_without - rmse_with) / rmse_without * 100

                    comparison_data.append(
                        {
                            "Method": method,
                            "Oracle Coverage": f"{oracle_cov:.0%}",
                            "RMSE (No Cal)": f"{rmse_without:.4f}",
                            "RMSE (With Cal)": f"{rmse_with:.4f}",
                            "Improvement": f"{improvement:+.1f}%",
                        }
                    )

    return pd.DataFrame(comparison_data)


def create_iic_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create table comparing IIC on/off for DR methods."""
    if df.empty:
        return pd.DataFrame()

    # Focus on DR methods that support IIC
    dr_methods = ["dr-cpo", "tmle", "mrdr"]
    dr_df = df[df["Estimator"].isin(dr_methods)]

    comparison_data = []

    for method in dr_methods:
        method_df = dr_df[dr_df["Estimator"] == method]

        # Compare at different oracle coverages
        for oracle_cov in [0.05, 0.10, 0.25]:
            # With IIC
            with_iic = method_df[
                (method_df["OracleCoverage"] == oracle_cov)
                & (method_df["UseIIC"] == True)
            ]

            # Without IIC
            without_iic = method_df[
                (method_df["OracleCoverage"] == oracle_cov)
                & (method_df["UseIIC"] == False)
            ]

            if not with_iic.empty and not without_iic.empty:
                # Average across sample sizes
                se_with = (
                    with_iic["MeanCIWidth_mean"].mean() / 3.92
                )  # Convert CI width to SE
                se_without = without_iic["MeanCIWidth_mean"].mean() / 3.92

                if not np.isnan(se_with) and not np.isnan(se_without):
                    reduction = (se_without - se_with) / se_without * 100

                    comparison_data.append(
                        {
                            "Method": method,
                            "Oracle Coverage": f"{oracle_cov:.0%}",
                            "SE (No IIC)": f"{se_without:.4f}",
                            "SE (With IIC)": f"{se_with:.4f}",
                            "SE Reduction": f"{reduction:+.1f}%",
                        }
                    )

    return pd.DataFrame(comparison_data)


def plot_rmse_by_configuration(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot RMSE for different configurations."""
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    oracle_coverages = sorted(df["OracleCoverage"].unique())[:4]

    for idx, oracle_cov in enumerate(oracle_coverages):
        ax = axes[idx]
        cov_df = df[df["OracleCoverage"] == oracle_cov]

        # Prepare data for plotting
        plot_data = []
        for _, row in cov_df.iterrows():
            est_name = row["Estimator"]
            # Handle new unified "ips" estimator
            if est_name == "ips":
                if row["UseCalibration"]:
                    est_name = "ips-cal"
                else:
                    est_name = "ips-raw"
            elif row["UseCalibration"]:
                est_name += "\n(cal)"

            # Add weight mode if Horvitz-Thompson
            if row.get("WeightMode", "hajek") == "raw":
                est_name += "-HT"

            if row["UseIIC"]:
                est_name += "+IIC"

            plot_data.append(
                {
                    "Method": est_name,
                    "Sample Size": row["SampleSize"],
                    "RMSE": row["RMSEvsOracle_mean"],
                    "SE": row["RMSEvsOracle_std"],
                }
            )

        plot_df = pd.DataFrame(plot_data)

        # Create grouped bar plot
        sample_sizes = sorted(plot_df["Sample Size"].unique())
        methods = plot_df["Method"].unique()

        x = np.arange(len(methods))
        width = 0.8 / len(sample_sizes)

        for i, size in enumerate(sample_sizes):
            size_data = plot_df[plot_df["Sample Size"] == size]
            values = []
            errors = []
            for method in methods:
                method_data = size_data[size_data["Method"] == method]
                if not method_data.empty:
                    values.append(method_data["RMSE"].iloc[0])
                    errors.append(method_data["SE"].iloc[0])
                else:
                    values.append(0)
                    errors.append(0)

            ax.bar(
                x + i * width, values, width, yerr=errors, label=f"n={size}", capsize=2
            )

        ax.set_xlabel("Method")
        ax.set_ylabel("RMSE")
        ax.set_title(f"Oracle Coverage = {oracle_cov:.0%}")
        ax.set_xticks(x + width * (len(sample_sizes) - 1) / 2)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("RMSE by Estimator Configuration", fontsize=14)
    plt.tight_layout()

    output_path = output_dir / "rmse_by_configuration.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved RMSE plot to {output_path}")


def save_all_tables(aggregated: pd.DataFrame, output_dir: Path) -> None:
    """Save all tables to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main comparison table
    main_table = create_main_comparison_table(aggregated)
    main_table.to_csv(output_dir / "main_comparison.csv", index=False)
    logger.info(f"Saved main comparison table to {output_dir / 'main_comparison.csv'}")

    # Calibration comparison
    cal_table = create_calibration_comparison_table(aggregated)
    if not cal_table.empty:
        cal_table.to_csv(output_dir / "calibration_comparison.csv", index=False)
        logger.info(
            f"Saved calibration comparison to {output_dir / 'calibration_comparison.csv'}"
        )

    # IIC comparison
    iic_table = create_iic_comparison_table(aggregated)
    if not iic_table.empty:
        iic_table.to_csv(output_dir / "iic_comparison.csv", index=False)
        logger.info(f"Saved IIC comparison to {output_dir / 'iic_comparison.csv'}")

    # Full aggregated results
    aggregated.to_csv(output_dir / "full_aggregated_results.csv", index=False)
    logger.info(
        f"Saved full aggregated results to {output_dir / 'full_aggregated_results.csv'}"
    )


def print_summary(results: List[Dict[str, Any]], aggregated: pd.DataFrame) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("UNIFIED EXPERIMENT ANALYSIS SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print(f"\nTotal experiments run: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print(f"\nFailed experiments:")
        for r in failed[:5]:  # Show first 5 failures
            spec = r["spec"]
            print(
                f"  - {spec['estimator']} (n={spec.get('sample_size')}, oracle={spec.get('oracle_coverage')}): {r.get('error', 'Unknown error')}"
            )

    if not aggregated.empty:
        print(f"\nUnique configurations: {len(aggregated)}")
        print(f"Estimators tested: {aggregated['Estimator'].nunique()}")
        print(f"Sample sizes: {sorted(aggregated['SampleSize'].unique())}")
        print(f"Oracle coverages: {sorted(aggregated['OracleCoverage'].unique())}")

        # Best performing methods
        print("\n" + "-" * 50)
        print("BEST PERFORMING METHODS (by RMSE)")
        print("-" * 50)

        for oracle_cov in sorted(aggregated["OracleCoverage"].unique()):
            cov_df = aggregated[aggregated["OracleCoverage"] == oracle_cov]
            if "RMSEvsOracle_mean" in cov_df.columns:
                best = cov_df.nsmallest(3, "RMSEvsOracle_mean")
                print(f"\nOracle Coverage = {oracle_cov:.0%}:")
                for _, row in best.iterrows():
                    method = row["Estimator"]
                    if row["UseCalibration"]:
                        method += " (cal)"
                    if row["UseIIC"]:
                        method += " + IIC"
                    print(
                        f"  {method}: RMSE={row['RMSEvsOracle_mean']:.4f}, n={row['SampleSize']}"
                    )

        # Calibration impact summary
        print("\n" + "-" * 50)
        print("CALIBRATION IMPACT (DR methods average)")
        print("-" * 50)

        dr_methods = ["dr-cpo", "tmle", "mrdr", "stacked-dr"]
        dr_df = aggregated[aggregated["Estimator"].isin(dr_methods)]

        if not dr_df.empty and "RMSEvsOracle_mean" in dr_df.columns:
            with_cal_df = dr_df[dr_df["UseCalibration"] == True]
            without_cal_df = dr_df[dr_df["UseCalibration"] == False]

            if not with_cal_df.empty and not without_cal_df.empty:
                with_cal = with_cal_df["RMSEvsOracle_mean"].mean()
                without_cal = without_cal_df["RMSEvsOracle_mean"].mean()

                if not np.isnan(with_cal) and not np.isnan(without_cal):
                    improvement = (without_cal - with_cal) / without_cal * 100
                    print(f"Average RMSE with calibration: {with_cal:.4f}")
                    print(f"Average RMSE without calibration: {without_cal:.4f}")
                    print(f"Overall improvement: {improvement:+.1f}%")


def main() -> None:
    """Run complete analysis."""
    # Load results
    results = load_results()

    if not results:
        logger.error("No results found. Please run experiments first.")
        return

    # Create diagnostic summary
    df = create_diagnostic_summary(results)

    if df.empty:
        logger.error("No successful experiments to analyze.")
        return

    # Aggregate by scenario
    aggregated = aggregate_by_scenario(df)

    # Create output directory
    output_dir = Path("results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all tables
    save_all_tables(aggregated, output_dir)

    # Create plots
    plot_rmse_by_configuration(aggregated, output_dir)

    # Print summary
    print_summary(results, aggregated)

    print(f"\n{'=' * 70}")
    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
