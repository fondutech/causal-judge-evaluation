#!/usr/bin/env python3
"""Rich analysis output that combines modular structure with informative displays.

This provides the detailed, interpretive output of analyze_simple.py
while using the modular analysis functions.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Import modular analysis functions
from analysis import (
    load_results,
    add_ablation_config,
    add_quadrant_classification,
    compute_rmse_metrics,
    compute_debiased_rmse,
    compute_coverage_metrics,
    compute_diagnostic_metrics,
)

# Policy definitions
POLICIES = ["clone", "parallel_universe_prompt", "premium", "unhelpful"]
WELL_BEHAVED_POLICIES = ["clone", "parallel_universe_prompt", "premium"]


def print_header(title: str, width: int = 120):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_detailed_rmse_analysis(results: List[Dict[str, Any]]):
    """Print detailed RMSE analysis with interpretations."""
    print_header("DETAILED RMSE ANALYSIS")

    # Compute RMSE metrics
    rmse_df = compute_rmse_metrics(results)
    if rmse_df.empty:
        print("No RMSE data available")
        return

    # Group by configuration and compute statistics
    config_stats = (
        rmse_df.groupby("config_string")
        .agg(
            {
                "overall_rmse": ["mean", "std", "min", "max", "count"],
                "clone_mse": lambda x: np.sqrt(np.nanmean(x)),
                "parallel_universe_prompt_mse": lambda x: np.sqrt(np.nanmean(x)),
                "premium_mse": lambda x: np.sqrt(np.nanmean(x)),
                "unhelpful_mse": lambda x: np.sqrt(np.nanmean(x)),
            }
        )
        .round(4)
    )

    # Rename columns for clarity
    config_stats.columns = [
        "rmse_mean",
        "rmse_std",
        "rmse_min",
        "rmse_max",
        "n_exp",
        "clone_rmse",
        "para_rmse",
        "premium_rmse",
        "unhelpful_rmse",
    ]

    # Sort by mean RMSE
    config_stats = config_stats.sort_values("rmse_mean")

    print("\n📊 RMSE by Configuration (sorted by performance):")
    print("-" * 120)
    print(
        f"{'Configuration':<45} {'RMSE':<12} {'Clone':<10} {'ParaU':<10} {'Premium':<10} {'Unhelpful':<10} {'N':<5}"
    )
    print("-" * 120)

    for config, row in config_stats.iterrows():
        # Parse configuration for better display
        parts = config.split("_")
        estimator = parts[0]
        weight_cal = "✓" if "WCal" in config else "✗"
        iic = "✓" if "IIC" in config else "✗"
        reward_mode = parts[-1] if len(parts) > 3 else "?"

        display_name = f"{estimator:<20} WCal:{weight_cal} IIC:{iic} {reward_mode:<8}"

        print(
            f"{display_name:<45} "
            f"{row['rmse_mean']:.4f}±{row['rmse_std']:.4f} "
            f"{row['clone_rmse']:.4f}     "
            f"{row['para_rmse']:.4f}     "
            f"{row['premium_rmse']:.4f}     "
            f"{row['unhelpful_rmse']:.4f}     "
            f"{int(row['n_exp'])}"
        )

    # Find winners
    print("\n🏆 Best Configurations:")
    best_overall = config_stats.index[0]
    print(
        f"  • Best Overall RMSE: {best_overall} ({config_stats.loc[best_overall, 'rmse_mean']:.4f})"
    )

    # Best for unhelpful policy
    best_unhelpful = config_stats.sort_values("unhelpful_rmse").index[0]
    print(
        f"  • Best for Unhelpful: {best_unhelpful} ({config_stats.loc[best_unhelpful, 'unhelpful_rmse']:.4f})"
    )

    # Insights
    print("\n💡 Key Insights:")

    # Check if weight calibration helps
    wcal_configs = [c for c in config_stats.index if "WCal" in c and "NoWCal" not in c]
    no_wcal_configs = [c for c in config_stats.index if "NoWCal" in c]

    if wcal_configs and no_wcal_configs:
        wcal_mean = config_stats.loc[wcal_configs, "rmse_mean"].mean()
        no_wcal_mean = config_stats.loc[no_wcal_configs, "rmse_mean"].mean()
        improvement = (no_wcal_mean - wcal_mean) / no_wcal_mean * 100
        if improvement > 0:
            print(
                f"  • Weight calibration improves RMSE by {improvement:.1f}% on average"
            )
        else:
            print(
                f"  • Weight calibration hurts RMSE by {-improvement:.1f}% on average"
            )

    # Check if IIC helps
    iic_configs = [c for c in config_stats.index if "IIC" in c and "NoIIC" not in c]
    no_iic_configs = [c for c in config_stats.index if "NoIIC" in c]

    if iic_configs and no_iic_configs:
        iic_mean = config_stats.loc[iic_configs, "rmse_mean"].mean()
        no_iic_mean = config_stats.loc[no_iic_configs, "rmse_mean"].mean()
        improvement = (no_iic_mean - iic_mean) / no_iic_mean * 100
        if abs(improvement) < 1:
            print(f"  • IIC has minimal impact on RMSE (< 1% difference)")
        elif improvement > 0:
            print(f"  • IIC improves RMSE by {improvement:.1f}% on average")
        else:
            print(f"  • IIC hurts RMSE by {-improvement:.1f}% on average")

    # Unhelpful policy analysis
    unhelpful_ratios = config_stats["unhelpful_rmse"] / config_stats["rmse_mean"]
    print(f"  • Unhelpful policy is {unhelpful_ratios.mean():.1f}x harder than average")

    # Estimator comparison
    estimator_groups = {}
    for config in config_stats.index:
        base_est = config.split("_")[0]
        if base_est not in estimator_groups:
            estimator_groups[base_est] = []
        estimator_groups[base_est].append(config)

    print("\n📈 Estimator Comparison (best configuration for each):")
    for est, configs in estimator_groups.items():
        best_config = config_stats.loc[configs, "rmse_mean"].idxmin()
        best_rmse = config_stats.loc[best_config, "rmse_mean"]
        print(f"  • {est}: {best_rmse:.4f} ({best_config})")


def print_coverage_analysis(results: List[Dict[str, Any]]):
    """Print detailed confidence interval coverage analysis."""
    print_header("CONFIDENCE INTERVAL COVERAGE ANALYSIS")

    coverage_df = compute_coverage_metrics(results)
    if coverage_df.empty:
        print("No coverage data available")
        return

    # Add configuration
    for result in results:
        idx = coverage_df[
            (coverage_df["estimator"] == result["spec"]["estimator"])
            & (
                coverage_df["seed"]
                == result.get("seed", result["spec"].get("seed_base", 0))
            )
        ].index
        if len(idx) > 0:
            coverage_df.loc[idx, "config_string"] = result.get(
                "config_string", "unknown"
            )

    # Compute coverage statistics by configuration
    print("\n📊 95% CI Coverage by Configuration (target = 95%):")
    print("-" * 120)
    print(
        f"{'Configuration':<45} {'Clone':<12} {'ParaU':<12} {'Premium':<12} {'Unhelpful':<12}"
    )
    print("-" * 120)

    for config in coverage_df["config_string"].unique():
        if pd.isna(config):
            continue
        config_data = coverage_df[coverage_df["config_string"] == config]

        # Parse configuration for display
        parts = config.split("_")
        estimator = parts[0]
        weight_cal = "✓" if "WCal" in config else "✗"
        iic = "✓" if "IIC" in config else "✗"
        reward_mode = parts[-1] if len(parts) > 3 else "?"

        display_name = f"{estimator:<20} WCal:{weight_cal} IIC:{iic} {reward_mode:<8}"

        coverages = []
        for policy in POLICIES:
            col = f"{policy}_covered"
            if col in config_data.columns:
                cov = config_data[col].mean() * 100
                coverages.append(f"{cov:.1f}%")
            else:
                coverages.append("N/A")

        print(
            f"{display_name:<45} {coverages[0]:<12} {coverages[1]:<12} {coverages[2]:<12} {coverages[3]:<12}"
        )

    print("\n💡 Coverage Insights:")

    # Check for undercoverage
    for policy in POLICIES:
        col = f"{policy}_covered"
        if col in coverage_df.columns:
            avg_coverage = coverage_df[col].mean() * 100
            if avg_coverage < 90:
                print(f"  ⚠️  {policy}: Severe undercoverage ({avg_coverage:.1f}%)")
            elif avg_coverage < 93:
                print(f"  • {policy}: Slight undercoverage ({avg_coverage:.1f}%)")
            elif avg_coverage > 97:
                print(f"  • {policy}: Slight overcoverage ({avg_coverage:.1f}%)")

    # Check if coverage varies by configuration
    coverage_variance = {}
    for policy in POLICIES:
        col = f"{policy}_covered"
        if col in coverage_df.columns:
            by_config = coverage_df.groupby("config_string")[col].mean() * 100
            coverage_variance[policy] = by_config.std()

    high_variance_policies = [p for p, v in coverage_variance.items() if v > 5]
    if high_variance_policies:
        print(
            f"  • High coverage variance across configs: {', '.join(high_variance_policies)}"
        )


def print_diagnostics_summary(results: List[Dict[str, Any]]):
    """Print diagnostic metrics summary."""
    print_header("DIAGNOSTIC METRICS")

    diag_df = compute_diagnostic_metrics(results)
    if diag_df.empty:
        print("No diagnostic data available")
        return

    # Add configuration
    for result in results:
        idx = diag_df[
            (diag_df["estimator"] == result["spec"]["estimator"])
            & (
                diag_df["seed"]
                == result.get("seed", result["spec"].get("seed_base", 0))
            )
        ].index
        if len(idx) > 0:
            diag_df.loc[idx, "config_string"] = result.get("config_string", "unknown")

    print("\n📊 Effective Sample Size (ESS) Analysis:")
    print("-" * 100)

    # Group by configuration and show ESS for key policies
    for config in diag_df["config_string"].unique():
        if pd.isna(config):
            continue
        config_data = diag_df[diag_df["config_string"] == config]

        print(f"\n{config}:")

        for policy in ["clone", "unhelpful"]:
            ess_col = f"{policy}_ess_pct"
            tail_col = f"{policy}_tail_alpha"

            if ess_col in config_data.columns:
                ess = config_data[ess_col].mean()
                if ess < 10:
                    status = "🔴 CRITICAL"
                elif ess < 30:
                    status = "🟡 WARNING"
                else:
                    status = "🟢 GOOD"

                tail_alpha = (
                    config_data[tail_col].mean()
                    if tail_col in config_data.columns
                    else np.nan
                )

                print(
                    f"  {policy:<20}: ESS={ess:.1f}% {status}  Tail α={tail_alpha:.2f}"
                )

    print("\n💡 Diagnostic Insights:")

    # Check for problematic policies
    for policy in POLICIES:
        ess_col = f"{policy}_ess_pct"
        if ess_col in diag_df.columns:
            avg_ess = diag_df[ess_col].mean()
            if avg_ess < 10:
                print(f"  🔴 {policy}: Critical overlap issue (ESS={avg_ess:.1f}%)")
            elif avg_ess < 30:
                print(f"  🟡 {policy}: Poor overlap (ESS={avg_ess:.1f}%)")

    # Check tail behavior
    for policy in POLICIES:
        tail_col = f"{policy}_tail_alpha"
        if tail_col in diag_df.columns:
            avg_tail = diag_df[tail_col].mean()
            if avg_tail < 2:
                print(f"  ⚠️  {policy}: Heavy tails detected (α={avg_tail:.2f})")


def print_recommendations(results: List[Dict[str, Any]]):
    """Print final recommendations based on analysis."""
    print_header("RECOMMENDATIONS", width=80)

    rmse_df = compute_rmse_metrics(results)
    coverage_df = compute_coverage_metrics(results)

    if rmse_df.empty:
        print("Insufficient data for recommendations")
        return

    # Find best overall configuration
    config_performance = (
        rmse_df.groupby("config_string")["overall_rmse"].mean().sort_values()
    )

    print("\n🏆 WINNER:")
    best_config = config_performance.index[0]
    print(f"  {best_config}")
    print(f"  RMSE: {config_performance.iloc[0]:.4f}")

    print("\n📋 METHOD RANKINGS:")
    for i, (config, rmse) in enumerate(config_performance.items(), 1):
        if i > 5:  # Show top 5
            break
        print(f"  {i}. {config:<40} RMSE={rmse:.4f}")

    print("\n🔍 KEY FINDINGS:")

    # Check estimator types
    estimator_groups = {}
    for config in config_performance.index:
        base = config.split("_")[0]
        if base not in estimator_groups:
            estimator_groups[base] = []
        estimator_groups[base].append(config_performance[config])

    best_estimator = min(estimator_groups.items(), key=lambda x: np.mean(x[1]))[0]
    print(f"  • Best estimator family: {best_estimator}")

    # Check calibration impact
    wcal_configs = [
        c for c in config_performance.index if "WCal" in c and "NoWCal" not in c
    ]
    no_wcal_configs = [c for c in config_performance.index if "NoWCal" in c]

    if wcal_configs and no_wcal_configs:
        wcal_perf = config_performance[wcal_configs].mean()
        no_wcal_perf = config_performance[no_wcal_configs].mean()
        if wcal_perf < no_wcal_perf:
            print(f"  • Weight calibration improves performance")
        else:
            print(f"  • Weight calibration hurts performance")

    print("\n⚠️  WARNINGS:")

    # Check for undercoverage
    if not coverage_df.empty:
        for policy in POLICIES:
            col = f"{policy}_covered"
            if col in coverage_df.columns:
                avg_cov = coverage_df[col].mean() * 100
                if avg_cov < 90:
                    print(f"  • {policy}: Severe CI undercoverage ({avg_cov:.1f}%)")

    # Check for poor overlap
    diag_df = compute_diagnostic_metrics(results)
    if not diag_df.empty:
        for policy in POLICIES:
            ess_col = f"{policy}_ess_pct"
            if ess_col in diag_df.columns:
                avg_ess = diag_df[ess_col].mean()
                if avg_ess < 10:
                    print(f"  • {policy}: Critical overlap (ESS={avg_ess:.1f}%)")


def main():
    """Main analysis function with rich output."""
    parser = argparse.ArgumentParser(description="Rich ablation analysis")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("results/all_experiments.jsonl"),
        help="Path to results JSONL file",
    )
    parser.add_argument(
        "--analysis",
        choices=["all", "rmse", "coverage", "diagnostics", "summary"],
        default="all",
        help="Type of analysis to perform",
    )
    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"❌ Results file not found: {args.results_file}")
        return

    # Load and prepare results
    print(f"Loading results from {args.results_file}...")
    results = load_results(args.results_file)

    if not results:
        print("❌ No valid results found!")
        return

    print(f"✅ Loaded {len(results)} successful experiments")

    # Add configurations
    add_ablation_config(results)
    add_quadrant_classification(results)

    # Print header
    print("\n" + "🔬" * 40)
    print(" " * 35 + "ABLATION ANALYSIS REPORT")
    print("🔬" * 40)

    # Run analyses
    if args.analysis in ["rmse", "all"]:
        print_detailed_rmse_analysis(results)

    if args.analysis in ["coverage", "all"]:
        print_coverage_analysis(results)

    if args.analysis in ["diagnostics", "all"]:
        print_diagnostics_summary(results)

    if args.analysis in ["all", "summary"]:
        print_recommendations(results)

    print("\n" + "=" * 80)
    print("Analysis complete! 🎉")
    print("=" * 80)


if __name__ == "__main__":
    main()
