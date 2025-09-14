"""Report generation for ablation experiments."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from .rmse import compute_rmse_metrics, aggregate_rmse_by_quadrant
from .constants import POLICIES, WELL_BEHAVED_POLICIES
from .constants import POLICIES, WELL_BEHAVED_POLICIES
from .diagnostics import compute_diagnostic_metrics, compute_boundary_analysis
from .ranking import compute_ranking_metrics


POLICIES = ["clone", "parallel_universe_prompt", "premium", "unhelpful"]
WELL_BEHAVED_POLICIES = ["clone", "parallel_universe_prompt", "premium"]


def print_summary_tables(results: List[Dict[str, Any]], verbose: bool = True) -> None:
    """Print comprehensive summary tables.

    Args:
        results: List of experiment results
        verbose: Whether to print detailed tables
    """
    print(f"\nLoaded {len(results)} experiments")
    print("=" * 80)

    # 1. RMSE Summary
    print("\n1. ROOT MEAN SQUARED ERROR (RMSE) SUMMARY")
    print("=" * 80)

    rmse_df = compute_rmse_metrics(results)
    if not rmse_df.empty:
        # Group by estimator configuration
        # Check which columns are available
        agg_dict = {"overall_rmse": ["mean", "std"]}

        # Add policy-specific columns if they exist
        for policy in WELL_BEHAVED_POLICIES:
            rmse_col = f"{policy}_rmse"
            mse_col = f"{policy}_mse"
            if rmse_col in rmse_df.columns:
                agg_dict[rmse_col] = "mean"
            elif mse_col in rmse_df.columns:
                # Convert MSE to RMSE for display
                rmse_df[rmse_col] = np.sqrt(rmse_df[mse_col])
                agg_dict[rmse_col] = "mean"

        rmse_summary = (
            rmse_df.groupby(["estimator", "use_weight_calibration", "use_iic"])
            .agg(agg_dict)
            .round(4)
        )

        print(rmse_summary.to_string())
    else:
        print("No RMSE data available")

    # 2. Coverage Summary
    print("\n2. CONFIDENCE INTERVAL COVERAGE (Target: 95%)")
    print("=" * 80)

    coverage_df = compute_coverage_metrics(results)
    if not coverage_df.empty:
        coverage_summary = aggregate_coverage_by_estimator(coverage_df)

        if not coverage_summary.empty:
            # Format for display
            display_cols = ["estimator", "n_experiments", "calibration_score"]
            for policy in WELL_BEHAVED_POLICIES:
                col = f"{policy}_coverage_pct"
                if col in coverage_summary.columns:
                    display_cols.append(col)

            print(coverage_summary[display_cols].to_string(index=False))
    else:
        print("No coverage data available")

    # 3. Bias Analysis
    print("\n3. BIAS ANALYSIS")
    print("=" * 80)

    bias_df = compute_bias_analysis(results)
    if not bias_df.empty:
        # Display key bias metrics
        display_cols = [
            "estimator",
            "overall_mean_bias",
            "overall_mean_abs_bias",
            "bias_pattern",
        ]
        for policy in WELL_BEHAVED_POLICIES:
            mean_col = f"{policy}_mean_bias"
            if mean_col in bias_df.columns:
                display_cols.append(mean_col)

        print(bias_df[display_cols].to_string(index=False))
    else:
        print("No bias data available")

    # 4. Ranking Metrics
    if verbose:
        print("\n4. RANKING ACCURACY")
        print("=" * 80)

        ranking_results = compute_ranking_metrics(results)
        if not ranking_results["aggregated"].empty:
            display_cols = [
                "estimator",
                "n_experiments",
                "mean_kendall_tau",
                "top1_accuracy",
                "mean_rank_error",
            ]
            print(ranking_results["aggregated"][display_cols].to_string(index=False))
        else:
            print("No ranking data available")

    # 5. Diagnostic Metrics
    if verbose:
        print("\n5. DIAGNOSTIC METRICS")
        print("=" * 80)

        diag_df = compute_diagnostic_metrics(results)
        if not diag_df.empty:
            # Show mean ESS and tail index for each estimator
            summary_data = []
            for estimator in diag_df["estimator"].unique():
                est_df = diag_df[diag_df["estimator"] == estimator]
                row = {"estimator": estimator}

                # Compute mean across well-behaved policies
                for policy in WELL_BEHAVED_POLICIES:
                    ess_col = f"{policy}_ess"
                    tail_col = f"{policy}_tail_index"

                    if ess_col in est_df.columns:
                        row[f"{policy}_ess"] = est_df[ess_col].mean()
                    if tail_col in est_df.columns:
                        row[f"{policy}_tail"] = est_df[tail_col].mean()

                summary_data.append(row)

            summary_df = pd.DataFrame(summary_data)
            print(summary_df.to_string(index=False))
        else:
            print("No diagnostic data available")


def print_quadrant_comparison(
    results: List[Dict[str, Any]], metrics: Optional[List[str]] = None
) -> None:
    """Print quadrant comparison analysis.

    Args:
        results: List of experiment results with quadrant classification
        metrics: List of metrics to include (default: ["rmse", "coverage"])
    """
    if metrics is None:
        metrics = ["rmse", "coverage"]

    print("\nQUADRANT COMPARISON ANALYSIS")
    print("=" * 80)
    print("Quadrants: SL=Small-Low, SH=Small-High, LL=Large-Low, LH=Large-High")
    print("(Sample size × Oracle coverage)\n")

    quadrant_order = [
        "Small-LowOracle",
        "Small-HighOracle",
        "Large-LowOracle",
        "Large-HighOracle",
    ]

    # RMSE by quadrant
    if "rmse" in metrics:
        print("\n1. RMSE BY QUADRANT")
        print("-" * 60)

        rmse_df = compute_rmse_metrics(results)
        if not rmse_df.empty:
            rmse_quad = aggregate_rmse_by_quadrant(rmse_df)

            if not rmse_quad.empty:
                # Pivot for display - use the aggregated column name
                pivot_df = rmse_quad.pivot_table(
                    index="estimator",
                    columns="quadrant",
                    values="overall_rmse_mean",
                    aggfunc="first",  # Already aggregated
                )

                # Reorder columns
                cols_present = [q for q in quadrant_order if q in pivot_df.columns]
                if cols_present:
                    pivot_df = pivot_df[cols_present]
                    pivot_df["mean_rmse"] = pivot_df.mean(axis=1)
                    pivot_df = pivot_df.sort_values("mean_rmse")

                    print(pivot_df.round(4).to_string())
        else:
            print("No RMSE data available")

    # Coverage by quadrant
    if "coverage" in metrics:
        print("\n2. CALIBRATION SCORE BY QUADRANT (Lower is Better)")
        print("-" * 60)

        coverage_df = compute_coverage_metrics(results)
        if not coverage_df.empty:
            # Group by estimator and quadrant - check if quadrant column exists
            if "quadrant" not in coverage_df.columns:
                print("No quadrant data available for coverage analysis")
                return
            grouped = coverage_df.groupby(["estimator", "quadrant"])

            calib_data = []
            for (estimator, quadrant), group in grouped:
                # Compute calibration score for well-behaved policies
                calib_scores = []
                for policy in WELL_BEHAVED_POLICIES:
                    covered_col = f"{policy}_covered"
                    if covered_col in group.columns:
                        coverage_pct = group[covered_col].mean() * 100
                        calib_scores.append(abs(coverage_pct - 95.0))

                if calib_scores:
                    calib_data.append(
                        {
                            "estimator": estimator,
                            "quadrant": quadrant,
                            "calibration_score": np.mean(calib_scores),
                        }
                    )

            if calib_data:
                calib_df = pd.DataFrame(calib_data)

                # Pivot for display
                pivot_df = calib_df.pivot_table(
                    index="estimator",
                    columns="quadrant",
                    values="calibration_score",
                    aggfunc="mean",
                )

                # Reorder columns
                cols_present = [q for q in quadrant_order if q in pivot_df.columns]
                if cols_present:
                    pivot_df = pivot_df[cols_present]
                    pivot_df["mean_calib"] = pivot_df.mean(axis=1)
                    pivot_df = pivot_df.sort_values("mean_calib")

                    print(pivot_df.round(1).to_string())
        else:
            print("No coverage data available")


def generate_latex_tables(
    results: List[Dict[str, Any]], output_dir: Optional[str] = None
) -> Dict[str, str]:
    """Generate LaTeX tables for paper.

    Args:
        results: List of experiment results
        output_dir: Directory to save LaTeX files (optional)

    Returns:
        Dictionary mapping table names to LaTeX strings
    """
    tables = {}

    # Main RMSE table
    rmse_df = compute_rmse_metrics(results)
    if not rmse_df.empty:
        # Format for LaTeX
        rmse_summary = (
            rmse_df.groupby("estimator")
            .agg(
                {
                    "overall_rmse": ["mean", "std"],
                    "clone_rmse": "mean",
                    "parallel_universe_prompt_rmse": "mean",
                    "premium_rmse": "mean",
                }
            )
            .round(4)
        )

        latex_str = rmse_summary.to_latex(
            caption="Root Mean Squared Error by Estimator",
            label="tab:rmse",
            escape=False,
        )
        tables["rmse"] = latex_str

    # Coverage table
    coverage_df = compute_coverage_metrics(results)
    if not coverage_df.empty:
        coverage_summary = aggregate_coverage_by_estimator(coverage_df)

        if not coverage_summary.empty:
            # Select key columns
            display_df = coverage_summary[
                ["estimator", "n_experiments", "calibration_score"]
            ].copy()

            for policy in WELL_BEHAVED_POLICIES:
                col = f"{policy}_coverage_pct"
                if col in coverage_summary.columns:
                    display_df[policy] = coverage_summary[col].round(1)

            latex_str = display_df.to_latex(
                index=False,
                caption="Confidence Interval Coverage (Target: 95%)",
                label="tab:coverage",
                escape=False,
            )
            tables["coverage"] = latex_str

    # Save to files if directory provided
    if output_dir:
        import os

        os.makedirs(output_dir, exist_ok=True)

        for name, latex in tables.items():
            filepath = os.path.join(output_dir, f"{name}_table.tex")
            with open(filepath, "w") as f:
                f.write(latex)
            print(f"Saved {name} table to {filepath}")

    return tables
