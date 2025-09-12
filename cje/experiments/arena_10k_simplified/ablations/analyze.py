#!/usr/bin/env python3
"""Clean entry point for ablation analysis.

Replaces the monolithic analyze_simple.py with modular analysis.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Import from the new modular analysis package
from analysis import (
    load_results,
    add_ablation_config,
    add_quadrant_classification,
    compute_rmse_metrics,
    compute_debiased_rmse,
    aggregate_rmse_by_quadrant,
    compute_coverage_metrics,
    compute_interval_scores,
    aggregate_coverage_by_estimator,
    compute_bias_analysis,
    compute_bias_by_quadrant,
    compute_diagnostic_metrics,
    compute_boundary_analysis,
    compare_ips_diagnostics,
    compute_ranking_metrics,
    compute_pairwise_preferences,
    compute_ranking_by_quadrant,
    print_summary_tables,
    print_quadrant_comparison,
    generate_latex_tables,
)


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze ablation experiment results")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("all_experiments.jsonl"),
        help="Path to results JSONL file",
    )
    parser.add_argument(
        "--analysis",
        choices=["summary", "rmse", "coverage", "bias", "diagnostics", "ranking", "all"],
        default="all",
        help="Type of analysis to perform",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        help="Directory to save LaTeX tables",
    )
    args = parser.parse_args()

    if not args.results_file.exists():
        print(f"Results file not found: {args.results_file}")
        return

    # Load results
    print(f"Loading results from {args.results_file}...")
    results = load_results(args.results_file)
    
    if not results:
        print("No valid results found!")
        return
    
    print(f"Loaded {len(results)} successful experiments")
    
    # Add ablation configuration and quadrant classification
    add_ablation_config(results)
    add_quadrant_classification(results)
    
    # Initialize for later use
    debiased_df = pd.DataFrame()
    
    # Run requested analyses
    if args.analysis in ["rmse", "all"]:
        print("\n" + "=" * 80)
        print("ROOT MEAN SQUARED ERROR ANALYSIS")
        print("=" * 80)
        
        rmse_df = compute_rmse_metrics(results)
        if not rmse_df.empty:
            # Compute debiased RMSE separately  
            debiased_df = compute_debiased_rmse(results)
            
            # Display summary by configuration
            print("\nRMSE by Configuration (averaged across seeds):")
            config_rmse = rmse_df.groupby("config_string")["overall_rmse"].agg(
                ["mean", "std", "count"]
            ).round(4)
            print(config_rmse.to_string())
            
            # Also show by base estimator
            print("\nRMSE by Base Estimator (averaged across all configurations):")
            estimator_rmse = rmse_df.groupby("estimator")["overall_rmse"].agg(
                ["mean", "std", "count"]
            ).round(4)
            print(estimator_rmse.to_string())
            
            if args.verbose:
                # Aggregate by quadrant
                quadrant_rmse = aggregate_rmse_by_quadrant(rmse_df)
                if not quadrant_rmse.empty:
                    print("\nRMSE by Quadrant:")
                    print(quadrant_rmse.to_string())
    
    if args.analysis in ["coverage", "all"]:
        print("\n" + "=" * 80)
        print("CONFIDENCE INTERVAL COVERAGE ANALYSIS")
        print("=" * 80)
        
        coverage_df = compute_coverage_metrics(results)
        if not coverage_df.empty:
            # Aggregate by estimator
            coverage_summary = aggregate_coverage_by_estimator(coverage_df)
            
            if not coverage_summary.empty:
                print("\nCoverage by Estimator Configuration:")
                display_cols = ["estimator", "calibration_score", "n_experiments"]
                for policy in ["clone", "parallel_universe_prompt", "premium"]:
                    col = f"{policy}_coverage_pct"
                    if col in coverage_summary.columns:
                        display_cols.append(col)
                print(coverage_summary[display_cols].round(1).to_string(index=False))
                
            if args.verbose:
                # Add interval scores
                interval_df = compute_interval_scores(results)
                if not interval_df.empty:
                    print("\nInterval Scores (lower is better):")
                    for policy in ["clone", "parallel_universe_prompt", "premium"]:
                        col = f"{policy}_interval_score"
                        if col in interval_df.columns:
                            mean_score = interval_df[col].mean()
                            print(f"  {policy}: {mean_score:.4f}")
    
    if args.analysis in ["bias", "all"]:
        print("\n" + "=" * 80)
        print("BIAS ANALYSIS")
        print("=" * 80)
        
        bias_df = compute_bias_analysis(results)
        if not bias_df.empty:
            print("\nBias Patterns by Estimator:")
            display_cols = [
                "estimator", "overall_mean_bias", "overall_mean_abs_bias",
                "overall_max_abs_bias", "bias_pattern"
            ]
            print(bias_df[display_cols].round(4).to_string(index=False))
            
            if args.verbose:
                # Bias by quadrant
                quad_bias = compute_bias_by_quadrant(results)
                if not quad_bias.empty:
                    print("\nBias by Quadrant:")
                    print(quad_bias.round(4).to_string(index=False))
    
    if args.analysis in ["diagnostics", "all"]:
        print("\n" + "=" * 80)
        print("DIAGNOSTIC METRICS")
        print("=" * 80)
        
        diag_df = compute_diagnostic_metrics(results)
        if not diag_df.empty:
            print("\nEffective Sample Size (ESS%) by Estimator:")
            # Show mean ESS for each estimator
            for estimator in diag_df["estimator"].unique():
                est_df = diag_df[diag_df["estimator"] == estimator]
                ess_vals = []
                for policy in ["clone", "parallel_universe_prompt", "premium"]:
                    col = f"{policy}_ess"
                    if col in est_df.columns:
                        ess_vals.append(est_df[col].mean())
                if ess_vals:
                    print(f"  {estimator}: {np.mean(ess_vals):.1f}%")
        
        if args.verbose:
            # IPS comparison
            ips_comparison = compare_ips_diagnostics(results)
            if ips_comparison.get("ess") is not None and not ips_comparison["ess"].empty:
                print("\nIPS vs Calibrated IPS Comparison:")
                print("ESS:")
                print(ips_comparison["ess"].round(1).to_string(index=False))
                print("\nTail Index:")
                print(ips_comparison["tail_index"].round(2).to_string(index=False))
        
        # Boundary analysis
        boundary_df = compute_boundary_analysis(results)
        if not boundary_df.empty:
            print("\nCalibration Boundary Proximity:")
            for _, row in boundary_df.iterrows():
                print(f"\n{row['method']}:")
                for policy in ["clone", "parallel_universe_prompt", "premium"]:
                    dist_col = f"{policy}_mean_boundary_dist"
                    flag_col = f"{policy}_pct_flagged"
                    if dist_col in row and flag_col in row:
                        print(f"  {policy}: dist={row[dist_col]:.3f}, flagged={row[flag_col]:.1f}%")
    
    
    if args.analysis in ["ranking", "all"]:
        print("\n" + "=" * 80)
        print("RANKING ACCURACY")
        print("=" * 80)
        
        ranking_results = compute_ranking_metrics(results)
        if not ranking_results["aggregated"].empty:
            print("\nRanking Metrics by Estimator:")
            display_cols = [
                "estimator", "n_experiments", "mean_kendall_tau",
                "mean_spearman_rho", "top1_accuracy"
            ]
            final_cols = [c for c in display_cols if c in ranking_results["aggregated"].columns]
            print(ranking_results["aggregated"][final_cols].round(3).to_string(index=False))
        
        if args.verbose:
            # Pairwise preferences
            pairwise_df = compute_pairwise_preferences(results)
            if not pairwise_df.empty:
                print("\nPairwise Preference Accuracy:")
                print(pairwise_df.round(1).to_string())
            
            # Ranking by quadrant
            quad_ranking = compute_ranking_by_quadrant(results)
            if not quad_ranking.empty:
                print("\nRanking Accuracy by Quadrant:")
                print(quad_ranking.round(3).to_string(index=False))
    
    if args.analysis == "summary":
        # Show basic statistics only
        print("\n" + "=" * 80)
        print("EXPERIMENT STATISTICS")
        print("=" * 80)
        
        # Count by configuration
        config_counts = {}
        estimator_counts = {}
        for r in results:
            config = r.get("config_string", "unknown")
            est = r.get("spec", {}).get("estimator", "unknown")
            config_counts[config] = config_counts.get(config, 0) + 1
            estimator_counts[est] = estimator_counts.get(est, 0) + 1
        
        print("\nExperiments by Configuration:")
        for config, count in sorted(config_counts.items()):
            print(f"  {config}: {count}")
        
        print("\nExperiments by Base Estimator:")
        for est, count in sorted(estimator_counts.items()):
            print(f"  {est}: {count}")
        
        # Count by quadrant
        quadrant_counts = {}
        for r in results:
            quad = r.get("quadrant", "unknown")
            quadrant_counts[quad] = quadrant_counts.get(quad, 0) + 1
        
        print("\nExperiments by Quadrant:")
        for quad in ["Small-LowOracle", "Small-HighOracle", "Large-LowOracle", "Large-HighOracle"]:
            if quad in quadrant_counts:
                print(f"  {quad}: {quadrant_counts[quad]}")
    
    # Comprehensive summary for "all" mode with verbose
    if args.analysis == "all" and args.verbose:
        print("\n" + "=" * 160)
        print("COMPREHENSIVE ABLATION RESULTS SUMMARY")
        print("=" * 160)
        print(f"\nDataset: {len(results)} experiments")
        
        # Show the full detailed analysis like analyze_simple.py
        print_summary_tables(results, verbose=True)
        print_quadrant_comparison(results, metrics=["rmse", "coverage", "bias"])
        
        # Also show debiased RMSE comparison
        if not debiased_df.empty:
            print("\n" + "=" * 80)
            print("NOISE-DEBIASED RMSE")
            print("=" * 80)
            debiased_summary = debiased_df.groupby("estimator")["overall_rmse_debiased"].mean().sort_values()
            for est, rmse in debiased_summary.items():
                print(f"  {est}: {rmse:.4f}")
    
    # Generate LaTeX tables if requested
    if args.latex_output:
        print(f"\nGenerating LaTeX tables to {args.latex_output}...")
        generate_latex_tables(results, output_dir=str(args.latex_output))
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()