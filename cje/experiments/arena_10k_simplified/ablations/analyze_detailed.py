#!/usr/bin/env python3
"""
Detailed ablation analysis matching the rich output of analyze_simple.py
but using the modular analysis functions.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Import modular functions
from analysis import (
    load_results,
    add_ablation_config,
    add_quadrant_classification,
)

# Import shared constants
from analysis.constants import POLICIES, WELL_BEHAVED_POLICIES, QUADRANT_ORDER, QUADRANT_ABBREVIATIONS


def create_method_key(result: Dict[str, Any]) -> str:
    """Create a detailed method key showing all configuration dimensions."""
    spec = result.get("spec", {})
    estimator = spec.get("estimator", "unknown")
    use_weight_cal = result.get("use_weight_calibration", False)
    use_iic = result.get("use_iic", False)
    
    # For IPS, weight calibration is always true
    if estimator == "calibrated-ips":
        return f"{estimator} (calib=True, iic={use_iic})"
    else:
        return f"{estimator} (calib={use_weight_cal}, iic={use_iic})"


def print_rmse_tables(results: List[Dict[str, Any]]):
    """Print detailed RMSE tables like analyze_simple.py."""
    print("=" * 160)
    print("1. RMSE PERFORMANCE BY POLICY AND QUADRANT")
    print("=" * 160)
    
    # Group results by method and quadrant
    rmse_by_method_quad = {}
    oracle_var_by_method_quad = {}
    
    for result in results:
        method_key = create_method_key(result)
        quadrant = result.get("quadrant", "Unknown")
        
        if method_key not in rmse_by_method_quad:
            rmse_by_method_quad[method_key] = {}
            oracle_var_by_method_quad[method_key] = {}
        if quadrant not in rmse_by_method_quad[method_key]:
            rmse_by_method_quad[method_key][quadrant] = {p: [] for p in POLICIES}
            rmse_by_method_quad[method_key][quadrant]["overall_sqerrs"] = []
            rmse_by_method_quad[method_key][quadrant]["overall_ovars"] = []
            oracle_var_by_method_quad[method_key][quadrant] = {p: [] for p in POLICIES}
        
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})
        n_oracle_truth = result.get("n_oracle_truth", 4989)
        
        # Collect squared errors and oracle variances for each policy
        for policy in POLICIES:
            if policy in estimates and policy in oracle_truths:
                sqerr = (estimates[policy] - oracle_truths[policy]) ** 2
                rmse_by_method_quad[method_key][quadrant][policy].append(sqerr)
                
                # Oracle variance
                truth = oracle_truths[policy]
                oracle_var = min(truth * (1 - truth), 0.25) / n_oracle_truth
                oracle_var_by_method_quad[method_key][quadrant][policy].append(oracle_var)
        
        # Overall metric for well-behaved policies
        if all(p in estimates and p in oracle_truths for p in WELL_BEHAVED_POLICIES):
            overall_sqerrs = [
                (estimates[p] - oracle_truths[p]) ** 2 for p in WELL_BEHAVED_POLICIES
            ]
            overall_ovars = [
                min(oracle_truths[p] * (1 - oracle_truths[p]), 0.25) / n_oracle_truth 
                for p in WELL_BEHAVED_POLICIES
            ]
            rmse_by_method_quad[method_key][quadrant]["overall_sqerrs"].extend(overall_sqerrs)
            rmse_by_method_quad[method_key][quadrant]["overall_ovars"].extend(overall_ovars)
    
    # Create summary table
    summary_rows = []
    
    for method, quad_data in rmse_by_method_quad.items():
        row = {"method": method}
        
        # Aggregate across quadrants
        all_overall_sqerrs = []
        all_overall_ovars = []
        policy_sqerrs = {p: [] for p in POLICIES}
        policy_ovars = {p: [] for p in POLICIES}
        
        for quad in QUADRANT_ORDER:
            if quad in quad_data:
                # Per-quadrant overall
                if quad_data[quad]["overall_sqerrs"]:
                    abbr = QUADRANT_ABBREVIATIONS[quad]
                    row[f"{abbr}_Over"] = np.sqrt(
                        np.mean(quad_data[quad]["overall_sqerrs"])
                    )
                    all_overall_sqerrs.extend(quad_data[quad]["overall_sqerrs"])
                    if "overall_ovars" in quad_data[quad]:
                        all_overall_ovars.extend(quad_data[quad]["overall_ovars"])
                
                # Collect for aggregates
                for policy in POLICIES:
                    if quad_data[quad][policy]:
                        policy_sqerrs[policy].extend(quad_data[quad][policy])
                        if policy in oracle_var_by_method_quad[method][quad]:
                            policy_ovars[policy].extend(oracle_var_by_method_quad[method][quad][policy])
        
        # Compute aggregates (standard and debiased)
        if all_overall_sqerrs:
            row["Overall"] = np.sqrt(np.mean(all_overall_sqerrs))
            
            # Debiased RMSE
            if all_overall_ovars:
                debiased_mse = np.mean(all_overall_sqerrs) - np.mean(all_overall_ovars)
                row["Overall_Deb"] = np.sqrt(max(debiased_mse, 0.0))
        
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy_sqerrs[policy]:
                if policy == "parallel_universe_prompt":
                    row["ParaU"] = np.sqrt(np.mean(policy_sqerrs[policy]))
                else:
                    row[policy.capitalize()] = np.sqrt(np.mean(policy_sqerrs[policy]))
        
        summary_rows.append(row)
    
    # Sort by overall RMSE
    df = pd.DataFrame(summary_rows).sort_values("Overall", na_position="last")
    
    # Print table with formatting
    print(f"{'Method':<40} {'Overall':<8} {'Debiased':<8} {'Clone':<8} {'ParaU':<8} {'Premium':<8} "
          f"{'SL_Over':<8} {'SH_Over':<8} {'LL_Over':<8} {'LH_Over':<8}")
    print("-" * 160)
    
    for _, row in df.iterrows():
        print(f"{row['method']:<40} "
              f"{row.get('Overall', np.nan):<8.4f} "
              f"{row.get('Overall_Deb', np.nan):<8.4f} "
              f"{row.get('Clone', np.nan):<8.4f} "
              f"{row.get('ParaU', np.nan):<8.4f} "
              f"{row.get('Premium', np.nan):<8.4f} "
              f"{row.get('SL_Over', np.nan):<8.4f} "
              f"{row.get('SH_Over', np.nan):<8.4f} "
              f"{row.get('LL_Over', np.nan):<8.4f} "
              f"{row.get('LH_Over', np.nan):<8.4f}")


def print_bias_tables(results: List[Dict[str, Any]]):
    """Print bias analysis."""
    print("\n" + "=" * 160)
    print("2. BIAS ANALYSIS (Estimate - Truth)")
    print("=" * 160)
    
    bias_by_method = {}
    
    for result in results:
        method_key = create_method_key(result)
        
        if method_key not in bias_by_method:
            bias_by_method[method_key] = {p: [] for p in POLICIES}
        
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})
        
        for policy in POLICIES:
            if policy in estimates and policy in oracle_truths:
                bias = estimates[policy] - oracle_truths[policy]
                bias_by_method[method_key][policy].append(bias)
    
    # Create summary
    summary_rows = []
    for method, policy_bias in bias_by_method.items():
        row = {"method": method}
        
        # Calculate mean bias and std for each policy
        for policy in WELL_BEHAVED_POLICIES:
            if policy_bias[policy]:
                mean_bias = np.mean(policy_bias[policy])
                std_bias = np.std(policy_bias[policy])
                if policy == "parallel_universe_prompt":
                    row["ParaU_bias"] = mean_bias
                    row["ParaU_std"] = std_bias
                else:
                    row[f"{policy.capitalize()}_bias"] = mean_bias
                    row[f"{policy.capitalize()}_std"] = std_bias
        
        # Overall bias
        all_biases = []
        for policy in WELL_BEHAVED_POLICIES:
            all_biases.extend(policy_bias[policy])
        if all_biases:
            row["Overall_bias"] = np.mean(all_biases)
            row["Overall_std"] = np.std(all_biases)
        
        summary_rows.append(row)
    
    # Sort by absolute overall bias
    df = pd.DataFrame(summary_rows)
    df["abs_bias"] = df["Overall_bias"].abs()
    df = df.sort_values("abs_bias", na_position="last")
    
    print(f"{'Method':<40} {'Overall':<12} {'Clone':<12} {'ParaU':<12} {'Premium':<12}")
    print(f"{'':40} {'Bias (Std)':<12} {'Bias (Std)':<12} {'Bias (Std)':<12} {'Bias (Std)':<12}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        overall = f"{row.get('Overall_bias', 0):.4f} ({row.get('Overall_std', 0):.4f})"
        clone = f"{row.get('Clone_bias', 0):.4f} ({row.get('Clone_std', 0):.4f})"
        parau = f"{row.get('ParaU_bias', 0):.4f} ({row.get('ParaU_std', 0):.4f})"
        premium = f"{row.get('Premium_bias', 0):.4f} ({row.get('Premium_std', 0):.4f})"
        
        print(f"{row['method']:<40} {overall:<12} {clone:<12} {parau:<12} {premium:<12}")


def print_coverage_tables(results: List[Dict[str, Any]]):
    """Print confidence interval coverage analysis."""
    print("\n" + "=" * 160)
    print("3. CONFIDENCE INTERVAL COVERAGE BY POLICY AND QUADRANT (Target: 95%)")
    print("=" * 160)
    
    coverage_by_method = {}
    
    for result in results:
        method_key = create_method_key(result)
        
        if method_key not in coverage_by_method:
            coverage_by_method[method_key] = {p: [] for p in POLICIES}
        
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})
        confidence_intervals = result.get("confidence_intervals", {})
        
        for policy in POLICIES:
            if all(policy in d for d in [estimates, oracle_truths, confidence_intervals]):
                ci = confidence_intervals[policy]
                if ci and len(ci) == 2:
                    covered = ci[0] <= oracle_truths[policy] <= ci[1]
                    coverage_by_method[method_key][policy].append(covered)
    
    # Create summary
    summary_rows = []
    for method, policy_coverage in coverage_by_method.items():
        row = {"method": method}
        
        # Calculate coverage percentages
        calib_scores = []
        for policy in WELL_BEHAVED_POLICIES:
            if policy_coverage[policy]:
                cov_pct = np.mean(policy_coverage[policy]) * 100
                if policy == "parallel_universe_prompt":
                    row["ParaU%"] = cov_pct
                else:
                    row[f"{policy.capitalize()}%"] = cov_pct
                calib_scores.append(abs(cov_pct - 95.0))
        
        if calib_scores:
            row["CalibScore"] = np.mean(calib_scores)
        
        summary_rows.append(row)
    
    # Sort by calibration score
    df = pd.DataFrame(summary_rows).sort_values("CalibScore", na_position="last")
    
    print(f"{'Method':<40} {'CalibScore':<10} {'Clone%':<7} {'ParaU%':<7} {'Premium%':<8}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        print(f"{row['method']:<40} "
              f"{row.get('CalibScore', np.nan):<10.1f} "
              f"{row.get('Clone%', np.nan):<7.1f} "
              f"{row.get('ParaU%', np.nan):<7.1f} "
              f"{row.get('Premium%', np.nan):<8.1f}")


def print_boundary_analysis_table(results: List[Dict[str, Any]]):
    """Print calibration boundary proximity analysis."""
    print("\n" + "=" * 160)
    print("4. CALIBRATION BOUNDARY ANALYSIS")
    print("=" * 160)
    
    # Only analyze calibrated methods
    calibrated_methods = ["calibrated-ips", "dr-cpo", "oc-dr-cpo", "orthogonalized-ips"]
    
    boundary_stats = {}
    
    for result in results:
        spec = result.get("spec", {})
        estimator = spec.get("estimator")
        
        if estimator not in calibrated_methods:
            continue
            
        method_key = create_method_key(result)
        
        if method_key not in boundary_stats:
            boundary_stats[method_key] = {
                "total": 0,
                "flagged": {p: 0 for p in POLICIES},
                "min_dist": {p: [] for p in POLICIES}
            }
        
        boundary_stats[method_key]["total"] += 1
        
        cal_min = result.get("calibrated_reward_min")
        cal_max = result.get("calibrated_reward_max")
        estimates = result.get("estimates", {})
        
        if cal_min is None or cal_max is None:
            continue
        
        # Check each policy
        for policy in POLICIES:
            if policy in estimates:
                reward = estimates[policy]
                dist_to_boundary = min(abs(reward - cal_min), abs(reward - cal_max))
                boundary_stats[method_key]["min_dist"][policy].append(dist_to_boundary)
                
                # Flag if problematic
                cal_range = cal_max - cal_min
                threshold = min(0.2 * cal_range, 0.15)
                if dist_to_boundary < threshold or reward < cal_min or reward > cal_max:
                    boundary_stats[method_key]["flagged"][policy] += 1
    
    # Print summary
    print("\nMethods with potential calibration boundary issues:")
    print(f"{'Method':<40} {'Clone':<12} {'ParaU':<12} {'Premium':<12} {'Unhelpful':<12}")
    print(f"{'':40} {'% Flagged':<12} {'% Flagged':<12} {'% Flagged':<12} {'% Flagged':<12}")
    print("-" * 100)
    
    for method, stats in sorted(boundary_stats.items()):
        if stats["total"] == 0:
            continue
        
        row_vals = [method]
        for policy in POLICIES:
            pct_flagged = 100.0 * stats["flagged"][policy] / stats["total"]
            if policy == "parallel_universe_prompt":
                row_vals.append(f"{pct_flagged:.1f}%")
            else:
                row_vals.append(f"{pct_flagged:.1f}%")
        
        print(f"{row_vals[0]:<40} {row_vals[1]:<12} {row_vals[2]:<12} {row_vals[3]:<12} {row_vals[4]:<12}")


def print_diagnostics_tables(results: List[Dict[str, Any]]):
    """Print ESS and tail index diagnostics."""
    print("\n" + "=" * 160)
    print("5. EFFECTIVE SAMPLE SIZE AND TAIL INDEX DIAGNOSTICS")
    print("=" * 160)
    
    diag_by_method = {}
    
    for result in results:
        method_key = create_method_key(result)
        
        if method_key not in diag_by_method:
            diag_by_method[method_key] = {
                "ess": {p: [] for p in POLICIES},
                "tail": {p: [] for p in POLICIES}
            }
        
        # Collect ESS and tail indices
        for policy in POLICIES:
            ess_key = f"ess_relative"
            if ess_key in result and policy in result[ess_key]:
                diag_by_method[method_key]["ess"][policy].append(
                    result[ess_key][policy]
                )
            
            tail_key = f"tail_alpha"
            if tail_key in result and policy in result[tail_key]:
                diag_by_method[method_key]["tail"][policy].append(
                    result[tail_key][policy]
                )
    
    # Print ESS table
    print("\nEffective Sample Size (% of total samples):")
    print(f"{'Method':<40} {'Clone':<10} {'ParaU':<10} {'Premium':<10} {'Unhelpful':<10}")
    print("-" * 90)
    
    for method, diags in sorted(diag_by_method.items()):
        ess_vals = []
        for policy in POLICIES:
            if diags["ess"][policy]:
                avg_ess = np.mean(diags["ess"][policy])
                if policy == "parallel_universe_prompt":
                    ess_vals.append(f"{avg_ess:.1f}%")
                else:
                    ess_vals.append(f"{avg_ess:.1f}%")
            else:
                ess_vals.append("N/A")
        
        print(f"{method:<40} {ess_vals[0]:<10} {ess_vals[1]:<10} {ess_vals[2]:<10} {ess_vals[3]:<10}")
    
    # Print tail index table
    print("\nTail Index (α < 2 indicates heavy tails):")
    print(f"{'Method':<40} {'Clone':<10} {'ParaU':<10} {'Premium':<10} {'Unhelpful':<10}")
    print("-" * 90)
    
    for method, diags in sorted(diag_by_method.items()):
        tail_vals = []
        for policy in POLICIES:
            if diags["tail"][policy]:
                avg_tail = np.mean(diags["tail"][policy])
                tail_vals.append(f"{avg_tail:.2f}")
            else:
                tail_vals.append("N/A")
        
        print(f"{method:<40} {tail_vals[0]:<10} {tail_vals[1]:<10} {tail_vals[2]:<10} {tail_vals[3]:<10}")


def print_ranking_analysis(results: List[Dict[str, Any]]):
    """Print policy ranking accuracy analysis."""
    print("\n" + "=" * 160)
    print("6. POLICY RANKING ACCURACY")
    print("=" * 160)
    
    from scipy.stats import kendalltau, spearmanr
    
    ranking_by_method = {}
    
    for result in results:
        method_key = create_method_key(result)
        
        if method_key not in ranking_by_method:
            ranking_by_method[method_key] = {
                "kendall_tau": [],
                "spearman_rho": [],
                "top1_correct": [],
                "total": 0
            }
        
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})
        
        # Get values for well-behaved policies only
        est_values = []
        true_values = []
        policies_ordered = []
        
        for policy in WELL_BEHAVED_POLICIES:
            if policy in estimates and policy in oracle_truths:
                est_values.append(estimates[policy])
                true_values.append(oracle_truths[policy])
                policies_ordered.append(policy)
        
        if len(est_values) >= 2:
            ranking_by_method[method_key]["total"] += 1
            
            # Compute rank correlations
            tau, _ = kendalltau(true_values, est_values)
            rho, _ = spearmanr(true_values, est_values)
            
            if not np.isnan(tau):
                ranking_by_method[method_key]["kendall_tau"].append(tau)
            if not np.isnan(rho):
                ranking_by_method[method_key]["spearman_rho"].append(rho)
            
            # Check if top-1 is correct
            best_true_idx = np.argmax(true_values)
            best_est_idx = np.argmax(est_values)
            top1_correct = (best_true_idx == best_est_idx)
            ranking_by_method[method_key]["top1_correct"].append(top1_correct)
    
    # Create summary
    summary_rows = []
    for method, metrics in ranking_by_method.items():
        if metrics["total"] == 0:
            continue
            
        row = {
            "method": method,
            "n_experiments": metrics["total"],
            "kendall_tau": np.mean(metrics["kendall_tau"]) if metrics["kendall_tau"] else np.nan,
            "spearman_rho": np.mean(metrics["spearman_rho"]) if metrics["spearman_rho"] else np.nan,
            "top1_accuracy": 100.0 * np.mean(metrics["top1_correct"]) if metrics["top1_correct"] else 0.0
        }
        summary_rows.append(row)
    
    # Sort by Kendall tau
    df = pd.DataFrame(summary_rows).sort_values("kendall_tau", ascending=False, na_position="last")
    
    print("\nRanking accuracy for well-behaved policies (clone, parallel_universe_prompt, premium):")
    print(f"{'Method':<40} {'Kendall τ':<10} {'Spearman ρ':<10} {'Top-1 Acc%':<10} {'N':<5}")
    print("-" * 75)
    
    for _, row in df.iterrows():
        print(f"{row['method']:<40} "
              f"{row['kendall_tau']:<10.3f} "
              f"{row['spearman_rho']:<10.3f} "
              f"{row['top1_accuracy']:<10.1f} "
              f"{row['n_experiments']:<5.0f}")
    
    # Additional insights
    print("\nInterpretation:")
    print("• Kendall τ: Rank correlation (-1 to 1, higher is better)")
    print("• Spearman ρ: Another rank correlation metric")
    print("• Top-1 Acc: % of times the best policy was correctly identified")


def print_summary_insights(results: List[Dict[str, Any]]):
    """Print summary insights and recommendations."""
    print("\n" + "=" * 160)
    print("SUMMARY INSIGHTS")
    print("=" * 160)
    
    # Find best method by RMSE
    method_rmse = {}
    for result in results:
        method_key = create_method_key(result)
        rmse = result.get("rmse_vs_oracle")
        if rmse is not None:
            if method_key not in method_rmse:
                method_rmse[method_key] = []
            method_rmse[method_key].append(rmse)
    
    avg_rmse = {m: np.mean(vals) for m, vals in method_rmse.items()}
    if avg_rmse:
        best_method = min(avg_rmse.items(), key=lambda x: x[1])
        print(f"\n🏆 Best RMSE: {best_method[0]} ({best_method[1]:.4f})")
    
    # Check weight calibration impact
    wcal_methods = [m for m in avg_rmse.keys() if "calib=True" in m]
    no_wcal_methods = [m for m in avg_rmse.keys() if "calib=False" in m]
    
    if wcal_methods and no_wcal_methods:
        wcal_avg = np.mean([avg_rmse[m] for m in wcal_methods])
        no_wcal_avg = np.mean([avg_rmse[m] for m in no_wcal_methods])
        if wcal_avg < no_wcal_avg:
            improvement = (no_wcal_avg - wcal_avg) / no_wcal_avg * 100
            print(f"• Weight calibration improves RMSE by {improvement:.1f}%")
        else:
            degradation = (wcal_avg - no_wcal_avg) / no_wcal_avg * 100
            print(f"• Weight calibration degrades RMSE by {degradation:.1f}%")
    
    # Check IIC impact
    iic_methods = [m for m in avg_rmse.keys() if "iic=True" in m]
    no_iic_methods = [m for m in avg_rmse.keys() if "iic=False" in m]
    
    if iic_methods and no_iic_methods:
        iic_avg = np.mean([avg_rmse[m] for m in iic_methods])
        no_iic_avg = np.mean([avg_rmse[m] for m in no_iic_methods])
        if abs(iic_avg - no_iic_avg) / no_iic_avg < 0.01:
            print(f"• IIC has minimal impact (<1% difference)")
        elif iic_avg < no_iic_avg:
            improvement = (no_iic_avg - iic_avg) / no_iic_avg * 100
            print(f"• IIC improves RMSE by {improvement:.1f}%")
        else:
            degradation = (iic_avg - no_iic_avg) / no_iic_avg * 100
            print(f"• IIC degrades RMSE by {degradation:.1f}%")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Detailed ablation analysis")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("results/all_experiments.jsonl"),
        help="Path to results JSONL file",
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
    
    print(f"Dataset: {len(results)} experiments")
    
    # Add configuration details
    add_ablation_config(results)
    add_quadrant_classification(results)
    
    # Print header
    print("\n" + "=" * 160)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 160)
    
    # Print detailed tables
    print_rmse_tables(results)           # Table 1
    print_bias_tables(results)            # Table 2
    print_coverage_tables(results)        # Table 3
    print_boundary_analysis_table(results)  # Table 4
    print_diagnostics_tables(results)     # Table 5
    print_ranking_analysis(results)       # Table 6
    print_summary_insights(results)
    
    print("\n" + "=" * 160)
    print("Analysis complete")
    print("=" * 160)


if __name__ == "__main__":
    main()