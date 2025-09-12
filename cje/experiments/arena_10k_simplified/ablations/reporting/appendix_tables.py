"""
Appendix tables for comprehensive diagnostics.

Generates tables A1-A6 for the paper appendix with detailed breakdowns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json


def generate_quadrant_leaderboard(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate Table A1: Quadrant-specific leaderboard.
    
    Shows RMSE_d and CalibScore broken down by data regime quadrant.
    
    Args:
        results: List of experiment results
        
    Returns:
        DataFrame with quadrant-specific metrics
    """
    from .paper_tables import create_config_key
    
    metrics_by_quadrant = {}
    
    for result in results:
        config_key = create_config_key(result)
        
        # Compute quadrant from spec
        spec = result.get("spec", {})
        size = spec.get("sample_size", 0)
        coverage = spec.get("oracle_coverage", 0)
        
        if size <= 1000:
            size_label = "Small"
        else:
            size_label = "Large"
            
        if coverage <= 0.1:
            cov_label = "LowOracle"
        else:
            cov_label = "HighOracle"
            
        quadrant = f"{size_label}-{cov_label}"
        
        if quadrant not in metrics_by_quadrant:
            metrics_by_quadrant[quadrant] = {}
        if config_key not in metrics_by_quadrant[quadrant]:
            metrics_by_quadrant[quadrant][config_key] = {
                'rmse': [],
                'coverage': []
            }
        
        # Collect RMSE
        rmse = result.get("rmse_vs_oracle")
        if rmse is not None:
            metrics_by_quadrant[quadrant][config_key]['rmse'].append(rmse)
        
        # Collect coverage for calibration score
        robust_cis = result.get("robust_confidence_intervals") or result.get("confidence_intervals", {})
        oracle_truths = result.get("oracle_truths", {})
        
        covered = []
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy in robust_cis and policy in oracle_truths:
                ci = robust_cis[policy]
                truth = oracle_truths[policy]
                covered.append(ci[0] <= truth <= ci[1])
        
        if covered:
            coverage = np.mean(covered)
            metrics_by_quadrant[quadrant][config_key]['coverage'].append(coverage)
    
    # Build table with quadrants as columns
    rows = []
    all_configs = set()
    for q_metrics in metrics_by_quadrant.values():
        all_configs.update(q_metrics.keys())
    
    for config in sorted(all_configs):
        row = {'Estimator': config}
        
        for quadrant in ['Small-LowOracle', 'Small-HighOracle', 'Large-LowOracle', 'Large-HighOracle']:
            abbrev = {'Small-LowOracle': 'SL', 'Small-HighOracle': 'SH',
                     'Large-LowOracle': 'LL', 'Large-HighOracle': 'LH'}[quadrant]
            
            if quadrant in metrics_by_quadrant and config in metrics_by_quadrant[quadrant]:
                rmse_vals = metrics_by_quadrant[quadrant][config]['rmse']
                cov_vals = metrics_by_quadrant[quadrant][config]['coverage']
                
                if rmse_vals:
                    row[f'{abbrev}_RMSE'] = np.mean(rmse_vals)
                else:
                    row[f'{abbrev}_RMSE'] = np.nan
                    
                if cov_vals:
                    row[f'{abbrev}_Calib'] = abs(np.mean(cov_vals) - 0.95) * 100
                else:
                    row[f'{abbrev}_Calib'] = np.nan
            else:
                row[f'{abbrev}_RMSE'] = np.nan
                row[f'{abbrev}_Calib'] = np.nan
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_bias_patterns_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate Table A2: Bias Patterns.
    
    Shows mean bias, mean |bias|, and per-policy biases with significance.
    
    Args:
        results: List of experiment results
        
    Returns:
        DataFrame with bias analysis
    """
    from .paper_tables import create_config_key
    
    bias_by_config = {}
    
    for result in results:
        config_key = create_config_key(result)
        
        if config_key not in bias_by_config:
            bias_by_config[config_key] = {
                'all_errors': [],
                'clone_errors': [],
                'parallel_errors': [],
                'premium_errors': [],
                'unhelpful_errors': []
            }
        
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})
        
        # Collect errors by policy
        for policy in ["clone", "parallel_universe_prompt", "premium", "unhelpful"]:
            if policy in estimates and policy in oracle_truths:
                error = estimates[policy] - oracle_truths[policy]
                
                if policy == "parallel_universe_prompt":
                    bias_by_config[config_key]['parallel_errors'].append(error)
                else:
                    bias_by_config[config_key][f'{policy}_errors'].append(error)
                
                # Add to overall (well-behaved only)
                if policy != "unhelpful":
                    bias_by_config[config_key]['all_errors'].append(error)
    
    # Compute statistics
    rows = []
    for config, errors_dict in bias_by_config.items():
        row = {'Estimator': config}
        
        # Overall bias (well-behaved only)
        if errors_dict['all_errors']:
            all_errors = errors_dict['all_errors']
            row['Mean_Bias'] = np.mean(all_errors)
            row['Mean_Abs_Bias'] = np.mean(np.abs(all_errors))
            row['Bias_SE'] = np.std(all_errors) / np.sqrt(len(all_errors))
            
            # Classify pattern
            if row['Mean_Bias'] < -0.005:
                row['Pattern'] = 'Negative'
            elif row['Mean_Bias'] > 0.005:
                row['Pattern'] = 'Positive'
            elif row['Mean_Abs_Bias'] < 0.01:
                row['Pattern'] = 'Unbiased'
            else:
                row['Pattern'] = 'Mixed'
        
        # Per-policy bias with t-stats
        for policy, key in [('clone', 'clone_errors'), 
                            ('parallel', 'parallel_errors'),
                            ('premium', 'premium_errors')]:
            if errors_dict[key]:
                errors = errors_dict[key]
                mean_bias = np.mean(errors)
                se_bias = np.std(errors) / np.sqrt(len(errors))
                t_stat = abs(mean_bias / se_bias) if se_bias > 0 else 0
                
                row[f'{policy}_bias'] = mean_bias
                row[f'{policy}_t'] = t_stat
                row[f'{policy}_sig'] = '*' if t_stat > 2 else ''
        
        rows.append(row)
    
    return pd.DataFrame(rows).sort_values('Mean_Abs_Bias')


def generate_overlap_diagnostics_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate Table A3: Overlap & Tail Diagnostics.
    
    Buckets ESS%, Tail index, and Hellinger affinity into Good/OK/Poor.
    
    Args:
        results: List of experiment results
        
    Returns:
        DataFrame with bucketed diagnostics
    """
    from .paper_tables import create_config_key
    
    diagnostics_by_config = {}
    
    for result in results:
        config_key = create_config_key(result)
        
        if config_key not in diagnostics_by_config:
            diagnostics_by_config[config_key] = {
                'ess_pct': [],
                'tail_index': [],
                'hellinger': []
            }
        
        # Extract diagnostics - ESS is at top level as dict by policy
        ess_rel = result.get("ess_relative")
        if ess_rel is not None and isinstance(ess_rel, dict):
            # Average ESS across well-behaved policies and convert to percentage
            ess_values = []
            for policy in ["clone", "parallel_universe_prompt", "premium"]:
                if policy in ess_rel:
                    ess_values.append(ess_rel[policy])
            if ess_values:
                avg_ess_pct = np.mean(ess_values) * 100
                diagnostics_by_config[config_key]['ess_pct'].append(avg_ess_pct)
        
        # Check if there's diagnostics dict for other metrics
        diagnostics = result.get("diagnostics", {})
        if diagnostics:
            tail_idx = diagnostics.get("tail_index")
            if tail_idx is not None and tail_idx > 0:  # Filter invalid
                diagnostics_by_config[config_key]['tail_index'].append(tail_idx)
            
            hellinger = diagnostics.get("hellinger_affinity")
            if hellinger is not None:
                diagnostics_by_config[config_key]['hellinger'].append(hellinger)
    
    # Bucket and aggregate
    rows = []
    for config, diag_dict in diagnostics_by_config.items():
        row = {'Estimator': config}
        
        # ESS% bucketing (>20 good, 10-20 OK, <10 poor)
        if diag_dict['ess_pct']:
            ess_vals = diag_dict['ess_pct']
            row['ESS_Good'] = np.mean([e > 20 for e in ess_vals]) * 100
            row['ESS_OK'] = np.mean([10 <= e <= 20 for e in ess_vals]) * 100
            row['ESS_Poor'] = np.mean([e < 10 for e in ess_vals]) * 100
            row['ESS_Median'] = np.median(ess_vals)
        
        # Tail index bucketing (>2 finite variance, 1.5-2 OK, <1.5 poor)
        if diag_dict['tail_index']:
            tail_vals = diag_dict['tail_index']
            row['Tail_Good'] = np.mean([t > 2 for t in tail_vals]) * 100
            row['Tail_OK'] = np.mean([1.5 <= t <= 2 for t in tail_vals]) * 100
            row['Tail_Poor'] = np.mean([t < 1.5 for t in tail_vals]) * 100
            row['Tail_Median'] = np.median(tail_vals)
        
        # Hellinger bucketing (>0.5 good, 0.3-0.5 OK, <0.3 poor)
        if diag_dict['hellinger']:
            hell_vals = diag_dict['hellinger']
            row['Hell_Good'] = np.mean([h > 0.5 for h in hell_vals]) * 100
            row['Hell_OK'] = np.mean([0.3 <= h <= 0.5 for h in hell_vals]) * 100
            row['Hell_Poor'] = np.mean([h < 0.3 for h in hell_vals]) * 100
            row['Hell_Median'] = np.median(hell_vals)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_oracle_adjustment_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate Table A4: Oracle Adjustment Share.
    
    Shows the proportion of uncertainty from oracle slice calibration.
    
    Args:
        results: List of experiment results
        
    Returns:
        DataFrame with OA analysis
    """
    from .paper_tables import create_config_key
    
    oa_by_config = {}
    
    for result in results:
        config_key = create_config_key(result)
        
        if config_key not in oa_by_config:
            oa_by_config[config_key] = {
                'oa_shares': [],
                'coverage_base': [],
                'coverage_oa': []
            }
        
        # Check if we have OA information
        base_ses = result.get("standard_errors", {})
        robust_ses = result.get("robust_standard_errors", {})
        
        # Calculate OA share if we have both
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy in base_ses and policy in robust_ses:
                base_se = base_ses[policy]
                robust_se = robust_ses[policy]
                
                if base_se > 0 and robust_se > 0:
                    # OA share = (robust^2 - base^2) / robust^2
                    oa_share = max(0, (robust_se**2 - base_se**2) / robust_se**2)
                    oa_by_config[config_key]['oa_shares'].append(oa_share)
        
        # Coverage with and without OA
        robust_cis = result.get("robust_confidence_intervals") or result.get("confidence_intervals", {})
        base_cis = result.get("confidence_intervals", {})
        oracle_truths = result.get("oracle_truths", {})
        
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy in oracle_truths:
                truth = oracle_truths[policy]
                
                if policy in base_cis:
                    ci = base_cis[policy]
                    oa_by_config[config_key]['coverage_base'].append(
                        ci[0] <= truth <= ci[1]
                    )
                
                if policy in robust_cis:
                    ci = robust_cis[policy]
                    oa_by_config[config_key]['coverage_oa'].append(
                        ci[0] <= truth <= ci[1]
                    )
    
    # Aggregate
    rows = []
    for config, oa_dict in oa_by_config.items():
        row = {'Estimator': config}
        
        if oa_dict['oa_shares']:
            row['OA_Share_Mean'] = np.mean(oa_dict['oa_shares']) * 100
            row['OA_Share_Median'] = np.median(oa_dict['oa_shares']) * 100
            row['OA_Share_Max'] = np.max(oa_dict['oa_shares']) * 100
        
        if oa_dict['coverage_base']:
            row['Coverage_Base'] = np.mean(oa_dict['coverage_base']) * 100
        
        if oa_dict['coverage_oa']:
            row['Coverage_OA'] = np.mean(oa_dict['coverage_oa']) * 100
        
        if 'Coverage_Base' in row and 'Coverage_OA' in row:
            row['Coverage_Diff'] = row['Coverage_OA'] - row['Coverage_Base']
        
        rows.append(row)
    
    return pd.DataFrame(rows).sort_values('OA_Share_Mean', ascending=False)


def generate_boundary_outlier_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate Table A5: Calibration Boundary Analysis with RCS-Lite.
    
    Shows distance to boundaries and outlier detection rates.
    
    Args:
        results: List of experiment results
        
    Returns:
        DataFrame with boundary analysis
    """
    from .paper_tables import create_config_key
    
    boundary_by_config = {}
    
    for result in results:
        config_key = create_config_key(result)
        
        if config_key not in boundary_by_config:
            boundary_by_config[config_key] = {
                'min_distances': [],
                'outlier_flags': [],
                'unhelpful_distances': [],
                'cal_ranges': [],
                'cal_mins': [],
                'cal_maxs': []
            }
        
        # Get calibrated reward range
        cal_min = result.get("calibrated_reward_min")
        cal_max = result.get("calibrated_reward_max")
        
        if cal_min is not None and cal_max is not None:
            estimates = result.get("estimates", {})
            
            # Store calibration range info for RCS check
            cal_range = cal_max - cal_min
            boundary_by_config[config_key]['cal_ranges'].append(cal_range)
            boundary_by_config[config_key]['cal_mins'].append(cal_min)
            boundary_by_config[config_key]['cal_maxs'].append(cal_max)
            
            for policy in ["clone", "parallel_universe_prompt", "premium"]:
                if policy in estimates:
                    est = estimates[policy]
                    # Distance to nearest boundary
                    dist = min(est - cal_min, cal_max - est)
                    boundary_by_config[config_key]['min_distances'].append(dist)
            
            # Special handling for unhelpful
            if "unhelpful" in estimates:
                est = estimates["unhelpful"]
                dist = min(est - cal_min, cal_max - est)
                boundary_by_config[config_key]['unhelpful_distances'].append(dist)
                
                # Check if outlier (using adaptive threshold)
                cal_range = cal_max - cal_min
                threshold = min(0.2 * cal_range, 0.15)
                is_outlier = dist < threshold
                boundary_by_config[config_key]['outlier_flags'].append(is_outlier)
    
    # Aggregate
    rows = []
    for config, boundary_dict in boundary_by_config.items():
        row = {'Estimator': config}
        
        if boundary_dict['min_distances']:
            row['Mean_Dist_Boundary'] = np.mean(boundary_dict['min_distances'])
            row['Min_Dist_Boundary'] = np.min(boundary_dict['min_distances'])
            row['Pct_Near_Boundary'] = np.mean([d < 0.1 for d in boundary_dict['min_distances']]) * 100
        
        if boundary_dict['unhelpful_distances']:
            row['Unhelpful_Mean_Dist'] = np.mean(boundary_dict['unhelpful_distances'])
            row['Unhelpful_Min_Dist'] = np.min(boundary_dict['unhelpful_distances'])
        
        if boundary_dict['outlier_flags']:
            row['Outlier_Rate'] = np.mean(boundary_dict['outlier_flags']) * 100
        
        # Simple RCS-Lite check based on calibration range
        if boundary_dict['cal_ranges']:
            avg_range = np.mean(boundary_dict['cal_ranges'])
            avg_min = np.mean(boundary_dict['cal_mins'])
            avg_max = np.mean(boundary_dict['cal_maxs'])
            
            # Good support: range > 0.4 AND covers [0.1, 0.9]
            if avg_range > 0.4 and avg_min <= 0.1 and avg_max >= 0.9:
                row['Support'] = 'Good'
            # OK support: range > 0.3 AND covers [0.2, 0.8]  
            elif avg_range > 0.3 and avg_min <= 0.2 and avg_max >= 0.8:
                row['Support'] = 'OK'
            # Weak support: everything else
            else:
                row['Support'] = 'Weak'
        
        rows.append(row)
    
    return pd.DataFrame(rows).sort_values('Pct_Near_Boundary', ascending=False)


def generate_runtime_complexity_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate Table A6: Runtime & Complexity.
    
    Shows runtime, fold counts, and computational complexity.
    
    Args:
        results: List of experiment results
        
    Returns:
        DataFrame with runtime analysis
    """
    from .paper_tables import create_config_key
    
    runtime_by_config = {}
    
    for result in results:
        config_key = create_config_key(result)
        spec = result.get("spec", {})
        
        if config_key not in runtime_by_config:
            runtime_by_config[config_key] = {
                'runtimes': [],
                'sample_sizes': [],
                'estimator': spec.get("estimator", "unknown")
            }
        
        runtime = result.get("runtime_s")
        if runtime is not None:
            runtime_by_config[config_key]['runtimes'].append(runtime)
            runtime_by_config[config_key]['sample_sizes'].append(
                spec.get("sample_size", 0)
            )
    
    # Aggregate and compute complexity
    rows = []
    for config, runtime_dict in runtime_by_config.items():
        row = {'Estimator': config}
        
        if runtime_dict['runtimes']:
            row['Runtime_Median'] = np.median(runtime_dict['runtimes'])
            row['Runtime_P90'] = np.percentile(runtime_dict['runtimes'], 90)
            
            # Estimate computational complexity
            estimator = runtime_dict['estimator']
            if estimator in ['raw-ips', 'calibrated-ips', 'orthogonalized-ips']:
                row['Complexity'] = 'O(n)'
                row['N_Folds'] = 0
            elif estimator.startswith('stacked'):
                row['Complexity'] = 'O(M*K*n)'
                row['N_Folds'] = 20
                row['M_Components'] = 5 if estimator == 'stacked-dr' else 4
            else:
                row['Complexity'] = 'O(K*n)'
                row['N_Folds'] = 20
            
            # Runtime per sample (normalized)
            if runtime_dict['sample_sizes']:
                avg_n = np.mean(runtime_dict['sample_sizes'])
                row['Runtime_per_1k'] = row['Runtime_Median'] / (avg_n / 1000)
        
        rows.append(row)
    
    return pd.DataFrame(rows).sort_values('Runtime_Median')


def format_appendix_latex(df: pd.DataFrame, table_num: str, caption: str) -> str:
    """Format appendix table as LaTeX."""
    latex = []
    latex.append(f"\\begin{{table}}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{tab:{table_num}}}")
    
    # Generate column specification based on DataFrame
    n_cols = len(df.columns)
    col_spec = "l|" + "c" * (n_cols - 1)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")
    
    # Header
    headers = [col.replace('_', ' ') for col in df.columns]
    latex.append(" & ".join(headers) + " \\\\")
    latex.append("\\midrule")
    
    # Data rows
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                cells.append("--")
            elif isinstance(val, (int, np.integer)):
                cells.append(str(val))
            elif isinstance(val, (float, np.floating)):
                if val < 0.01:
                    cells.append(f"{val:.4f}")
                elif val < 1:
                    cells.append(f"{val:.3f}")
                elif val < 100:
                    cells.append(f"{val:.1f}")
                else:
                    cells.append(f"{val:.0f}")
            else:
                cells.append(str(val))
        latex.append(" & ".join(cells) + " \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def main():
    """Generate all appendix tables."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate appendix tables")
    parser.add_argument("--results", type=Path, default=Path("results/all_experiments.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("tables/appendix/"))
    parser.add_argument("--format", choices=["dataframe", "latex", "markdown"], default="latex")
    args = parser.parse_args()
    
    # Load results
    results = []
    with open(args.results) as f:
        for line in f:
            results.append(json.loads(line))
    
    # Create output directory
    args.output.mkdir(exist_ok=True, parents=True)
    
    # Generate tables
    tables = [
        ("A1", "Quadrant Leaderboard", generate_quadrant_leaderboard),
        ("A2", "Bias Patterns", generate_bias_patterns_table),
        ("A3", "Overlap & Tail Diagnostics", generate_overlap_diagnostics_table),
        ("A4", "Oracle Adjustment Share", generate_oracle_adjustment_table),
        ("A5", "Calibration Boundary Analysis", generate_boundary_outlier_table),
        ("A6", "Runtime & Complexity", generate_runtime_complexity_table),
    ]
    
    for table_num, caption, generator in tables:
        print(f"Generating Table {table_num}: {caption}...")
        df = generator(results)
        
        if args.format == "latex":
            latex = format_appendix_latex(df, table_num, caption)
            (args.output / f"table{table_num}.tex").write_text(latex)
        elif args.format == "markdown":
            (args.output / f"table{table_num}.md").write_text(df.to_markdown())
        else:
            df.to_csv(args.output / f"table{table_num}.csv")
    
    print(f"Appendix tables written to {args.output}/")


if __name__ == "__main__":
    main()