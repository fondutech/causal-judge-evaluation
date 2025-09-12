#!/usr/bin/env python3
"""Generate quadrant-specific leaderboards."""

import json
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
from pathlib import Path

# Add reporting to path
sys.path.append('reporting')
from paper_tables import compute_debiased_rmse, compute_interval_score_oa, compute_calibration_score, compute_se_geomean, compute_ranking_metrics, create_config_key, compute_aggregate_score

def generate_quadrant_leaderboards(results_file: str = "results/all_experiments.jsonl"):
    """Generate leaderboard for each quadrant."""
    
    # Load results
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    # Define quadrants based on actual data
    quadrants = {
        'Small-Low': {'size': [500, 1000], 'coverage': [0.05, 0.10]},
        'Small-High': {'size': [500, 1000], 'coverage': [0.25, 0.50, 1.00]},
        'Large-Low': {'size': [2500, 5000], 'coverage': [0.05, 0.10]},
        'Large-High': {'size': [2500, 5000], 'coverage': [0.25, 0.50, 1.00]}
    }
    
    all_quadrant_tables = []
    
    # Process each quadrant
    for quad_name, criteria in quadrants.items():
        # Filter results for this quadrant
        quad_results = [r for r in results if 
                        r['spec']['sample_size'] in criteria['size'] and 
                        r['spec']['oracle_coverage'] in criteria['coverage']]
        
        if not quad_results:
            continue
        
        # Compute metrics ONLY for this quadrant's results
        rmse_d = compute_debiased_rmse(quad_results)
        interval_scores = compute_interval_score_oa(quad_results)
        calib_scores = compute_calibration_score(quad_results)
        se_geomeans = compute_se_geomean(quad_results)
        ranking_metrics = compute_ranking_metrics(quad_results)
        
        # Create rows for this quadrant's estimators
        rows = []
        configs_in_quad = set()
        
        for result in quad_results:
            config = create_config_key(result)
            configs_in_quad.add(config)
        
        for config in configs_in_quad:
            row = {
                'Estimator': config,
                'RMSE_d': rmse_d.get(config, np.nan),
                'IntervalScore_OA': interval_scores.get(config, np.nan),
                'CalibScore': calib_scores.get(config, np.nan) * 100 if config in calib_scores else np.nan,
                'SE_GeoMean': se_geomeans.get(config, np.nan),
                'Kendall_tau': ranking_metrics.get(config, {}).get('kendall_tau', np.nan),
                'Top1_Acc': ranking_metrics.get(config, {}).get('top1_accuracy', np.nan)
            }
            rows.append(row)
        
        if not rows:
            continue
            
        df = pd.DataFrame(rows)
        
        # Compute aggregate scores with quadrant-specific normalization
        normalize_bounds = {
            'RMSE_d': (df['RMSE_d'].min(), df['RMSE_d'].max()),
            'IntervalScore_OA': (df['IntervalScore_OA'].min(), df['IntervalScore_OA'].max()),
            'CalibScore': (df['CalibScore'].min(), df['CalibScore'].max()),
            'SE_GeoMean': (df['SE_GeoMean'].min(), df['SE_GeoMean'].max()),
            'Kendall_tau': (df['Kendall_tau'].min(), df['Kendall_tau'].max()),
            'Top1_Acc': (df['Top1_Acc'].min(), df['Top1_Acc'].max())
        }
        
        df['AggScore'] = df.apply(lambda row: compute_aggregate_score(row, normalize_bounds), axis=1)
        df = df.sort_values('AggScore', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        
        # Store quadrant info
        size_label = "Small" if 500 in criteria['size'] or 1000 in criteria['size'] else "Large"
        cov_label = "Low" if 0.05 in criteria['coverage'] or 0.10 in criteria['coverage'] else "High"
        
        # Format for display
        print(f'\n## {quad_name} Quadrant Leaderboard')
        print(f'Size: {size_label} ({criteria["size"]}), Coverage: {cov_label} ({criteria["coverage"]})')
        print(f'Total experiments in quadrant: {len(quad_results)}')
        print()
        print('| Rank | Estimator | Score | RMSE^d | IS^OA | CalibScore | SE_GM | K-tau | Top-1 |')
        print('|---:|:---|---:|---:|---:|---:|---:|---:|---:|')
        
        for _, row in df.iterrows():
            est_short = row['Estimator'][:35]
            # Handle NaN values for display
            rmse = f"{row['RMSE_d']:.4f}" if not pd.isna(row['RMSE_d']) else "—"
            is_oa = f"{row['IntervalScore_OA']:.4f}" if not pd.isna(row['IntervalScore_OA']) else "—"
            calib = f"{row['CalibScore']:.1f}" if not pd.isna(row['CalibScore']) else "—"
            se_gm = f"{row['SE_GeoMean']:.4f}" if not pd.isna(row['SE_GeoMean']) else "—"
            ktau = f"{row['Kendall_tau']:.3f}" if not pd.isna(row['Kendall_tau']) else "—"
            top1 = f"{row['Top1_Acc']:.0f}" if not pd.isna(row['Top1_Acc']) else "—"
            
            print(f"| {row['Rank']} | {est_short} | {row['AggScore']:.1f} | {rmse} | {is_oa} | {calib} | {se_gm} | {ktau} | {top1} |")
        
        # Save to file
        all_quadrant_tables.append({
            'quadrant': quad_name,
            'dataframe': df
        })
    
    # Also create a combined markdown file
    output_dir = Path("tables/quadrant")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    with open(output_dir / "quadrant_leaderboards.md", "w") as f:
        f.write("# Quadrant-Specific Leaderboards\n\n")
        
        for table_info in all_quadrant_tables:
            quad_name = table_info['quadrant']
            df = table_info['dataframe']
            
            f.write(f"## {quad_name} Quadrant\n\n")
            
            # Save the markdown table
            df_display = df[['Rank', 'Estimator', 'AggScore', 'RMSE_d', 'IntervalScore_OA', 
                           'CalibScore', 'SE_GeoMean', 'Kendall_tau', 'Top1_Acc']].copy()
            
            # Round for display
            df_display['AggScore'] = df_display['AggScore'].round(1)
            df_display['RMSE_d'] = df_display['RMSE_d'].round(4)
            df_display['IntervalScore_OA'] = df_display['IntervalScore_OA'].round(4)
            df_display['CalibScore'] = df_display['CalibScore'].round(1)
            df_display['SE_GeoMean'] = df_display['SE_GeoMean'].round(4)
            df_display['Kendall_tau'] = df_display['Kendall_tau'].round(3)
            df_display['Top1_Acc'] = df_display['Top1_Acc'].round(0)
            
            f.write(df_display.to_markdown(index=False))
            f.write("\n\n")
    
    print(f"\nQuadrant leaderboards saved to {output_dir}/")

if __name__ == "__main__":
    generate_quadrant_leaderboards()