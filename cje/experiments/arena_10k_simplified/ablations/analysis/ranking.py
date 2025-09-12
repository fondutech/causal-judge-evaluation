"""Policy ranking analysis for ablation experiments."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy.stats import kendalltau, spearmanr

from .constants import POLICIES, WELL_BEHAVED_POLICIES
from .constants import POLICIES, WELL_BEHAVED_POLICIES
WELL_BEHAVED_POLICIES = ["clone", "parallel_universe_prompt", "premium"]


def compute_ranking_metrics(results: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """Compute ranking metrics (Kendall tau, top-1 accuracy, etc.).
    
    Args:
        results: List of experiment results
        
    Returns:
        Dictionary with per-experiment and aggregated ranking metrics
    """
    per_experiment_rows = []
    
    for result in results:
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})
        spec = result.get("spec", {})
        
        # Skip if missing data
        if not estimates or not oracle_truths:
            continue
        
        # Get values for well-behaved policies only
        est_values = []
        true_values = []
        for policy in WELL_BEHAVED_POLICIES:
            if policy in estimates and policy in oracle_truths:
                est_values.append(estimates[policy])
                true_values.append(oracle_truths[policy])
        
        if len(est_values) < 2:
            continue
        
        # Compute ranking metrics
        row = {
            "estimator": spec.get("estimator"),
            "sample_size": spec.get("sample_size"),
            "oracle_coverage": spec.get("oracle_coverage"),
            "quadrant": result.get("quadrant", "Unknown"),
            "seed": spec.get("seed_base", 0)
        }
        
        # Kendall's tau
        tau, _ = kendalltau(est_values, true_values)
        row["kendall_tau"] = tau
        
        # Spearman's rho
        rho, _ = spearmanr(est_values, true_values)
        row["spearman_rho"] = rho
        
        # Top-1 accuracy (did we get the best policy right?)
        best_est_idx = np.argmax(est_values)
        best_true_idx = np.argmax(true_values)
        row["top1_correct"] = best_est_idx == best_true_idx
        
        # Top-2 accuracy (are the top 2 policies correct, any order?)
        if len(est_values) >= 3:
            top2_est = set(np.argsort(est_values)[-2:])
            top2_true = set(np.argsort(true_values)[-2:])
            row["top2_correct"] = top2_est == top2_true
        else:
            row["top2_correct"] = np.nan
        
        # Ranking error (sum of position differences)
        est_ranks = np.argsort(np.argsort(est_values))
        true_ranks = np.argsort(np.argsort(true_values))
        row["rank_error"] = np.sum(np.abs(est_ranks - true_ranks))
        
        per_experiment_rows.append(row)
    
    per_experiment_df = pd.DataFrame(per_experiment_rows)
    
    # Aggregate by estimator configuration
    aggregated_rows = []
    
    if not per_experiment_df.empty:
        grouped = per_experiment_df.groupby("estimator")
        
        for estimator, group in grouped:
            agg_row = {
                "estimator": estimator,
                "n_experiments": len(group),
                "mean_kendall_tau": group["kendall_tau"].mean(),
                "std_kendall_tau": group["kendall_tau"].std(),
                "mean_spearman_rho": group["spearman_rho"].mean(),
                "std_spearman_rho": group["spearman_rho"].std(),
                "top1_accuracy": group["top1_correct"].mean() * 100,
                "mean_rank_error": group["rank_error"].mean()
            }
            
            if "top2_correct" in group.columns:
                agg_row["top2_accuracy"] = group["top2_correct"].mean() * 100
            
            aggregated_rows.append(agg_row)
    
    aggregated_df = pd.DataFrame(aggregated_rows)
    
    if not aggregated_df.empty:
        aggregated_df = aggregated_df.sort_values("mean_kendall_tau", ascending=False)
    
    return {
        "per_experiment": per_experiment_df,
        "aggregated": aggregated_df
    }


def compute_pairwise_preferences(
    results: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Compute pairwise preference accuracy between policies.
    
    Args:
        results: List of experiment results
        
    Returns:
        DataFrame with pairwise preference accuracies
    """
    # Track pairwise comparisons
    pairwise_correct: Dict[Tuple[str, str, str], int] = {}
    pairwise_total: Dict[Tuple[str, str, str], int] = {}
    
    for result in results:
        estimator = result.get("spec", {}).get("estimator")
        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})
        
        if not estimator or not estimates or not oracle_truths:
            continue
        
        # Check each pair of policies
        for i, policy1 in enumerate(WELL_BEHAVED_POLICIES):
            for policy2 in WELL_BEHAVED_POLICIES[i+1:]:
                if policy1 in estimates and policy2 in estimates and \
                   policy1 in oracle_truths and policy2 in oracle_truths:
                    
                    key = (estimator, policy1, policy2)
                    
                    if key not in pairwise_total:
                        pairwise_total[key] = 0
                        pairwise_correct[key] = 0
                    
                    pairwise_total[key] += 1
                    
                    # Check if ordering is correct
                    est_order = estimates[policy1] > estimates[policy2]
                    true_order = oracle_truths[policy1] > oracle_truths[policy2]
                    
                    if est_order == true_order:
                        pairwise_correct[key] += 1
    
    # Create summary table
    rows = []
    for key in pairwise_total:
        estimator, policy1, policy2 = key
        accuracy = 100.0 * pairwise_correct[key] / pairwise_total[key]
        
        rows.append({
            "estimator": estimator,
            "comparison": f"{policy1} vs {policy2}",
            "accuracy_pct": accuracy,
            "n_comparisons": pairwise_total[key]
        })
    
    df = pd.DataFrame(rows)
    
    if not df.empty:
        # Pivot for better presentation
        df = df.pivot_table(
            index="estimator",
            columns="comparison",
            values="accuracy_pct",
            aggfunc="mean"
        )
        df["mean_accuracy"] = df.mean(axis=1)
        df = df.sort_values("mean_accuracy", ascending=False)
    
    return df


def compute_ranking_by_quadrant(
    results: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Compute ranking metrics by data regime quadrant.
    
    Args:
        results: List of experiment results
        
    Returns:
        DataFrame with ranking metrics by estimator and quadrant
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
    
    # Compute metrics for each group
    for (estimator, quadrant), group_results in grouped_data.items():
        kendall_taus = []
        top1_correct = []
        
        for result in group_results:
            estimates = result.get("estimates", {})
            oracle_truths = result.get("oracle_truths", {})
            
            # Get values for well-behaved policies
            est_values = []
            true_values = []
            for policy in WELL_BEHAVED_POLICIES:
                if policy in estimates and policy in oracle_truths:
                    est_values.append(estimates[policy])
                    true_values.append(oracle_truths[policy])
            
            if len(est_values) >= 2:
                tau, _ = kendalltau(est_values, true_values)
                kendall_taus.append(tau)
                
                if len(est_values) >= 3:
                    best_est = np.argmax(est_values)
                    best_true = np.argmax(true_values)
                    top1_correct.append(best_est == best_true)
        
        if kendall_taus:
            row = {
                "estimator": estimator,
                "quadrant": quadrant,
                "n_experiments": len(group_results),
                "mean_kendall_tau": np.mean(kendall_taus),
                "std_kendall_tau": np.std(kendall_taus)
            }
            
            if top1_correct:
                row["top1_accuracy"] = np.mean(top1_correct) * 100
            
            rows.append(row)
    
    return pd.DataFrame(rows).sort_values(["estimator", "quadrant"])