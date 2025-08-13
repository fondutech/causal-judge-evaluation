"""
Display and formatting utilities for diagnostics.
"""

from typing import Dict, Any
import numpy as np


def create_weight_summary_table(all_diagnostics: Dict[str, Any]) -> str:
    """Create a formatted table of weight diagnostics.
    
    Args:
        all_diagnostics: Dictionary of WeightDiagnostics objects by policy
    
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("\nWeight Summary")
    lines.append("-" * 70)
    lines.append(f"{'Policy':<30} {'ESS':>8} {'Mean Weight':>12} {'Status':<10}")
    lines.append("-" * 70)
    
    for policy, diag in all_diagnostics.items():
        # Handle both WeightDiagnostics objects and dicts
        if hasattr(diag, 'ess_fraction'):
            ess = diag.ess_fraction
            mean_w = diag.mean_weight
            status = diag.consistency_flag
        else:
            # Assume it's a dict
            ess = diag.get('ess_fraction', 0.0)
            mean_w = diag.get('mean_weight', 1.0)
            status = diag.get('consistency_flag', 'UNKNOWN')
        
        lines.append(
            f"{policy:<30} {ess:>7.1%} {mean_w:>12.4f} {status:<10}"
        )
    
    return "\n".join(lines)


def format_dr_diagnostic_summary(diagnostics: Dict[str, Any]) -> str:
    """Format DR diagnostics as a readable summary table.
    
    Args:
        diagnostics: Dictionary of DR diagnostics by policy
    
    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 100)
    lines.append("DR DIAGNOSTICS SUMMARY")
    lines.append("=" * 100)
    
    # Header
    lines.append(
        f"{'Policy':<20} {'DM':>7} {'IPS':>7} {'DR±SE':<20} "
        f"{'Score(mean±se, p)':<25} {'RMSE(R,g)':>10} {'|IF| tail(p99/p5)':>17}"
    )
    lines.append("-" * 100)
    
    # Per-policy rows
    worst_if_tail = 0.0
    r2_values = []
    max_score_z = 0.0
    
    for policy, diag in diagnostics.items():
        if isinstance(diag, dict):
            dm = diag.get("dm_mean", 0.0)
            ips = diag.get("ips_corr_mean", 0.0)
            dr = diag.get("dr_estimate", 0.0)
            
            # Standard error (from influence functions if available)
            if "if_std" in diag and "n_samples" in diag:
                se = diag["if_std"] / np.sqrt(diag["n_samples"])
            else:
                se = 0.0
            
            # Score test (for TMLE)
            score_mean = diag.get("score_mean", 0.0)
            score_se = diag.get("score_se", 0.0)
            score_p = diag.get("score_p", 1.0)
            score_str = f"{score_mean:>7.3f}±{score_se:.3f} (p={score_p:.2f})"
            
            # Outcome model RMSE
            rmse = diag.get("residual_rmse", np.nan)
            
            # IF tail ratio
            if_tail = diag.get("if_tail_ratio_99_5", 0.0)
            
            lines.append(
                f"{policy:<20} {dm:>7.3f} {ips:>7.3f} {dr:>7.3f}±{se:.3f}  "
                f"{score_str:<25} {rmse:>10.3f} {if_tail:>17.1f}"
            )
            
            # Track worst metrics
            worst_if_tail = max(worst_if_tail, if_tail)
            if "r2_oof" in diag and not np.isnan(diag["r2_oof"]):
                r2_values.append(diag["r2_oof"])
            if "score_z" in diag:
                max_score_z = max(max_score_z, abs(diag["score_z"]))
    
    lines.append("-" * 100)
    
    # Summary statistics
    lines.append(f"Worst IF tail ratio (p99/p5): {worst_if_tail:.1f}")
    if r2_values:
        lines.append(f"R² OOF range: [{min(r2_values):.3f}, {max(r2_values):.3f}]")
    
    # TMLE-specific
    if "tmle_max_score_z" in diagnostics:
        lines.append(f"TMLE max |score z|: {diagnostics['tmle_max_score_z']:.2f} (should be ~0)")
    elif max_score_z > 0:
        lines.append(f"TMLE max |score z|: {max_score_z:.2f} (should be ~0)")
    
    lines.append("=" * 100)
    
    # Warnings
    if worst_if_tail > 100:
        lines.append("\n⚠️  Warning: Heavy-tailed influence functions detected")
        lines.append("   Consider using more fresh draws or checking policy overlap")
    
    return "\n".join(lines)