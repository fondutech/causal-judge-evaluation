"""Simple visualization utilities for weight diagnostics."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from pathlib import Path


def plot_weight_distributions(
    weights_dict: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Plot weight distributions for multiple policies.
    
    Args:
        weights_dict: Dict mapping policy names to weight arrays
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    n_policies = len(weights_dict)
    fig, axes = plt.subplots(n_policies, 2, figsize=figsize)
    
    # Handle single policy case
    if n_policies == 1:
        axes = axes.reshape(1, -1)
    
    for i, (policy_name, weights) in enumerate(weights_dict.items()):
        # Left: Weight histogram (log scale)
        ax_hist = axes[i, 0]
        
        # Filter out non-finite weights
        finite_weights = weights[np.isfinite(weights) & (weights > 0)]
        if len(finite_weights) > 0:
            log_weights = np.log10(finite_weights)
            ax_hist.hist(log_weights, bins=30, alpha=0.7, edgecolor='black')
            ax_hist.axvline(0, color='red', linestyle='--', alpha=0.5, label='Weight=1')
            ax_hist.set_xlabel('Log₁₀(Weight)')
            ax_hist.set_ylabel('Count')
            ax_hist.legend()
        else:
            ax_hist.text(0.5, 0.5, 'No finite weights', ha='center', va='center')
        
        ax_hist.set_title(f'{policy_name} - Weight Distribution')
        
        # Right: Weight vs index scatter
        ax_scatter = axes[i, 1]
        
        # Plot all weights (including zeros)
        indices = np.arange(len(weights))
        ax_scatter.scatter(indices, weights, alpha=0.5, s=10)
        ax_scatter.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Weight=1')
        ax_scatter.set_xlabel('Sample Index')
        ax_scatter.set_ylabel('Weight')
        ax_scatter.set_yscale('log')
        ax_scatter.set_ylim(bottom=1e-10)
        ax_scatter.legend()
        
        # Add statistics
        ess = compute_ess(weights)
        ess_pct = ess / len(weights) * 100
        ax_scatter.set_title(f'ESS: {ess_pct:.1f}% | Mean: {weights.mean():.3f}')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_ess_comparison(
    weights_dict: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot ESS comparison across policies.
    
    Args:
        weights_dict: Dict mapping policy names to weight arrays
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    # Calculate ESS for each policy
    policy_names = list(weights_dict.keys())
    ess_values = []
    ess_percentages = []
    
    for weights in weights_dict.values():
        ess = compute_ess(weights)
        ess_values.append(ess)
        ess_percentages.append(ess / len(weights) * 100)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(policy_names, ess_percentages, alpha=0.7, edgecolor='black')
    
    # Add reference lines
    ax.axhline(10, color='orange', linestyle='--', alpha=0.7, label='10% ESS')
    ax.axhline(50, color='green', linestyle='--', alpha=0.7, label='50% ESS')
    
    # Labels
    ax.set_ylabel('Effective Sample Size (%)')
    ax.set_title('ESS Comparison Across Policies')
    ax.set_ylim(0, max(100, max(ess_percentages) * 1.1))
    
    # Add value labels on bars
    for bar, ess_pct, ess_val in zip(bars, ess_percentages, ess_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{ess_pct:.1f}%\n({ess_val:.0f})',
                ha='center', va='bottom')
    
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_weight_summary(
    weights_dict: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Create a summary plot with multiple weight diagnostics.
    
    Args:
        weights_dict: Dict mapping policy names to weight arrays
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. ESS comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    policy_names = list(weights_dict.keys())
    ess_percentages = [compute_ess(w) / len(w) * 100 for w in weights_dict.values()]
    
    ax1.bar(policy_names, ess_percentages, alpha=0.7, edgecolor='black')
    ax1.axhline(10, color='orange', linestyle='--', alpha=0.7)
    ax1.axhline(50, color='green', linestyle='--', alpha=0.7)
    ax1.set_ylabel('ESS (%)')
    ax1.set_title('Effective Sample Size by Policy')
    ax1.set_ylim(0, max(100, max(ess_percentages) * 1.1))
    
    # 2. Weight ranges (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (name, weights) in enumerate(weights_dict.items()):
        finite_w = weights[np.isfinite(weights) & (weights > 0)]
        if len(finite_w) > 0:
            # Box plot of log weights
            ax2.boxplot(np.log10(finite_w), positions=[i], widths=0.6,
                       showfliers=False, patch_artist=True,
                       boxprops=dict(alpha=0.7))
    
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5, label='Weight=1')
    ax2.set_xticks(range(len(policy_names)))
    ax2.set_xticklabels(policy_names, rotation=45, ha='right')
    ax2.set_ylabel('Log₁₀(Weight)')
    ax2.set_title('Weight Ranges (Box Plot)')
    ax2.legend()
    
    # 3. Extreme weights table (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    table_data = []
    headers = ['Policy', 'Min', 'Max', 'Zero', '>100']
    
    for name, weights in weights_dict.items():
        finite_w = weights[np.isfinite(weights)]
        min_w = finite_w.min() if len(finite_w) > 0 else np.nan
        max_w = finite_w.max() if len(finite_w) > 0 else np.nan
        n_zero = np.sum(weights == 0)
        n_large = np.sum(weights > 100)
        table_data.append([name, f'{min_w:.2e}', f'{max_w:.2e}', 
                          str(n_zero), str(n_large)])
    
    table = ax3.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax3.set_title('Extreme Weight Statistics')
    
    # 4. Mean weight comparison (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    mean_weights = [w.mean() for w in weights_dict.values()]
    bars = ax4.bar(policy_names, mean_weights, alpha=0.7, edgecolor='black')
    ax4.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Expected=1.0')
    ax4.set_ylabel('Mean Weight')
    ax4.set_title('Mean Weight by Policy')
    ax4.legend()
    
    # Add value labels
    for bar, mean_w in zip(bars, mean_weights):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean_w:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.suptitle('Weight Diagnostics Summary', fontsize=14)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_calibration_comparison(
    judge_scores: np.ndarray,
    oracle_labels: np.ndarray,
    calibrated_scores: Optional[np.ndarray] = None,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot calibration comparison (reliability diagram).
    
    Args:
        judge_scores: Raw judge scores
        oracle_labels: True oracle labels
        calibrated_scores: Calibrated judge scores (optional)
        n_bins: Number of bins for grouping
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Bin the scores
    bins = np.linspace(0, 1, n_bins + 1)
    
    # Plot raw scores
    bin_indices = np.digitize(judge_scores, bins) - 1
    mean_pred_raw = []
    mean_true_raw = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_pred_raw.append(judge_scores[mask].mean())
            mean_true_raw.append(oracle_labels[mask].mean())
    
    ax.scatter(mean_pred_raw, mean_true_raw, s=100, alpha=0.7, 
               label='Raw Judge Scores', edgecolor='black')
    
    # Plot calibrated scores if provided
    if calibrated_scores is not None:
        bin_indices_cal = np.digitize(calibrated_scores, bins) - 1
        mean_pred_cal = []
        mean_true_cal = []
        
        for i in range(n_bins):
            mask = bin_indices_cal == i
            if mask.sum() > 0:
                mean_pred_cal.append(calibrated_scores[mask].mean())
                mean_true_cal.append(oracle_labels[mask].mean())
        
        ax.scatter(mean_pred_cal, mean_true_cal, s=100, alpha=0.7,
                   label='Calibrated Scores', marker='s', edgecolor='black')
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    
    # Labels
    ax.set_xlabel('Mean Predicted Score')
    ax.set_ylabel('Mean Oracle Label')
    ax.set_title('Calibration Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compute_ess(weights: np.ndarray) -> float:
    """Compute Effective Sample Size from importance weights.
    
    ESS = (sum w)^2 / sum(w^2)
    
    Args:
        weights: Importance weights
        
    Returns:
        Effective sample size
    """
    finite_weights = weights[np.isfinite(weights) & (weights > 0)]
    if len(finite_weights) == 0:
        return 0.0
    
    sum_w = np.sum(finite_weights)
    sum_w2 = np.sum(finite_weights ** 2)
    
    if sum_w2 == 0:
        return 0.0
    
    return float((sum_w ** 2) / sum_w2)