#!/usr/bin/env python3
"""
Analyze importance weights for Arena 10K data.
Uses built-in weight diagnostics to identify potential issues.
"""

import sys
from pathlib import Path
import json
import numpy as np
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.loggers import PrecomputedSampler
from cje.utils.weight_diagnostics import (
    diagnose_weights_with_overlap,
    create_weight_summary_table,
    format_weight_diagnostics,
    format_overlap_diagnostics,
    compute_overlap_diagnostics,
)
from cje.utils.importance_weights import compute_weight_statistics

console = Console()


def plot_weight_distribution(weights_dict, output_file="weight_distributions.png"):
    """Create visualization of weight distributions."""
    n_policies = len(weights_dict)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (policy, weights) in enumerate(weights_dict.items()):
        ax = axes[idx]

        # Filter out NaN values
        valid_weights = weights[~np.isnan(weights)]

        if len(valid_weights) > 0:
            # Histogram on log scale
            log_weights = np.log10(valid_weights + 1e-10)
            ax.hist(log_weights, bins=30, alpha=0.7, edgecolor="black")
            ax.axvline(x=0, color="red", linestyle="--", label="Weight=1")

            # Add statistics
            ax.text(
                0.02,
                0.98,
                f"Mean: {np.mean(valid_weights):.2f}\n"
                f"Median: {np.median(valid_weights):.2f}\n"
                f"Max: {np.max(valid_weights):.2f}\n"
                f"ESS%: {100 * (np.sum(valid_weights)**2) / np.sum(valid_weights**2) / len(weights):.1f}%",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            ax.set_xlabel("Log10(Weight)")
            ax.set_ylabel("Count")
            ax.set_title(f"{policy} Weight Distribution")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No valid weights", transform=ax.transAxes, ha="center")
            ax.set_title(f"{policy} - No Data")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    console.print(f"\n📊 Saved weight distribution plot to {output_file}")


def main():
    """Analyze importance weights with detailed diagnostics."""
    console.print("[bold cyan]Importance Weight Analysis for Arena 10K[/bold cyan]")
    console.print("=" * 60)

    # Load P0 data
    p0_file = "../data/p0_scored_deterministic.jsonl"

    if not Path(p0_file).exists():
        console.print(f"[red]Error: P0 file not found: {p0_file}[/red]")
        return 1

    console.print("Loading data...")
    p0_data = []
    with open(p0_file) as f:
        for line in f:
            p0_data.append(json.loads(line))

    console.print(f"Loaded {len(p0_data)} samples")

    # Extract target policies
    if p0_data and "target_logps" in p0_data[0]:
        target_policies = list(p0_data[0]["target_logps"].keys())
    else:
        console.print("[red]Error: No target_logps found in data[/red]")
        return 1

    # Create PrecomputedSampler
    sampler = PrecomputedSampler(data=p0_data, target_policies=target_policies)

    # Extract contexts and responses
    contexts = [item["prompt"] for item in p0_data]
    responses = [item["response"] for item in p0_data]

    # Compute importance weights
    console.print("\n[bold]Computing importance weights...[/bold]")
    weights_matrix, weight_stats = sampler.importance_weights_matrix(
        contexts, responses, show_progress=False
    )

    # Use built-in weight statistics
    console.print("\n[bold]Weight Statistics (from compute_weight_statistics):[/bold]")
    detailed_stats = compute_weight_statistics(weights_matrix, target_policies)

    # Display overall statistics
    console.print(f"\nTotal samples: {detailed_stats['n_samples']}")
    console.print(f"Number of policies: {detailed_stats['n_policies']}")

    # Create summary table
    table = Table(title="Policy Weight Statistics")
    table.add_column("Policy", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("ESS%", justify="right", style="magenta")
    table.add_column("Valid", justify="right")

    for policy in target_policies:
        stats = detailed_stats["policy_stats"][policy]
        table.add_row(
            policy,
            f"{stats['mean']:.3f}",
            f"{stats['std']:.3f}",
            f"{stats['min']:.3f}",
            f"{stats['max']:.3f}",
            f"{stats['ess_percentage']:.1f}%",
            f"{stats['n_valid']}/{detailed_stats['n_samples']}",
        )

    console.print(table)

    # Run detailed diagnostics for each policy
    console.print("\n[bold]Detailed Weight Diagnostics:[/bold]")

    all_diagnostics = {}
    weights_for_plot = {}

    for policy_idx, policy in enumerate(target_policies):
        # Extract log probabilities
        behavior_logprobs = [item["total_logprob"] for item in p0_data]
        target_logprobs = [item["target_logps"][policy] for item in p0_data]

        # Get weights for this policy
        policy_weights = weights_matrix[:, policy_idx]
        weights_for_plot[policy] = policy_weights

        # Diagnose weights with overlap
        diagnostics = diagnose_weights_with_overlap(
            weights=policy_weights.tolist(),
            behavior_logprobs=behavior_logprobs,
            target_logprobs=target_logprobs,
            policy_name=policy,
            expected_weight=1.0 if policy == "pi_clone" else None,
        )

        all_diagnostics[policy] = diagnostics

        # Print detailed diagnostics
        console.print(f"\n{format_weight_diagnostics(diagnostics)}")

        # Compute and display overlap diagnostics
        overlap_diag = compute_overlap_diagnostics(behavior_logprobs, target_logprobs)
        console.print(format_overlap_diagnostics(overlap_diag, policy))

    # Create summary table
    console.print("\n" + create_weight_summary_table(all_diagnostics))

    # Identify problematic samples
    console.print("\n[bold]Sample-Level Analysis:[/bold]")

    # Find samples with extreme weights
    extreme_samples = []
    for i in range(len(p0_data)):
        sample_info = {
            "index": i,
            "prompt_preview": p0_data[i]["prompt"][:50] + "...",
            "extreme_weights": {},
        }

        for policy_idx, policy in enumerate(target_policies):
            weight = weights_matrix[i, policy_idx]
            if not np.isnan(weight) and (weight > 100 or weight < 0.01):
                sample_info["extreme_weights"][policy] = weight

        if sample_info["extreme_weights"]:
            extreme_samples.append(sample_info)

    if extreme_samples:
        console.print(f"\nFound {len(extreme_samples)} samples with extreme weights:")
        for sample in extreme_samples[:5]:  # Show first 5
            console.print(f"\nSample {sample['index']}: {sample['prompt_preview']}")
            for policy, weight in sample["extreme_weights"].items():
                console.print(f"  {policy}: {weight:.2e}")

    # Plot weight distributions
    plot_weight_distribution(weights_for_plot)

    # Recommendations
    console.print("\n[bold]Recommendations:[/bold]")

    critical_policies = [
        p for p, d in all_diagnostics.items() if d.consistency_flag == "CRITICAL"
    ]
    warning_policies = [
        p for p, d in all_diagnostics.items() if d.consistency_flag == "WARNING"
    ]

    if critical_policies:
        console.print(f"\n❌ Critical issues found for: {', '.join(critical_policies)}")
        console.print("   - Consider using SNIPS or weight clipping")
        console.print("   - These policies have very poor overlap with P0")

    if warning_policies:
        console.print(f"\n⚠️  Warnings for: {', '.join(warning_policies)}")
        console.print("   - Monitor estimates carefully")
        console.print("   - Consider increasing sample size")

    # ESS analysis
    min_ess_policy = min(all_diagnostics.items(), key=lambda x: x[1].ess_fraction)
    console.print(
        f"\n📊 Lowest ESS: {min_ess_policy[0]} with {min_ess_policy[1].ess_fraction:.1%}"
    )
    console.print(
        f"   Effective samples: {min_ess_policy[1].ess_fraction * len(p0_data):.0f} out of {len(p0_data)}"
    )

    console.print("\n✅ Weight analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
