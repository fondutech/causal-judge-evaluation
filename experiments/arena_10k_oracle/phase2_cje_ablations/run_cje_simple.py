#!/usr/bin/env python3
"""
Simple CJE runner for Arena 10K data.
Uses precomputed log probabilities to estimate policy values.
"""

import sys
from pathlib import Path
import json
import numpy as np
from rich.console import Console
from rich.table import Table

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.loggers import PrecomputedSampler
from cje.estimators import (
    MultiIPSEstimator,
    MultiSNIPSEstimator,
    CalibratedIPSEstimator,
)

console = Console()


def main():
    """Run CJE estimators on Arena 10K data."""
    console.print("[bold cyan]CJE Analysis on Arena 10K Data[/bold cyan]")
    console.print("=" * 50)

    # Load P0 data (contains log probabilities)
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

    console.print(f"Target policies: {target_policies}")

    # Create PrecomputedSampler
    sampler = PrecomputedSampler(data=p0_data, target_policies=target_policies)

    # Extract data for fitting
    contexts = [item["prompt"] for item in p0_data]
    responses = [item["response"] for item in p0_data]
    judge_scores = [item["judge_score"] for item in p0_data]

    # Compute importance weights
    console.print("\nComputing importance weights...")
    weights_matrix, weight_stats = sampler.importance_weights_matrix(
        contexts, responses, show_progress=False
    )

    # Display weight statistics
    console.print("\n[bold]Weight Statistics:[/bold]")
    for policy_idx, policy in enumerate(target_policies):
        policy_weights = weights_matrix[:, policy_idx]
        valid_weights = policy_weights[~np.isnan(policy_weights)]
        if len(valid_weights) > 0:
            console.print(f"  {policy}:")
            console.print(f"    Mean weight: {np.mean(valid_weights):.3f}")
            console.print(f"    Max weight: {np.max(valid_weights):.3f}")
            console.print(f"    Min weight: {np.min(valid_weights):.3f}")

    # Run estimators
    results = {}

    # IPS
    console.print("\n[bold]Running IPS estimator...[/bold]")
    ips = MultiIPSEstimator(sampler)
    ips_estimates = {}
    for policy_idx, policy in enumerate(target_policies):
        policy_weights = weights_matrix[:, policy_idx]
        valid_mask = ~np.isnan(policy_weights)
        if np.any(valid_mask):
            weighted_rewards = (
                policy_weights[valid_mask] * np.array(judge_scores)[valid_mask]
            )
            ips_estimates[policy] = {
                "mean": np.mean(weighted_rewards),
                "n": np.sum(valid_mask),
            }
    results["IPS"] = ips_estimates

    # SNIPS
    console.print("\n[bold]Running SNIPS estimator...[/bold]")
    snips = MultiSNIPSEstimator(sampler)
    snips_estimates = {}
    for policy_idx, policy in enumerate(target_policies):
        policy_weights = weights_matrix[:, policy_idx]
        valid_mask = ~np.isnan(policy_weights)
        if np.any(valid_mask):
            valid_weights = policy_weights[valid_mask]
            valid_scores = np.array(judge_scores)[valid_mask]
            # Self-normalized IPS
            snips_estimates[policy] = {
                "mean": np.sum(valid_weights * valid_scores) / np.sum(valid_weights),
                "n": np.sum(valid_mask),
            }
    results["SNIPS"] = snips_estimates

    # Display results
    console.print("\n[bold green]Results Summary[/bold green]")

    table = Table(title="Policy Value Estimates")
    table.add_column("Estimator", style="cyan")
    for policy in target_policies:
        table.add_column(policy, justify="right")

    for estimator_name, estimates in results.items():
        row = [estimator_name]
        for policy in target_policies:
            if policy in estimates:
                value = estimates[policy]["mean"]
                row.append(f"{value:.3f}")
            else:
                row.append("N/A")
        table.add_row(*row)

    console.print(table)

    # Load oracle labels if available
    oracle_file = "../data/labeling/oracle_labels_validation_detailed.jsonl"
    if Path(oracle_file).exists():
        console.print("\n[bold]Oracle Ground Truth[/bold]")
        oracle_scores = {}
        with open(oracle_file) as f:
            for line in f:
                item = json.loads(line)
                policy = item["policy"]
                if policy not in oracle_scores:
                    oracle_scores[policy] = []
                oracle_scores[policy].append(
                    item.get("y_true", item.get("oracle_full", 0))
                )

        oracle_row = ["Oracle"]
        for policy in target_policies:
            if policy in oracle_scores:
                mean_score = np.mean(oracle_scores[policy])
                oracle_row.append(f"{mean_score:.3f}")
            else:
                oracle_row.append("N/A")

        # Add oracle to table
        table.add_row(*oracle_row)
        console.print(table)

    console.print("\n✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
