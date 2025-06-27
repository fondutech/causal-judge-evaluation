#!/usr/bin/env python3
"""
Run CJE ablations using precomputed data.

This script uses CJE's PrecomputedMultiTargetSampler to run experiments
without needing to call any APIs.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from rich.console import Console
from rich.table import Table

from cje.loggers.precomputed_sampler import PrecomputedMultiTargetSampler
from cje.estimators.ips_only_estimators import IPS, SNIPS
from cje.estimators.doubly_robust_estimators import (
    MultiDRCPOEstimator,
    MultiMRDREstimator,
)

console = Console()


def load_precomputed_data(
    data_dir: Path,
) -> Tuple[Dict[Tuple[str, str], List[float]], List[Dict], List[str], int]:
    """Load precomputed data from files."""
    # Load lookup table
    lookup_file = data_dir / "logp_lookup.json"
    with open(lookup_file, "r") as f:
        lookup_data = json.load(f)

    # Convert string keys back to tuples
    logp_lookup = {}
    for key_str, logps in lookup_data["lookup"].items():
        ctx, resp = key_str.split("|||")
        logp_lookup[(ctx, resp)] = logps

    # Load rows
    rows = []
    rows_file = data_dir / "precomputed_rows.jsonl"
    with open(rows_file, "r") as f:
        for line in f:
            rows.append(json.loads(line))

    return (logp_lookup, rows, lookup_data["policy_names"], lookup_data["n_policies"])


def run_cje_ablations(
    logp_lookup: Dict[Tuple[str, str], List[float]],
    rows: List[Dict[str, Any]],
    policy_names: List[str],
    n_policies: int,
) -> Dict[str, Any]:
    """Run CJE ablations with different estimators."""

    # Create precomputed sampler
    sampler = PrecomputedMultiTargetSampler(
        logp_lookup=logp_lookup, n_policies=n_policies
    )

    # Prepare data for estimators
    contexts = [row["context"] for row in rows]
    responses = [row["response"] for row in rows]
    rewards = [row["reward"] for row in rows]
    logp_behavior = [row["logp"] for row in rows]

    # Get importance weights
    console.print("\n[blue]Computing importance weights...[/blue]")
    weights_matrix, weight_stats = sampler.importance_weights_matrix(
        contexts=contexts,
        responses=responses,
        logp_behavior=logp_behavior,
        stabilize=True,
        return_stats=True,
    )

    # Print weight statistics
    console.print("\n[bold]📊 Importance Weight Statistics[/bold]")
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("ESS Percentage", f"{weight_stats['ess_percentage']:.1f}%")
    table.add_row(
        "Clipped Samples",
        f"{weight_stats['n_clipped']} ({weight_stats['clip_fraction']*100:.1f}%)",
    )
    table.add_row(
        "Weight Range",
        f"[{weight_stats['weight_range'][0]:.3f}, {weight_stats['weight_range'][1]:.3f}]",
    )

    for i, (policy_name, ess) in enumerate(
        zip(policy_names, weight_stats["ess_values"])
    ):
        table.add_row(f"ESS {policy_name}", f"{ess:.1f} ({ess/len(rows)*100:.1f}%)")

    console.print(table)

    # Run different estimators
    results = {}

    console.print("\n[bold]🧮 Running Estimators[/bold]")

    # Format data for estimators
    estimator_data = []
    for i, row in enumerate(rows):
        est_row = {
            "context": row["context"],
            "response": row["response"],
            "reward": row["reward"],
            "logp": row["logp"],
            "logp_target_all": [
                logp_lookup[(row["context"], row["response"])][j]
                for j in range(n_policies)
            ],
        }
        estimator_data.append(est_row)

    results_table = Table(title="CJE Ablation Results")
    results_table.add_column("Estimator", style="cyan")
    for policy_name in policy_names:
        results_table.add_column(f"{policy_name}", style="green")

    # Create estimators with the sampler
    estimators = [
        ("IPS", IPS(sampler=sampler)),
        ("SNIPS", SNIPS(sampler=sampler)),
        ("DRCPO", MultiDRCPOEstimator(sampler=sampler, k=5)),
        ("MRDR", MultiMRDREstimator(sampler=sampler, k=5)),
    ]

    for estimator_name, estimator in estimators:
        try:
            console.print(f"\n[blue]Running {estimator_name}...[/blue]")

            # Run estimation
            result = estimator.estimate_from_logs(estimator_data)

            # Store results
            results[estimator_name] = {
                "estimates": result.estimates,
                "standard_errors": result.standard_errors,
                "confidence_intervals": [
                    (est - 1.96 * np.sqrt(se), est + 1.96 * np.sqrt(se))
                    for est, se in zip(result.estimates, result.standard_errors)
                ],
            }

            # Add to table
            row_values = [estimator_name]
            for j, policy_name in enumerate(policy_names):
                est = result.estimates[j]
                se = np.sqrt(result.standard_errors[j])
                row_values.append(f"{est:.3f} ± {se:.3f}")

            results_table.add_row(*row_values)

        except Exception as e:
            console.print(f"[red]Error running {estimator_name}: {e}[/red]")
            results[estimator_name] = {"error": str(e)}

    console.print("\n")
    console.print(results_table)

    return results


def analyze_results(results: Dict[str, Any], policy_names: List[str]) -> None:
    """Analyze and visualize ablation results."""
    console.print("\n[bold]📈 Analysis[/bold]")

    # Compare estimators
    console.print("\n[yellow]Estimator Comparison:[/yellow]")

    valid_estimators = {k: v for k, v in results.items() if "error" not in v}

    if len(valid_estimators) > 1:
        # Find estimator with lowest variance
        min_variance_estimator = None
        min_variance = float("inf")

        for est_name, est_results in valid_estimators.items():
            avg_variance = np.mean(est_results["standard_errors"])
            if avg_variance < min_variance:
                min_variance = avg_variance
                min_variance_estimator = est_name

        console.print(f"  • Lowest variance: {min_variance_estimator}")

        # Check consistency across estimators
        estimates_matrix = np.array(
            [res["estimates"] for res in valid_estimators.values()]
        )
        estimate_std = np.std(estimates_matrix, axis=0)

        console.print(f"  • Cross-estimator consistency (std dev):")
        for i, policy_name in enumerate(policy_names):
            console.print(f"    - {policy_name}: {estimate_std[i]:.4f}")

    # Policy comparison
    console.print("\n[yellow]Policy Comparison:[/yellow]")

    # Use DRCPO if available, otherwise IPS
    best_estimator = "DRCPO" if "DRCPO" in valid_estimators else "IPS"
    if best_estimator in valid_estimators:
        estimates = valid_estimators[best_estimator]["estimates"]

        # Find best policy
        best_idx = np.argmax(estimates)
        best_policy = policy_names[best_idx]

        console.print(
            f"  • Best policy: {best_policy} (score: {estimates[best_idx]:.3f})"
        )
        console.print(f"  • Policy ranking:")

        sorted_indices = np.argsort(estimates)[::-1]
        for rank, idx in enumerate(sorted_indices):
            console.print(f"    {rank+1}. {policy_names[idx]}: {estimates[idx]:.3f}")


def save_results(results: Dict[str, Any], output_file: Path) -> None:
    """Save results to file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]💾 Results saved to {output_file}[/green]")


def main():
    """Run CJE ablations with precomputed data."""
    console.print(
        "[bold blue]🚀 Running CJE Ablations with Precomputed Data[/bold blue]"
    )

    # Check if precomputed data exists
    data_dir = Path("precomputed_data")
    if not data_dir.exists():
        console.print(f"\n[red]Error: Precomputed data not found at {data_dir}[/red]")
        console.print("[yellow]Run prepare_precomputed_data.py first![/yellow]")
        return

    # Load data
    console.print("\n📂 Loading precomputed data...")
    logp_lookup, rows, policy_names, n_policies = load_precomputed_data(data_dir)

    console.print(f"  ✓ Loaded {len(rows)} rows")
    console.print(f"  ✓ {n_policies} target policies: {', '.join(policy_names)}")

    # Run ablations
    results = run_cje_ablations(logp_lookup, rows, policy_names, n_policies)

    # Analyze results
    analyze_results(results, policy_names)

    # Save results
    output_file = Path("ablation_results.json")
    save_results(results, output_file)

    console.print("\n[bold green]✅ CJE ablations complete![/bold green]")


if __name__ == "__main__":
    main()
