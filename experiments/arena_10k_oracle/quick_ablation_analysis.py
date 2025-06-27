#!/usr/bin/env python3
"""
Quick ablation analysis focusing on IPS-based estimators.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from rich.console import Console
from rich.table import Table
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

console = Console()


def load_data(judge_type: str):
    """Load scored data files."""
    data_dir = Path("data")

    # Load P0
    p0_file = data_dir / f"p0_scored_{judge_type}.jsonl"
    p0_data = []
    with open(p0_file) as f:
        for line in f:
            p0_data.append(json.loads(line))

    # Load targets
    targets_file = data_dir / f"targets_scored_{judge_type}.jsonl"
    if not targets_file.exists():
        console.print(f"[yellow]No targets file for {judge_type}[/yellow]")
        return p0_data, []

    targets_data = []
    with open(targets_file) as f:
        for line in f:
            targets_data.append(json.loads(line))

    return p0_data, targets_data


def compute_ips_estimates(
    p0_data: List[Dict], targets_data: List[Dict]
) -> Dict[str, float]:
    """Compute simple IPS estimates."""

    # Group targets by policy
    targets_by_policy = {"pi_bad": [], "pi_bigger_model": [], "pi_cot": []}
    for entry in targets_data:
        policy = entry.get("policy") or entry.get("model")
        if policy in targets_by_policy:
            targets_by_policy[policy].append(entry)

    # Compute mean scores for each policy
    results = {}

    # P0 mean
    p0_scores = [d["judge_score"]["mean"] for d in p0_data]
    results["pi_0"] = {
        "mean": np.mean(p0_scores),
        "std": np.std(p0_scores) / np.sqrt(len(p0_scores)),
        "n": len(p0_scores),
    }

    # Target policy means (simple average since we have unit weights for clone policies)
    for policy, policy_data in targets_by_policy.items():
        if policy_data:
            scores = [d["judge_score"]["mean"] for d in policy_data]
            results[policy] = {
                "mean": np.mean(scores),
                "std": np.std(scores) / np.sqrt(len(scores)),
                "n": len(scores),
            }
        else:
            results[policy] = {"mean": 0.0, "std": 0.0, "n": 0}

    return results


def main():
    """Run quick analysis."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge-types", nargs="+", default=["deterministic", "uncertainty"]
    )
    args = parser.parse_args()

    console.print("\n[bold cyan]🔬 Quick Arena 10K Analysis[/bold cyan]")
    console.print("=" * 60)

    # Create results table
    table = Table(title="Mean Judge Scores by Policy")
    table.add_column("Judge Type", style="cyan")
    table.add_column("π_0 (baseline)", justify="right")
    table.add_column("π_bad", justify="right", style="red")
    table.add_column("π_bigger", justify="right", style="yellow")
    table.add_column("π_cot", justify="right", style="green")

    all_results = {}

    for judge_type in args.judge_types:
        console.print(f"\nProcessing {judge_type} scores...")

        # Load data
        p0_data, targets_data = load_data(judge_type)
        console.print(f"  P0: {len(p0_data)} entries")
        console.print(f"  Targets: {len(targets_data)} entries")

        if not targets_data:
            console.print(f"[yellow]Skipping {judge_type} - no target data[/yellow]")
            continue

        # Compute estimates
        results = compute_ips_estimates(p0_data, targets_data)
        all_results[judge_type] = results

        # Add to table
        table.add_row(
            judge_type.capitalize(),
            f"{results['pi_0']['mean']:.3f} ± {results['pi_0']['std']:.3f}",
            f"{results['pi_bad']['mean']:.3f} ± {results['pi_bad']['std']:.3f}",
            f"{results['pi_bigger_model']['mean']:.3f} ± {results['pi_bigger_model']['std']:.3f}",
            f"{results['pi_cot']['mean']:.3f} ± {results['pi_cot']['std']:.3f}",
        )

    # Display table
    console.print("\n")
    console.print(table)

    # Save results
    output_file = Path("quick_analysis_results.json")
    with open(output_file, "w") as f:
        json.dump(
            {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": all_results},
            f,
            indent=2,
        )

    console.print(f"\n💾 Results saved to: {output_file}")

    # Analysis insights
    if (
        len(all_results) > 1
        and "deterministic" in all_results
        and "uncertainty" in all_results
    ):
        console.print("\n[bold cyan]📊 Insights:[/bold cyan]")

        # Compare uncertainty impact
        for policy in ["pi_bad", "pi_bigger_model", "pi_cot"]:
            det_mean = all_results["deterministic"][policy]["mean"]
            unc_mean = all_results["uncertainty"][policy]["mean"]
            diff = unc_mean - det_mean
            if det_mean != 0:
                pct = diff / det_mean * 100
                console.print(f"  {policy}: Uncertainty adds {diff:+.3f} ({pct:+.1f}%)")
            else:
                console.print(f"  {policy}: Det={det_mean:.3f}, Unc={unc_mean:.3f}")

        # Warn about incomplete data
        det_n = sum(
            all_results["deterministic"][p]["n"]
            for p in ["pi_bad", "pi_bigger_model", "pi_cot"]
        )
        if det_n < 30000:
            console.print(
                f"\n[yellow]⚠️  Deterministic has only {det_n:,}/30,000 target samples[/yellow]"
            )


if __name__ == "__main__":
    main()
