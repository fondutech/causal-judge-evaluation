#!/usr/bin/env python3
"""
Run CJE ablations directly without the full pipeline.

This script runs ablations using our precomputed data in a more direct way,
bypassing some of the pipeline complexity.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


def load_all_data() -> Tuple[List[Dict], Dict[str, List[float]]]:
    """Load all available data for ablations."""
    base_dir = Path("../data")

    # Load teacher forcing data (prefer fixed version)
    tf_file = base_dir / "p0_with_target_logps_fixed.jsonl"
    if not tf_file.exists():
        tf_file = base_dir / "p0_with_target_logps.checkpoint.jsonl"
        if not tf_file.exists():
            tf_file = base_dir / "p0_with_target_logps.jsonl"

    all_data = []
    policy_logps = defaultdict(list)

    console.print(f"📂 Loading teacher forcing data from {tf_file}...")

    with open(tf_file, "r") as f:
        for line in f:
            item = json.loads(line)
            all_data.append(item)

            # Extract policy log probs
            for policy_name, logp in item.get("target_logps", {}).items():
                policy_logps[policy_name].append(logp)

    # Load P0 scores
    score_files = [
        base_dir / "p0_scored_deterministic.jsonl",
        base_dir / "p0_scored_uncertainty.jsonl",
    ]

    scores_by_id = {}
    for score_file in score_files:
        if score_file.exists():
            console.print(f"📂 Loading scores from {score_file}...")
            with open(score_file, "r") as f:
                for line in f:
                    item = json.loads(line)
                    prompt_id = item.get("prompt_id")
                    if prompt_id:
                        scores_by_id[prompt_id] = item.get(
                            "judge_score", {"mean": 0.5, "variance": 0.01}
                        )

    # Merge scores into data
    for item in all_data:
        prompt_id = item.get("prompt_id")
        if prompt_id in scores_by_id:
            item["judge_score"] = scores_by_id[prompt_id]
        else:
            item["judge_score"] = {"mean": 0.5, "variance": 0.01}

    return all_data, dict(policy_logps)


def compute_importance_weights(
    data: List[Dict], policy_names: List[str]
) -> Dict[str, np.ndarray]:
    """Compute importance weights for each policy."""
    weights = defaultdict(list)

    for item in data:
        p0_logp = item["total_logprob"]
        target_logps = item.get("target_logps", {})

        for policy_name in policy_names:
            if policy_name in target_logps:
                target_logp = target_logps[policy_name]
                # Importance weight = exp(log P(response|context, π_target) - log P(response|context, π_0))
                weight = np.exp(np.clip(target_logp - p0_logp, -20, 20))
                weights[policy_name].append(weight)
            else:
                weights[policy_name].append(1.0)  # Default to 1 if missing

    return {k: np.array(v) for k, v in weights.items()}


def compute_ips_estimates(
    data: List[Dict], weights: Dict[str, np.ndarray]
) -> Dict[str, Dict]:
    """Compute IPS estimates for each policy."""
    rewards = np.array([item["judge_score"]["mean"] for item in data])
    n = len(rewards)

    results = {}

    for policy_name, policy_weights in weights.items():
        # Standard IPS
        ips_estimate = np.mean(policy_weights * rewards)

        # Self-normalized IPS (SNIPS)
        snips_estimate = np.sum(policy_weights * rewards) / np.sum(policy_weights)

        # Compute ESS
        ess = (
            (np.sum(policy_weights) ** 2) / np.sum(policy_weights**2)
            if np.sum(policy_weights) > 0
            else 0
        )
        ess_percentage = (ess / n) * 100

        # Compute variance (simplified)
        ips_variance = np.var(policy_weights * rewards) / n

        results[policy_name] = {
            "ips": ips_estimate,
            "snips": snips_estimate,
            "ips_std": np.sqrt(ips_variance),
            "ess": ess,
            "ess_percentage": ess_percentage,
            "weight_mean": np.mean(policy_weights),
            "weight_std": np.std(policy_weights),
            "weight_range": (np.min(policy_weights), np.max(policy_weights)),
        }

    return results


def display_results(results: Dict[str, Dict], n_samples: int) -> None:
    """Display ablation results in a nice format."""
    console.print("\n[bold]📊 CJE Ablation Results[/bold]")
    console.print(f"Based on {n_samples} samples with teacher forcing\n")

    # Main results table
    table = Table(title="Policy Evaluation Results")
    table.add_column("Policy", style="cyan")
    table.add_column("IPS Estimate", style="green")
    table.add_column("SNIPS Estimate", style="green")
    table.add_column("ESS %", style="yellow")
    table.add_column("Weight Range", style="magenta")

    policy_names = sorted(results.keys())
    for policy_name in policy_names:
        res = results[policy_name]
        table.add_row(
            policy_name,
            f"{res['ips']:.3f} ± {res['ips_std']:.3f}",
            f"{res['snips']:.3f}",
            f"{res['ess_percentage']:.1f}%",
            f"[{res['weight_range'][0]:.2e}, {res['weight_range'][1]:.2e}]",
        )

    console.print(table)

    # Weight statistics table
    weight_table = Table(title="Importance Weight Statistics")
    weight_table.add_column("Policy", style="cyan")
    weight_table.add_column("Mean Weight", style="green")
    weight_table.add_column("Std Weight", style="yellow")
    weight_table.add_column("ESS", style="magenta")

    for policy_name in policy_names:
        res = results[policy_name]
        weight_table.add_row(
            policy_name,
            f"{res['weight_mean']:.3f}",
            f"{res['weight_std']:.3f}",
            f"{res['ess']:.1f}",
        )

    console.print("\n")
    console.print(weight_table)

    # Analysis
    console.print("\n[bold]📈 Analysis[/bold]")

    # Find best policy by IPS
    best_policy_ips = max(policy_names, key=lambda p: results[p]["ips"])
    console.print(
        f"\n✅ Best policy (IPS): {best_policy_ips} ({results[best_policy_ips]['ips']:.3f})"
    )

    # Find best policy by SNIPS
    best_policy_snips = max(policy_names, key=lambda p: results[p]["snips"])
    console.print(
        f"✅ Best policy (SNIPS): {best_policy_snips} ({results[best_policy_snips]['snips']:.3f})"
    )

    # ESS warnings
    low_ess_policies = [p for p in policy_names if results[p]["ess_percentage"] < 5]
    if low_ess_policies:
        console.print(f"\n⚠️  Low ESS warning for: {', '.join(low_ess_policies)}")
        console.print(
            "   Consider using DRCPO/MRDR estimators or increasing sample size"
        )

    # Policy ranking
    console.print("\n[yellow]Policy Ranking (by SNIPS):[/yellow]")
    ranked_policies = sorted(
        policy_names, key=lambda p: results[p]["snips"], reverse=True
    )
    for rank, policy in enumerate(ranked_policies, 1):
        console.print(f"  {rank}. {policy}: {results[policy]['snips']:.3f}")


def save_results(results: Dict, output_file: Path) -> None:
    """Save results to JSON file."""
    # Convert numpy types to Python types
    clean_results = {}
    for policy, res in results.items():
        clean_results[policy] = {
            k: (
                float(v)
                if isinstance(v, (np.floating, np.integer))
                else (
                    (float(v[0]), float(v[1]))
                    if isinstance(v, tuple) and len(v) == 2
                    else v
                )
            )
            for k, v in res.items()
        }

    with open(output_file, "w") as f:
        json.dump(clean_results, f, indent=2)

    console.print(f"\n💾 Results saved to {output_file}")


def main():
    """Run direct CJE ablations."""
    console.print("[bold blue]🚀 Running Direct CJE Ablations[/bold blue]")
    console.print("This bypasses the full pipeline and works directly with our data\n")

    # Load all data
    data, policy_logps = load_all_data()

    if not data:
        console.print("[red]❌ No teacher forcing data found![/red]")
        console.print("Run 02c_compute_target_logprobs.py first")
        return

    console.print(f"\n✅ Loaded {len(data)} samples with teacher forcing")

    # Get policy names
    policy_names = sorted(policy_logps.keys())
    console.print(
        f"✅ Found {len(policy_names)} target policies: {', '.join(policy_names)}"
    )

    # Compute importance weights
    console.print("\n🔄 Computing importance weights...")
    weights = compute_importance_weights(data, policy_names)

    # Compute estimates
    console.print("🧮 Computing IPS estimates...")
    results = compute_ips_estimates(data, weights)

    # Display results
    display_results(results, len(data))

    # Save results
    output_file = Path("direct_ablation_results.json")
    save_results(results, output_file)

    console.print("\n[bold green]✅ Direct ablations complete![/bold green]")


if __name__ == "__main__":
    main()
