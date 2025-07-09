#!/usr/bin/env python3
"""
Correct CJE analysis for Arena 10K data.
Uses ONLY P0 data for importance sampling as it should be.
"""

import sys
from pathlib import Path
import json
import numpy as np
from rich.console import Console
from rich.table import Table
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.loggers import PrecomputedSampler

console = Console()


def load_p0_data(filepath):
    """Load P0 data which contains everything needed for importance sampling."""
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_oracle_data(filepath):
    """Load target responses for oracle evaluation (optional)."""
    if not Path(filepath).exists():
        return None

    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_importance_weights(p0_data, clip_log_ratio=50.0):
    """Compute importance weights with optional clipping."""
    weights_by_policy = defaultdict(list)
    weight_stats = defaultdict(
        lambda: {"min": float("inf"), "max": 0, "extreme_count": 0}
    )

    for record in p0_data:
        p0_logp = record["total_logprob"]

        if p0_logp is None:
            continue

        for policy, target_logp in record["target_logps"].items():
            if target_logp is None:
                weights_by_policy[policy].append(np.nan)
                continue

            # Compute log ratio with clipping
            log_ratio = target_logp - p0_logp

            # Track extreme values before clipping
            if abs(log_ratio) > 5.0:
                weight_stats[policy]["extreme_count"] += 1

            # Clip to prevent numerical issues
            clipped_ratio = np.clip(log_ratio, -clip_log_ratio, clip_log_ratio)

            if clipped_ratio != log_ratio:
                console.print(
                    f"[yellow]Clipped log ratio for {policy}: {log_ratio:.2f} → {clipped_ratio:.2f}[/yellow]"
                )

            weight = np.exp(clipped_ratio)
            weights_by_policy[policy].append(weight)

            # Update stats
            weight_stats[policy]["min"] = min(weight_stats[policy]["min"], weight)
            weight_stats[policy]["max"] = max(weight_stats[policy]["max"], weight)

    return weights_by_policy, weight_stats


def compute_ips_estimates(p0_data, weights_by_policy):
    """Compute IPS estimates for each policy."""
    estimates = {}

    for policy, weights in weights_by_policy.items():
        rewards = []
        valid_weights = []

        for i, record in enumerate(p0_data):
            if i < len(weights) and not np.isnan(weights[i]):
                rewards.append(record["judge_score"])
                valid_weights.append(weights[i])

        if valid_weights:
            weighted_rewards = np.array(rewards) * np.array(valid_weights)
            estimates[policy] = {
                "mean": np.mean(weighted_rewards),
                "std": np.std(weighted_rewards),
                "n": len(valid_weights),
                "ess": compute_ess(valid_weights),
            }
        else:
            estimates[policy] = {"mean": np.nan, "std": np.nan, "n": 0, "ess": 0}

    return estimates


def compute_snips_estimates(p0_data, weights_by_policy):
    """Compute Self-Normalized IPS estimates for each policy."""
    estimates = {}

    for policy, weights in weights_by_policy.items():
        rewards = []
        valid_weights = []

        for i, record in enumerate(p0_data):
            if i < len(weights) and not np.isnan(weights[i]):
                rewards.append(record["judge_score"])
                valid_weights.append(weights[i])

        if valid_weights:
            weights_array = np.array(valid_weights)
            rewards_array = np.array(rewards)

            # Self-normalized estimator
            numerator = np.sum(weights_array * rewards_array)
            denominator = np.sum(weights_array)

            estimates[policy] = {
                "mean": numerator / denominator if denominator > 0 else np.nan,
                "n": len(valid_weights),
                "ess": compute_ess(valid_weights),
            }
        else:
            estimates[policy] = {"mean": np.nan, "n": 0, "ess": 0}

    return estimates


def compute_ess(weights):
    """Compute Effective Sample Size."""
    weights = np.array(weights)
    if len(weights) == 0:
        return 0
    sum_weights = np.sum(weights)
    sum_squared = np.sum(weights**2)
    if sum_squared == 0:
        return 0
    return (sum_weights**2) / sum_squared


def compute_oracle_estimates(oracle_data):
    """Compute oracle estimates from actual policy responses."""
    if oracle_data is None:
        return None

    estimates = defaultdict(lambda: {"scores": [], "mean": 0, "std": 0})

    for record in oracle_data:
        policy = record["policy"]
        score = record.get("judge_score")
        if score is not None:
            estimates[policy]["scores"].append(score)

    # Compute means and stds
    for policy, data in estimates.items():
        if data["scores"]:
            data["mean"] = np.mean(data["scores"])
            data["std"] = np.std(data["scores"])
            data["n"] = len(data["scores"])

    return dict(estimates)


def main():
    """Run correct CJE analysis using only P0 data."""
    console.print("[bold cyan]Correct CJE Analysis on Arena 10K Data[/bold cyan]")
    console.print("=" * 60)

    # File paths
    p0_file = "../data/p0_scored_deterministic.jsonl"
    oracle_file = "../data/targets_scored_deterministic.jsonl"

    # Load P0 data (contains everything needed for importance sampling)
    console.print("\n📄 Loading P0 data (for importance sampling)...")
    p0_data = load_p0_data(p0_file)
    console.print(f"✅ Loaded {len(p0_data)} P0 samples")

    # Get target policies from first record
    target_policies = list(p0_data[0]["target_logps"].keys())
    console.print(f"📊 Target policies: {', '.join(target_policies)}")

    # Compute importance weights
    console.print("\n⚖️  Computing importance weights...")
    weights_by_policy, weight_stats = compute_importance_weights(p0_data)

    # Display weight statistics
    weight_table = Table(title="Importance Weight Statistics")
    weight_table.add_column("Policy", style="cyan")
    weight_table.add_column("Mean", justify="right")
    weight_table.add_column("Median", justify="right")
    weight_table.add_column("Min", justify="right")
    weight_table.add_column("Max", justify="right")
    weight_table.add_column("ESS%", justify="right", style="magenta")
    weight_table.add_column("Extreme", justify="right", style="yellow")

    for policy in target_policies:
        weights = [w for w in weights_by_policy[policy] if not np.isnan(w)]
        if weights:
            ess = compute_ess(weights)
            ess_pct = 100 * ess / len(weights)
            weight_table.add_row(
                policy,
                f"{np.mean(weights):.3f}",
                f"{np.median(weights):.3f}",
                f"{weight_stats[policy]['min']:.3f}",
                f"{weight_stats[policy]['max']:.3f}",
                f"{ess_pct:.1f}%",
                f"{weight_stats[policy]['extreme_count']}",
            )

    console.print("\n", weight_table)

    # Compute estimates
    console.print("\n📈 Computing policy value estimates...")
    ips_estimates = compute_ips_estimates(p0_data, weights_by_policy)
    snips_estimates = compute_snips_estimates(p0_data, weights_by_policy)

    # Load oracle data if available
    console.print("\n🔮 Loading oracle data (for comparison)...")
    oracle_data = load_oracle_data(oracle_file)
    oracle_estimates = compute_oracle_estimates(oracle_data) if oracle_data else None

    # Display results
    results_table = Table(title="Policy Value Estimates")
    results_table.add_column("Policy", style="cyan")
    results_table.add_column("IPS", justify="right")
    results_table.add_column("SNIPS", justify="right", style="green")
    if oracle_estimates:
        results_table.add_column("Oracle", justify="right", style="yellow")

    for policy in target_policies:
        row = [policy]

        # IPS
        if policy in ips_estimates and not np.isnan(ips_estimates[policy]["mean"]):
            row.append(f"{ips_estimates[policy]['mean']:.3f}")
        else:
            row.append("N/A")

        # SNIPS
        if policy in snips_estimates and not np.isnan(snips_estimates[policy]["mean"]):
            row.append(f"{snips_estimates[policy]['mean']:.3f}")
        else:
            row.append("N/A")

        # Oracle
        if oracle_estimates and policy in oracle_estimates:
            row.append(f"{oracle_estimates[policy]['mean']:.3f}")

        results_table.add_row(*row)

    console.print("\n", results_table)

    # Special check for pi_clone
    if "pi_clone" in weights_by_policy:
        pi_clone_weights = [w for w in weights_by_policy["pi_clone"] if not np.isnan(w)]
        if pi_clone_weights:
            median_weight = np.median(pi_clone_weights)
            if abs(median_weight - 1.0) > 0.1:
                console.print(
                    f"\n[yellow]⚠️  Warning: pi_clone median weight is {median_weight:.3f} (expected ~1.0)[/yellow]"
                )
                console.print(
                    "[yellow]This indicates potential issues with log probability computation.[/yellow]"
                )

    console.print("\n✅ Analysis complete!")
    console.print(
        "\n[dim]Note: This analysis uses ONLY P0 data as required for importance sampling.[/dim]"
    )
    console.print(
        "[dim]Target responses are used only for oracle comparison, not for weight computation.[/dim]"
    )


if __name__ == "__main__":
    main()
