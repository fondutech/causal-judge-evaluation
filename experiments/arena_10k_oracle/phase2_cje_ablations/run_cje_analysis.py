#!/usr/bin/env python3
"""
CJE analysis using ONLY library estimators - no manual reimplementation.
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
from cje.estimators import IPS, SNIPS, CalibratedIPS

console = Console()


def prepare_data_for_library():
    """Load and prepare data in the format expected by CJE library."""

    # Load P0 scored data
    p0_data_path = "../data/p0_scored_deterministic.jsonl"
    p0_data = []
    with open(p0_data_path) as f:
        for line in f:
            p0_data.append(json.loads(line))

    # Load logprobs data and transform to PrecomputedSampler format
    logprobs_data = []
    logprobs_path = "../phase1_dataset_preparation/data/logprobs.jsonl"

    with open(logprobs_path) as f:
        for line in f:
            item = json.loads(line)
            # Transform to PrecomputedSampler format
            transformed = {
                "prompt": item["prompt"],
                "response": item["p0_response"],
                "prompt_id": item["prompt_id"],
                "total_logprob": item["logprobs"]["p0"],  # Base policy logprob
                "target_logps": {  # Target policy logprobs
                    k: v for k, v in item["logprobs"].items() if k != "p0"
                },
            }
            logprobs_data.append(transformed)

    # Create logs for estimators (matching p0_data order)
    logs = []
    for item in p0_data:
        logs.append(
            {
                "context": item["prompt"],
                "response": item["response"],
                "reward": item["judge_score"],
                "logp": item["total_logprob"],  # Required by IPS estimators
            }
        )

    return logprobs_data, logs, p0_data


def load_oracle_scores():
    """Load oracle scores for comparison."""
    oracle_path = "../data/targets_scored_deterministic.jsonl"
    oracle_scores = {}

    with open(oracle_path) as f:
        for line in f:
            item = json.loads(line)
            policy = item["policy"]
            prompt_id = item["prompt_id"]
            if policy not in oracle_scores:
                oracle_scores[policy] = {}
            oracle_scores[policy][prompt_id] = item["judge_score"]

    # Compute means
    oracle_means = {}
    for policy, scores_dict in oracle_scores.items():
        oracle_means[policy] = np.mean(list(scores_dict.values()))

    return oracle_means


def main():
    console.print("[bold]CJE Analysis Using Library Estimators Only[/bold]")
    console.print("=" * 60)

    # Prepare data
    console.print("\n📄 Preparing data...")
    logprobs_data, logs, p0_data = prepare_data_for_library()
    console.print(f"✅ Loaded {len(logs)} samples")

    # Create PrecomputedSampler
    console.print("\n🔧 Creating PrecomputedSampler...")
    target_policies = list(logprobs_data[0]["target_logps"].keys())
    sampler = PrecomputedSampler(
        data=logprobs_data,
        target_policies=target_policies,
        max_importance_weight=50,  # Use our weight clipping
    )
    console.print(
        f"✅ Sampler initialized with {sampler.K} target policies: {', '.join(target_policies)}"
    )

    # Get weight statistics before running estimators
    contexts = [log["context"] for log in logs]
    responses = [log["response"] for log in logs]

    console.print("\n📊 Computing importance weights...")
    weight_matrix, weight_stats = sampler.importance_weights_matrix(
        contexts, responses, show_progress=False
    )

    # Display weight statistics
    weight_table = Table(title="Raw Importance Weight Statistics")
    weight_table.add_column("Policy", style="cyan")
    weight_table.add_column("Mean", justify="right")
    weight_table.add_column("Median", justify="right")
    weight_table.add_column("Min", justify="right")
    weight_table.add_column("Max", justify="right")
    weight_table.add_column("ESS%", justify="right", style="magenta")

    for policy, stats in weight_stats["policy_stats"].items():
        # Get median from weight matrix
        policy_idx = target_policies.index(policy)
        policy_weights = weight_matrix[:, policy_idx]
        valid_weights = policy_weights[~np.isnan(policy_weights)]
        median = np.median(valid_weights) if len(valid_weights) > 0 else np.nan

        weight_table.add_row(
            policy,
            f"{stats['mean']:.3f}",
            f"{median:.3f}",
            f"{stats['min']:.3f}",
            f"{stats['max']:.3f}",
            f"{stats.get('ess_percentage', np.nan):.1f}%",
        )

    console.print(weight_table)

    # Create estimators
    console.print("\n🧮 Running estimators...")

    estimators = {
        "IPS": IPS(sampler, stabilize_weights=False),  # No additional stabilization
        "SNIPS": SNIPS(sampler),
        "CalibratedIPS": CalibratedIPS(
            sampler, clip_min=0.1, clip_max=50.0, n_folds=5, max_calibrated_weight=100.0
        ),
    }

    # Fit all estimators
    results = {}
    for name, estimator in estimators.items():
        console.print(f"\n[bold]{name}:[/bold]")
        estimator.fit(logs)
        results[name] = estimator.estimate()

    # Load oracle values for comparison
    oracle_means = load_oracle_scores()

    # Display results
    results_table = Table(title="Policy Value Estimates")
    results_table.add_column("Policy", style="cyan")
    results_table.add_column("IPS", justify="right")
    results_table.add_column("SNIPS", justify="right")
    results_table.add_column("CalibratedIPS", justify="right", style="green")
    results_table.add_column("Oracle", justify="right", style="yellow")

    for i, policy in enumerate(target_policies):
        results_table.add_row(
            policy,
            f"{results['IPS'].v_hat[i]:.3f} ± {results['IPS'].se[i]:.3f}",
            f"{results['SNIPS'].v_hat[i]:.3f} ± {results['SNIPS'].se[i]:.3f}",
            f"{results['CalibratedIPS'].v_hat[i]:.3f} ± {results['CalibratedIPS'].se[i]:.3f}",
            f"{oracle_means.get(policy, np.nan):.3f}",
        )

    console.print("\n", results_table)

    # Show metadata for each estimator
    console.print("\n[bold]Estimator Metadata:[/bold]")
    for name, result in results.items():
        console.print(f"\n{name}:")
        for key, value in result.metadata.items():
            if value is not None:
                console.print(f"  {key}: {value}")

    # Performance comparison
    console.print("\n[bold]Performance Comparison (RMSE vs Oracle):[/bold]")
    for name, result in results.items():
        errors = []
        for i, policy in enumerate(target_policies):
            if policy in oracle_means:
                error = (result.v_hat[i] - oracle_means[policy]) ** 2
                errors.append(error)

        rmse = np.sqrt(np.mean(errors)) if errors else np.nan
        console.print(f"  {name}: RMSE = {rmse:.3f}")

    # Pi_clone health check
    pi_clone_idx = target_policies.index("pi_clone")
    console.print("\n[bold]Pi_clone Health Check:[/bold]")
    console.print(
        f"  Raw mean weight: {weight_stats['policy_stats']['pi_clone']['mean']:.3f}"
    )
    for name, result in results.items():
        console.print(
            f"  {name}: {result.v_hat[pi_clone_idx]:.3f} (oracle: {oracle_means['pi_clone']:.3f})"
        )


if __name__ == "__main__":
    main()
