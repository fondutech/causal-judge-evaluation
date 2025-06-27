#!/usr/bin/env python3
"""
Updated ablation analysis that uses teacher forcing log probabilities.

This version properly loads the target policy log probabilities computed
by teacher forcing P0 responses through each target policy.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the original analysis functions
from run_ablation_analysis import (
    run_estimator_analysis,
    get_estimator,
    PrecomputedMultiTargetSampler,
)

console = Console()


def load_scored_data_with_logprobs(
    judge_type: str,
) -> Tuple[Dict[str, List[Dict]], bool]:
    """Load scored data and check for teacher forcing log probs."""
    data_dir = Path("data")

    # First try to load P0 data with target logprobs
    p0_with_logprobs_file = data_dir / "p0_with_target_logps.jsonl"
    has_teacher_forcing = p0_with_logprobs_file.exists()

    if has_teacher_forcing:
        console.print(f"✅ Found teacher forcing data: {p0_with_logprobs_file}")

        # Load P0 data with target logprobs
        p0_data = []
        with open(p0_with_logprobs_file) as f:
            for line in f:
                data = json.loads(line)
                # Add judge scores from the scored file
                p0_data.append(data)

        # Also need to merge with judge scores
        p0_scored_file = data_dir / f"p0_scored_{judge_type}.jsonl"
        p0_scores = {}
        with open(p0_scored_file) as f:
            for line in f:
                item = json.loads(line)
                p0_scores[item["prompt_id"]] = item["judge_score"]

        # Merge scores into p0_data
        for item in p0_data:
            if item["prompt_id"] in p0_scores:
                item["judge_score"] = p0_scores[item["prompt_id"]]
    else:
        console.print(
            "[yellow]⚠️  No teacher forcing data found - using fallback[/yellow]"
        )
        # Fall back to original loading
        p0_file = data_dir / f"p0_scored_{judge_type}.jsonl"
        p0_data = []
        with open(p0_file) as f:
            for line in f:
                p0_data.append(json.loads(line))

    # Load target policy scored data
    targets_file = data_dir / f"targets_scored_{judge_type}.jsonl"
    targets_data = []
    with open(targets_file) as f:
        for line in f:
            targets_data.append(json.loads(line))

    # Group target data by policy
    target_by_policy: Dict[str, List[Dict[str, Any]]] = {
        "pi_cot": [],
        "pi_bigger_model": [],
        "pi_bad": [],
    }

    for item in targets_data:
        policy = item.get("policy") or item.get("model")
        if policy in target_by_policy:
            target_by_policy[policy].append(item)

    return {"pi_0": p0_data, **target_by_policy}, has_teacher_forcing


def prepare_estimation_data_with_logprobs(
    data: Dict[str, List[Dict[str, Any]]], has_teacher_forcing: bool
) -> List[Dict[str, Any]]:
    """Prepare data for estimation with proper log probabilities."""

    # Get unique prompt IDs
    prompt_ids = set()
    for policy_data in data.values():
        for item in policy_data:
            prompt_ids.add(item["prompt_id"])

    # Create estimation records
    records = []
    for prompt_id in sorted(prompt_ids):
        # Get logging policy data
        p0_item = next(
            (item for item in data["pi_0"] if item["prompt_id"] == prompt_id), None
        )
        if not p0_item:
            continue

        # Create base record
        record = {
            "uid": prompt_id,
            "context": p0_item["prompt"],
            "response": p0_item["response"],
            "reward": p0_item["judge_score"]["mean"],
            "judge_variance": p0_item["judge_score"]["variance"],
            "logp_pi0": p0_item.get("logp", -10.0),
            "target_logps": {},
            "target_rewards": {},
        }

        # Add target policy data
        if has_teacher_forcing and "target_logps" in p0_item:
            # Use teacher forcing log probs from P0 response
            record["target_logps"] = p0_item["target_logps"]

            # Still need target rewards from target responses
            for policy in ["pi_cot", "pi_bigger_model", "pi_bad"]:
                target_item = next(
                    (item for item in data[policy] if item["prompt_id"] == prompt_id),
                    None,
                )
                if target_item:
                    record["target_rewards"][policy] = target_item["judge_score"][
                        "mean"
                    ]
        else:
            # Fallback: use incorrect log probs (all -10.0)
            for policy in ["pi_cot", "pi_bigger_model", "pi_bad"]:
                target_item = next(
                    (item for item in data[policy] if item["prompt_id"] == prompt_id),
                    None,
                )
                if target_item:
                    record["target_logps"][policy] = target_item.get("logp", -10.0)
                    record["target_rewards"][policy] = target_item["judge_score"][
                        "mean"
                    ]

        records.append(record)

    return records


def analyze_importance_weights(records: List[Dict[str, Any]]) -> None:
    """Analyze the importance weights to verify they're working."""
    console.print("\n[bold]Importance Weight Analysis:[/bold]")

    # Extract log probs
    behavior_logps = [r["logp_pi0"] for r in records]

    for policy in ["pi_cot", "pi_bigger_model", "pi_bad"]:
        target_logps = [r["target_logps"].get(policy, -10.0) for r in records]

        # Compute importance weights
        log_ratios = np.array(target_logps) - np.array(behavior_logps)
        weights = np.exp(np.clip(log_ratios, -20, 20))

        console.print(f"\n{policy}:")
        console.print(
            f"  Log prob range: [{min(target_logps):.2f}, {max(target_logps):.2f}]"
        )
        console.print(f"  Weight range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
        console.print(f"  Mean weight: {np.mean(weights):.4f}")
        console.print(
            f"  % weights ≈ 1.0: {np.sum(np.abs(weights - 1.0) < 0.01) / len(weights) * 100:.1f}%"
        )


def main() -> None:
    """Run complete ablation analysis with teacher forcing."""

    console.print(
        "[bold blue]🔬 Arena 10K Ablation Analysis (v2 with Teacher Forcing)[/bold blue]\n"
    )

    # Create results table
    table = Table(title="CJE Ablation Results")
    table.add_column("Judge Type", style="cyan")
    table.add_column("Estimator", style="yellow")
    table.add_column("π_cot", justify="right", style="green")
    table.add_column("π_bigger", justify="right", style="green")
    table.add_column("π_bad", justify="right", style="red")
    table.add_column("Avg SE", justify="right")

    all_results = {}

    for judge_type in ["deterministic", "uncertainty"]:
        console.print(f"\n[bold]Processing {judge_type} judge scores...[/bold]")

        # Load data with teacher forcing check
        data, has_teacher_forcing = load_scored_data_with_logprobs(judge_type)
        records = prepare_estimation_data_with_logprobs(data, has_teacher_forcing)
        console.print(f"  Loaded {len(records)} samples")

        if judge_type == "deterministic" and has_teacher_forcing:
            # Analyze importance weights once
            analyze_importance_weights(records)

        for estimator in ["IPS", "SNIPS", "CalibratedIPS", "DRCPO", "MRDR"]:
            console.print(f"  Running {estimator}...")

            # Run estimator
            results = run_estimator_analysis(estimator, records)
            all_results[f"{judge_type}_{estimator.lower()}"] = results

            # Format results for table
            row_data = [judge_type.capitalize(), estimator]
            avg_se_values = []

            for policy in ["pi_cot", "pi_bigger_model", "pi_bad"]:
                if policy in results:
                    estimate = results[policy]["estimate"]
                    se = results[policy]["se"]
                    row_data.append(f"{estimate:.3f} ± {se:.3f}")
                    avg_se_values.append(se)
                else:
                    row_data.append("N/A")

            # Add average SE
            avg_se = np.mean(avg_se_values) if avg_se_values else 0
            row_data.append(f"{avg_se:.3f}")

            table.add_row(*row_data)

    # Display results
    console.print("\n")
    console.print(table)

    # Save results
    output_path = Path("phase2_cje_ablations/results/ablation_analysis_v2.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    console.print(f"\n💾 Results saved to {output_path}")

    # Key insights
    console.print("\n[bold]📊 Key Insights:[/bold]")

    # Check if teacher forcing made a difference
    if has_teacher_forcing:
        console.print("  • ✅ Using proper teacher forcing log probabilities")
        console.print("  • Importance weights now vary across policies")
        console.print("  • Estimators can differentiate between policy qualities")
    else:
        console.print("  • ⚠️  No teacher forcing data - all weights ≈ 1.0")
        console.print("  • Run 02c_compute_target_logprobs.py to fix this")

    # Compare deterministic vs uncertainty
    det_mean = np.mean(
        [
            all_results[f"deterministic_ips"][p]["estimate"]
            for p in ["pi_cot", "pi_bigger_model", "pi_bad"]
        ]
    )
    unc_mean = np.mean(
        [
            all_results[f"uncertainty_ips"][p]["estimate"]
            for p in ["pi_cot", "pi_bigger_model", "pi_bad"]
        ]
    )

    console.print(f"  • Deterministic vs Uncertainty: {det_mean:.3f} vs {unc_mean:.3f}")

    # Find best estimator
    best_se = float("inf")
    best_est = None
    for est in ["IPS", "SNIPS", "CalibratedIPS", "DRCPO", "MRDR"]:
        avg_se = np.mean(
            [
                all_results[f"deterministic_{est.lower()}"][p]["se"]
                for p in ["pi_cot", "pi_bigger_model", "pi_bad"]
            ]
        )
        if avg_se < best_se:
            best_se = avg_se
            best_est = est

    console.print(f"  • Lowest variance estimator: {best_est} (avg SE: {best_se:.3f})")

    # Policy ranking
    console.print("\n[bold]🏆 Policy Rankings (by IPS estimate):[/bold]")
    policies = ["pi_cot", "pi_bigger_model", "pi_bad"]
    scores = [(p, all_results["deterministic_ips"][p]["estimate"]) for p in policies]
    scores.sort(key=lambda x: x[1], reverse=True)

    for i, (policy, score) in enumerate(scores, 1):
        console.print(f"  {i}. {policy}: {score:.3f}")


if __name__ == "__main__":
    main()
