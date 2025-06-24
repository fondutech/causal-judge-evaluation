#!/usr/bin/env python3
# mypy: disable-error-code="attr-defined,var-annotated,call-arg,arg-type,name-defined"
"""
Phase 1 - Step 4d: Add judge scores WITH UNCERTAINTY to target policy responses.

This script scores responses from all target policies (π_cot, π_bigger_model, π_bad)
using confidence interval based uncertainty quantification.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Dict, Any
from rich.console import Console
from rich.progress import track
import numpy as np

console = Console()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.judge.factory import JudgeFactory
from cje.judge.schemas import JudgeScore
from add_judge_scores import update_row_with_score


def score_target_responses(
    input_file: Path,
    output_file: Path,
    model: str = "accounts/fireworks/models/llama4-scout-instruct-basic",
    temperature: float = 0.0,
    batch_size: int = 32,
) -> None:
    """Score target policy responses with uncertainty-aware judge."""

    # Load target responses
    console.print(f"Loading target responses from {input_file}")
    with open(input_file) as f:
        responses = [json.loads(line) for line in f]

    # Group by policy for better tracking
    policies = {}
    for resp in responses:
        policy = resp.get("policy", "unknown")
        if policy not in policies:
            policies[policy] = []
        policies[policy].append(resp)

    console.print(
        f"Found {len(responses)} total responses across {len(policies)} policies:"
    )
    for policy, items in policies.items():
        console.print(f"  - {policy}: {len(items)} responses")

    # Create confidence interval judge
    judge_config = {
        "provider": "fireworks",
        "model_name": model,
        "template": "confidence_interval",
        "temperature": temperature,
        "uncertainty_method": "confidence_interval",
    }

    judge_instance = JudgeFactory.create(judge_config)
    console.print(f"Created CI judge: {model}")
    console.print(f"📊 Scores will include 95% confidence intervals")

    # Score all responses
    all_scored = []

    for policy, policy_responses in policies.items():
        console.print(
            f"\n[bold blue]Scoring {policy} responses with uncertainty...[/bold blue]"
        )

        # Process in batches
        for i in track(
            range(0, len(policy_responses), batch_size), description=f"Scoring {policy}"
        ):
            batch = policy_responses[i : i + batch_size]

            # Create judge samples
            samples = [
                JudgeSample(
                    context=row["prompt"],
                    response=row["response"],
                    judge_context={"request_idx": idx},
                )
                for idx, row in enumerate(batch, start=i)
            ]

            # Score batch
            scores = judge_instance.score_batch(samples)

            # Update rows with scores
            for row, score in zip(batch, scores):
                scored_row = update_row_with_score(row, score, "judge_score")
                all_scored.append(scored_row)

    # Save scored responses
    console.print(
        f"\n[bold green]Saving scored responses to {output_file}[/bold green]"
    )
    with open(output_file, "w") as f:
        for row in all_scored:
            f.write(json.dumps(row) + "\n")

    # Print statistics
    console.print(f"\n[bold]Statistics:[/bold]")
    console.print(f"Total responses scored: {len(all_scored)}")

    # Per-policy statistics
    for policy in policies.keys():
        policy_rows = [r for r in all_scored if r.get("policy") == policy]
        if policy_rows:
            scores = [r["judge_score"] for r in policy_rows]
            variances = [r.get("judge_score_variance", 0.0) for r in policy_rows]

            console.print(f"\n{policy}:")
            console.print(f"  Count: {len(scores)}")
            console.print(f"  Mean score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            console.print(f"  Score range: [{min(scores):.3f}, {max(scores):.3f}]")
            console.print(f"  Mean variance: {np.mean(variances):.4f}")
            console.print(
                f"  Variance range: [{min(variances):.4f}, {max(variances):.4f}]"
            )

    # Compare uncertainty across policies
    console.print(f"\n[bold]Uncertainty Analysis:[/bold]")
    policy_variances = {}
    for policy in policies.keys():
        policy_rows = [r for r in all_scored if r.get("policy") == policy]
        if policy_rows:
            variances = [r.get("judge_score_variance", 0.0) for r in policy_rows]
            policy_variances[policy] = {
                "mean": np.mean(variances),
                "std": np.std(variances),
                "median": np.median(variances),
            }

    # Sort by mean variance
    sorted_policies = sorted(policy_variances.items(), key=lambda x: x[1]["mean"])
    console.print("\nPolicies ranked by judge uncertainty (lowest to highest):")
    for policy, stats in sorted_policies:
        console.print(
            f"  {policy}: mean_var={stats['mean']:.4f}, "
            f"std={stats['std']:.4f}, median={stats['median']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add judge scores WITH UNCERTAINTY to target policy responses"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="../data/target_ground_truth.jsonl",
        help="Input file with target policy responses",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../data/targets_scored_uncertainty.jsonl",
        help="Output file for scored responses",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="accounts/fireworks/models/llama4-scout-instruct-basic",
        help="Judge model",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 recommended for CI consistency)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for judge API calls",
    )

    args = parser.parse_args()

    console.print(
        f"🔬 [bold blue]Arena 10K Dataset Preparation - Step 4d: Target Policy UNCERTAINTY Scores[/bold blue]"
    )
    console.print(f"📊 Scores will include 95% confidence intervals")
    console.print(f"📈 Variance calculated from CI width")

    score_target_responses(
        input_file=Path(args.input),
        output_file=Path(args.output),
        model=args.model,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
