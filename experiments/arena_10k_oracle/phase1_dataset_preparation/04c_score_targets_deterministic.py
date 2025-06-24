#!/usr/bin/env python3
# mypy: disable-error-code="attr-defined,var-annotated,call-arg,arg-type,name-defined"
"""
Phase 1 - Step 4c: Add DETERMINISTIC judge scores to target policy responses.

This script scores responses from all target policies (π_cot, π_bigger_model, π_bad)
using deterministic scoring (variance=0).
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Dict, Any
from rich.console import Console
from rich.progress import track

console = Console()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.judge.factory import JudgeFactory
from cje.judge.schemas import JudgeScore

# from add_judge_scores import update_row_with_score  # Not used


def score_target_responses(
    input_file: Path,
    output_file: Path,
    model: str = "accounts/fireworks/models/llama4-scout-instruct-basic",
    temperature: float = 0.0,
    batch_size: int = 32,
) -> None:
    """Score target policy responses with deterministic judge."""

    # Load target responses
    console.print(f"Loading target responses from {input_file}")
    with open(input_file) as f:
        responses = [json.loads(line) for line in f]

    # Group by policy for better tracking
    policies: Dict[str, List[Dict[str, Any]]] = {}
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

    # Create deterministic judge
    judge_instance = JudgeFactory.create(
        model=model,
        provider="fireworks",
        temperature=temperature,
        uncertainty_method="deterministic",
    )
    console.print(f"Created deterministic judge: {model}")

    # Score all responses
    all_scored = []

    for policy, policy_responses in policies.items():
        console.print(f"\n[bold blue]Scoring {policy} responses...[/bold blue]")

        # Process in batches
        for i in track(
            range(0, len(policy_responses), batch_size), description=f"Scoring {policy}"
        ):
            batch = policy_responses[i : i + batch_size]

            # Create judge samples
            samples = [
                dict(
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
                scored_row = row.copy()
                scored_row["judge_score"] = {
                    "mean": score.mean,
                    "variance": score.variance,
                }
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
            console.print(f"\n{policy}:")
            console.print(f"  Count: {len(scores)}")
            console.print(f"  Mean score: {sum(scores) / len(scores):.3f}")
            console.print(f"  Min: {min(scores):.3f}, Max: {max(scores):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add DETERMINISTIC judge scores to target policy responses"
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
        default="../data/targets_scored_deterministic.jsonl",
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
        help="Sampling temperature (0 for deterministic)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for judge API calls",
    )

    args = parser.parse_args()

    console.print(
        f"🔬 [bold blue]Arena 10K Dataset Preparation - Step 4c: Target Policy DETERMINISTIC Scores[/bold blue]"
    )
    console.print(f"📊 All scores will have variance = 0.0")

    score_target_responses(
        input_file=Path(args.input),
        output_file=Path(args.output),
        model=args.model,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
