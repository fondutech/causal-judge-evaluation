#!/usr/bin/env python3
"""
Phase 1 - Step 4c: Add DETERMINISTIC judge scores to target policy responses.

This script scores responses from all target policies (π_cot, π_bigger_model, π_bad)
using deterministic scoring (variance=0).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from cje.utils.checkpointing import CheckpointManager as CM
from rich.console import Console
from rich.progress import track

console = Console()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.judge.factory import JudgeFactory
from cje.utils.checkpointing import CheckpointManager, BatchProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add DETERMINISTIC judge scores to target policy responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="../data/target_responses.jsonl",
        help="Input file with target responses (all policies combined)",
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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for judge API calls",
    )

    args = parser.parse_args()

    console.print(
        f"🔬 [bold blue]Arena 10K Dataset Preparation - Step 4c: DETERMINISTIC Judge Scores for Targets[/bold blue]"
    )
    console.print(f"📊 All scores will have variance = 0.0")

    # Load responses
    console.print(f"📄 Loading target responses from {args.input}")
    with open(args.input) as f:
        responses = [json.loads(line) for line in f]
    console.print(f"📊 Loaded {len(responses)} responses")

    # Group by policy for statistics
    policies = {}
    for resp in responses:
        policy = resp.get("policy", "unknown")
        if policy not in policies:
            policies[policy] = 0
        policies[policy] += 1

    console.print(f"📊 Responses by policy:")
    for policy, count in policies.items():
        console.print(f"  - {policy}: {count} responses")

    # Create judge
    console.print(f"\n🔧 Creating deterministic judge with model: {args.model}")
    judge = JudgeFactory.create(
        model=args.model,
        provider="fireworks",
        temperature=0.0,
        uncertainty_method="deterministic",  # Forces variance=0
    )

    # Set up checkpointing
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.jsonl")
    checkpoint_mgr: CM = CheckpointManager(
        checkpoint_path=str(checkpoint_path),
        get_uid_fn=lambda x: f"{x.get('prompt_id', 'unknown')}_{x.get('policy', 'unknown')}",
    )

    # Process in batches with checkpointing
    processor = BatchProcessor(
        batch_size=args.batch_size, checkpoint_manager=checkpoint_mgr
    )

    def score_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score a batch of responses."""
        results = []
        for response in batch:
            try:
                # Judge the response
                score = judge.score(
                    context=response["prompt"],
                    response=response["response"],
                )

                # Add score to response
                scored = response.copy()
                scored["judge_score"] = {
                    "mean": score.mean,
                    "variance": score.variance,
                }
                results.append(scored)
            except Exception as e:
                console.print(
                    f"[red]❌ Error scoring {response.get('prompt_id', 'unknown')} ({response.get('policy', 'unknown')}): {e}[/red]"
                )
                # Skip this response
        return results

    # Process all responses
    console.print(f"\n🔬 Scoring responses...")
    scored_responses = processor.process_batches(
        responses, score_batch, description="Scoring with deterministic judge"
    )

    # Calculate statistics by policy
    console.print(f"\n📊 Scoring statistics by policy:")
    policy_stats = {}
    for resp in scored_responses:
        policy = resp.get("policy", "unknown")
        if policy not in policy_stats:
            policy_stats[policy] = {"count": 0, "sum": 0}

        if "judge_score" in resp:
            policy_stats[policy]["count"] += 1
            policy_stats[policy]["sum"] += resp["judge_score"]["mean"]

    for policy, stats in policy_stats.items():
        if stats["count"] > 0:
            avg = stats["sum"] / stats["count"]
            console.print(
                f"  - {policy}: avg score = {avg:.2f} ({stats['count']} scored)"
            )

    # Save final output
    console.print(
        f"\n💾 Saving {len(scored_responses)} scored responses to {args.output}"
    )
    with open(args.output, "w") as f:
        for response in scored_responses:
            f.write(json.dumps(response) + "\n")

    console.print(f"[green]✅ Successfully scored all target responses![/green]")


if __name__ == "__main__":
    main()
