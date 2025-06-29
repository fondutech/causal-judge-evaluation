#!/usr/bin/env python3
"""
Score the missing target policies for uncertainty.
Only scores pi_bigger_model and pi_cot since pi_bad is already done.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.judge.factory import JudgeFactory
from cje.utils.checkpointing import CheckpointManager, BatchProcessor

console = Console()


def main():
    """Score missing target policies."""

    # Setup paths
    targets_file = Path("../data/target_responses.jsonl")
    output_file = Path("../data/targets_scored_uncertainty_remaining.jsonl")

    console.print("🔬 Scoring Missing Target Policies for Uncertainty")
    console.print("=" * 60)

    # Load all target responses
    all_targets = []
    with open(targets_file) as f:
        for line in f:
            all_targets.append(json.loads(line))

    console.print(f"📊 Loaded {len(all_targets):,} total target responses")

    # Filter to only missing policies
    missing_policies = ["pi_bigger_model", "pi_cot"]
    targets_to_score = [t for t in all_targets if t.get("policy") in missing_policies]

    console.print(f"🎯 Found {len(targets_to_score):,} responses to score")
    console.print(f"   Policies: {', '.join(missing_policies)}")

    # Create uncertainty judge
    judge = JudgeFactory.create(
        model="accounts/fireworks/models/llama4-scout-instruct-basic",
        provider="fireworks",
        uncertainty_method="confidence_interval",
    )

    # Setup checkpointing
    checkpoint_path = output_file.with_suffix(".checkpoint.jsonl")
    checkpoint_mgr = CheckpointManager(
        checkpoint_path=str(checkpoint_path),
        get_uid_fn=lambda x: f"{x['prompt_id']}_{x['policy']}",
    )

    # Process in batches
    processor = BatchProcessor(batch_size=16, checkpoint_manager=checkpoint_mgr)

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
                    f"[red]❌ Error scoring {response.get('prompt_id')}: {e}[/red]"
                )
        return results

    # Process all responses
    start_time = time.time()
    scored_responses = processor.process_batches(
        targets_to_score, score_batch, description="Scoring missing policies"
    )

    # Save final output
    console.print(
        f"\n💾 Saving {len(scored_responses)} scored responses to {output_file}"
    )
    with open(output_file, "w") as f:
        for response in scored_responses:
            f.write(json.dumps(response) + "\n")

    elapsed = time.time() - start_time
    console.print(
        f"\n✅ Complete! Scored {len(scored_responses):,} responses in {elapsed:.1f}s"
    )
    console.print(f"   Rate: {len(scored_responses)/elapsed:.1f} responses/second")

    # Merge with existing uncertainty scores
    console.print("\n🔀 Merging with existing scores...")
    existing_file = Path("../data/targets_scored_uncertainty.jsonl")

    # Read existing
    existing_scores = []
    with open(existing_file) as f:
        for line in f:
            existing_scores.append(json.loads(line))

    # Combine
    all_scores = existing_scores + scored_responses
    console.print(f"   Total entries: {len(all_scores):,}")

    # Write combined file
    merged_file = existing_file.with_suffix(".merged.jsonl")
    with open(merged_file, "w") as f:
        for score in all_scores:
            f.write(json.dumps(score) + "\n")

    console.print(f"💾 Saved merged file to: {merged_file}")
    console.print("\n📊 Final coverage:")

    # Count by policy
    policy_counts = {}
    for score in all_scores:
        policy = score.get("policy", "unknown")
        policy_counts[policy] = policy_counts.get(policy, 0) + 1

    for policy, count in sorted(policy_counts.items()):
        console.print(f"   {policy}: {count:,}")


if __name__ == "__main__":
    main()
