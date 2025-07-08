#!/usr/bin/env python3
"""
Step 3a: Score all responses with deterministic judge.

This script scores all responses (P0 and target policies) using a deterministic judge.
Uses configuration from arena_10k.yaml for judge settings.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.judge import JudgeFactory
from cje.utils.progress import console
from cje.utils import CheckpointManager, BatchProcessor
from config_loader import load_arena_config


def score_responses_batch(batch: List[Dict[str, Any]], judge) -> List[Dict[str, Any]]:
    """Score a batch of responses."""
    results = []

    for item in batch:
        prompt_id = item["prompt_id"]
        prompt = item["prompt"]
        policy = item["policy"]
        response = item["response"]

        # Score the response
        try:
            score = judge.score(prompt, response)
            results.append(
                {
                    "prompt_id": prompt_id,
                    "policy": policy,
                    "judge_score": score.mean,
                    "judge_score_variance": 0.0,  # Deterministic has no variance
                    "response_length": len(response),
                }
            )
        except Exception as e:
            console.print(f"[red]Error scoring {prompt_id}/{policy}: {e}[/red]")
            results.append(
                {
                    "prompt_id": prompt_id,
                    "policy": policy,
                    "judge_score": None,
                    "judge_score_variance": None,
                    "response_length": len(response),
                    "error": str(e),
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Score all responses with deterministic judge"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/all_responses.jsonl",
        help="Input file with all responses (default: data/all_responses.jsonl)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/responses_scored_deterministic.jsonl",
        help="Output file for scored responses",
    )

    parser.add_argument(
        "--provider",
        type=str,
        help="Override judge provider from config",
    )

    args = parser.parse_args()

    # Load config
    config = load_arena_config()

    console.print("[bold cyan]Step 3a: Deterministic Judge Scoring[/bold cyan]")

    # Check input exists
    if not Path(args.input).exists():
        console.print(f"❌ [red]Error: {args.input} not found.[/red]")
        sys.exit(1)

    # Load all responses
    console.print(f"\n📄 Loading responses from {args.input}")
    all_items = []

    with open(args.input) as f:
        for line in f:
            data = json.loads(line)
            prompt_id = data["prompt_id"]
            prompt = data["prompt"]

            # Extract all responses for this prompt
            for policy_name, resp_data in data["responses"].items():
                all_items.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "policy": policy_name,
                        "response": resp_data["response"],
                        "model": resp_data["model"],
                    }
                )

    console.print(f"✅ Loaded {len(all_items)} total responses")

    # Count by policy
    policy_counts = {}
    for item in all_items:
        policy = item["policy"]
        policy_counts[policy] = policy_counts.get(policy, 0) + 1

    console.print("\n📊 Responses by policy:")
    for policy, count in sorted(policy_counts.items()):
        console.print(f"   {policy}: {count}")

    # Initialize judge
    provider = args.provider or config.judge_config["provider"]
    console.print(f"\n⚖️  Initializing deterministic judge:")
    console.print(f"   Provider: {provider}")
    console.print(f"   Model: {config.judge_config['model_name']}")

    judge = JudgeFactory.create(
        provider=provider,
        model=config.judge_config["model_name"],
        template="deterministic",
        uncertainty_method="deterministic",
    )

    # Process with checkpointing
    console.print(f"\n🔄 Scoring responses...")

    checkpoint_mgr = CheckpointManager(
        checkpoint_path=str(Path(args.output).with_suffix(".checkpoint.jsonl")),
        get_uid_fn=lambda x: f"{x['prompt_id']}_{x['policy']}",
    )

    processor = BatchProcessor(
        checkpoint_manager=checkpoint_mgr,
        batch_size=20,
    )

    results = processor.process_batches(
        all_items,
        lambda batch: score_responses_batch(batch, judge),
        description="Scoring responses",
    )

    # Save results
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Print summary
    console.print(f"\n✅ Scored {len(results)} responses")
    console.print(f"💾 Saved to: {args.output}")

    # Score statistics by policy
    console.print("\n📊 Score Statistics by Policy:")
    for policy in sorted(policy_counts.keys()):
        policy_scores = [
            r["judge_score"]
            for r in results
            if r["policy"] == policy and r["judge_score"] is not None
        ]
        if policy_scores:
            import numpy as np

            console.print(
                f"   {policy}: mean={np.mean(policy_scores):.3f}, std={np.std(policy_scores):.3f}"
            )

    # Clean up checkpoint
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.jsonl")
    if checkpoint_path.exists():
        console.print(f"\n🧹 Cleaning up checkpoint file")
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
