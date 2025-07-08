#!/usr/bin/env python3
"""
Step 3b: Score all responses with uncertainty-based judge.

This script scores all responses using confidence intervals to estimate
judge score variance. Uses fixed settings from config.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.judge import JudgeFactory
from cje.utils.progress import console
from cje.utils import CheckpointManager, BatchProcessor
from config_loader import load_arena_config


def score_responses_batch(batch: List[Dict[str, Any]], judge) -> List[Dict[str, Any]]:
    """Score a batch of responses with uncertainty."""
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
                    "judge_score_variance": score.variance,
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
    # No arguments - everything from config
    INPUT_FILE = "data/all_responses.jsonl"
    OUTPUT_FILE = "data/responses_scored_uncertainty.jsonl"

    # Load config
    config = load_arena_config()

    console.print("[bold cyan]Step 3b: Judge Scoring with Uncertainty[/bold cyan]")

    # Check input exists
    if not Path(INPUT_FILE).exists():
        console.print(f"❌ [red]Error: {INPUT_FILE} not found.[/red]")
        sys.exit(1)

    # Load all responses
    console.print(f"\n📄 Loading responses from {INPUT_FILE}")
    all_items = []

    with open(INPUT_FILE) as f:
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

    # Initialize judge with uncertainty
    console.print(f"\n⚖️  Initializing judge with uncertainty:")
    console.print(f"   Provider: {config.judge_config['provider']}")
    console.print(f"   Model: {config.judge_config['model_name']}")
    console.print(f"   Method: confidence_interval")
    console.print(f"   Temperature: 0.3")

    judge = JudgeFactory.create(
        provider=config.judge_config["provider"],
        model=config.judge_config["model_name"],
        template="deterministic",
        uncertainty_method="confidence_interval",
        temperature=0.3,  # Fixed for CI method
    )

    # Process with checkpointing
    console.print(f"\n🔄 Scoring responses with uncertainty...")
    console.print(f"   Note: This is slower due to multiple samples per response")

    checkpoint_mgr = CheckpointManager(
        checkpoint_path="data/checkpoint_uncertainty.jsonl",
        get_uid_fn=lambda x: f"{x['prompt_id']}_{x['policy']}",
    )

    processor = BatchProcessor(
        checkpoint_manager=checkpoint_mgr,
        batch_size=10,  # Smaller batch due to multiple samples
    )

    results = processor.process_batches(
        all_items,
        lambda batch: score_responses_batch(batch, judge),
        description="Scoring with uncertainty",
    )

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Print summary
    console.print(f"\n✅ Scored {len(results)} responses")
    console.print(f"💾 Saved to: {OUTPUT_FILE}")

    # Score statistics by policy
    console.print("\n📊 Score Statistics by Policy:")
    for policy in sorted(policy_counts.keys()):
        policy_scores = [
            (r["judge_score"], r.get("judge_score_variance", 0))
            for r in results
            if r["policy"] == policy and r["judge_score"] is not None
        ]
        if policy_scores:
            means = [s[0] for s in policy_scores]
            variances = [s[1] for s in policy_scores]
            console.print(
                f"   {policy}: mean={np.mean(means):.3f}, std={np.std(means):.3f}, "
                f"avg_variance={np.mean(variances):.4f}"
            )

    # Uncertainty analysis
    all_variances = [
        r.get("judge_score_variance", 0)
        for r in results
        if r.get("judge_score_variance") is not None
    ]
    if all_variances:
        console.print("\n📊 Uncertainty Analysis:")
        console.print(f"   Mean variance: {np.mean(all_variances):.4f}")
        console.print(f"   Std of variance: {np.std(all_variances):.4f}")
        console.print(
            f"   Min/Max variance: {np.min(all_variances):.4f} / {np.max(all_variances):.4f}"
        )
        high_uncertainty = sum(1 for v in all_variances if v > 0.01)
        console.print(
            f"   High uncertainty responses (var > 0.01): {high_uncertainty} ({high_uncertainty/len(all_variances)*100:.1f}%)"
        )

    # Clean up checkpoint
    checkpoint_path = Path("data/checkpoint_uncertainty.jsonl")
    if checkpoint_path.exists():
        console.print(f"\n🧹 Cleaning up checkpoint file")
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
