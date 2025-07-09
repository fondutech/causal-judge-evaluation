#!/usr/bin/env python3
"""
Step 3a: Score all responses with deterministic judge and write in Phase 2 format.

This script:
- Scores all responses (P0 and target policies) using a deterministic judge
- Merges P0 scores with log probabilities for importance weighting
- Writes separate files for P0 and target policies in Phase 2 format
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
# Add current directory to path for local imports
sys.path.append(str(Path(__file__).parent))

from cje.judge import JudgeFactory
from cje.utils.progress import console
from cje.utils import CheckpointManager, BatchProcessor
from config_loader import load_arena_config


def score_responses_batch(batch: List[Dict[str, Any]], judge) -> List[Dict[str, Any]]:
    """Score a batch of responses."""
    results = []

    for item in batch:
        try:
            score = judge.score(item["prompt"], item["response"])
            item["judge_score"] = score.mean
            # For deterministic judge, variance should be None
            if hasattr(score, "variance") and score.variance is not None:
                item["judge_score_variance"] = score.variance
        except Exception as e:
            console.print(f"[red]Error scoring response: {e}[/red]")
            item["judge_score"] = None
            item["error"] = str(e)

        results.append(item)

    return results


def create_phase2_record(
    prompt_id: str,
    prompt: str,
    response: str,
    policy: str,
    judge_score: float,
    model: str,
    temperature: float,
    logprobs: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Create a record in Phase 2 format."""

    record = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "response": response,
        "policy": policy,
        "judge_score": judge_score,
        "model": model,
        "temperature": temperature,
    }

    # For P0 responses, add log probabilities
    if policy == "p0" and logprobs:
        # Extract p0 log prob - use None for missing values, not 0.0!
        record["p0_logp"] = logprobs.get("p0", None)

        # Validate for suspicious zero values
        if record["p0_logp"] == 0.0:
            console.print(
                f"[yellow]Warning: Suspicious zero log prob for p0 response (len={len(response) if response else 0}) on {prompt_id}[/yellow]"
            )

        # Add target policy log probs - critical to use None for missing values
        for target_policy in ["pi_clone", "pi_cot", "pi_bigger_model", "pi_bad"]:
            target_logp = logprobs.get(target_policy, None)

            # Validate for suspicious zero values
            if target_logp == 0.0:
                console.print(
                    f"[yellow]Warning: Suspicious zero log prob for {target_policy} (response len={len(response) if response else 0}) on {prompt_id}[/yellow]"
                )

            record[target_policy] = {
                "logp": target_logp,
                "score": None,  # Will be filled when scoring target responses
                "response": None,  # Will be filled when scoring target responses
            }

    return record


def main():
    # No arguments - everything from config
    INPUT_FILE = "data/all_responses.jsonl"
    LOGPROBS_FILE = "data/logprobs.jsonl"

    # Create Phase 2 output directory
    phase2_dir = Path("../data")
    phase2_dir.mkdir(parents=True, exist_ok=True)

    P0_OUTPUT = phase2_dir / "p0_scored_deterministic.jsonl"
    TARGETS_OUTPUT = phase2_dir / "targets_scored_deterministic.jsonl"

    # Load config
    config = load_arena_config()

    console.print("[bold cyan]Step 3a: Deterministic Judge Scoring[/bold cyan]")

    # Check input exists
    if not Path(INPUT_FILE).exists():
        console.print(f"❌ [red]Error: {INPUT_FILE} not found.[/red]")
        sys.exit(1)

    # Load log probabilities if available
    logprobs_by_prompt = {}
    if Path(LOGPROBS_FILE).exists():
        console.print(f"\n📄 Loading log probabilities from {LOGPROBS_FILE}")
        with open(LOGPROBS_FILE) as f:
            for line in f:
                data = json.loads(line)
                logprobs_by_prompt[data["prompt_id"]] = data["logprobs"]
        console.print(f"✅ Loaded log probs for {len(logprobs_by_prompt)} prompts")

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
                        "response": resp_data["response"],
                        "policy": policy_name,
                        "model": resp_data.get("model", "unknown"),
                        "temperature": resp_data.get("temperature", 0.5),
                    }
                )

    console.print(f"✅ Loaded {len(all_items)} total responses")

    # Count by policy
    console.print("\n📊 Responses by policy:")
    policy_counts = {}
    for item in all_items:
        policy = item["policy"]
        policy_counts[policy] = policy_counts.get(policy, 0) + 1

    for policy, count in sorted(policy_counts.items()):
        console.print(f"   {policy}: {count}")

    # Initialize judge
    console.print(f"\n⚖️  Initializing deterministic judge:")
    console.print(f"   Provider: {config.judge_config['provider']}")
    console.print(f"   Model: {config.judge_config['model_name']}")

    judge = JudgeFactory.create(
        provider=config.judge_config["provider"],
        model=config.judge_config["model_name"],
        template="deterministic",
        uncertainty_method="deterministic",
    )

    # Process with checkpointing
    checkpoint_mgr = CheckpointManager(
        checkpoint_path="data/checkpoint_deterministic.jsonl",
        get_uid_fn=lambda x: f"{x['prompt_id']}_{x['policy']}",
    )

    processor = BatchProcessor(checkpoint_manager=checkpoint_mgr, batch_size=10)

    console.print("\n🔄 Scoring responses...")
    scored_items = processor.process_batches(
        all_items,
        lambda batch: score_responses_batch(batch, judge),
        description="Scoring responses",
    )

    # Separate P0 and target responses
    p0_records = []
    target_records = []

    for item in scored_items:
        if item.get("judge_score") is None:
            continue

        # Get log probs for this prompt (only for P0)
        logprobs = None
        if item["policy"] == "p0":
            logprobs = logprobs_by_prompt.get(item["prompt_id"])

        # Create Phase 2 format record
        record = create_phase2_record(
            prompt_id=item["prompt_id"],
            prompt=item["prompt"],
            response=item["response"],
            policy=item["policy"],
            judge_score=item["judge_score"],
            model=item["model"],
            temperature=item["temperature"],
            logprobs=logprobs,
        )

        # Add to appropriate list
        if item["policy"] == "p0":
            p0_records.append(record)
        else:
            target_records.append(record)

    # Write output files
    console.print(f"\n💾 Writing output files...")

    with open(P0_OUTPUT, "w") as f:
        for record in p0_records:
            f.write(json.dumps(record) + "\n")
    console.print(f"✅ Wrote {len(p0_records)} P0 records to {P0_OUTPUT}")

    with open(TARGETS_OUTPUT, "w") as f:
        for record in target_records:
            f.write(json.dumps(record) + "\n")
    console.print(f"✅ Wrote {len(target_records)} target records to {TARGETS_OUTPUT}")

    # Print statistics
    console.print("\n📊 Score Statistics by Policy:")
    for policy in sorted(policy_counts.keys()):
        if policy == "p0":
            policy_scores = [r["judge_score"] for r in p0_records]
        else:
            policy_scores = [
                r["judge_score"] for r in target_records if r["policy"] == policy
            ]

        if policy_scores:
            import numpy as np

            console.print(
                f"   {policy}: mean={np.mean(policy_scores):.3f}, std={np.std(policy_scores):.3f}"
            )

    # Clean up checkpoint
    checkpoint_path = Path("data/checkpoint_deterministic.jsonl")
    if checkpoint_path.exists():
        console.print(f"\n🧹 Cleaning up checkpoint file")
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
