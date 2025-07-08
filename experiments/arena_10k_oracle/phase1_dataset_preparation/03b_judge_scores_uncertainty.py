#!/usr/bin/env python3
"""
Step 3b: Score all responses with uncertainty judge and write in Phase 2 format.

This script:
- Scores all responses (P0 and target policies) using judge with uncertainty
- Merges P0 scores with log probabilities for importance weighting
- Writes separate files for P0 and target policies in Phase 2 format
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.judge import JudgeFactory
from cje.utils.progress import console
from cje.utils import CheckpointManager, BatchProcessor
from config_loader import load_arena_config


def score_responses_batch(
    batch: List[Dict[str, Any]], judge, uncertainty_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Score a batch of responses with uncertainty."""
    results = []

    for item in batch:
        try:
            score = judge.score(item["prompt"], item["response"], **uncertainty_config)
            item["judge_score"] = score.mean
            item["judge_score_variance"] = score.variance
        except Exception as e:
            console.print(
                f"[red]Error scoring {item['prompt_id']}/{item['policy']}: {e}[/red]"
            )
            item["judge_score"] = None
            item["judge_score_variance"] = None
            item["error"] = str(e)

        results.append(item)

    return results


def create_phase2_record(
    prompt_id: str,
    prompt: str,
    response: str,
    policy: str,
    judge_score: float,
    judge_score_variance: float,
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
        "judge_score_variance": judge_score_variance,
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

    P0_OUTPUT = phase2_dir / "p0_scored_uncertainty.jsonl"
    TARGETS_OUTPUT = phase2_dir / "targets_scored_uncertainty.jsonl"

    # Load config
    config = load_arena_config()

    console.print("[bold cyan]Step 3b: Judge Scoring with Uncertainty[/bold cyan]")

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

    # Initialize judge with uncertainty
    uncertainty_method = config.judge_config.get(
        "uncertainty_method", "confidence_interval"
    )
    uncertainty_config = config.judge_config.get("uncertainty_config", {})
    temperature = uncertainty_config.get("temperature", 0.3)

    console.print(f"\n⚖️  Initializing judge with uncertainty:")
    console.print(f"   Provider: {config.judge_config['provider']}")
    console.print(f"   Model: {config.judge_config['model_name']}")
    console.print(f"   Method: {uncertainty_method}")
    console.print(f"   Temperature: {temperature}")

    if uncertainty_method == "confidence_interval" and temperature != 0:
        console.print(
            f"[yellow]CI Judge works best with temperature=0, not {temperature}[/yellow]"
        )

    judge = JudgeFactory.create(
        provider=config.judge_config["provider"],
        model=config.judge_config["model_name"],
        template="deterministic",
        uncertainty_method=uncertainty_method,
    )

    # Process with checkpointing
    checkpoint_mgr = CheckpointManager(
        checkpoint_path="data/checkpoint_uncertainty.jsonl",
        get_uid_fn=lambda x: f"{x['prompt_id']}_{x['policy']}",
    )

    processor = BatchProcessor(checkpoint_manager=checkpoint_mgr, batch_size=5)

    console.print("\n🔄 Scoring responses with uncertainty...")
    console.print("   Note: This is slower due to multiple samples per response")

    scored_items = processor.process_batches(
        all_items,
        lambda batch: score_responses_batch(batch, judge, uncertainty_config),
        description="Scoring with uncertainty",
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
            judge_score_variance=item["judge_score_variance"],
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
            policy_records = p0_records
        else:
            policy_records = [r for r in target_records if r["policy"] == policy]

        if policy_records:
            scores = [r["judge_score"] for r in policy_records]
            variances = [r["judge_score_variance"] for r in policy_records]
            console.print(
                f"   {policy}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}, "
                f"avg_variance={np.mean(variances):.4f}"
            )

    # Uncertainty analysis
    all_records = p0_records + target_records
    all_variances = [r["judge_score_variance"] for r in all_records]

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
