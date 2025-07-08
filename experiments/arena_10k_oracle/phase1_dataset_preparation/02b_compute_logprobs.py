#!/usr/bin/env python3
"""
Step 2b: Compute log probabilities for P0 responses under all policies.

This script computes log P(P0_response|prompt, policy) for each policy
using robust teacher forcing. Critical for importance weighting.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils import RobustTeacherForcing, CheckpointManager, BatchProcessor
from cje.utils.progress import console
from cje.types.results import LogProbResult, LogProbStatus
from config_loader import load_arena_config


def compute_logprobs_batch(
    batch: List[Dict[str, Any]],
    teacher_forcing_instances: Dict[str, RobustTeacherForcing],
    policy_configs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute log probabilities for a batch of items."""
    results = []

    for item in batch:
        prompt_id = item["prompt_id"]
        prompt = item["prompt"]
        p0_response = item["p0_response"]

        result = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "p0_response": p0_response,
            "logprobs": {},
        }

        # Compute P0 response log prob under each policy
        for policy_name, tf_instance in teacher_forcing_instances.items():
            logprob_result = tf_instance.compute_log_prob(prompt, p0_response)

            if logprob_result.status == LogProbStatus.SUCCESS:
                result["logprobs"][policy_name] = logprob_result.value
            else:
                result["logprobs"][policy_name] = None
                console.print(
                    f"[yellow]Warning: Failed to compute {policy_name} log prob "
                    f"for prompt {prompt_id}: {logprob_result.error}[/yellow]"
                )

        results.append(result)

    return results


def main():
    # No arguments - fixed paths from convention
    INPUT_FILE = "data/all_responses.jsonl"
    OUTPUT_FILE = "data/logprobs.jsonl"

    console.print("[bold cyan]Step 2b: Compute Log Probabilities[/bold cyan]")

    # Check API key
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        console.print("❌ [red]Error: FIREWORKS_API_KEY not set![/red]")
        sys.exit(1)

    # Check input exists
    if not Path(INPUT_FILE).exists():
        console.print(
            f"❌ [red]Error: {INPUT_FILE} not found. Run 02_generate_responses.py first.[/red]"
        )
        sys.exit(1)

    # Load config
    config = load_arena_config()

    # Build full policy configurations
    policy_configs = {"p0": config.logging_policy}
    for policy in config.target_policies:
        policy_configs[policy["name"]] = policy

    # Load all responses
    console.print(f"\n📄 Loading all responses from {INPUT_FILE}")
    all_responses_data = []
    with open(INPUT_FILE) as f:
        for line in f:
            all_responses_data.append(json.loads(line))
    console.print(f"✅ Loaded {len(all_responses_data)} prompt entries")

    # Extract P0 responses for processing
    p0_data = []
    for entry in all_responses_data:
        if "p0" in entry["responses"]:
            p0_resp = entry["responses"]["p0"]
            p0_data.append(
                {
                    "prompt_id": entry["prompt_id"],
                    "prompt": entry["prompt"],
                    "p0_response": p0_resp["response"],
                    "metadata": entry.get("metadata", {}),
                }
            )

    console.print(f"📊 Processing {len(p0_data)} P0 responses")

    # Initialize teacher forcing for each policy with full config
    console.print("\n🤖 Initializing teacher forcing for all policies:")
    tf_instances = {}
    for policy_name, policy_config in policy_configs.items():
        # Extract relevant parameters
        model_name = policy_config["model_name"]
        temperature = policy_config.get("temperature", 0.5)
        system_prompt = policy_config.get("system_prompt", None)

        tf_instances[policy_name] = RobustTeacherForcing(
            provider="fireworks",
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        console.print(f"  ✅ {policy_name}: {model_name} (temp={temperature})")
        if system_prompt:
            console.print(f"     └─ System prompt: {system_prompt[:50]}...")

    # Process with checkpointing
    console.print(f"\n🔄 Computing log probabilities...")

    checkpoint_mgr = CheckpointManager(
        checkpoint_path="data/checkpoint_logprobs.jsonl",
        get_uid_fn=lambda x: x["prompt_id"],
    )

    processor = BatchProcessor(
        checkpoint_manager=checkpoint_mgr,
        batch_size=10,
    )

    results = processor.process_batches(
        p0_data,
        lambda batch: compute_logprobs_batch(batch, tf_instances, policy_configs),
        description="Computing log probs",
    )

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Validate results
    stats_by_policy = defaultdict(
        lambda: {"valid": 0, "null": 0, "zero": 0, "logprobs": []}
    )
    zero_samples = []

    for item in results:
        prompt_id = item["prompt_id"]
        p0_response = item.get("p0_response", "")

        for policy, logprob in item["logprobs"].items():
            if logprob is not None:
                stats_by_policy[policy]["valid"] += 1
                stats_by_policy[policy]["logprobs"].append(logprob)
                if logprob == 0.0 and len(p0_response.strip()) > 0:
                    stats_by_policy[policy]["zero"] += 1
                    zero_samples.append((prompt_id, policy, p0_response[:50]))
            else:
                stats_by_policy[policy]["null"] += 1

    # Print statistics
    console.print("\n📊 Log Probability Statistics:")
    for policy in sorted(stats_by_policy.keys()):
        stats = stats_by_policy[policy]
        if stats["logprobs"]:
            import numpy as np

            mean = np.mean(stats["logprobs"])
            std = np.std(stats["logprobs"])
            console.print(
                f"  {policy}: valid={stats['valid']}, failed={stats['null']}, "
                f"zero={stats['zero']}, mean={mean:.2f}, std={std:.2f}"
            )

    # Warn about zero values
    if zero_samples:
        console.print(
            f"\n[yellow]⚠️  Found {len(zero_samples)} samples with 0.0 log probs "
            f"for non-empty responses[/yellow]"
        )
        console.print("[yellow]First 3 examples:[/yellow]")
        for prompt_id, policy, response_preview in zero_samples[:3]:
            console.print(f"  - {prompt_id} ({policy}): {response_preview}...")

    # Show teacher forcing statistics
    console.print("\n📊 Teacher Forcing Method Usage:")
    for policy_name, tf in tf_instances.items():
        stats = tf.get_stats()
        if stats["total_calls"] > 0:
            console.print(f"  {policy_name}: {stats}")

    console.print(f"\n✅ [bold green]Saved results to {OUTPUT_FILE}[/bold green]")

    # Clean up checkpoint
    checkpoint_path = Path("data/checkpoint_logprobs.jsonl")
    if checkpoint_path.exists():
        console.print(f"🧹 Cleaning up checkpoint file")
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
