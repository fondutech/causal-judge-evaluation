#!/usr/bin/env python3
"""
Phase 1 - Step 2c: Compute target policy log probabilities using robust teacher forcing.

This script computes log P(p0_response | context, target_policy) for each
target policy using the robust teacher forcing implementation that correctly
handles tokenization boundaries.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from rich.console import Console
from rich.progress import track
from rich.table import Table
import numpy as np

console = Console()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils import RobustTeacherForcing
from cje.utils.checkpointing import CheckpointManager, BatchProcessor


def load_p0_data(p0_file: Path) -> List[Dict[str, Any]]:
    """Load P0 responses that need teacher forcing."""
    p0_data = []
    with open(p0_file) as f:
        for line in f:
            data = json.loads(line)
            p0_data.append(data)

    console.print(f"📊 Loaded {len(p0_data)} P0 responses")
    return p0_data


def initialize_target_policies() -> Dict[str, RobustTeacherForcing]:
    """Initialize teacher forcing for each target policy."""
    policies = {
        "pi_clone": "accounts/fireworks/models/llama4-scout-instruct-basic",
        "pi_cot": "accounts/fireworks/models/llama4-scout-instruct-basic",
        "pi_bigger_model": "accounts/fireworks/models/llama4-maverick-instruct-basic",
        "pi_bad": "accounts/fireworks/models/llama4-scout-instruct-basic",
    }

    console.print("\n🤖 Initializing target policies with robust teacher forcing:")

    tf_instances = {}
    for name, model in policies.items():
        try:
            tf = RobustTeacherForcing(provider="fireworks", model=model)
            tf_instances[name] = tf
            console.print(f"  ✅ {name}: {model}")
        except Exception as e:
            console.print(f"  ❌ {name}: Failed to initialize - {e}")
            raise

    return tf_instances


def compute_target_logprobs_batch(
    batch: List[Dict[str, Any]], tf_instances: Dict[str, RobustTeacherForcing]
) -> List[Dict[str, Any]]:
    """Compute log probabilities for a batch of samples."""
    results: List[Dict[str, Any]] = []

    for item in batch:
        try:
            prompt_id = item["prompt_id"]
            context = item["context"]
            response = item["response"]

            # Compute log probs for each target policy
            target_logps = {}

            for policy_name, tf in tf_instances.items():
                try:
                    # Use robust teacher forcing
                    result = tf.compute_log_prob(context, response)

                    if result.is_valid:
                        logp = result.value

                        # Additional validation
                        if logp > 0:
                            raise ValueError(f"Positive log prob: {logp}")
                        if logp == 0.0 and response:  # Empty response should be 0.0
                            console.print(
                                f"[yellow]Warning: Got 0.0 for non-empty response "
                                f"(policy: {policy_name}, prompt: {prompt_id})[/yellow]"
                            )
                            console.print(
                                f"[yellow]Response: '{response[:100]}...'[/yellow]"
                            )
                            console.print(
                                f"[yellow]Method: {result.metadata.get('method')}[/yellow]"
                            )

                        target_logps[policy_name] = float(logp)
                    else:
                        console.print(
                            f"[red]Teacher forcing failed for {policy_name} on {prompt_id}: "
                            f"{result.error}[/red]"
                        )
                        target_logps[policy_name] = None

                except Exception as e:
                    console.print(
                        f"[red]Error computing {policy_name} logp for {prompt_id}: {e}[/red]"
                    )
                    target_logps[policy_name] = None

            # Update item with target log probs
            item["target_logps"] = target_logps
            results.append(item)

        except Exception as e:
            console.print(
                f"[red]Error processing {item.get('prompt_id', 'unknown')}: {e}[/red]"
            )
            # Skip failed items rather than including them
            console.print(f"[red]Skipping item due to error[/red]")

    return results


def validate_results(data: List[Dict[str, Any]]) -> None:
    """Validate the computed log probabilities."""
    console.print("\n📊 Validation Results:")

    # Count samples with issues
    total = len(data)
    has_logps = sum(1 for d in data if "target_logps" in d)

    # Count by policy
    policy_stats = {}
    zero_count = 0

    for item in data:
        if "target_logps" not in item:
            continue

        for policy, logp in item["target_logps"].items():
            if policy not in policy_stats:
                policy_stats[policy] = {"valid": 0, "null": 0, "zero": 0}

            if logp is None:
                policy_stats[policy]["null"] += 1
            else:
                policy_stats[policy]["valid"] += 1
                if logp == 0.0 and item.get("response"):
                    policy_stats[policy]["zero"] += 1
                    zero_count += 1

    # Display stats
    table = Table(title="Log Probability Statistics")
    table.add_column("Policy", style="cyan")
    table.add_column("Valid", style="green")
    table.add_column("Failed", style="red")
    table.add_column("Zero Values", style="yellow")

    for policy, stats in sorted(policy_stats.items()):
        table.add_row(
            policy, str(stats["valid"]), str(stats["null"]), str(stats["zero"])
        )

    console.print(table)

    # Show sample of computed values
    console.print("\n📈 Sample Log Probabilities:")
    sample_size = min(5, len(data))
    for item in data[:sample_size]:
        if "target_logps" in item:
            console.print(f"  Prompt {item['prompt_id']}: {item['target_logps']}")

    # Warnings
    if zero_count > 0:
        console.print(
            f"\n[yellow]⚠️  Found {zero_count} samples with 0.0 log probabilities "
            f"for non-empty responses[/yellow]"
        )
        console.print(
            "[yellow]These should be investigated to ensure they are genuine.[/yellow]"
        )


def main() -> None:
    """Main execution function."""
    # Setup paths
    data_dir = Path(__file__).parent.parent / "data"
    p0_file = data_dir / "p0_replies.jsonl"
    output_file = data_dir / "p0_with_target_logps.jsonl"

    # Check if input exists
    if not p0_file.exists():
        console.print(f"[red]Error: {p0_file} not found. Run 02a first.[/red]")
        return

    # Load data
    p0_data = load_p0_data(p0_file)

    # Initialize robust teacher forcing
    console.print("\n🚀 Using Robust Teacher Forcing Implementation")
    console.print("This correctly handles tokenization boundaries and edge cases.")
    tf_instances = initialize_target_policies()

    # Show statistics
    console.print(f"\n📊 Will compute log probs for {len(tf_instances)} policies")

    # Process with checkpointing
    console.print("\n🔄 Computing target log probabilities...")

    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(
        checkpoint_path=str(output_file.with_suffix(".checkpoint.jsonl")),
        get_uid_fn=lambda x: x["prompt_id"],
    )

    # Process data
    processor = BatchProcessor(
        checkpoint_manager=checkpoint_mgr,
        batch_size=10,
    )

    results = processor.process_batches(
        p0_data,
        lambda batch: compute_target_logprobs_batch(batch, tf_instances),
        description="Computing log probs",
    )

    # Validate results
    validate_results(results)

    # Show teacher forcing statistics
    console.print("\n📊 Teacher Forcing Method Usage:")
    for policy_name, tf in tf_instances.items():
        stats = tf.get_stats()
        console.print(f"\n{policy_name}:")
        console.print(f"  Total calls: {stats['total_calls']}")
        console.print(f"  Method successes: {stats['method_successes']}")
        console.print(f"  Zero values: {stats['zero_values']}")

    console.print(f"\n✅ Saved results to {output_file}")
    console.print("\n🎯 Next step: Run 03_generate_oracle_labels.py")


if __name__ == "__main__":
    main()
