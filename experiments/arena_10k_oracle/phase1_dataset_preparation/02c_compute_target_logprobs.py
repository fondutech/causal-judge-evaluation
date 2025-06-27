#!/usr/bin/env python3
"""
Phase 1 - Step 2c: Compute target policy log probabilities for P0 responses.

This critical step computes log P(p0_response | context, target_policy) for each
target policy. These log probabilities are essential for importance weighting.

Without this step, all importance weights default to 1.0, making all policies
appear identical in the ablation analysis.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from rich.console import Console
from rich.progress import track
from rich.table import Table
import numpy as np

console = Console()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.loggers.api_policy import APIPolicyRunner
from cje.loggers.multi_target_sampler import MultiTargetSampler
from cje.utils.checkpointing import CheckpointManager, BatchProcessor


def load_p0_data(p0_file: Path) -> List[Dict[str, Any]]:
    """Load P0 responses that need teacher forcing."""
    p0_data = []
    with open(p0_file) as f:
        for line in f:
            data = json.loads(line)
            p0_data.append(data)

    console.print(f"📊 Loaded {len(p0_data)} P0 responses")

    # Check if we already have log probs
    sample = p0_data[0] if p0_data else {}
    has_logp = "logp" in sample

    if has_logp:
        logp_range = [
            min(d.get("logp", 0) for d in p0_data),
            max(d.get("logp", 0) for d in p0_data),
        ]
        console.print(
            f"✅ P0 log probs present: range [{logp_range[0]:.2f}, {logp_range[1]:.2f}]"
        )
    else:
        console.print(
            "[yellow]⚠️  P0 log probs not found - will need to compute[/yellow]"
        )

    return p0_data


def create_target_policy_runners(
    target_policies: Dict[str, Dict[str, Any]],
) -> Dict[str, APIPolicyRunner]:
    """Create API runners for each target policy."""
    runners = {}

    for policy_name, config in target_policies.items():
        console.print(f"🔧 Creating runner for {policy_name}: {config['model_name']}")

        runner = APIPolicyRunner(
            provider=config["provider"],
            model_name=config["model_name"],
            temperature=0.0,  # Always use temperature=0 for teacher forcing
            system_prompt=config.get("system_prompt"),
            user_message_template=config.get("user_message_template", "{context}"),
            batch_size=1,  # Teacher forcing is done one at a time
        )

        runners[policy_name] = runner

    return runners


def compute_target_logprobs_batch(
    batch: List[Dict[str, Any]], runners: Dict[str, APIPolicyRunner]
) -> List[Dict[str, Any]]:
    """Compute target policy log probs for a batch of P0 responses.

    CRITICAL: Teacher forcing requires using the EXACT input sequence that was
    fed to the logging policy P0. In our case, P0 was run with:
    - No system prompt (system_prompt=None)
    - Default user template: "{context}"
    - Template format: "llama4"

    Therefore we use the raw prompt/context directly.
    """
    results = []

    for item in batch:
        context = item["prompt"]
        response = item["response"]
        prompt_id = item["prompt_id"]

        # Compute log prob for each target policy
        target_logps = {}

        for policy_name, runner in runners.items():
            try:
                # Teacher force the P0 response through target policy
                # IMPORTANT: We use the raw context here because P0 was generated
                # without any system prompt or special formatting (just {context})
                logp = runner.log_prob(context, response)

                # Validate the result
                if not isinstance(logp, (int, float)):
                    raise ValueError(f"Invalid log prob type: {type(logp)}")
                if logp > 0:
                    raise ValueError(f"Positive log prob: {logp}")
                if logp == 0.0:
                    raise ValueError("Exactly 0.0 log prob (suspicious)")

                target_logps[policy_name] = float(logp)
            except Exception as e:
                console.print(
                    f"[red]Error computing {policy_name} logp for {prompt_id}: {e}[/red]"
                )
                # Mark as None - NEVER use fake values
                target_logps[policy_name] = None

        # Create result with all data
        result = item.copy()
        result["target_logps"] = target_logps
        results.append(result)

    return results


def main():
    """Compute target policy log probabilities for P0 responses."""

    console.print(
        "[bold blue]🔬 Phase 1 - Step 2c: Computing Target Policy Log Probabilities[/bold blue]"
    )
    console.print("This step teacher-forces P0 responses through target policies\n")

    # Define target policies (same as in 02b)
    target_policies = {
        "pi_cot": {
            "provider": "fireworks",
            "model_name": "accounts/fireworks/models/llama4-scout-instruct-basic",
            "temperature": 0.5,
            "system_prompt": "You are a helpful assistant. Always think step by step before answering.",
            "user_message_template": "Let's work through this step by step.\n\n{context}\n\nLet me think about this carefully:",
            "description": "Chain-of-thought reasoning policy",
        },
        "pi_bigger_model": {
            "provider": "fireworks",
            "model_name": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "temperature": 0.5,
            "system_prompt": None,
            "user_message_template": "{context}",
            "description": "Larger model (70B vs 8B baseline)",
        },
        "pi_bad": {
            "provider": "fireworks",
            "model_name": "accounts/fireworks/models/llama4-scout-instruct-basic",
            "temperature": 1.5,
            "system_prompt": "You are learning to be helpful. Keep responses brief.",
            "user_message_template": "{context}\n\n(Reply briefly):",
            "description": "High temperature + brevity constraints",
        },
    }

    # Load P0 data
    p0_file = Path("../data/p0_replies.jsonl")
    if not p0_file.exists():
        console.print(f"[red]❌ P0 file not found: {p0_file}[/red]")
        return

    p0_data = load_p0_data(p0_file)

    # Create runners
    runners = create_target_policy_runners(target_policies)

    # Set up checkpointing
    output_file = Path("../data/p0_with_target_logps.jsonl")
    checkpoint_path = output_file.with_suffix(".checkpoint.jsonl")

    checkpoint_mgr = CheckpointManager(
        checkpoint_path=str(checkpoint_path), get_uid_fn=lambda x: x["prompt_id"]
    )

    # Process in batches
    batch_size = 16  # Small batches for teacher forcing
    processor = BatchProcessor(batch_size=batch_size, checkpoint_manager=checkpoint_mgr)

    console.print(f"\n🚀 Processing {len(p0_data)} P0 responses...")
    console.print(f"   Batch size: {batch_size}")
    console.print(f"   Target policies: {list(target_policies.keys())}")

    start_time = time.time()

    # Process with progress tracking
    results = processor.process_batches(
        p0_data,
        lambda batch: compute_target_logprobs_batch(batch, runners),
        description="Computing target log probs",
    )

    elapsed = time.time() - start_time

    # Save final results
    console.print(f"\n💾 Saving {len(results)} results to {output_file}")
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Display statistics
    console.print(f"\n✅ Complete! Processed {len(results)} items in {elapsed:.1f}s")
    console.print(f"   Rate: {len(results)/elapsed:.1f} items/second")

    # Show sample log prob ranges
    if results:
        table = Table(title="Target Policy Log Probability Ranges")
        table.add_column("Policy", style="cyan")
        table.add_column("Min", justify="right")
        table.add_column("Mean", justify="right")
        table.add_column("Max", justify="right")

        for policy in target_policies:
            logps = [
                r["target_logps"][policy]
                for r in results
                if policy in r.get("target_logps", {})
            ]
            if logps:
                table.add_row(
                    policy,
                    f"{min(logps):.2f}",
                    f"{np.mean(logps):.2f}",
                    f"{max(logps):.2f}",
                )

        console.print("\n")
        console.print(table)

    console.print("\n📊 Next steps:")
    console.print("1. This data can now be used for importance weighting")
    console.print("2. Run ablation analysis with proper log probabilities")
    console.print("3. Compare estimator performance with real importance weights")


if __name__ == "__main__":
    main()
