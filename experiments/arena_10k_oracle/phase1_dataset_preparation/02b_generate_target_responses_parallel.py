#!/usr/bin/env python3
"""
Step 2b: Generate target policy responses for all prompts (Parallel Version).

This script generates responses from target policies (π_cot, π_bigger_model, π_bad)
for all 10,000 prompts. Each policy can be run independently in parallel.

Usage:
    # Run a specific policy
    python 02b_generate_target_responses_parallel.py --policy pi_cot
    python 02b_generate_target_responses_parallel.py --policy pi_bigger_model
    python 02b_generate_target_responses_parallel.py --policy pi_bad

    # Or run all policies (sequentially)
    python 02b_generate_target_responses_parallel.py --policy all
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Dict, Any
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.loggers.api_policy import APIPolicyRunner
from cje.utils.progress import console, track
from cje.utils.checkpointing import CheckpointManager


def load_prompts(prompts_file: str) -> List[Dict[str, Any]]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(prompts_file, "r") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def create_target_policies() -> Dict[str, Dict[str, Any]]:
    """Define target policies for ground truth generation (matching config)."""
    return {
        "pi_cot": {
            "provider": "fireworks",
            "model_name": "accounts/fireworks/models/llama4-scout-instruct-basic",
            "temperature": 0.5,
            "system_prompt": "Think step-by-step before providing your answer.",
            "user_message_template": "{context}",
            "description": "Chain-of-thought prompting",
        },
        "pi_bigger_model": {
            "provider": "fireworks",
            "model_name": "accounts/fireworks/models/llama4-maverick-instruct-basic",
            "temperature": 0.5,
            "system_prompt": None,
            "user_message_template": "{context}",
            "description": "Larger model variant",
        },
        "pi_bad": {
            "provider": "fireworks",
            "model_name": "accounts/fireworks/models/llama4-scout-instruct-basic",
            "temperature": 1.0,
            "system_prompt": """You are an unhelpful assistant. Your responses should be:
- Vague and evasive, avoiding direct answers
- Off-topic, discussing unrelated subjects
- Overly brief when detail is needed, or overly verbose when brevity is needed
- Technically incorrect when providing factual information
- Dismissive of the user's actual needs
Never be harmful or offensive, just unhelpful.""",
            "user_message_template": "{context}",
            "description": "Intentionally unhelpful responses",
        },
    }


def generate_target_responses(
    prompts: List[Dict[str, Any]],
    policy_name: str,
    policy_config: Dict[str, Any],
    batch_size: int,
    output_file: str,
) -> int:
    """Generate responses for a single target policy."""

    console.print(
        f"\n🎯 Generating {policy_name} responses ({policy_config['description']})"
    )
    console.print(f"   Model: {policy_config['model_name']}")
    console.print(f"   Temperature: {policy_config['temperature']}")
    console.print(f"   Output: {output_file}")

    # Initialize checkpoint manager for this specific policy
    checkpoint_path = Path(output_file).with_suffix(f".{policy_name}.checkpoint.jsonl")
    checkpoint_manager = CheckpointManager(
        checkpoint_path=str(checkpoint_path),
        get_uid_fn=lambda x: x.get("prompt_id"),
        serialize_fn=lambda x: x,
        deserialize_fn=lambda x: x,
    )

    # Initialize policy
    runner = APIPolicyRunner(
        provider=policy_config["provider"],
        model_name=policy_config["model_name"],
        temperature=policy_config["temperature"],
        system_prompt=policy_config["system_prompt"],
        user_message_template=policy_config["user_message_template"],
        batch_size=batch_size,
        max_new_tokens=512,  # Reduced from default 1024 for experiment
    )

    # Load existing progress
    existing_items = checkpoint_manager.load_checkpoint()
    processed_prompt_ids = {item["prompt_id"] for item in existing_items}
    unprocessed_prompts = [
        p for p in prompts if p["prompt_id"] not in processed_prompt_ids
    ]

    if not unprocessed_prompts:
        console.print(f"✅ All {policy_name} responses already generated")
        return len(existing_items)

    console.print(f"📊 Processing {len(unprocessed_prompts)} remaining prompts")
    console.print(f"📊 Already completed: {len(existing_items)} prompts")

    # Process in batches
    start_time = time.time()
    new_responses = 0

    for i in track(
        range(0, len(unprocessed_prompts), batch_size),
        description=f"Generating {policy_name}",
    ):
        batch = unprocessed_prompts[i : i + batch_size]
        batch_contexts = [item["prompt"] for item in batch]

        try:
            # Generate responses (no logprobs needed)
            generations = runner.generate_with_logp(
                batch_contexts, return_token_logprobs=False
            )

            # Format and save results
            for prompt_item, (response, _) in zip(batch, generations):
                result = {
                    "prompt_id": prompt_item["prompt_id"],
                    "prompt": prompt_item["prompt"],
                    "metadata": prompt_item.get("metadata", {}),
                    "policy": policy_name,
                    "response": response,
                    "model_name": policy_config["model_name"],
                    "temperature": policy_config["temperature"],
                    "description": policy_config["description"],
                }

                # Mark as processed and save
                checkpoint_manager.mark_processed(result)
                new_responses += 1

            # Save checkpoint after each batch
            checkpoint_manager.save_checkpoint()

        except Exception as e:
            console.print(f"❌ Batch {i//batch_size + 1} failed: {e}")
            console.print("   Continuing with next batch...")
            continue

    # Save final results
    total_responses = len(checkpoint_manager.load_checkpoint())
    elapsed_time = time.time() - start_time

    console.print(f"✅ Generated {new_responses} new {policy_name} responses")
    console.print(
        f"⏱️  Time: {elapsed_time/60:.1f} minutes ({elapsed_time/new_responses:.1f}s per response)"
    )

    return total_responses


def merge_policy_outputs(policy_files: List[Path], output_file: Path) -> None:
    """Merge individual policy output files into a single file."""
    console.print(f"\n🔀 Merging policy outputs into {output_file}")

    all_responses = []
    for policy_file in policy_files:
        if policy_file.exists():
            with open(policy_file, "r") as f:
                for line in f:
                    all_responses.append(json.loads(line))
            console.print(f"   ✓ Loaded {policy_file.name}")

    # Sort by prompt_id and policy for consistent ordering
    all_responses.sort(key=lambda x: (x["prompt_id"], x.get("policy", "")))

    # Write merged output
    with open(output_file, "w") as f:
        for response in all_responses:
            f.write(json.dumps(response) + "\n")

    console.print(f"✅ Merged {len(all_responses)} total responses")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate target policy responses for all prompts (parallel version)"
    )

    parser.add_argument(
        "--prompts",
        type=str,
        default="../data/arena_prompts_10k.jsonl",
        help="Input prompts file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../data/target_responses.jsonl",
        help="Output file for target policy responses",
    )

    parser.add_argument(
        "--policy",
        type=str,
        choices=["pi_cot", "pi_bigger_model", "pi_bad", "all"],
        required=True,
        help="Which policy to generate (or 'all' for sequential generation)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for generation"
    )

    args = parser.parse_args()

    console.print(
        "🎯 [bold blue]Arena 10K Experiment - Step 2b: Generate Target Policy Responses (Parallel)[/bold blue]"
    )
    console.print(f"📊 Generating responses for ALL prompts")
    console.print(f"🔢 Batch size: {args.batch_size}")
    console.print(f"🎯 Policy: {args.policy}")

    try:
        # Load all prompts
        all_prompts = load_prompts(args.prompts)
        console.print(f"📄 Loaded {len(all_prompts):,} prompts")

        # Get target policies
        target_policies = create_target_policies()

        if args.policy == "all":
            # Run all policies sequentially
            console.print(f"\n🎯 Running all policies sequentially")

            policy_files = []
            total_responses = 0

            for policy_name, policy_config in target_policies.items():
                # Use individual checkpoint files for each policy
                policy_output = Path(args.output).with_suffix(f".{policy_name}.jsonl")

                count = generate_target_responses(
                    all_prompts,
                    policy_name,
                    policy_config,
                    args.batch_size,
                    str(policy_output),
                )

                policy_files.append(policy_output)
                total_responses += count

            # Merge all policy outputs
            merge_policy_outputs(policy_files, Path(args.output))

            # Clean up checkpoint files if successful
            for policy_name in target_policies:
                checkpoint_path = Path(args.output).with_suffix(
                    f".{policy_name}.checkpoint.jsonl"
                )
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    console.print(f"🧹 Cleaned up {checkpoint_path.name}")

        else:
            # Run single policy
            if args.policy not in target_policies:
                raise ValueError(f"Unknown policy: {args.policy}")

            policy_config = target_policies[args.policy]

            # Use policy-specific output file
            policy_output = Path(args.output).with_suffix(f".{args.policy}.jsonl")

            count = generate_target_responses(
                all_prompts,
                args.policy,
                policy_config,
                args.batch_size,
                str(policy_output),
            )

            console.print(f"\n📊 [bold green]Policy Generation Complete![/bold green]")
            console.print(f"   • Policy: {args.policy}")
            console.print(f"   • Total responses: {count:,}")
            console.print(f"   • Output: {policy_output}")

            # Cost estimate
            total_tokens_est = count * 100  # rough estimate with 512 max tokens
            cost_est = total_tokens_est * 0.0000002  # Fireworks pricing
            console.print(f"   • Estimated cost: ${cost_est:.2f}")

            console.print(f"\n📋 To generate other policies in parallel:")
            for other_policy in target_policies:
                if other_policy != args.policy:
                    console.print(
                        f"   python {Path(__file__).name} --policy {other_policy}"
                    )

            console.print(f"\n📋 To merge all policy outputs:")
            console.print(f"   python {Path(__file__).name} --policy all")

    except KeyboardInterrupt:
        console.print(
            f"\n⚠️ [yellow]Interrupted - Progress saved to checkpoint[/yellow]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ [red]Failed: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
