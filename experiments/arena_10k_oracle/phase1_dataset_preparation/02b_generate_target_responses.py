#!/usr/bin/env python3
"""
Step 2b: Generate target policy responses for ground truth validation.

This script generates responses from target policies (π_cot, π_bigger_model, etc.)
for human labeling. These responses establish the ground truth that CJE aims to predict.

Note: No teacher forcing needed - we only need the response text for human evaluation.

Usage:
    python 02b_generate_target_ground_truth.py --prompts ../data/arena_prompts_10k.jsonl --output ../data/target_ground_truth.jsonl
"""

import argparse
import json
import random
from pathlib import Path
import sys
from typing import List, Dict, Any

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


def sample_ground_truth_prompts(
    prompts: List[Dict[str, Any]], n_samples: int, seed: int = 42
) -> List[Dict[str, Any]]:
    """Sample prompts for ground truth validation."""
    random.seed(seed)
    if len(prompts) < n_samples:
        console.print(f"⚠️  Only {len(prompts)} prompts available, using all")
        return prompts

    sampled = random.sample(prompts, n_samples)
    console.print(f"🎲 Sampled {len(sampled)} prompts for ground truth validation")
    return sampled


def create_target_policies() -> Dict[str, Dict[str, Any]]:
    """Define target policies for ground truth generation (matching config).

    Note: pi_clone is excluded as it doesn't need human labels - CJE validates
    it should return neutral results since it's identical to π₀.
    """
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
    checkpoint_manager: CheckpointManager,
) -> List[Dict[str, Any]]:
    """Generate responses for a single target policy."""

    console.print(
        f"\n🎯 Generating {policy_name} responses ({policy_config['description']})"
    )
    console.print(f"   Model: {policy_config['model_name']}")
    console.print(f"   Temperature: {policy_config['temperature']}")

    # Initialize policy
    runner = APIPolicyRunner(
        provider=policy_config["provider"],
        model_name=policy_config["model_name"],
        temperature=policy_config["temperature"],
        system_prompt=policy_config["system_prompt"],
        user_message_template=policy_config["user_message_template"],
        batch_size=batch_size,
        max_new_tokens=512,  # Reduced from default 1024 for experiment
        # No completions template needed - just generation
    )

    # Load existing progress
    all_existing_items = checkpoint_manager.load_checkpoint()

    # Filter to only this policy's existing items
    existing_items = [
        item for item in all_existing_items if item.get("policy") == policy_name
    ]
    results = existing_items.copy()

    # For unprocessed check, we need to see which prompts have been done for THIS policy
    processed_prompt_ids = {item["prompt_id"] for item in existing_items}
    unprocessed_prompts = [
        p for p in prompts if p["prompt_id"] not in processed_prompt_ids
    ]

    if not unprocessed_prompts:
        console.print(f"✅ All {policy_name} responses already generated")
        return results

    console.print(f"📊 Processing {len(unprocessed_prompts)} remaining prompts")

    # Process in batches
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

            # Format results
            batch_results = []
            for j, (prompt_item, (response, _)) in enumerate(zip(batch, generations)):
                result = {
                    "prompt_id": prompt_item["prompt_id"],
                    "prompt": prompt_item["prompt"],
                    "metadata": prompt_item.get(
                        "metadata", {}
                    ),  # Keep only minimal metadata
                    "policy": policy_name,
                    "response": response,
                    "model_name": policy_config["model_name"],
                    "temperature": policy_config["temperature"],
                    "description": policy_config["description"],
                }
                batch_results.append(result)
                results.append(result)

                # Mark as processed
                checkpoint_manager.mark_processed(result)

            # Save checkpoint after each batch
            checkpoint_manager.save_checkpoint()

        except Exception as e:
            console.print(f"❌ Batch {i//batch_size + 1} failed: {e}")
            console.print("   Continuing with next batch...")
            continue

    console.print(
        f"✅ Generated {len(results) - len(existing_items)} new {policy_name} responses (total: {len(results)})"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate target policy responses for ground truth validation"
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
        default="../data/target_ground_truth.jsonl",
        help="Output file for target policy responses",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of prompts to sample for ground truth",
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for generation"
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    args = parser.parse_args()

    console.print(
        "🎯 [bold blue]Arena 10K Experiment - Step 2b: Generate Target Ground Truth[/bold blue]"
    )
    console.print(f"📊 Samples: {args.samples}")
    console.print(f"🔢 Batch size: {args.batch_size}")

    try:
        # Load and sample prompts
        all_prompts = load_prompts(args.prompts)
        ground_truth_prompts = sample_ground_truth_prompts(
            all_prompts, args.samples, args.seed
        )

        # Get target policies
        target_policies = create_target_policies()
        console.print(f"\n🎯 Target policies: {list(target_policies.keys())}")

        # Initialize checkpoint manager with composite key for prompt_id and policy
        checkpoint_manager = CheckpointManager(
            checkpoint_path=args.output,
            get_uid_fn=lambda x: f"{x.get('prompt_id')}_{x.get('policy', 'unknown')}",
            serialize_fn=lambda x: x,
            deserialize_fn=lambda x: x,
        )

        # Generate responses for each target policy
        all_results = []
        for policy_name, policy_config in target_policies.items():
            policy_results = generate_target_responses(
                ground_truth_prompts,
                policy_name,
                policy_config,
                args.batch_size,
                checkpoint_manager,
            )
            all_results.extend(policy_results)

        # Final checkpoint save
        checkpoint_manager.save_checkpoint()

        # Summary
        console.print(
            f"\n📊 [bold green]Ground Truth Generation Complete![/bold green]"
        )
        # Count actual responses in the output file
        total_responses = len(checkpoint_manager.load_checkpoint())
        console.print(f"   • Total responses: {total_responses:,}")
        console.print(f"   • Policies: {len(target_policies)}")
        console.print(f"   • Prompts per policy: {args.samples}")
        console.print(f"   • Output: {args.output}")

        # Cost estimate
        total_tokens_est = len(all_results) * 100  # rough estimate with 512 max tokens
        cost_est = total_tokens_est * 0.0000002  # Fireworks pricing
        console.print(f"   • Estimated cost: ${cost_est:.2f}")

        console.print(f"\n📋 Next steps:")
        console.print(f"1. Export for labeling: python 03_export_for_labeling.py")
        console.print(f"2. Run judge scoring: python 04_add_judge_scores.py")
        console.print(f"2. Export for human labeling")
        console.print(f"3. Collect human labels for validation")

    except Exception as e:
        console.print(f"\n❌ [red]Failed: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
