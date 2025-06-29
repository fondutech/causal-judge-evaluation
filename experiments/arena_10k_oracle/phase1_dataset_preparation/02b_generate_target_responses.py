#!/usr/bin/env python3
"""
Step 2b: Generate target policy responses for all prompts using async.

This script generates responses from target policies (π_cot, π_bigger_model, π_bad)
for all 10,000 prompts. Uses async to efficiently batch API calls.

Usage:
    python 02b_generate_target_responses.py

    # Run only specific policies:
    python 02b_generate_target_responses.py --policies pi_cot pi_bigger_model

    # Adjust concurrency:
    python 02b_generate_target_responses.py --max-concurrent 50
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Tuple
import time
import os
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.console import Console
from rich.table import Table

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.loggers.api_policy import APIPolicyRunner
from cje.utils.checkpointing import CheckpointManager

console = Console()


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
        "pi_clone": {
            "provider": "fireworks",
            "model_name": "accounts/fireworks/models/llama4-scout-instruct-basic",
            "temperature": 0.5,
            "system_prompt": None,
            "user_message_template": "{context}",
            "description": "Clone of P0 policy (same model and settings)",
        },
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


async def generate_batch_async(
    runner: APIPolicyRunner,
    batch_contexts: List[str],
    semaphore: asyncio.Semaphore,
) -> List[Tuple[str, Any]]:
    """Generate responses for a batch of prompts with rate limiting."""
    async with semaphore:
        # Run the synchronous API call in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: runner.generate_with_logp(
                batch_contexts, return_token_logprobs=False
            ),
        )


async def generate_policy_responses_async(
    prompts: List[Dict[str, Any]],
    policy_name: str,
    policy_config: Dict[str, Any],
    batch_size: int,
    max_concurrent: int,
    output_dir: Path,
    progress: Progress,
    task_id: Any,
) -> Dict[str, Any]:
    """Generate responses for a single policy using async batching."""

    # Policy-specific checkpoint file
    checkpoint_path = output_dir / f"target_responses_{policy_name}.checkpoint.jsonl"
    checkpoint_manager = CheckpointManager(
        checkpoint_path=str(checkpoint_path),
        get_uid_fn=lambda x: x.get("prompt_id"),
        serialize_fn=lambda x: x,
        deserialize_fn=lambda x: x,
    )

    # Initialize policy runner
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

    # Update progress
    progress.update(task_id, completed=len(existing_items))

    if not unprocessed_prompts:
        return {
            "policy": policy_name,
            "total": len(existing_items),
            "new": 0,
            "status": "completed",
        }

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Process in batches
    start_time = time.time()
    new_responses = 0
    errors = 0

    # Create all batch tasks
    tasks = []
    for i in range(0, len(unprocessed_prompts), batch_size):
        batch = unprocessed_prompts[i : i + batch_size]
        batch_contexts = [item["prompt"] for item in batch]
        tasks.append((batch, generate_batch_async(runner, batch_contexts, semaphore)))

    # Process all batches concurrently
    for batch, task in tasks:
        try:
            generations = await task

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

            # Update progress
            progress.update(task_id, completed=len(existing_items) + new_responses)

        except Exception as e:
            errors += 1
            console.print(f"[red]Error in {policy_name} batch: {e}[/red]")
            continue

    # Calculate statistics
    total_responses = len(checkpoint_manager.load_checkpoint())
    elapsed_time = time.time() - start_time

    return {
        "policy": policy_name,
        "total": total_responses,
        "new": new_responses,
        "elapsed_time": elapsed_time,
        "status": "completed",
        "errors": errors,
    }


async def generate_all_policies_async(
    prompts: List[Dict[str, Any]],
    target_policies: Dict[str, Dict[str, Any]],
    batch_size: int,
    max_concurrent: int,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Generate responses for all policies concurrently."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=1,
    ) as progress:
        # Create tasks for each policy
        tasks = {}
        for policy_name in target_policies.keys():
            task_id = progress.add_task(
                f"[cyan]{policy_name}[/cyan]", total=len(prompts)
            )
            tasks[policy_name] = task_id

        # Run all policies concurrently
        policy_tasks = []
        for policy_name, policy_config in target_policies.items():
            task = generate_policy_responses_async(
                prompts,
                policy_name,
                policy_config,
                batch_size,
                max_concurrent
                // len(target_policies),  # Divide concurrency among policies
                output_dir,
                progress,
                tasks[policy_name],
            )
            policy_tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*policy_tasks, return_exceptions=True)

        # Handle any exceptions
        final_results: List[Dict[str, Any]] = []
        for i, result in enumerate(results):
            policy_name = list(target_policies.keys())[i]
            if isinstance(result, Exception):
                console.print(f"[red]Policy {policy_name} failed: {result}[/red]")
                final_results.append(
                    {
                        "policy": policy_name,
                        "total": 0,
                        "new": 0,
                        "status": "failed",
                        "error": str(result),
                    }
                )
            else:
                # result is guaranteed to be Dict[str, Any] here
                final_results.append(result)  # type: ignore
                # Update progress bar to show completion
                progress.update(
                    tasks[policy_name], description=f"[green]✓ {policy_name}[/green]"
                )

        return final_results


def merge_policy_checkpoints(output_dir: Path, output_file: Path) -> None:
    """Merge individual policy checkpoint files into the final output."""
    console.print(f"\n🔀 Merging policy outputs into {output_file}")

    all_responses = []

    # Load each policy's checkpoint
    for policy_name in ["pi_cot", "pi_bigger_model", "pi_bad"]:
        checkpoint_file = (
            output_dir / f"target_responses_{policy_name}.checkpoint.jsonl"
        )
        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                for line in f:
                    all_responses.append(json.loads(line))
            console.print(f"   ✓ Loaded {policy_name} responses")

    # Sort by prompt_id and policy for consistent ordering
    all_responses.sort(key=lambda x: (x["prompt_id"], x.get("policy", "")))

    # Write merged output
    with open(output_file, "w") as f:
        for response in all_responses:
            f.write(json.dumps(response) + "\n")

    console.print(f"✅ Merged {len(all_responses)} total responses")

    # Clean up checkpoint files
    for policy_name in ["pi_cot", "pi_bigger_model", "pi_bad"]:
        checkpoint_file = (
            output_dir / f"target_responses_{policy_name}.checkpoint.jsonl"
        )
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            console.print(f"🧹 Cleaned up {checkpoint_file.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate target policy responses using async API calls"
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
        "--policies",
        nargs="+",
        choices=["pi_cot", "pi_bigger_model", "pi_bad"],
        default=["pi_cot", "pi_bigger_model", "pi_bad"],
        help="Which policies to generate (default: all)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for generation"
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=30,
        help="Maximum concurrent API requests (default: 30)",
    )

    args = parser.parse_args()

    console.print(
        "🎯 [bold blue]Arena 10K Experiment - Step 2b: Generate Target Policy Responses[/bold blue]"
    )
    console.print(f"📊 Generating responses for ALL prompts")
    console.print(f"🔢 Batch size: {args.batch_size}")
    console.print(f"⚡ Max concurrent requests: {args.max_concurrent}")
    console.print(f"🎯 Policies: {', '.join(args.policies)}")

    try:
        # Check for required API key
        if not os.environ.get("FIREWORKS_API_KEY"):
            console.print(
                "\n❌ [red]Error: FIREWORKS_API_KEY environment variable not set![/red]"
            )
            console.print(
                "   Please set it with: export FIREWORKS_API_KEY='your-api-key'"
            )
            sys.exit(1)

        # Load all prompts
        all_prompts = load_prompts(args.prompts)
        console.print(f"📄 Loaded {len(all_prompts):,} prompts")

        # Get target policies
        all_target_policies = create_target_policies()
        target_policies = {
            k: v for k, v in all_target_policies.items() if k in args.policies
        }

        # Create output directory
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run async generation
        console.print(
            f"\n🚀 Starting async generation for {len(target_policies)} policies...\n"
        )

        start_time = time.time()

        # Run the async function
        results = asyncio.run(
            generate_all_policies_async(
                all_prompts,
                target_policies,
                args.batch_size,
                args.max_concurrent,
                output_dir,
            )
        )

        total_time = time.time() - start_time

        # Merge results
        merge_policy_checkpoints(output_dir, Path(args.output))

        # Create summary table
        table = Table(title="Generation Summary", show_header=True)
        table.add_column("Policy", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Total", justify="right", style="green")
        table.add_column("New", justify="right", style="yellow")
        table.add_column("Errors", justify="right", style="red")

        total_responses = 0
        total_new = 0

        for result in sorted(results, key=lambda x: x["policy"]):
            status = result.get("status", "unknown")
            if status == "completed":
                status_icon = "✅"
            elif status == "failed":
                status_icon = "❌"
            else:
                status_icon = "❓"

            table.add_row(
                result["policy"],
                status_icon,
                f"{result.get('total', 0):,}",
                f"{result.get('new', 0):,}",
                f"{result.get('errors', 0):,}" if result.get("errors", 0) > 0 else "-",
            )

            total_responses += result.get("total", 0)
            total_new += result.get("new", 0)

        console.print(f"\n")
        console.print(table)

        # Summary
        console.print(f"\n📊 [bold green]Generation Complete![/bold green]")
        console.print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        console.print(f"   • Total responses: {total_responses:,}")
        console.print(f"   • New responses: {total_new:,}")
        console.print(f"   • Output: {args.output}")

        # Cost estimate
        total_tokens_est = total_responses * 100  # rough estimate with 512 max tokens
        cost_est = total_tokens_est * 0.0000002  # Fireworks pricing
        console.print(f"   • Estimated cost: ${cost_est:.2f}")

        console.print(f"\n📋 Next steps:")
        console.print(f"1. Generate oracle labels: python 03_generate_oracle_labels.py")
        console.print(
            f"2. Score with judges: python 04c_score_targets_deterministic.py"
        )
        console.print(
            f"3. Score with uncertainty: python 04d_score_targets_uncertainty.py"
        )

    except KeyboardInterrupt:
        console.print(
            f"\n⚠️ [yellow]Interrupted - Progress saved to checkpoints[/yellow]"
        )
        console.print("Re-run the same command to resume from where you left off")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ [red]Failed: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
