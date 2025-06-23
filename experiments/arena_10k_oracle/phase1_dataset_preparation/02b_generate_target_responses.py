#!/usr/bin/env python3
"""
Step 2b: Generate target policy responses for all prompts in parallel.

This script generates responses from target policies (π_cot, π_bigger_model, π_bad)
for all 10,000 prompts. Runs all three policies in parallel by default.

Usage:
    python 02b_generate_target_responses.py

    # Run only specific policies:
    python 02b_generate_target_responses.py --policies pi_cot pi_bigger_model

    # Run sequentially instead of parallel:
    python 02b_generate_target_responses.py --sequential
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Queue, Process
import threading
from rich.progress import (
    Progress,
    TaskID,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.loggers.api_policy import APIPolicyRunner
from cje.utils.progress import console
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


def progress_monitor(
    status_queue: "multiprocessing.queues.Queue[Any]",
    total_prompts: int,
    policies: List[str],
) -> None:
    """Monitor progress from all worker processes and display unified progress bars."""

    # Initialize progress tracking
    policy_progress = {
        policy: {"completed": 0, "total": total_prompts} for policy in policies
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        # Create tasks for each policy
        tasks = {}
        for policy in policies:
            tasks[policy] = progress.add_task(
                f"[cyan]{policy}[/cyan]", total=total_prompts
            )

        # Monitor status updates
        active_policies = set(policies)
        while active_policies:
            try:
                update = status_queue.get(timeout=1.0)

                if update["type"] == "progress":
                    policy = update["policy"]
                    completed = update["completed"]
                    policy_progress[policy]["completed"] = completed
                    progress.update(tasks[policy], completed=completed)

                elif update["type"] == "complete":
                    policy = update["policy"]
                    active_policies.discard(policy)
                    progress.update(
                        tasks[policy],
                        completed=policy_progress[policy]["total"],
                        description=f"[green]✓ {policy}[/green]",
                    )

                elif update["type"] == "error":
                    policy = update["policy"]
                    active_policies.discard(policy)
                    progress.update(
                        tasks[policy], description=f"[red]✗ {policy} (failed)[/red]"
                    )

            except:
                # Timeout is fine, just check if we should continue
                pass


def generate_single_policy_with_progress(
    prompts: List[Dict[str, Any]],
    policy_name: str,
    policy_config: Dict[str, Any],
    batch_size: int,
    output_dir: str,
    status_queue: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generate responses for a single target policy with progress reporting."""

    # Policy-specific checkpoint file
    checkpoint_path = (
        Path(output_dir) / f"target_responses_{policy_name}.checkpoint.jsonl"
    )
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

    # Report initial status
    if status_queue:
        status_queue.put(
            {
                "type": "progress",
                "policy": policy_name,
                "completed": len(existing_items),
                "total": len(prompts),
            }
        )

    if not unprocessed_prompts:
        if status_queue:
            status_queue.put(
                {
                    "type": "complete",
                    "policy": policy_name,
                    "total": len(existing_items),
                    "new": 0,
                }
            )
        return {
            "policy": policy_name,
            "total": len(existing_items),
            "new": 0,
            "status": "completed",
        }

    # Process in batches
    start_time = time.time()
    new_responses = 0

    try:
        for i in range(0, len(unprocessed_prompts), batch_size):
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

                # Report progress
                if status_queue:
                    status_queue.put(
                        {
                            "type": "progress",
                            "policy": policy_name,
                            "completed": len(existing_items) + new_responses,
                            "total": len(prompts),
                        }
                    )

            except Exception as e:
                # Log error but continue
                if status_queue:
                    status_queue.put(
                        {
                            "type": "batch_error",
                            "policy": policy_name,
                            "batch": i // batch_size + 1,
                            "error": str(e),
                        }
                    )
                continue

        # Calculate statistics
        total_responses = len(checkpoint_manager.load_checkpoint())
        elapsed_time = time.time() - start_time

        if status_queue:
            status_queue.put(
                {
                    "type": "complete",
                    "policy": policy_name,
                    "total": total_responses,
                    "new": new_responses,
                    "elapsed_time": elapsed_time,
                }
            )

        return {
            "policy": policy_name,
            "total": total_responses,
            "new": new_responses,
            "elapsed_time": elapsed_time,
            "status": "completed",
        }

    except Exception as e:
        if status_queue:
            status_queue.put({"type": "error", "policy": policy_name, "error": str(e)})
        return {
            "policy": policy_name,
            "total": len(existing_items),
            "new": new_responses,
            "status": "failed",
            "error": str(e),
        }


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
        description="Generate target policy responses for all prompts (parallel by default)"
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
        "--sequential",
        action="store_true",
        help="Run policies sequentially instead of in parallel",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum number of parallel workers (default: 3)",
    )

    args = parser.parse_args()

    console.print(
        "🎯 [bold blue]Arena 10K Experiment - Step 2b: Generate Target Policy Responses[/bold blue]"
    )
    console.print(f"📊 Generating responses for ALL prompts")
    console.print(f"🔢 Batch size: {args.batch_size}")
    console.print(f"🎯 Policies: {', '.join(args.policies)}")
    console.print(
        f"⚙️  Mode: {'Sequential' if args.sequential else f'Parallel (max {args.max_workers} workers)'}"
    )

    try:
        # Load all prompts
        all_prompts = load_prompts(args.prompts)
        console.print(f"📄 Loaded {len(all_prompts):,} prompts")

        # Get target policies
        all_target_policies = create_target_policies()
        target_policies = {
            k: v for k, v in all_target_policies.items() if k in args.policies
        }

        # Create output directory for checkpoints
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        if args.sequential:
            # Sequential execution with progress bars
            console.print(f"\n📋 Running policies sequentially...")
            results = []

            for policy_name, policy_config in target_policies.items():
                console.print(
                    f"\n🎯 Starting {policy_name} ({policy_config['description']})"
                )

                # Create a simple progress bar for sequential mode
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        f"[cyan]{policy_name}[/cyan]", total=len(all_prompts)
                    )

                    # Create a queue for progress updates
                    import queue

                    status_queue: queue.Queue[Any] = queue.Queue()

                    # Run in a thread so we can update progress
                    from threading import Thread

                    result_container: List[Dict[str, Any]] = []

                    def run_policy() -> None:
                        result = generate_single_policy_with_progress(
                            all_prompts,
                            policy_name,
                            policy_config,
                            args.batch_size,
                            str(output_dir),
                            status_queue,
                        )
                        result_container.append(result)

                    thread = Thread(target=run_policy)
                    thread.start()

                    # Monitor progress
                    while thread.is_alive():
                        try:
                            update = status_queue.get(timeout=0.1)
                            if update["type"] == "progress":
                                progress.update(task, completed=update["completed"])
                        except:
                            pass

                    thread.join()
                    progress.update(task, completed=len(all_prompts))

                    if result_container:
                        results.append(result_container[0])
        else:
            # Parallel execution with unified progress display
            console.print(
                f"\n🚀 Starting parallel generation with {len(target_policies)} policies..."
            )
            console.print(
                f"📊 Progress bars will show real-time status for each policy:\n"
            )

            # Create a queue for progress updates
            manager = multiprocessing.Manager()
            status_queue = manager.Queue()

            # Start progress monitor in a separate thread
            monitor_thread = threading.Thread(
                target=progress_monitor,
                args=(status_queue, len(all_prompts), list(target_policies.keys())),
            )
            monitor_thread.start()

            with ProcessPoolExecutor(
                max_workers=min(args.max_workers, len(target_policies))
            ) as executor:
                # Submit all tasks
                future_to_policy = {
                    executor.submit(
                        generate_single_policy_with_progress,
                        all_prompts,
                        policy_name,
                        policy_config,
                        args.batch_size,
                        str(output_dir),
                        status_queue,
                    ): policy_name
                    for policy_name, policy_config in target_policies.items()
                }

                # Collect results as they complete
                results = []
                for future in as_completed(future_to_policy):
                    policy_name = future_to_policy[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append(
                            {
                                "policy": policy_name,
                                "total": 0,
                                "new": 0,
                                "status": "failed",
                                "error": str(e),
                            }
                        )

            # Wait for monitor thread to finish
            monitor_thread.join(timeout=5.0)

        total_time = time.time() - start_time

        # Merge all results
        merge_policy_checkpoints(output_dir, Path(args.output))

        # Summary
        console.print(
            f"\n📊 [bold green]Target Policy Generation Complete![/bold green]"
        )
        console.print(f"⏱️  Total time: {total_time/60:.1f} minutes")

        total_responses = sum(r["total"] for r in results)
        new_responses = sum(r["new"] for r in results)

        console.print(f"\n📈 Results by policy:")
        for result in sorted(results, key=lambda x: x["policy"]):
            status_icon = "✅" if result["status"] == "completed" else "❌"
            console.print(
                f"   {status_icon} {result['policy']}: {result['total']:,} total ({result['new']:,} new)"
            )

        console.print(f"\n📊 Overall statistics:")
        console.print(f"   • Total responses: {total_responses:,}")
        console.print(f"   • New responses: {new_responses:,}")
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
    # Set multiprocessing start method to avoid issues on macOS
    multiprocessing.set_start_method("spawn", force=True)
    main()
