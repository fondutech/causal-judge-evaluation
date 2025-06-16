#!/usr/bin/env python3
"""
Step 2: Generate π₀ (logging policy) responses with log probabilities.

Improved version with atomic checkpointing to prevent duplicates.

Usage:
    python 02_generate_logs.py --input ../data/prompts.jsonl --output ../data/p0_replies.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.loggers.api_policy import APIPolicyRunner
from cje.utils.progress import console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)


class AtomicCheckpointManager:
    """Manages atomic checkpointing with no duplicates."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.processed_items: List[Dict[str, Any]] = []
        self.processed_indices: set = set()

    def load(self) -> bool:
        """Load checkpoint if it exists. Returns True if loaded."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r") as f:
                    for line in f:
                        item = json.loads(line)
                        if "prompt_id" in item:
                            self.processed_indices.add(item["prompt_id"])
                            self.processed_items.append(item)
                console.print(
                    f"📥 Loaded {len(self.processed_items)} items from checkpoint"
                )
                return True
            except Exception as e:
                console.print(f"[yellow]⚠️  Failed to load checkpoint: {e}[/yellow]")
        return False

    def save_batch(self, new_items: List[Dict[str, Any]]) -> None:
        """Save a batch of new items atomically."""
        # Filter out any duplicates
        unique_items = []
        for item in new_items:
            if item.get("prompt_id") not in self.processed_indices:
                unique_items.append(item)
                self.processed_indices.add(item["prompt_id"])
                self.processed_items.append(item)

        if not unique_items:
            return

        # Write all processed items to temp file
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            for item in self.processed_items:
                f.write(json.dumps(item) + "\n")

        # Atomic rename
        shutil.move(str(temp_path), str(self.checkpoint_path))
        console.print(
            f"[green]✓ Saved {len(unique_items)} new items (total: {len(self.processed_items)})[/green]"
        )

    def is_processed(self, prompt_id: str) -> bool:
        """Check if a prompt has been processed."""
        return prompt_id in self.processed_indices

    def get_processed_items(self) -> List[Dict[str, Any]]:
        """Get all processed items."""
        return self.processed_items


def load_prompts(input_path: str) -> List[Dict[str, Any]]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(input_path, "r") as f:
        for line in f:
            prompts.append(json.loads(line))

    console.print(f"📄 Loaded {len(prompts):,} prompts from {input_path}")
    return prompts


def get_api_key() -> str:
    """Get Fireworks API key from environment or AWS Secrets Manager."""
    # First try environment variable
    env_key = os.getenv("FIREWORKS_API_KEY")
    if env_key:
        console.print("✅ [green]Using Fireworks API key from environment[/green]")
        return env_key

    # Try AWS Secrets Manager
    try:
        from cje.utils.aws_secrets import get_api_key_from_secrets

        console.print("🔐 Retrieving Fireworks API key from AWS Secrets Manager...")

        api_key = get_api_key_from_secrets(
            secret_name="cje/prod/api-keys",
            key="FIREWORKS_API_KEY",
            env_var_name="FIREWORKS_API_KEY",
            cache_in_env=True,
        )
        console.print("✅ [green]Successfully retrieved API key[/green]")
        return str(api_key)

    except Exception as e:
        console.print(f"⚠️ [yellow]AWS Secrets Manager failed: {e}[/yellow]")
        console.print("💡 Please set FIREWORKS_API_KEY environment variable")
        raise ValueError("No Fireworks API key available")


def generate_with_retry(
    runner: APIPolicyRunner,
    prompts: List[str],
    temperature: float,
    max_new_tokens: int,
    max_retries: int = 3,
) -> List[Tuple[str, float]]:
    """Generate responses with automatic retry on failure."""

    for attempt in range(max_retries):
        try:
            # Use two-pass teacher forcing for consistent logprobs
            results = runner.generate_with_consistent_logp(
                prompts,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            return results
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff
                console.print(
                    f"[yellow]⚠️  API error: {e}. Retrying in {wait_time}s...[/yellow]"
                )
                time.sleep(wait_time)
            else:
                raise
    # This should never be reached due to the raise above
    return []


def generate_logging_policy_responses(
    prompts: List[Dict[str, Any]],
    model_name: str = "accounts/fireworks/models/llama4-scout-instruct-basic",
    temperature: float = 0.5,
    max_new_tokens: int = 1024,
    batch_size: int = 16,
    checkpoint_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Generate π₀ responses with log probabilities.

    Uses atomic checkpointing to prevent duplicates.
    """
    # Setup checkpointing
    checkpoint_manager = AtomicCheckpointManager(
        checkpoint_path or Path("checkpoint.jsonl")
    )
    checkpoint_manager.load()

    # Initialize API runner
    runner = APIPolicyRunner(
        provider="fireworks",
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    # Display configuration
    console.print(f"🔬 [bold blue]Generating π₀ (logging policy) responses[/bold blue]")
    console.print(f"📊 Model: {model_name}")
    console.print(f"🌡️  Temperature: {temperature}")
    console.print(f"📝 Max tokens: {max_new_tokens}")
    console.print(f"📦 Batch size: {batch_size}")

    # Filter out already processed prompts
    prompts_to_process = [
        p for p in prompts if not checkpoint_manager.is_processed(p["prompt_id"])
    ]

    if not prompts_to_process:
        console.print("[green]✅ All prompts already processed![/green]")
        return checkpoint_manager.get_processed_items()

    console.print(f"📋 Processing {len(prompts_to_process)} remaining prompts")

    # Process with progress tracking
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        task = progress.add_task(
            "Generating π₀ responses", total=len(prompts_to_process)
        )

        # Process in batches
        for batch_start in range(0, len(prompts_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts_to_process))
            batch = prompts_to_process[batch_start:batch_end]

            # Extract prompt texts
            prompt_texts = [p["prompt"] for p in batch]

            try:
                # Generate responses with log probabilities
                responses_with_logp = generate_with_retry(
                    runner, prompt_texts, temperature, max_new_tokens
                )

                # Format results
                batch_results = []
                for prompt_data, (response, logp) in zip(batch, responses_with_logp):
                    result = {
                        **prompt_data,  # Keep all original prompt data
                        "response": response,
                        "total_logprob": float(logp),
                        "logging_policy": {
                            "model_name": model_name,
                            "temperature": temperature,
                            "max_new_tokens": max_new_tokens,
                            "provider": "fireworks",
                        },
                        "timestamp": time.time(),
                    }
                    batch_results.append(result)

                # Save batch atomically
                checkpoint_manager.save_batch(batch_results)
                progress.update(task, advance=len(batch))

            except Exception as e:
                console.print(
                    f"\n[red]❌ Failed to generate batch {batch_start}-{batch_end}: {e}[/red]"
                )
                # Continue with next batch instead of failing completely
                continue

            # Rate limit protection
            if batch_end < len(prompts_to_process):
                time.sleep(0.5)

    total_time = time.time() - start_time
    all_results = checkpoint_manager.get_processed_items()

    console.print(f"\n✅ [green]Total responses: {len(all_results):,}[/green]")
    console.print(f"⏱️  Generation time: {total_time/60:.1f} minutes")
    if len(prompts_to_process) > 0:
        console.print(
            f"📈 Average: {total_time/len(prompts_to_process):.2f} seconds per sample"
        )

    return all_results


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save results to JSONL file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    console.print(
        f"💾 [green]Saved {len(results):,} responses to {output_file}[/green]"
    )

    # Calculate and display cost estimate
    # Rough estimate: ~$0.0004 per 1K tokens generated
    avg_response_tokens = 200  # Conservative estimate
    total_tokens = len(results) * avg_response_tokens
    estimated_cost = (total_tokens / 1000) * 0.0004

    console.print(f"💰 Estimated generation cost: ~${estimated_cost:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate π₀ (logging policy) responses for Arena 10K experiment"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="../data/prompts.jsonl",
        help="Input prompts file from Step 1",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../data/p0_replies.jsonl",
        help="Output file for responses",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint file for resumable generation (default: auto-generated)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for API calls"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="accounts/fireworks/models/llama4-scout-instruct-basic",
        help="Model name",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens",
    )

    args = parser.parse_args()

    console.print(
        f"🔬 [bold blue]Arena 10K Experiment - Step 2: Generate π₀ Responses[/bold blue]"
    )

    try:
        # Ensure we have API key
        get_api_key()

        # Load prompts
        prompts = load_prompts(args.input)

        # Set checkpoint path
        checkpoint_path = (
            Path(args.checkpoint)
            if args.checkpoint
            else Path(args.output).with_suffix(".checkpoint.jsonl")
        )

        # Generate responses
        results = generate_logging_policy_responses(
            prompts,
            model_name=args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            batch_size=args.batch_size,
            checkpoint_path=checkpoint_path,
        )

        # Save final results
        save_results(results, args.output)

        # Clean up checkpoint if successful
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            console.print("🧹 Cleaned up checkpoint file")

        console.print(f"\n✅ [bold green]π₀ generation complete![/bold green]")
        console.print(f"📁 Output: {args.output}")
        console.print(f"Next step: python 03_add_judge_scores.py")

    except KeyboardInterrupt:
        console.print(
            f"\n⚠️ [yellow]Interrupted - Progress saved to checkpoint[/yellow]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ [red]Failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
