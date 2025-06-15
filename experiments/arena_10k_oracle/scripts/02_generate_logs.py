#!/usr/bin/env python3
"""
Step 2: Generate π₀ (logging policy) responses with log probabilities.

This script:
1. Loads prompts from Step 1
2. Generates responses using Llama-3-34B-Instruct
3. Computes exact token-level log probabilities
4. Saves responses with metadata for downstream processing

Usage:
    python 02_generate_logs.py --input ../data/prompts.jsonl --output ../data/p0_replies.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.loggers.api_policy import APIPolicyRunner
from cje.utils.checkpointing import create_jsonl_checkpoint_manager, BatchProcessor
from cje.utils.progress import console


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


def generate_logging_policy_responses(
    prompts: List[Dict[str, Any]],
    model_name: str = "accounts/fireworks/models/llama4-scout-instruct-basic",
    temperature: float = 0.5,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
    batch_size: int = 16,
    checkpoint_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate π₀ responses with log probabilities.

    Uses model and parameters from experiment config.
    Computes exact token-level log probabilities via teacher forcing.
    """
    # Setup checkpointing
    checkpoint_manager = create_jsonl_checkpoint_manager(checkpoint_path)

    # Create batch processor
    batch_processor = BatchProcessor(
        batch_size=batch_size,
        checkpoint_manager=checkpoint_manager,
        continue_on_error=True,
    )

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
    console.print(f"🌡️  Temperature: {temperature}, Top-p: {top_p}")
    console.print(f"📝 Max tokens: {max_new_tokens}")
    console.print(f"📦 Batch size: {batch_size}")

    def process_batch(batch_prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of prompts to generate responses."""
        # Extract prompt texts
        prompt_texts = [p["prompt"] for p in batch_prompts]

        # Generate responses with log probabilities
        responses_with_logp = runner.generate_with_consistent_logp(
            prompt_texts,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        # Format results
        batch_results = []
        for prompt_data, (response, logp) in zip(batch_prompts, responses_with_logp):
            result = {
                **prompt_data,  # Keep all original prompt data
                "response": response,
                "total_logprob": float(logp),
                "logging_policy": {
                    "model_name": model_name,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                    "provider": "fireworks",
                },
                "timestamp": time.time(),
            }
            batch_results.append(result)

        return batch_results

    # Process all prompts with progress tracking
    start_time = time.time()

    try:
        results = batch_processor.process_batches(
            prompts,
            process_batch,
            description="Generating π₀ responses",
            auto_save_frequency=1,  # Save after every batch
        )

        total_time = time.time() - start_time
        console.print(f"\n✅ [green]Generated {len(results):,} responses[/green]")
        console.print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        if len(results) > 0:
            console.print(
                f"📈 Average: {total_time/len(results):.2f} seconds per sample"
            )

        return results

    except KeyboardInterrupt:
        console.print(
            f"\n⚠️ [yellow]Interrupted - Progress saved to checkpoint[/yellow]"
        )
        return checkpoint_manager.get_processed_items()
    except Exception as e:
        console.print(f"❌ [red]Error: {e}[/red]")
        if checkpoint_manager:
            console.print(f"💾 Partial progress saved to: {checkpoint_path}")
        raise


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
        description="Generate π₀ (logging policy) responses for Arena 10K experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python 02_generate_logs.py --input ../data/prompts.jsonl
  
  # With checkpointing (recommended)
  python 02_generate_logs.py --input ../data/prompts.jsonl --checkpoint checkpoint.jsonl
  
  # Custom batch size for memory efficiency
  python 02_generate_logs.py --input ../data/prompts.jsonl --batch-size 8
  
  # Resume interrupted job
  python 02_generate_logs.py --input ../data/prompts.jsonl --checkpoint checkpoint.jsonl
        """,
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
        "--checkpoint", type=str, help="Checkpoint file for resumable generation"
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for API calls"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="accounts/fireworks/models/llama4-scout-instruct-basic",
        help="Model name (default: from config)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature (default: 0.5 from config)",
    )

    parser.add_argument(
        "--top-p", type=float, default=0.95, help="Top-p sampling (default: 0.95)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024 from config)",
    )

    parser.add_argument(
        "--cleanup-checkpoint",
        action="store_true",
        help="Delete checkpoint file on successful completion",
    )

    args = parser.parse_args()

    console.print(
        f"🔬 [bold blue]Arena 10K Experiment - Step 2: Generate Logging Policy[/bold blue]"
    )

    # Auto-enable checkpointing for large runs
    if not args.checkpoint:
        prompts = load_prompts(args.input)
        if len(prompts) > 1000:
            args.checkpoint = args.output.replace(".jsonl", "_checkpoint.jsonl")
            console.print(f"📝 Auto-enabling checkpointing: {args.checkpoint}")

    try:
        # Get API key
        api_key = get_api_key()

        # Load prompts
        prompts = load_prompts(args.input)

        # Generate responses
        results = generate_logging_policy_responses(
            prompts,
            model_name=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens,
            batch_size=args.batch_size,
            checkpoint_path=args.checkpoint,
        )

        # Save results
        save_results(results, args.output)

        # Show sample output
        if results:
            sample = results[0]
            console.print(f"\n📝 Sample response:")
            console.print(f"  Prompt: {sample['prompt'][:100]}...")
            console.print(f"  Response: {sample['response'][:100]}...")
            console.print(f"  Log prob: {sample['total_logprob']:.3f}")

        # Cleanup checkpoint if requested
        if args.checkpoint and os.path.exists(args.checkpoint):
            if args.cleanup_checkpoint:
                try:
                    os.remove(args.checkpoint)
                    console.print(f"🗑️  [green]Cleaned up checkpoint[/green]")
                except Exception as e:
                    console.print(
                        f"⚠️  [yellow]Could not delete checkpoint: {e}[/yellow]"
                    )
            else:
                console.print(f"🗑️  You can now delete checkpoint: {args.checkpoint}")

        console.print(f"\n✅ [bold green]Step 2 complete![/bold green]")
        console.print(f"📄 Output: {args.output}")

    except KeyboardInterrupt:
        console.print("\n⚠️ [yellow]Interrupted by user[/yellow]")
        if args.checkpoint:
            console.print(f"💾 Resume with the same command")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ [red]Failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
