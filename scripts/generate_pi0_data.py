#!/usr/bin/env python3
"""
Minimal π₀ data generation - Steps 1-3 from analysis plan:
1. Download ChatBot Arena corpus
2. Generate π₀ answers (Llama-3-8B)
3. Teacher-forced scoring (propensities)

Usage: python scripts/generate_pi0_data.py [--samples 1000] [--output pi0_data.jsonl]

Features:
- Progress tracking with estimated completion time
- Checkpointing for interruption recovery
- Batch processing for memory efficiency
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional


def get_fireworks_api_key() -> str:
    """Get Fireworks API key from AWS Secrets Manager or environment variable"""
    from cje.utils.progress import console

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
        console.print(
            "✅ [green]Successfully retrieved Fireworks API key from AWS Secrets Manager[/green]"
        )
        return api_key

    except Exception as e:
        console.print(f"⚠️ [yellow]AWS Secrets Manager failed: {e}[/yellow]")
        console.print(
            "💡 Please set FIREWORKS_API_KEY environment variable or configure AWS Secrets Manager"
        )
        raise ValueError("No Fireworks API key available")


def download_corpus(sample_limit: int) -> List[Dict[str, Any]]:
    """Download ChatBot Arena corpus"""
    from cje.utils.progress import console

    console.print(f"Downloading {sample_limit:,} ChatBot Arena samples...")

    try:
        from datasets import load_dataset

        ds = load_dataset("agie-ai/lmsys-chatbot_arena_conversations", split="train")

        contexts = []
        seen_contexts = set()

        for i, row in enumerate(ds):
            if len(contexts) >= sample_limit:
                break

            conversation_a = row.get("conversation_a", [])
            if not conversation_a or conversation_a[0].get("role") != "user":
                continue

            context = conversation_a[0].get("content", "").strip()
            if not context or context in seen_contexts:
                continue

            seen_contexts.add(context)
            contexts.append(
                {"uid": f"arena_{row.get('conversation_id', i)}", "context": context}
            )

        console.print(f"✅ [green]{len(contexts):,} unique contexts[/green]")
        return contexts

    except Exception as e:
        console.print(f"⚠️ [yellow]Dataset error: {e} - Using mock data[/yellow]")
        return [
            {
                "uid": f"mock_{i}",
                "context": f"Mock prompt {i}: Please explain quantum computing.",
            }
            for i in range(min(sample_limit, 10))
        ]


def generate_pi0_responses(
    contexts: List[Dict[str, Any]],
    model_name: str = "accounts/fireworks/models/llama4-scout-instruct-basic",
    temperature: float = 0.4,
    max_new_tokens: int = 1024,
    batch_size: int = 16,
    checkpoint_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate π₀ responses with log probabilities.

    Features:
    - Progress tracking with time estimates using shared CJE utilities
    - Checkpointing for resumable generation
    - Batch processing for efficiency
    """
    from cje.loggers.api_policy import APIPolicyRunner
    from cje.utils.checkpointing import create_jsonl_checkpoint_manager, BatchProcessor
    from cje.utils.progress import console

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
    console.print(f"🔬 [bold blue]Generating π₀ responses[/bold blue]")
    console.print(f"📊 Model: {model_name}")
    console.print(f"🌡️ Temperature: {temperature}, Max tokens: {max_new_tokens}")
    console.print(f"📦 Batch size: {batch_size}")

    def process_batch(batch_contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of contexts to generate responses."""
        # Extract contexts for generation
        contexts_list = [ctx["context"] for ctx in batch_contexts]

        # Generate responses
        responses_with_logp = runner.generate_with_consistent_logp(
            contexts_list,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
        )

        # Format results
        batch_results = []
        for ctx, (response, logp) in zip(batch_contexts, responses_with_logp):
            batch_results.append(
                {
                    "uid": ctx["uid"],
                    "context": ctx["context"],
                    "response": response,
                    "logp": float(logp),
                    "action": {
                        "model_name": model_name,
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                    },
                }
            )

        return batch_results

    # Process all contexts with progress tracking and checkpointing
    start_time = time.time()

    try:
        results = batch_processor.process_batches(
            contexts,
            process_batch,
            description="Generating π₀ responses",
            auto_save_frequency=1,  # Save after every batch
        )

        total_time = time.time() - start_time
        console.print(f"\n✅ [green]Generated {len(results):,} total responses[/green]")
        console.print(f"⏱️ Total time: {total_time/60:.1f} minutes")
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


def save_data(data: List[Dict[str, Any]], output_path: str) -> None:
    """Save to JSONL"""
    from cje.utils.progress import console

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    console.print(f"✅ [green]Saved to {output_path}[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Generate π₀ data with progress tracking and checkpointing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/generate_pi0_data.py --samples 1000

  # With checkpointing (recommended for large runs)
  python scripts/generate_pi0_data.py --samples 10000 --checkpoint checkpoint.jsonl

  # Resume interrupted job
  python scripts/generate_pi0_data.py --samples 10000 --checkpoint checkpoint.jsonl

  # Custom batch size and model
  python scripts/generate_pi0_data.py --samples 5000 --batch-size 32 --model accounts/fireworks/models/llama-v3p1-70b-instruct

  # Auto-cleanup checkpoint on success
  python scripts/generate_pi0_data.py --samples 1000 --cleanup-checkpoint
        """,
    )
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument("--output", default="pi0_data.jsonl", help="Output file path")
    parser.add_argument(
        "--checkpoint", help="Checkpoint file for resumable generation (recommended)"
    )
    parser.add_argument(
        "--cleanup-checkpoint",
        action="store_true",
        help="Automatically delete checkpoint file on successful completion",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for API calls"
    )
    parser.add_argument(
        "--model",
        default="accounts/fireworks/models/llama4-scout-instruct-basic",
        help="Model name for generation",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.4, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Maximum tokens to generate"
    )
    args = parser.parse_args()

    from cje.utils.checkpointing import auto_enable_checkpointing
    from cje.utils.progress import console

    console.print(
        f"🔬 [bold blue]π₀ Data Generation[/bold blue] - {args.samples:,} samples"
    )

    # Auto-enable checkpointing using shared utility
    checkpoint_path = auto_enable_checkpointing(args.output, args.checkpoint)
    if checkpoint_path and not args.checkpoint:
        console.print(f"📝 Auto-enabling checkpointing: {checkpoint_path}")

    try:
        # Get API key (AWS Secrets Manager or environment)
        api_key = get_fireworks_api_key()

        # Generate data
        contexts = download_corpus(args.samples)
        pi0_data = generate_pi0_responses(
            contexts,
            model_name=args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            batch_size=args.batch_size,
            checkpoint_path=checkpoint_path,
        )

        # Save final data (different from checkpoint to avoid confusion)
        save_data(pi0_data, args.output)

        console.print(f"\n📊 [green]Generated {len(pi0_data):,} samples[/green]")
        console.print(f"💰 Estimated cost: ~${len(pi0_data) * 0.0004:.2f}")
        console.print(f"💾 Saved to: {args.output}")

        # Handle checkpoint cleanup
        if checkpoint_path and os.path.exists(checkpoint_path):
            if args.cleanup_checkpoint:
                try:
                    os.remove(checkpoint_path)
                    console.print(
                        f"🗑️ [green]Cleaned up checkpoint: {checkpoint_path}[/green]"
                    )
                except Exception as e:
                    console.print(
                        f"⚠️ [yellow]Could not delete checkpoint {checkpoint_path}: {e}[/yellow]"
                    )
            else:
                console.print(f"🗑️ You can now delete checkpoint: {checkpoint_path}")
                console.print(
                    "   Or use --cleanup-checkpoint flag for automatic cleanup"
                )

        if pi0_data:
            sample = pi0_data[0]
            console.print(f"📝 Sample logp: {sample['logp']:.3f}")

    except KeyboardInterrupt:
        console.print(
            f"\n⚠️ [yellow]Interrupted - Progress saved to checkpoint[/yellow]"
        )
        if checkpoint_path:
            console.print(f"💾 Resume with: python {' '.join(sys.argv)}")
    except Exception as e:
        console.print(f"❌ [red]Error: {e}[/red]")
        if checkpoint_path and os.path.exists(checkpoint_path):
            console.print(f"💾 Partial progress saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
