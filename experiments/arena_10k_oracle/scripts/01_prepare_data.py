#!/usr/bin/env python3
"""
Step 1: Download and prepare ChatBot Arena data for Arena 10K experiment.

This script:
1. Downloads ChatBot Arena conversations from HuggingFace
2. Extracts first-turn user prompts
3. Samples 10,000 prompts with a fixed seed
4. Saves to JSONL format for downstream processing

Usage:
    python 01_prepare_data.py [--samples 10000] [--output ../data/prompts.jsonl]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils.progress import console


def download_and_prepare_corpus(
    sample_limit: int = 10000,
    seed: int = 42,
    output_path: str = "../data/prompts.jsonl",
) -> List[Dict[str, Any]]:
    """Download ChatBot Arena corpus and prepare prompts."""

    console.print(
        f"📥 [bold blue]Downloading ChatBot Arena conversations...[/bold blue]"
    )

    try:
        from datasets import load_dataset

        # Download the dataset
        ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")
        console.print(f"✅ [green]Downloaded {len(ds):,} conversations[/green]")

        # Extract first-turn user prompts
        prompts = []
        seen_prompts = set()

        for idx, row in enumerate(ds):
            if not row.get("conversation"):
                continue

            # Get first user message
            first_turn = None
            for msg in row["conversation"]:
                if msg.get("role") == "user":
                    first_turn = msg
                    break

            if not first_turn:
                continue

            prompt_text = first_turn.get("content", "").strip()

            # Skip empty or duplicate prompts
            if not prompt_text or prompt_text in seen_prompts:
                continue

            seen_prompts.add(prompt_text)
            prompts.append(
                {
                    "prompt_id": f"arena_{idx}",
                    "prompt": prompt_text,
                    "metadata": {
                        "source": "chatbot_arena",
                        "conversation_id": row.get("conversation_id", idx),
                        "timestamp": row.get("tstamp", None),
                        "model_a": row.get("model_a", "unknown"),
                        "model_b": row.get("model_b", "unknown"),
                    },
                }
            )

            # Progress update every 5000
            if len(prompts) % 5000 == 0:
                console.print(f"📊 Extracted {len(prompts):,} unique prompts...")

        console.print(
            f"✅ [green]Extracted {len(prompts):,} unique prompts total[/green]"
        )

        # Sample if we have more than requested
        if len(prompts) > sample_limit:
            console.print(f"🎲 Sampling {sample_limit:,} prompts with seed={seed}")
            import random

            random.seed(seed)
            prompts = random.sample(prompts, sample_limit)

            # Re-number the prompt IDs after sampling
            for i, prompt in enumerate(prompts):
                prompt["prompt_id"] = f"arena_sampled_{i}"

        # Save to JSONL
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for prompt in prompts:
                f.write(json.dumps(prompt) + "\n")

        console.print(
            f"💾 [green]Saved {len(prompts):,} prompts to {output_file}[/green]"
        )

        # Print statistics
        console.print("\n📊 [bold]Dataset Statistics:[/bold]")
        console.print(f"  • Total conversations downloaded: {len(ds):,}")
        console.print(f"  • Unique prompts extracted: {len(seen_prompts):,}")
        console.print(f"  • Final sample size: {len(prompts):,}")

        # Show a few examples
        console.print("\n📝 [bold]Sample prompts:[/bold]")
        for i in range(min(3, len(prompts))):
            prompt_text = prompts[i]["prompt"]
            if len(prompt_text) > 100:
                prompt_text = prompt_text[:97] + "..."
            console.print(f"  {i+1}. {prompt_text}")

        return prompts

    except Exception as e:
        console.print(f"❌ [red]Error downloading dataset: {e}[/red]")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare ChatBot Arena data for Arena 10K experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Download 10k samples
  python 01_prepare_data.py
  
  # Custom sample size
  python 01_prepare_data.py --samples 5000
  
  # Custom output location
  python 01_prepare_data.py --output /path/to/prompts.jsonl
        """,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of prompts to sample (default: 10000)",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling (default: 42)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../data/prompts.jsonl",
        help="Output file path (default: ../data/prompts.jsonl)",
    )

    args = parser.parse_args()

    console.print(
        f"🔬 [bold blue]Arena 10K Experiment - Step 1: Data Preparation[/bold blue]"
    )
    console.print(f"📊 Target sample size: {args.samples:,}")
    console.print(f"🎲 Random seed: {args.seed}")

    try:
        prompts = download_and_prepare_corpus(
            sample_limit=args.samples, seed=args.seed, output_path=args.output
        )

        console.print(f"\n✅ [bold green]Step 1 complete![/bold green]")
        console.print(f"📄 Output: {args.output}")
        console.print(f"📊 Prompts ready: {len(prompts):,}")

    except KeyboardInterrupt:
        console.print("\n⚠️ [yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ [red]Failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
