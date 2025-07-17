#!/usr/bin/env python3
"""
Prepare ChatBot Arena data for CJE experiments.

Downloads ChatBot Arena conversations and extracts first-turn prompts.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


def prepare_arena_prompts(
    n_samples: int = 1000, output_file: str = "arena_prompts.jsonl", seed: int = 42
) -> List[Dict[str, Any]]:
    """Download and prepare Arena prompts."""

    print(f"Downloading ChatBot Arena conversations...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library required. Install with: pip install datasets")
        return []

    # Download dataset
    dataset = load_dataset("lmsys/chatbot_arena_conversations", split="train")
    print(f"Downloaded {len(dataset):,} conversations")

    # Extract unique first-turn prompts
    prompts = []
    seen = set()

    for i, row in enumerate(dataset):
        # Get conversation (handle different dataset formats)
        conv = row.get("conversation_a") or row.get("conversation") or []

        # Find first user message
        user_prompt = None
        for msg in conv:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_prompt = msg.get("content", "").strip()
                break

        if not user_prompt or user_prompt in seen:
            continue

        # Filter English only (simple heuristic)
        if any(ord(c) > 127 for c in user_prompt[:100]):  # Non-ASCII in first 100 chars
            continue

        seen.add(user_prompt)
        prompts.append(
            {
                "prompt_id": f"arena_{i}",
                "prompt": user_prompt,
                "source": "chatbot_arena",
            }
        )

        if len(prompts) % 1000 == 0:
            print(f"Extracted {len(prompts):,} prompts...")

    print(f"Total unique prompts: {len(prompts):,}")

    # Sample if needed
    random.seed(seed)
    if len(prompts) > n_samples:
        prompts = random.sample(prompts, n_samples)
        print(f"Sampled {n_samples:,} prompts")

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")

    print(f"✓ Saved to {output_path}")
    return prompts


def main():
    """Run data preparation."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Arena prompts")
    parser.add_argument("--samples", type=int, default=1000, help="Number of prompts")
    parser.add_argument(
        "--output", type=str, default="data/arena_prompts.jsonl", help="Output file"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    prepare_arena_prompts(
        n_samples=args.samples, output_file=args.output, seed=args.seed
    )


if __name__ == "__main__":
    main()
