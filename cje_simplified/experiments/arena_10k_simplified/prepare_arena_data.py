#!/usr/bin/env python3
"""
Prepare ChatBot Arena data for CJE experiments.

Downloads ChatBot Arena conversations and extracts unique prompts.
This follows the key insight from the old codebase: deduplicate contexts
to ensure we're comparing fresh responses from our policies.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


def is_problematic_prompt(prompt: str) -> bool:
    """Check if a prompt is likely to cause oracle scoring failures.
    
    This filters out prompts that:
    1. Ask to spell offensive words letter by letter
    2. Contains patterns that lead to problematic content generation
    
    Args:
        prompt: The prompt to check
        
    Returns:
        True if the prompt should be filtered out
    """
    prompt_lower = prompt.lower()
    
    # Pattern 1: Letter-by-letter spelling that could form offensive words
    letter_patterns = [
        # N-I-G-G-E-R patterns
        r"write.*letters.*n.*i.*g.*g.*e.*r",
        r"spell.*n.*i.*g.*g.*e.*r", 
        r"sequence.*n.*i.*g.*g.*e.*r",
        # F-A-G-G-O-T patterns  
        r"write.*letters.*f.*a.*g.*g.*o.*t",
        r"spell.*f.*a.*g.*g.*o.*t",
        r"sequence.*f.*a.*g.*g.*o.*t",
    ]
    
    for pattern in letter_patterns:
        if re.search(pattern, prompt_lower):
            return True
    
    # Pattern 2: Explicit requests to spell offensive words
    offensive_words = ["nigger", "faggot", "retard"]
    for word in offensive_words:
        if word in prompt_lower:
            return True
    
    return False


def prepare_arena_prompts(
    n_samples: int = 1000,
    output_file: str = "arena_prompts.jsonl",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Download and prepare unique Arena prompts (first turn only).

    Args:
        n_samples: Number of unique prompts to extract
        output_file: Where to save the prompts
        seed: Random seed for sampling

    Returns:
        List of prompt dictionaries
    """
    print(f"Downloading ChatBot Arena conversations...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library required. Install with: pip install datasets")
        return []

    # Download dataset
    dataset = load_dataset("agie-ai/lmsys-chatbot_arena_conversations", split="train")
    print(f"Downloaded {len(dataset):,} conversations")

    # Extract unique prompts (key insight from old codebase)
    prompts = []
    seen = set()

    for i, row in enumerate(dataset):
        conv_id = row.get("conversation_id", f"conv_{i}")
        conversation = row.get("conversation_a", [])
        language = row.get("language", "unknown")

        # Skip non-English conversations
        if language not in ["English", "english", "en", "EN"]:
            continue

        # Extract first user turn only
        first_user_prompt = None
        for msg in conversation:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "").strip()
                if content:
                    first_user_prompt = content
                    break

        if not first_user_prompt:
            continue

        # Filter out problematic prompts that cause oracle scoring issues
        if is_problematic_prompt(first_user_prompt):
            continue

        # Skip duplicates (critical for proper policy comparison)
        if first_user_prompt in seen:
            continue
        seen.add(first_user_prompt)

        prompts.append(
            {
                "prompt_id": f"arena_{i}",
                "prompt": first_user_prompt,
                "language": language,
            }
        )

        if len(prompts) >= n_samples:
            break

    print(f"Extracted {len(prompts):,} unique English prompts")

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


def main() -> None:
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
        n_samples=args.samples,
        output_file=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
