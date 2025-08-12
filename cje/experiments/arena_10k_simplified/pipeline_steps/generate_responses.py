#!/usr/bin/env python3
"""
Generate responses for Arena prompts using different policies.

Uses Fireworks API with different system prompts to simulate different policies:
- base/clone: Helpful assistant
- unhelpful: Deliberately confusing assistant
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests  # type: ignore

import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))  # Add arena_10k_simplified to path

from policy_config import get_all_policies


def load_existing_responses(output_file: str) -> Dict[str, Dict]:
    """Load existing responses from file.

    Returns:
        Dictionary mapping prompt_id to response data
    """
    existing = {}
    corrupted_lines = 0
    if Path(output_file).exists():
        with open(output_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    data = json.loads(line)
                    if "prompt_id" in data:
                        existing[data["prompt_id"]] = data
                except json.JSONDecodeError as e:
                    corrupted_lines += 1
                    print(f"  ⚠️  Skipping corrupted line {line_num}: {e}")
                    continue  # Skip corrupted lines

    if corrupted_lines > 0:
        print(f"  ⚠️  Found {corrupted_lines} corrupted lines during resume")

    return existing


def generate_responses(
    prompts_file: str,
    output_file: str,
    model: str,
    temperature: float = 0.7,
    max_responses: Optional[int] = None,
    policy_name: str = "base",
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1000,
    batch_size: Optional[int] = None,
) -> List[Dict]:
    """Generate responses for prompts using Fireworks API.

    Args:
        prompts_file: Path to JSONL file with prompts
        output_file: Where to save responses
        model: Fireworks model identifier
        temperature: Sampling temperature
        max_responses: Limit number of responses to generate
        policy_name: Name of the policy (for tracking)
        system_prompt: System prompt to use
        max_tokens: Maximum number of tokens to generate

    Returns:
        List of response dictionaries
    """
    # Check for API key
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY environment variable required")

    # Load existing responses if resuming
    existing_responses = load_existing_responses(output_file) if batch_size else {}

    # Load prompts
    prompts = []
    with open(prompts_file, "r") as f:
        for line in f:
            prompt_data = json.loads(line)
            # Skip if already exists and using batching
            if batch_size and prompt_data.get("prompt_id") in existing_responses:
                continue
            prompts.append(prompt_data)

    if max_responses:
        # Adjust for existing responses
        total_needed = max_responses - len(existing_responses)
        prompts = prompts[:total_needed]

    if not prompts:
        print(
            f"✓ All {len(existing_responses)} responses already exist for {policy_name}"
        )
        return list(existing_responses.values())

    print(f"Generating {len(prompts)} new responses with {policy_name} policy...")
    if existing_responses:
        print(
            f"  📂 Resuming from previous run: {len(existing_responses)} already completed"
        )
        print(f"  🔄 Continuing with {len(prompts)} remaining responses")
    print(f"Model: {model}, Temperature: {temperature}, Max tokens: {max_tokens}")
    print(f"System prompt: {system_prompt[:50]}...")
    if batch_size:
        print(f"Batch size: {batch_size} (saving progress incrementally)")

    # Fireworks API endpoint
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Setup output file for appending if using batching
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # For batching, we'll write to temp file then rename atomically
    temp_file = None
    output_f = None
    if batch_size:
        # Copy existing file to temp if it exists
        temp_file = f"{output_file}.tmp"
        if output_path.exists():
            shutil.copy2(output_file, temp_file)
        output_f = open(temp_file, "a")

    # Generate responses
    results = list(existing_responses.values()) if batch_size else []

    try:
        for i, prompt_data in enumerate(prompts):
            prompt = prompt_data["prompt"]

            try:
                # Prepare messages with system prompt
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]

                # Call Fireworks API
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()

                response_data = response.json()

                result = {
                    "prompt_id": prompt_data["prompt_id"],
                    "prompt": prompt,
                    "response": response_data["choices"][0]["message"]["content"],
                    "policy": policy_name,
                    "model": model,
                    "temperature": temperature,
                }

                # Save immediately if using batching
                if batch_size and output_f is not None:
                    output_f.write(json.dumps(result) + "\n")
                    output_f.flush()  # Ensure written to disk
                    # Save progress every batch_size responses
                    if (i + 1) % batch_size == 0:
                        total_so_far = len(existing_responses) + i + 1
                        print(
                            f"  💾 Progress saved: {total_so_far} total responses ({i + 1} new this run)"
                        )
                else:
                    results.append(result)

                if not batch_size and (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{len(prompts)} responses...")

            except Exception as e:
                print(f"Error on prompt {prompt_data.get('prompt_id', i)}: {e}")
                # Add failed result
                result = {
                    "prompt_id": prompt_data["prompt_id"],
                    "prompt": prompt,
                    "response": None,
                    "policy": policy_name,
                    "error": str(e),
                }

                if batch_size and output_f is not None:
                    output_f.write(json.dumps(result) + "\n")
                    output_f.flush()
                else:
                    results.append(result)

    finally:
        # Always close file if using batching
        if batch_size and output_f:
            output_f.close()
            # Atomic rename from temp to final
            if temp_file and Path(temp_file).exists():
                os.replace(temp_file, output_file)
                total_results = len(existing_responses) + len(prompts)
                print(f"✓ Saved {total_results} total responses to {output_path}")

    # Handle return for batch mode
    if batch_size:
        # Return all results including existing
        return list(load_existing_responses(output_file).values())
    else:
        # Save all results at once (original behavior)
        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        print(f"✓ Saved {len(results)} responses to {output_path}")
        return results


def main() -> None:
    """Generate responses for different policies."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts", default="data/arena_prompts.jsonl", help="Input prompts file"
    )
    parser.add_argument(
        "--output-dir", default="data/responses", help="Output directory"
    )
    parser.add_argument("--max-responses", type=int, help="Limit number of responses")
    parser.add_argument(
        "--max-tokens", type=int, default=1000, help="Maximum tokens per response"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Save progress every N responses (0 to disable)",
    )

    args = parser.parse_args()

    # Get policies from centralized configuration
    policies = get_all_policies()

    for policy in policies:
        output_file = f"{args.output_dir}/{policy['name']}_responses.jsonl"
        generate_responses(
            prompts_file=args.prompts,
            output_file=output_file,
            model=policy["model"],
            temperature=policy["temperature"],
            policy_name=policy["name"],
            system_prompt=policy["system_prompt"],
            max_responses=args.max_responses,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size if args.batch_size > 0 else None,
        )


if __name__ == "__main__":
    main()
