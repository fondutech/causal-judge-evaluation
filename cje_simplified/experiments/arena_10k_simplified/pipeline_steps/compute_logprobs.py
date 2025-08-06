#!/usr/bin/env python3
"""
Compute log probabilities for responses using teacher forcing.

Simplified version with single API call per sample.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))  # Add arena_10k_simplified to path

from cje_simplified import compute_chat_logprob
from policy_config import POLICIES, get_policy_config, POLICY_NAMES


def load_existing_logprobs(output_file: str) -> Dict[str, Dict]:
    """Load existing log probabilities from file.

    Returns:
        Dictionary mapping prompt_id to logprob data
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


def compute_logprobs_for_responses(
    base_responses_file: str,
    output_file: str,
    max_samples: Optional[int] = None,
    policy_name: str = "base",
    batch_size: Optional[int] = None,
) -> List[Dict]:
    """Compute log probabilities for BASE policy responses under a given policy's model.

    Simplified to use single API call per sample.

    Args:
        base_responses_file: Path to BASE policy responses JSONL file
        output_file: Where to save log probabilities
        max_samples: Limit number of samples to process
        policy_name: Name of the policy whose model to use for computing log probs
        batch_size: Save progress every N samples (for resume capability)

    Returns:
        List of dictionaries with log probability results
    """
    if not os.getenv("FIREWORKS_API_KEY"):
        raise ValueError("FIREWORKS_API_KEY environment variable required")

    # Get policy configuration
    policy_config = get_policy_config(policy_name)
    model = policy_config["model"]
    temperature = policy_config["temperature"]
    system_prompt = policy_config["system_prompt"]
    template_config = policy_config.get(
        "template_config"
    )  # Get template config from policy

    # Load existing logprobs if resuming
    existing_logprobs = load_existing_logprobs(output_file) if batch_size else {}

    # Load BASE policy responses
    responses = []
    with open(base_responses_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if data.get("response"):  # Skip failed responses
                # Skip if already computed and using batching
                if batch_size and data.get("prompt_id") in existing_logprobs:
                    continue
                responses.append(data)

    if max_samples:
        # Adjust for existing logprobs
        total_needed = max_samples - len(existing_logprobs)
        responses = responses[:total_needed]

    if not responses:
        print(
            f"✓ All {len(existing_logprobs)} log probs already computed for {policy_name}"
        )
        return list(existing_logprobs.values())

    print(f"Computing log probs for {len(responses)} BASE responses...")
    if existing_logprobs:
        print(
            f"  📂 Resuming from previous run: {len(existing_logprobs)} already completed"
        )
        print(f"  🔄 Continuing with {len(responses)} remaining computations")
    print(f"Using {policy_name} policy model: {model}")
    print(f"Temperature: {temperature}, System prompt: {system_prompt[:50]}...")
    if batch_size:
        print(f"Batch size: {batch_size} (saving progress incrementally)")

    # Log template configuration
    if template_config:
        print(f"Using explicit template config: {template_config.__class__.__name__}")
    else:
        print("Using auto-detected template for Fireworks model")

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

    # Compute log probabilities
    results: List[Dict[str, Any]] = (
        list(existing_logprobs.values()) if batch_size else []
    )
    failed_count = 0
    successful_count = 0

    try:
        for i, data in enumerate(responses):
            prompt = data["prompt"]
            response = data["response"]

            # Create chat format with system prompt
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]

            # Single API call with simple retry logic
            start_time = time.time()

            # Simple retry with exponential backoff
            max_retries = 3
            retry_delay = 1.0  # Initial delay in seconds

            for attempt in range(max_retries):
                result = compute_chat_logprob(
                    chat=chat,
                    model=model,
                    temperature=temperature,
                    template_config=template_config,
                )

                # Break if successful
                if result.is_valid and result.value is not None:
                    break

                # On failure, retry unless it's the last attempt
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Double the delay for next attempt
                    if attempt == 0:  # Only print on first retry
                        print(f"    Retrying after error: {result.error}")

            duration = time.time() - start_time

            # Record result
            if result.is_valid and result.value is not None:
                logprob = result.value
                error = None
                successful_count += 1
                print(
                    f"  [{i + 1}/{len(responses)}] {data['prompt_id']}: {logprob:.3f} ({duration:.1f}s)"
                )
            else:
                logprob = None
                error = result.error or "Unknown error"
                failed_count += 1
                print(
                    f"  [{i + 1}/{len(responses)}] {data['prompt_id']}: FAILED - {error}"
                )

            output_record = {
                "prompt_id": data["prompt_id"],
                "prompt": prompt,
                "response": response,
                "source_policy": "base",  # Always computing base responses
                "eval_model": model,
                "logprob": logprob,
                "error": error,
            }

            # Save immediately if using batching
            if batch_size and output_f:
                output_f.write(json.dumps(output_record) + "\n")
                output_f.flush()  # Ensure written to disk
                # Save progress message every batch_size computations
                if (i + 1) % batch_size == 0:
                    total_so_far = len(existing_logprobs) + i + 1
                    print(
                        f"  💾 Progress saved: {total_so_far} total log probs ({i + 1} new this run)"
                    )
            else:
                results.append(output_record)

    finally:
        # Always close file if using batching
        if batch_size and output_f:
            output_f.close()
            # Atomic rename from temp to final
            if temp_file and Path(temp_file).exists():
                os.replace(temp_file, output_file)
                total_results = len(existing_logprobs) + len(responses)
                print(f"\n✓ Saved {total_results} total log probs to {output_path}")

    # Handle return for batch mode
    if batch_size:
        # Return all results including existing
        return list(load_existing_logprobs(output_file).values())
    else:
        # Save all results at once (original behavior)
        print(f"\nSaving results to {output_file}")
        with open(output_file, "w") as f:
            for record in results:
                f.write(json.dumps(record) + "\n")

    # Print summary
    print(f"\nSummary:")
    print(f"  Successful: {successful_count}/{len(responses)}")
    print(f"  Failed: {failed_count}/{len(responses)}")

    if successful_count > 0:
        logprobs = [r["logprob"] for r in results if r["logprob"] is not None]
        mean_logprob = sum(logprobs) / len(logprobs)
        min_logprob = min(logprobs)
        max_logprob = max(logprobs)
        print(f"  Mean logprob: {mean_logprob:.3f}")
        print(f"  Range: [{min_logprob:.3f}, {max_logprob:.3f}]")

    return results


def main() -> None:
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute log probabilities for responses (simplified single-sample version)"
    )
    parser.add_argument(
        "--responses-dir",
        required=True,
        help="Directory containing response JSONL files",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save log probability files"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=POLICY_NAMES,
        help="List of policies to compute log probs for (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Save progress every N log probs (0 to disable)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each policy
    base_responses_file = Path(args.responses_dir) / "base_responses.jsonl"
    if not base_responses_file.exists():
        raise FileNotFoundError(f"Base responses file not found: {base_responses_file}")

    for policy in args.policies:
        print(f"\n{'=' * 60}")
        print(f"Computing log probs for {policy} policy")
        print(f"{'=' * 60}")

        output_file = output_dir / f"{policy}_logprobs.jsonl"
        compute_logprobs_for_responses(
            base_responses_file=str(base_responses_file),
            output_file=str(output_file),
            max_samples=args.max_samples,
            policy_name=policy,
            batch_size=args.batch_size if args.batch_size > 0 else None,
        )


if __name__ == "__main__":
    main()
