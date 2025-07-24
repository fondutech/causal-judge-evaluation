#!/usr/bin/env python3
"""
Add judge scores to CJE dataset.

For the API judge with structured outputs, you need:
  pip install langchain-fireworks pydantic
"""

import json
from pathlib import Path
from typing import Any
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation_utils import FireworksEvaluator, DEFAULT_JUDGE_MODEL


def add_judge_scores(
    input_file: str,
    output_file: str,
    judge: Any,
    show_progress: bool = True,
) -> None:
    """Add judge scores to dataset.

    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file
        judge: Judge instance to use for scoring
        show_progress: Whether to show progress bar
    """
    print(f"Adding judge scores to dataset...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")

    # Load records
    records = []
    with open(input_file, "r") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    # Collect records that need scoring
    to_score_indices = []
    prompts = []
    responses = []

    for i, record in enumerate(records):
        # Ensure metadata exists
        if "metadata" not in record:
            record["metadata"] = {}

        # Always score (overwrite existing scores)

        # Check if we have prompt and response
        prompt = record.get("prompt", "")
        response = record.get("response", "")

        if prompt and response:
            to_score_indices.append(i)
            prompts.append(prompt)
            responses.append(response)

    if not to_score_indices:
        print("No records need scoring.")
        return

    print(f"\nScoring {len(to_score_indices)} records...")

    # Score in batch
    result = judge.score_batch(prompts, responses, show_progress=show_progress)

    # Update records with scores
    for idx, record_idx in enumerate(to_score_indices):
        records[record_idx]["metadata"]["judge_score"] = result.scores[idx]

    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\n✓ Added {len(to_score_indices)} judge scores")
    print(f"✓ Saved to {output_path}")

    # Print score statistics
    print(f"\nScore statistics:")
    print(f"  Mean: {result.mean_score:.3f}")
    print(f"  Std:  {result.std_score:.3f}")
    print(f"  Min:  {min(result.scores):.3f}")
    print(f"  Max:  {max(result.scores):.3f}")


def main() -> None:
    """Add judge scores with minimal command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Add judge scores to CJE dataset")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", help="Output JSONL file (defaults to input)")

    args = parser.parse_args()

    # Default output to input
    output_file = args.output or args.input

    # Create judge with default model
    judge = FireworksEvaluator(model=DEFAULT_JUDGE_MODEL)
    print(f"Using judge model: {judge.model}")

    # Add scores
    add_judge_scores(
        input_file=args.input,
        output_file=output_file,
        judge=judge,
    )


if __name__ == "__main__":
    main()
