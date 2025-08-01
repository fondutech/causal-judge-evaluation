#!/usr/bin/env python3
"""
Add oracle labels to all records in the CJE dataset.

Oracle labels represent ground truth evaluations used for validation
and calibration of judge scores.

For the API oracle with structured outputs, you need:
  pip install langchain-fireworks pydantic
"""

import json
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation_utils import FireworksEvaluator, DEFAULT_ORACLE_MODEL


def add_oracle_labels(
    input_file: str,
    output_file: str,
    oracle_field: str = "oracle_label",
    show_progress: bool = True,
) -> None:
    """Add oracle labels to all records in the dataset.

    Oracle labels serve as ground truth for validation and calibration.

    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file
        oracle_field: Name for oracle label field in metadata
        show_progress: Whether to show progress bar
    """

    print(f"Adding oracle labels to dataset...")
    print(f"Input: {input_file}")
    print(f"Oracle field: {oracle_field}")

    # Create oracle
    oracle = FireworksEvaluator(model=DEFAULT_ORACLE_MODEL)
    print(f"Using oracle model: {oracle.model}")

    # Load records
    records = []
    with open(input_file, "r") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    # Collect prompts and responses for all records
    prompts = []
    responses = []
    valid_indices = []

    for i, record in enumerate(records):
        prompt = record.get("prompt", "")
        response = record.get("response", "")
        if prompt and response:
            prompts.append(prompt)
            responses.append(response)
            valid_indices.append(i)

    if not prompts:
        print("No valid records to evaluate.")
        return

    # Score in batch with failure resilience
    print(f"\nEvaluating {len(prompts)} records...")
    result = oracle.score_batch(
        prompts, responses, show_progress=show_progress, desc="Oracle evaluations", skip_failures=True
    )

    # Update records with oracle labels
    for idx, record_idx in enumerate(valid_indices):
        # Ensure metadata exists
        if "metadata" not in records[record_idx]:
            records[record_idx]["metadata"] = {}

        # Handle failed scorings - set to None if scoring failed
        score = result.scores[idx] if result.scores[idx] is not None else None
        records[record_idx]["metadata"][oracle_field] = score

    oracle_count = len(result.scores)

    # Save output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\n✓ Added {oracle_count} oracle labels")
    print(f"✓ Saved to {output_path}")

    # Print statistics - filter out None values for failed scorings
    oracle_labels = [
        r["metadata"][oracle_field]
        for r in records
        if oracle_field in r.get("metadata", {}) and r["metadata"][oracle_field] is not None
    ]

    valid_scores = [s for s in result.scores if s is not None]
    failed_count = len(result.scores) - len(valid_scores)

    if valid_scores:
        print(f"\nOracle label statistics:")
        print(f"  Valid scores: {len(valid_scores)}")
        if failed_count > 0:
            print(f"  Failed scores: {failed_count}")
        print(f"  Mean: {np.mean(valid_scores):.3f}")
        print(f"  Std:  {np.std(valid_scores):.3f}")
        print(f"  Min:  {min(valid_scores):.3f}")
        print(f"  Max:  {max(valid_scores):.3f}")
    else:
        print(f"\nNo valid oracle scores obtained")

    # Compare with judge scores if available (only for valid oracle scores)
    judge_scores = []
    oracle_for_judged = []

    for r in records:
        metadata = r.get("metadata", {})
        if (oracle_field in metadata and "judge_score" in metadata and 
            metadata[oracle_field] is not None and metadata["judge_score"] is not None):
            judge_scores.append(metadata["judge_score"])
            oracle_for_judged.append(metadata[oracle_field])

    if judge_scores and oracle_for_judged:
        if len(judge_scores) > 1:  # Need at least 2 points for correlation
            correlation = np.corrcoef(judge_scores, oracle_for_judged)[0, 1]
            print(f"\nJudge-Oracle correlation: {correlation:.3f}")
        else:
            print(f"\nJudge-Oracle correlation: N/A (need at least 2 points)")

        mean_diff = np.mean(np.array(oracle_for_judged) - np.array(judge_scores))
        print(f"Mean difference (oracle - judge): {mean_diff:+.3f}")


def main() -> None:
    """Add oracle labels with command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Add oracle labels to CJE dataset")
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file with judge scores",
    )
    parser.add_argument(
        "--output",
        help="Output JSONL file (defaults to input file)",
    )

    args = parser.parse_args()

    # Default output to input
    output_file = args.output or args.input

    # Add oracle labels
    add_oracle_labels(
        input_file=args.input,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()
