#!/usr/bin/env python3
"""
Prepare data for CJE analysis by combining responses, log probs, and judge scores.

This creates the final dataset in the format expected by PrecomputedSampler.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


def prepare_cje_dataset(
    logprobs_dir: str,
    output_file: str,
    base_policy: str = "base",
    judge_score_range: tuple = (0, 1),
) -> List[Dict]:
    """Combine log probabilities from different policies into CJE format."""

    print("Preparing CJE dataset...")

    # Load all log prob files
    logprobs_by_prompt = defaultdict(dict)
    policies = set()

    logprobs_path = Path(logprobs_dir)
    for file in logprobs_path.glob("*_logprobs.jsonl"):
        policy = file.stem.replace("_logprobs", "")
        policies.add(policy)

        print(f"Loading {policy} log probabilities...")
        with open(file, "r") as f:
            for line in f:
                data = json.loads(line)
                prompt_id = data["prompt_id"]

                # Store data keyed by prompt_id
                if policy == base_policy:
                    # Base policy data
                    logprobs_by_prompt[prompt_id]["prompt"] = data["prompt"]
                    logprobs_by_prompt[prompt_id]["response"] = data["response"]
                    logprobs_by_prompt[prompt_id]["p0_logprob"] = data["logprob"]
                else:
                    # Target policy log probs
                    if "target_logps" not in logprobs_by_prompt[prompt_id]:
                        logprobs_by_prompt[prompt_id]["target_logps"] = {}
                    logprobs_by_prompt[prompt_id]["target_logps"][policy] = data[
                        "logprob"
                    ]

    print(f"Found {len(policies)} policies: {sorted(policies)}")

    # Create CJE format records
    records = []
    for prompt_id, data in logprobs_by_prompt.items():
        # Skip if missing base policy data
        if "p0_logprob" not in data or data["p0_logprob"] is None:
            continue

        # Skip if no valid target policies
        target_logps = data.get("target_logps", {})
        if not any(lp is not None for lp in target_logps.values()):
            continue

        # Add synthetic judge score (in practice, would come from a judge model)
        # For demo, we'll use a simple heuristic based on response length
        response_len = len(data.get("response", ""))
        judge_score = min(
            max(response_len / 500, judge_score_range[0]), judge_score_range[1]
        )

        record = {
            "prompt": data["prompt"],
            "response": data["response"],
            "p0_logprob": data["p0_logprob"],
            "target_logps": target_logps,
            "judge_score": judge_score,
            # Reward will be computed by calibration later
        }
        records.append(record)

    print(f"Created {len(records)} complete records")

    # Save dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"✓ Saved CJE dataset to {output_path}")

    # Print summary statistics
    print("\nDataset summary:")
    print(f"  Total samples: {len(records)}")
    print(f"  Base policy: {base_policy}")
    print(f"  Target policies: {sorted(p for p in policies if p != base_policy)}")

    valid_counts = defaultdict(int)
    for record in records:
        for policy, logprob in record["target_logps"].items():
            if logprob is not None:
                valid_counts[policy] += 1

    print("\nValid log probs per policy:")
    for policy, count in sorted(valid_counts.items()):
        print(f"  {policy}: {count}/{len(records)} ({100*count/len(records):.1f}%)")

    return records


def add_oracle_labels(
    dataset_file: str, oracle_fraction: float = 0.1, seed: int = 42
) -> None:
    """Add oracle labels to a fraction of the data for judge calibration."""

    import random

    random.seed(seed)

    # Load dataset
    records = []
    with open(dataset_file, "r") as f:
        for line in f:
            records.append(json.loads(line))

    # Select subset for oracle labeling
    n_oracle = int(len(records) * oracle_fraction)
    oracle_indices = set(random.sample(range(len(records)), n_oracle))

    print(
        f"Adding oracle labels to {n_oracle}/{len(records)} samples ({oracle_fraction:.0%})"
    )

    # Add oracle labels (in practice, these would come from human evaluation)
    for i, record in enumerate(records):
        if i in oracle_indices:
            # Synthetic oracle label with some noise
            base_score = record["judge_score"]
            oracle_label = min(max(base_score + 0.1 * random.gauss(0, 1), 0), 1)
            record["oracle_label"] = oracle_label

    # Save updated dataset
    with open(dataset_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"✓ Added oracle labels to dataset")


def main():
    """Prepare complete CJE dataset."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logprobs-dir", default="data/logprobs", help="Directory with log prob files"
    )
    parser.add_argument(
        "--output", default="data/cje_dataset.jsonl", help="Output CJE dataset"
    )
    parser.add_argument(
        "--base-policy", default="base", help="Name of base/behavior policy"
    )
    parser.add_argument(
        "--add-oracle", action="store_true", help="Add synthetic oracle labels"
    )
    parser.add_argument(
        "--oracle-fraction", type=float, default=0.1, help="Fraction to label"
    )

    args = parser.parse_args()

    # Prepare dataset
    prepare_cje_dataset(
        logprobs_dir=args.logprobs_dir,
        output_file=args.output,
        base_policy=args.base_policy,
    )

    # Optionally add oracle labels
    if args.add_oracle:
        add_oracle_labels(
            dataset_file=args.output, oracle_fraction=args.oracle_fraction
        )


if __name__ == "__main__":
    main()
