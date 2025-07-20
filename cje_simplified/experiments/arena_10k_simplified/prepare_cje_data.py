#!/usr/bin/env python3
"""
Prepare data for CJE analysis by combining BASE policy responses and log probs.

This creates the final dataset in the format expected by CJE analysis:
- Uses BASE policy responses for all samples
- Includes log probabilities under all policy models
- Adds judge scores to metadata for calibration
- Follows the core data model structure
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from typing import Dict, Set, Any

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))


def prepare_cje_dataset(
    logprobs_dir: str,
    output_file: str,
    base_policy: str = "base",
) -> List[Dict]:
    """Combine BASE policy responses with log probs from all policies.

    Creates records in the format expected by the CJE data model:
    - prompt: The input prompt
    - response: The BASE policy's response
    - base_policy_logprob: Log P(response | prompt) under base policy
    - target_policy_logprobs: Dict of log P(response | prompt) under each policy
    - metadata: Additional fields including judge_score for calibration
    """

    print("Preparing CJE dataset...")

    # Load all log prob files
    logprobs_by_prompt: Dict[str, Dict[str, Any]] = defaultdict(dict)
    policies: Set[str] = set()

    logprobs_path = Path(logprobs_dir)
    for file in logprobs_path.glob("*_logprobs.jsonl"):
        policy = file.stem.replace("_logprobs", "")
        policies.add(policy)

        print(f"Loading {policy} log probabilities...")
        with open(file, "r") as f:
            for line in f:
                data = json.loads(line)
                prompt_id = data["prompt_id"]

                # All files should have the same BASE responses
                if "prompt" not in logprobs_by_prompt[prompt_id]:
                    logprobs_by_prompt[prompt_id]["prompt"] = data["prompt"]
                    logprobs_by_prompt[prompt_id]["response"] = data["response"]
                    logprobs_by_prompt[prompt_id]["prompt_id"] = prompt_id

                # Store log prob under this policy's model
                if policy == base_policy:
                    logprobs_by_prompt[prompt_id]["base_policy_logprob"] = data[
                        "logprob"
                    ]
                else:
                    if "target_policy_logprobs" not in logprobs_by_prompt[prompt_id]:
                        logprobs_by_prompt[prompt_id]["target_policy_logprobs"] = {}
                    logprobs_by_prompt[prompt_id]["target_policy_logprobs"][policy] = (
                        data["logprob"]
                    )

    print(f"Found {len(policies)} policies: {sorted(policies)}")

    # Create CJE format records
    records = []
    for prompt_id, data in logprobs_by_prompt.items():
        # Skip if missing base policy data
        if "base_policy_logprob" not in data or data["base_policy_logprob"] is None:
            continue

        # Skip if no valid target policies
        target_logps = data.get("target_policy_logprobs", {})
        if not any(lp is not None for lp in target_logps.values()):
            continue

        # Create record following core data model structure
        record = {
            "prompt": data["prompt"],
            "response": data["response"],
            "base_policy_logprob": data["base_policy_logprob"],
            "target_policy_logprobs": target_logps,
            # Note: reward field is left empty - will be added by calibration
            # metadata contains fields for calibration (judge_score, oracle_label)
            "metadata": {
                "prompt_id": data.get("prompt_id", prompt_id),
            },
        }

        # Note: judge_score will be added by a separate script

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

    valid_counts: Dict[str, int] = defaultdict(int)
    for record in records:
        for policy, logprob in record["target_policy_logprobs"].items():
            if logprob is not None:
                valid_counts[policy] += 1

    print("\nValid log probs per policy:")
    for policy, count in sorted(valid_counts.items()):
        print(f"  {policy}: {count}/{len(records)} ({100*count/len(records):.1f}%)")

    return records


def main() -> None:
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

    args = parser.parse_args()

    # Prepare dataset
    prepare_cje_dataset(
        logprobs_dir=args.logprobs_dir,
        output_file=args.output,
        base_policy=args.base_policy,
    )


if __name__ == "__main__":
    main()
