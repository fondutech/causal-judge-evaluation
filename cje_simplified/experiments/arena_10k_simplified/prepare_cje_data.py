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
    responses_dir: str,
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

    # First, load base responses to get judge/oracle scores
    responses_by_prompt: Dict[str, Dict[str, Any]] = {}
    base_responses_file = Path(responses_dir) / f"{base_policy}_responses.jsonl"

    print(f"Loading base responses from {base_responses_file}...")
    with open(base_responses_file, "r") as f:
        for line in f:
            data = json.loads(line)
            prompt_id = data.get("prompt_id")
            if prompt_id:
                responses_by_prompt[prompt_id] = data

    print(f"Loaded {len(responses_by_prompt)} base responses with evaluation scores")

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

        # Get evaluation scores from base responses
        base_response_data = responses_by_prompt.get(prompt_id, {})
        metadata = {
            "prompt_id": data.get("prompt_id", prompt_id),
        }

        # Add judge and oracle scores if available
        if "metadata" in base_response_data:
            response_metadata = base_response_data["metadata"]
            if "judge_score" in response_metadata:
                metadata["judge_score"] = response_metadata["judge_score"]
            if "oracle_label" in response_metadata:
                metadata["oracle_label"] = response_metadata["oracle_label"]

        # Create record following core data model structure
        record = {
            "prompt": data["prompt"],
            "response": data["response"],
            "base_policy_logprob": data["base_policy_logprob"],
            "target_policy_logprobs": target_logps,
            # Note: reward field is left empty - will be added by calibration
            "metadata": metadata,
        }

        records.append(record)

    # Track dropped records
    total_prompts = len(logprobs_by_prompt)
    dropped_base = sum(
        1
        for d in logprobs_by_prompt.values()
        if "base_policy_logprob" not in d or d.get("base_policy_logprob") is None
    )
    dropped_all_targets = sum(
        1
        for d in logprobs_by_prompt.values()
        if "base_policy_logprob" in d
        and d.get("base_policy_logprob") is not None
        and not any(
            lp is not None for lp in d.get("target_policy_logprobs", {}).values()
        )
    )

    print(f"Created {len(records)} complete records from {total_prompts} prompts")
    if dropped_base > 0:
        print(f"⚠️  Dropped {dropped_base} records with null base policy logprob")
    if dropped_all_targets > 0:
        print(f"⚠️  Dropped {dropped_all_targets} records with all target logprobs null")

    # Save dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"✓ Saved CJE dataset to {output_path}")

    # Warn if too few samples
    if len(records) < 10:
        print(
            f"\n⚠️  WARNING: Only {len(records)} samples in dataset. Minimum 10 recommended for reliable CJE analysis."
        )

    # Print summary statistics
    print("\nDataset summary:")
    print(f"  Total samples: {len(records)}")
    print(f"  Base policy: {base_policy}")
    print(f"  Target policies: {sorted(p for p in policies if p != base_policy)}")

    # Check how many records have evaluation scores
    with_judge = sum(1 for r in records if "judge_score" in r.get("metadata", {}))
    with_oracle = sum(1 for r in records if "oracle_label" in r.get("metadata", {}))
    print(f"\nEvaluation scores:")
    print(
        f"  Records with judge scores: {with_judge}/{len(records)} ({100*with_judge/len(records):.1f}%)"
    )
    print(
        f"  Records with oracle labels: {with_oracle}/{len(records)} ({100*with_oracle/len(records):.1f}%)"
    )

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
        "--responses-dir",
        default="data/responses",
        help="Directory with response files",
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
        responses_dir=args.responses_dir,
        output_file=args.output,
        base_policy=args.base_policy,
    )


if __name__ == "__main__":
    main()
