#!/usr/bin/env python3
"""
Prepare CJE datasets with different oracle coverage levels.

This script reads the immutable response and logprob files, then creates
datasets where only a fraction of samples have oracle labels for calibration.
The remaining samples use calibrated judge scores.
"""

import argparse
import json
import random
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from cje_simplified import (
    JudgeCalibrator,
    calibrate_judge_scores,
)


def load_response_data(response_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load all response files."""
    response_data = {}

    for policy_file in response_dir.glob("*_responses.jsonl"):
        policy = policy_file.stem.replace("_responses", "")
        responses = []

        with open(policy_file) as f:
            for line in f:
                data = json.loads(line)
                responses.append(data)

        response_data[policy] = responses
        print(f"  Loaded {len(responses)} {policy} responses")

    return response_data


def load_logprob_data(logprob_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load all logprob files."""
    logprob_data = {}

    for policy_file in logprob_dir.glob("*_logprobs.jsonl"):
        policy = policy_file.stem.replace("_logprobs", "")
        logprobs = []

        with open(policy_file) as f:
            for line in f:
                data = json.loads(line)
                logprobs.append(data)

        logprob_data[policy] = logprobs
        print(f"  Loaded {len(logprobs)} {policy} logprobs")

    return logprob_data


def create_ablation_dataset(
    response_data: Dict[str, List[Dict]],
    logprob_data: Dict[str, List[Dict]],
    oracle_fraction: float,
    seed: int,
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
) -> List[Dict[str, Any]]:
    """Create dataset with specified oracle coverage."""

    random.seed(seed)
    np.random.seed(seed)

    # Use base policy responses as reference
    base_responses = response_data["base"]
    n_samples = len(base_responses)

    # Randomly select samples for oracle calibration
    n_oracle = int(n_samples * oracle_fraction)
    oracle_indices = set(random.sample(range(n_samples), n_oracle))

    print(f"\nCalibration setup:")
    print(f"  Total samples: {n_samples}")
    print(f"  Oracle samples: {n_oracle} ({oracle_fraction:.1%})")
    print(f"  Random seed: {seed}")

    # Collect all judge scores and oracle labels
    all_judges = []
    all_oracles = []

    for i, resp in enumerate(base_responses):
        # Judge and oracle scores are in metadata
        metadata = resp.get("metadata", {})
        judge_score = metadata.get(judge_field)
        oracle_label = metadata.get(oracle_field)

        if judge_score is None or oracle_label is None:
            raise ValueError(f"Missing judge or oracle at index {i}")

        all_judges.append(judge_score)
        all_oracles.append(oracle_label)

    # Create oracle mask for calibration
    oracle_mask = np.zeros(len(all_judges), dtype=bool)
    for i in oracle_indices:
        oracle_mask[i] = True

    # Get oracle labels for calibration subset
    oracle_labels_for_calibration = []
    for i in range(len(all_judges)):
        if oracle_mask[i]:
            oracle_labels_for_calibration.append(all_oracles[i])

    # Calibrate judge scores using only the selected subset
    print(f"\nCalibrating with {sum(oracle_mask)} oracle samples...")
    calibrated_scores, diagnostics = calibrate_judge_scores(
        judge_scores=np.array(all_judges),
        oracle_labels=np.array(oracle_labels_for_calibration),
        oracle_mask=oracle_mask,
        k_folds=5,
    )

    print(f"  Calibration RMSE: {diagnostics['rmse']:.3f}")
    print(f"  Coverage (±0.1): {diagnostics['coverage']:.1%}")

    # Create CJE dataset
    cje_samples = []

    for i in range(n_samples):
        # Get base response data
        base_resp = base_responses[i]
        prompt = base_resp["prompt"]
        response = base_resp["response"]

        # Collect target policy logprobs
        target_logprobs = {}
        for policy in response_data:
            if policy == "base":
                continue
            target_logprobs[policy] = logprob_data[policy][i]["logprob"]

        # Create sample
        sample = {
            "prompt": prompt,
            "response": response,
            "reward": float(calibrated_scores[i]),  # Use calibrated reward
            "base_policy_logprob": logprob_data["base"][i]["logprob"],
            "target_policy_logprobs": target_logprobs,
            "metadata": {
                judge_field: base_resp.get("metadata", {}).get(judge_field),
                oracle_field: base_resp.get("metadata", {}).get(oracle_field),
                "used_for_calibration": i in oracle_indices,
                "oracle_fraction": oracle_fraction,
                "calibration_seed": seed,
            },
        }

        cje_samples.append(sample)

    return cje_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare CJE dataset with specified oracle coverage"
    )
    parser.add_argument(
        "--response-dir",
        type=Path,
        default=Path("test_e2e_data/responses"),
        help="Directory containing response files",
    )
    parser.add_argument(
        "--logprob-dir",
        type=Path,
        default=Path("test_e2e_data/logprobs"),
        help="Directory containing logprob files",
    )
    parser.add_argument(
        "--oracle-fraction",
        type=float,
        required=True,
        help="Fraction of samples to use for oracle calibration (0.0-1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for CJE dataset",
    )
    parser.add_argument(
        "--judge-field",
        default="judge_score",
        help="Field name containing judge scores",
    )
    parser.add_argument(
        "--oracle-field",
        default="oracle_label",
        help="Field name containing oracle labels",
    )

    args = parser.parse_args()

    # Validate oracle fraction
    if not 0.0 < args.oracle_fraction <= 1.0:
        raise ValueError("oracle_fraction must be between 0 and 1")

    print("Preparing ablation dataset")
    print("=" * 50)

    # Load data
    print("\nLoading response data...")
    response_data = load_response_data(args.response_dir)

    print("\nLoading logprob data...")
    logprob_data = load_logprob_data(args.logprob_dir)

    # Validate data consistency
    policies = set(response_data.keys())
    if policies != set(logprob_data.keys()):
        raise ValueError("Mismatch between response and logprob policies")

    # Create dataset
    cje_samples = create_ablation_dataset(
        response_data,
        logprob_data,
        args.oracle_fraction,
        args.seed,
        args.judge_field,
        args.oracle_field,
    )

    # Write output
    print(f"\nWriting {len(cje_samples)} samples to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for sample in cje_samples:
            f.write(json.dumps(sample) + "\n")

    print("\n✓ Dataset created successfully!")

    # Summary statistics
    rewards = [s["reward"] for s in cje_samples]
    print(f"\nReward statistics:")
    print(f"  Mean: {np.mean(rewards):.3f}")
    print(f"  Std: {np.std(rewards):.3f}")
    print(f"  Range: [{np.min(rewards):.3f}, {np.max(rewards):.3f}]")


if __name__ == "__main__":
    main()
