#!/usr/bin/env python3
"""
Create a minimal test dataset for Arena 10K experiment.
This allows us to test the full pipeline quickly without processing 10,000 samples.
"""

import json
import random
from pathlib import Path


def create_test_dataset(n_samples: int = 20) -> None:
    """Create a small test dataset with all required files."""

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Create labeling directory
    labeling_dir = data_dir / "labeling"
    labeling_dir.mkdir(exist_ok=True)

    # Generate test prompts
    prompts = []
    for i in range(n_samples):
        prompts.append(
            {
                "prompt_id": f"test_{i:04d}",
                "prompt": f"Test prompt {i}: What is {i} + {i}?",
                "context": f"Test context for prompt {i}",
            }
        )

    # 1. Create logging policy responses with scores
    p0_scored_det = []
    p0_scored_unc = []

    for prompt in prompts:
        base_response = {
            "prompt_id": prompt["prompt_id"],
            "prompt": prompt["prompt"],
            "response": f"The answer is {int(prompt['prompt_id'].split('_')[1]) * 2}",
            "model": "pi_0",
            "logp": random.uniform(-10, -5),
        }

        # Deterministic scoring
        p0_scored_det.append(
            {
                **base_response,
                "judge_score": {"mean": random.uniform(0.6, 0.8), "variance": 0.0},
            }
        )

        # Uncertainty scoring
        p0_scored_unc.append(
            {
                **base_response,
                "judge_score": {
                    "mean": random.uniform(0.6, 0.8),
                    "variance": random.uniform(0.01, 0.05),
                },
            }
        )

    # 2. Create target policy responses with scores
    target_policies = ["pi_cot", "pi_bigger_model", "pi_bad"]
    targets_scored_det = []
    targets_scored_unc = []

    for prompt in prompts:
        for policy in target_policies:
            if policy == "pi_cot":
                response = f"Let me think step by step: {int(prompt['prompt_id'].split('_')[1])} + {int(prompt['prompt_id'].split('_')[1])} = {int(prompt['prompt_id'].split('_')[1]) * 2}"
                score_mean = random.uniform(0.8, 0.95)
            elif policy == "pi_bigger_model":
                response = f"The answer is {int(prompt['prompt_id'].split('_')[1]) * 2}. This is a simple addition."
                score_mean = random.uniform(0.85, 0.9)
            else:  # pi_bad
                response = f"I don't know, maybe {random.randint(0, 100)}?"
                score_mean = random.uniform(0.2, 0.4)

            base_response = {
                "prompt_id": prompt["prompt_id"],
                "prompt": prompt["prompt"],
                "response": response,
                "model": policy,
                "logp": random.uniform(-15, -8),
            }

            # Deterministic scoring
            targets_scored_det.append(
                {**base_response, "judge_score": {"mean": score_mean, "variance": 0.0}}
            )

            # Uncertainty scoring
            targets_scored_unc.append(
                {
                    **base_response,
                    "judge_score": {
                        "mean": score_mean,
                        "variance": random.uniform(0.01, 0.1),
                    },
                }
            )

    # 3. Create oracle labels
    oracle_calibration = []
    oracle_validation = []

    # Calibration: 25% of logging policy
    for i in range(n_samples // 4):
        oracle_calibration.append(
            {"prompt_id": prompts[i]["prompt_id"], "oracle_score": random.uniform(6, 8)}
        )

    # Validation: All target policies
    for prompt in prompts:
        for policy in target_policies:
            if policy == "pi_cot":
                oracle_score = random.uniform(8, 9.5)
            elif policy == "pi_bigger_model":
                oracle_score = random.uniform(7.5, 8.5)
            else:  # pi_bad
                oracle_score = random.uniform(2, 4)

            oracle_validation.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "model": policy,
                    "oracle_score": oracle_score,
                }
            )

    # Write all files
    print(f"📝 Writing test dataset with {n_samples} samples...")

    # Scored files
    with open(data_dir / "p0_scored_deterministic.jsonl", "w") as f:
        for item in p0_scored_det:
            f.write(json.dumps(item) + "\n")

    with open(data_dir / "p0_scored_uncertainty.jsonl", "w") as f:
        for item in p0_scored_unc:
            f.write(json.dumps(item) + "\n")

    with open(data_dir / "targets_scored_deterministic.jsonl", "w") as f:
        for item in targets_scored_det:
            f.write(json.dumps(item) + "\n")

    with open(data_dir / "targets_scored_uncertainty.jsonl", "w") as f:
        for item in targets_scored_unc:
            f.write(json.dumps(item) + "\n")

    # Oracle labels
    with open(labeling_dir / "oracle_labels_calibration_detailed.jsonl", "w") as f:
        for item in oracle_calibration:
            f.write(json.dumps(item) + "\n")

    with open(labeling_dir / "oracle_labels_validation_detailed.jsonl", "w") as f:
        for item in oracle_validation:
            f.write(json.dumps(item) + "\n")

    print("✅ Test dataset created successfully!")
    print(f"   - Logging policy samples: {n_samples}")
    print(f"   - Target policy samples: {n_samples * 3}")
    print(f"   - Oracle calibration labels: {n_samples // 4}")
    print(f"   - Oracle validation labels: {n_samples * 3}")


if __name__ == "__main__":
    create_test_dataset(20)
