"""Example usage of simplified CJE library - demonstrates all three workflows."""

import json
import os
import numpy as np
from typing import Dict, Any, List

# Import from the simplified CJE library
from cje import (
    PrecomputedSampler,
    CalibratedIPS,
    load_dataset_from_jsonl,
    calibrate_dataset,
    compute_teacher_forced_logprob,
    calibrate_judge_scores,
    Llama3TemplateConfig,
    HuggingFaceTemplateConfig,
    compute_chat_logprob,
    Dataset,
    Sample,
)
from cje.utils.diagnostics import compute_weight_diagnostics


def create_example_data() -> None:
    """Create example data files for all three workflows."""

    # 1. Data with oracle labels (for workflow 1)
    oracle_data = [
        {
            "prompt": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence...",
            "oracle_label": 0.82,  # Direct oracle label (ground truth)
            "base_policy_logprob": -35.704,
            "target_policy_logprobs": {
                "pi_cot": -40.123,
                "pi_bigger": -32.456,
                "pi_clone": -35.800,
            },
        },
        {
            "prompt": "Explain quantum computing",
            "response": "Quantum computing uses quantum mechanical phenomena...",
            "oracle_label": 0.71,
            "base_policy_logprob": -42.156,
            "target_policy_logprobs": {
                "pi_cot": -48.234,
                "pi_bigger": -38.901,
                "pi_clone": -42.200,
            },
        },
    ]

    # 2. Data with judge scores that need calibration (for workflow 2)
    judge_data = [
        {
            "prompt": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence...",
            "judge_score": 7.5,  # Raw score from judge (e.g., 0-10 scale)
            "oracle_label": 0.82,  # Oracle label for calibration
            "base_policy_logprob": -35.704,
            "target_policy_logprobs": {
                "pi_cot": -40.123,
                "pi_bigger": -32.456,
                "pi_clone": -35.800,
            },
        },
        {
            "prompt": "Explain quantum computing",
            "response": "Quantum computing uses quantum mechanical phenomena...",
            "judge_score": 6.8,
            "oracle_label": 0.71,  # Oracle label for calibration
            "base_policy_logprob": -42.156,
            "target_policy_logprobs": {
                "pi_cot": -48.234,
                "pi_bigger": -38.901,
                "pi_clone": -42.200,
            },
        },
        {
            "prompt": "What are neural networks?",
            "response": "Neural networks are computing systems inspired by...",
            "judge_score": 8.9,
            # No oracle label - will be calibrated based on above samples
            "base_policy_logprob": -28.493,
            "target_policy_logprobs": {
                "pi_cot": -33.567,
                "pi_bigger": -25.123,
                "pi_clone": -28.500,
            },
        },
        {
            "prompt": "Define reinforcement learning",
            "response": "Reinforcement learning is a type of machine learning...",
            "judge_score": 7.2,
            # No oracle label - will be calibrated based on above samples
            "base_policy_logprob": -31.245,
            "target_policy_logprobs": {
                "pi_cot": -36.789,
                "pi_bigger": -28.456,
                "pi_clone": -31.300,
            },
        },
    ]

    # 3. Data with pre-calibrated rewards (for workflow 3)
    calibrated_data = [
        {
            "prompt": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence...",
            "reward": 0.85,  # Pre-calibrated reward
            "base_policy_logprob": -35.704,
            "target_policy_logprobs": {
                "pi_cot": -40.123,
                "pi_bigger": -32.456,
                "pi_clone": -35.800,
            },
        },
        {
            "prompt": "Explain quantum computing",
            "response": "Quantum computing uses quantum mechanical phenomena...",
            "reward": 0.72,
            "base_policy_logprob": -42.156,
            "target_policy_logprobs": {
                "pi_cot": -48.234,
                "pi_bigger": -38.901,
                "pi_clone": -42.200,
            },
        },
    ]

    # Save all datasets
    with open("oracle_data.jsonl", "w") as f:
        for record in oracle_data:
            f.write(json.dumps(record) + "\n")

    with open("judge_data.jsonl", "w") as f:
        for record in judge_data:
            f.write(json.dumps(record) + "\n")

    with open("calibrated_data.jsonl", "w") as f:
        for record in calibrated_data:
            f.write(json.dumps(record) + "\n")

    print(f"Created oracle_data.jsonl with {len(oracle_data)} samples")
    print(
        f"Created judge_data.jsonl with {len(judge_data)} samples (2 with oracle labels)"
    )
    print(f"Created calibrated_data.jsonl with {len(calibrated_data)} samples")


def workflow_1_oracle_labels() -> None:
    """Workflow 1: Using oracle labels directly as rewards."""

    print("\n=== WORKFLOW 1: Oracle Labels as Rewards ===\n")

    # Load dataset without rewards
    dataset = load_dataset_from_jsonl("oracle_data.jsonl")
    print(f"Loaded {dataset.n_samples} samples")

    # Map oracle labels directly to rewards
    for sample in dataset.samples:
        if "oracle_label" in sample.metadata:
            sample.reward = sample.metadata["oracle_label"]

    # Verify rewards were set
    rewards_set = sum(1 for s in dataset.samples if s.reward is not None)
    print(f"Set rewards for {rewards_set}/{dataset.n_samples} samples")

    # Now ready for CJE estimation
    sampler = PrecomputedSampler(dataset)
    estimator = CalibratedIPS(sampler)
    results = estimator.fit_and_estimate()

    print(f"\nEstimates: {results.estimates}")
    print(f"Best policy: {sampler.target_policies[results.best_policy()]}")


def workflow_2_judge_calibration() -> None:
    """Workflow 2: Calibrating judge scores using oracle labels."""

    print("\n=== WORKFLOW 2: Judge Score Calibration ===\n")

    # Load dataset without specifying reward field
    dataset = load_dataset_from_jsonl("judge_data.jsonl")
    print(f"Loaded {dataset.n_samples} samples")

    # Check which samples have oracle labels
    oracle_count = sum(1 for s in dataset.samples if "oracle_label" in s.metadata)
    print(f"Oracle labels available for {oracle_count}/{dataset.n_samples} samples")

    # Calibrate judge scores to oracle labels
    calibrated_dataset, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
        # Use 2 folds for small example
    )

    print(f"\nCalibration statistics:")
    print(f"  Oracle samples: {cal_result.n_oracle}")
    print(f"  RMSE: {cal_result.calibration_rmse:.3f}")
    print(f"  Coverage (±0.1): {cal_result.coverage_at_01:.1%}")

    # Show calibration results
    print("\nCalibration results:")
    for i, sample in enumerate(calibrated_dataset.samples):
        judge = sample.metadata.get("judge_score", "N/A")
        reward = sample.reward
        oracle = sample.metadata.get("oracle_label", "N/A")
        oracle_str = (
            f" (oracle: {oracle:.2f})" if isinstance(oracle, (int, float)) else ""
        )
        print(f"  Sample {i}: judge={judge} → reward={reward:.3f}{oracle_str}")

    # Now ready for CJE estimation
    sampler = PrecomputedSampler(calibrated_dataset)
    estimator = CalibratedIPS(sampler)
    results = estimator.fit_and_estimate()

    print(f"\nEstimates: {results.estimates}")
    print(f"Best policy: {sampler.target_policies[results.best_policy()]}")


def workflow_3_pre_calibrated() -> None:
    """Workflow 3: Using pre-calibrated rewards."""

    print("\n=== WORKFLOW 3: Pre-calibrated Rewards ===\n")

    # Load dataset that already has rewards
    dataset = load_dataset_from_jsonl("calibrated_data.jsonl")
    print(f"Loaded {dataset.n_samples} samples")

    # Check that rewards exist
    rewards_exist = sum(1 for s in dataset.samples if s.reward is not None)
    print(f"Rewards present for {rewards_exist}/{dataset.n_samples} samples")

    # Can directly use for CJE estimation
    sampler = PrecomputedSampler(dataset)
    estimator = CalibratedIPS(sampler)
    results = estimator.fit_and_estimate()

    print(f"\nEstimates: {results.estimates}")
    print(f"Best policy: {sampler.target_policies[results.best_policy()]}")


def compute_new_log_probs() -> None:
    """Example of computing log probabilities for new data."""

    # Example 1: Simple completions format
    prompt = "What is 2+2?"
    response = "2+2 equals 4."

    # Compute log probability using teacher forcing
    # Note: Requires FIREWORKS_API_KEY environment variable
    result = compute_teacher_forced_logprob(
        prompt=prompt,
        response=response,
        model="accounts/fireworks/models/llama-v3p2-3b-instruct",
        temperature=1.0,
    )

    if result.is_valid:
        print(f"Log probability: {result.value:.3f}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"Error: {result.error}")
        print(f"Status: {result.status}")

    # Example 2: Chat format with explicit template
    print("\n--- Chat Format Example ---")

    chat = [
        {"role": "user", "content": "What is machine learning?"},
        {
            "role": "assistant",
            "content": "Machine learning is a branch of AI that enables computers to learn from data.",
        },
    ]

    # Use Llama 3 template configuration
    template_config = Llama3TemplateConfig()

    # Compute log probability for chat
    chat_result = compute_chat_logprob(
        chat=chat,
        model="accounts/fireworks/models/llama-v3p2-3b-instruct",
        template_config=template_config,
    )

    if chat_result.is_valid:
        print(f"Chat log probability: {chat_result.value:.3f}")
        print(f"Method: {chat_result.metadata.get('method', 'unknown')}")
    else:
        print(f"Chat error: {chat_result.error}")


def demonstrate_judge_calibration() -> None:
    """Demonstrate judge score calibration with oracle labels."""

    print("\n=== Judge Score Calibration (Standalone) ===\n")

    # Example: Raw judge scores and oracle labels for subset
    raw_judge_scores = np.array([0.7, 0.8, 0.6, 0.9, 0.75, 0.85, 0.65, 0.95])
    oracle_labels = np.array(
        [0.8, 0.9, 0.5, 0.95]
    )  # First 4 samples have oracle labels

    # Calibrate judge scores
    calibrated_scores, stats = calibrate_judge_scores(
        judge_scores=raw_judge_scores,
        oracle_labels=oracle_labels,
        # Use 2 folds for small example
    )

    print("Raw judge scores:      ", raw_judge_scores)
    print("Calibrated scores:     ", np.round(calibrated_scores, 3))
    print(f"\nCalibration statistics:")
    print(f"  Oracle samples: {stats['n_oracle']}")
    print(f"  RMSE: {stats['rmse']:.3f}")
    print(f"  Coverage (±0.1): {stats['coverage']:.1%}")

    # Show how calibration changes the scores
    print("\nScore changes:")
    for i in range(len(raw_judge_scores)):
        change = calibrated_scores[i] - raw_judge_scores[i]
        oracle_marker = " (oracle)" if i < len(oracle_labels) else ""
        print(
            f"  Sample {i}: {raw_judge_scores[i]:.2f} → {calibrated_scores[i]:.3f} "
            f"(change: {change:+.3f}){oracle_marker}"
        )


def advanced_diagnostics() -> None:
    """Demonstrate advanced weight diagnostics and API non-determinism detection."""

    print("\n=== Advanced Diagnostics ===\n")

    # Use workflow 3 data for diagnostics
    dataset = load_dataset_from_jsonl("calibrated_data.jsonl")
    sampler = PrecomputedSampler(dataset)
    estimator = CalibratedIPS(sampler, clip_weight=100.0)
    results = estimator.fit_and_estimate()

    # Weight diagnostics
    print("=== Weight Diagnostics ===\n")

    all_diagnostics = {}
    for policy in sampler.target_policies:
        weights = estimator.get_weights(policy)
        if weights is not None:
            diag = compute_weight_diagnostics(weights, policy)
            all_diagnostics[policy] = diag
            # Print diagnostic summary
            print(f"Policy {policy}:")
            print(f"  ESS: {diag['ess_fraction']:.1%}")
            print(
                f"  Weight range: [{diag['min_weight']:.2e}, {diag['max_weight']:.2e}]"
            )
            print(
                f"  Mean: {diag['mean_weight']:.4f}, Median: {diag['median_weight']:.4f}"
            )
            print()


def main() -> None:
    """Run all examples demonstrating the three workflows."""

    # Create example data for all workflows
    create_example_data()

    # Demonstrate all three workflows
    workflow_1_oracle_labels()
    workflow_2_judge_calibration()
    workflow_3_pre_calibrated()

    # Optionally compute new log probs (requires API key)
    if os.getenv("FIREWORKS_API_KEY"):
        print("\n=== Computing New Log Probabilities ===\n")
        compute_new_log_probs()
    else:
        print("\nSkipping log prob computation (no FIREWORKS_API_KEY)")

    # Demonstrate standalone judge calibration
    demonstrate_judge_calibration()

    # Show advanced diagnostics
    advanced_diagnostics()

    # Cleanup
    for f in ["oracle_data.jsonl", "judge_data.jsonl", "calibrated_data.jsonl"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
