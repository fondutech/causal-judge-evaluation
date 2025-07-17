"""Example usage of simplified CJE library."""

import json
import os
import numpy as np
from typing import Dict, Any

# Import from the simplified CJE library
from cje_simplified import (
    PrecomputedSampler,
    CalibratedIPS,
    load_dataset_with_calibration,
    compute_teacher_forced_logprob,
    diagnose_weights,
    create_weight_summary_table,
    detect_api_nondeterminism,
    calibrate_judge_scores,
    Llama3TemplateConfig,
    Llama4TemplateConfig,
    HuggingFaceTemplateConfig,
    convert_chat_for_teacher_forcing,
    compute_chat_logprob,
)


def create_example_data() -> None:
    """Create example data files for demonstration."""

    # Example data with raw judge scores (not yet calibrated)
    raw_data = [
        {
            "prompt": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence...",
            "judge_score": 7.5,  # Raw score from judge (e.g., 0-10 scale)
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
            "base_policy_logprob": -31.245,
            "target_policy_logprobs": {
                "pi_cot": -36.789,
                "pi_bigger": -28.456,
                "pi_clone": -31.300,
            },
        },
    ]

    # Add oracle labels for first 2 samples (50% oracle fraction)
    # These represent true business KPIs (e.g., user satisfaction 0-1)
    raw_data[0]["oracle_label"] = 0.82
    raw_data[1]["oracle_label"] = 0.71

    # Save raw data
    with open("raw_data.jsonl", "w") as f:
        for record in raw_data:
            f.write(json.dumps(record) + "\n")

    print(f"Created raw_data.jsonl with {len(raw_data)} samples (2 with oracle labels)")


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
        tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",
    )

    if chat_result.is_valid:
        print(f"Chat log probability: {chat_result.value:.3f}")
        print(f"Method: {chat_result.metadata.get('method', 'unknown')}")
    else:
        print(f"Chat error: {chat_result.error}")


def prepare_data_for_cje() -> None:
    """Prepare data by calibrating judge scores to oracle labels."""

    print("\n=== Preparing Data for CJE ===\n")

    # Load data and calibrate judge scores in one step using SOLID factory
    dataset, stats = load_dataset_with_calibration(
        "raw_data.jsonl",
        judge_score_field="judge_score",
        oracle_label_field="oracle_label",
        k_folds=2,  # Use 2 folds for small example
    )

    print(f"Calibration statistics:")
    print(f"  Oracle samples: {stats['n_oracle']}")
    print(f"  RMSE: {stats['rmse']:.3f}")
    print(f"  Coverage (±0.1): {stats['coverage']:.1%}")

    # Save prepared data - convert Dataset to raw data for saving
    with open("cje_ready_data.jsonl", "w") as f:
        for sample in dataset.samples:
            record = {
                "prompt": sample.prompt,
                "response": sample.response,
                "reward": sample.reward,
                "base_policy_logprob": sample.base_policy_logprob,
                "target_policy_logprobs": sample.target_policy_logprobs,
                "judge_score": sample.metadata.get("judge_score"),  # Preserve original
                "oracle_label": sample.metadata.get(
                    "oracle_label"
                ),  # Preserve original
            }
            f.write(json.dumps(record) + "\n")

    print("\nData prepared and saved to cje_ready_data.jsonl")

    # Show example of calibration
    print("\nExample calibration:")
    for i in range(min(3, len(dataset.samples))):
        sample = dataset.samples[i]
        judge = sample.metadata.get("judge_score", "N/A")
        reward = sample.reward
        oracle = sample.metadata.get("oracle_label", "N/A")
        judge_str = f"{judge:.1f}" if isinstance(judge, (int, float)) else str(judge)
        oracle_str = (
            f"{oracle:.2f}" if isinstance(oracle, (int, float)) else str(oracle)
        )
        print(
            f"  Sample {i}: judge_score={judge_str} → reward={reward:.3f} (oracle: {oracle_str})"
        )


def run_cje_estimation() -> None:
    """Run CJE estimation on prepared data."""

    print("\n=== Running CJE Estimation ===\n")

    # Load data with calibrated rewards
    sampler = PrecomputedSampler.from_jsonl("cje_ready_data.jsonl")

    print(f"Loaded {sampler.n_samples} samples")
    print(f"Target policies: {sampler.target_policies}")
    print(f"Data summary: {sampler.summary()}")

    # Run calibrated IPS estimation
    estimator = CalibratedIPS(
        sampler,
        k_folds=2,  # Use 2 folds for small example
        clip_weight=100.0,
        random_seed=42,
    )

    results = estimator.fit_and_estimate()

    # Display results
    print("\n=== Estimation Results ===\n")
    print(f"Estimates: {results.estimates}")
    print(f"Standard errors: {results.standard_errors}")

    # Confidence intervals
    ci_lower, ci_upper = results.confidence_interval(0.95)
    print("\n95% Confidence Intervals:")
    for i, policy in enumerate(sampler.target_policies):
        print(f"  {policy}: [{ci_lower[i]:.3f}, {ci_upper[i]:.3f}]")

    # Best policy
    best_idx = results.best_policy()
    print(
        f"\nBest policy: {sampler.target_policies[best_idx]} "
        f"(estimate: {results.estimates[best_idx]:.3f})"
    )

    # Policy comparisons
    print("\n=== Policy Comparisons ===\n")
    comparison = results.compare_policies(0, 1)  # Compare first two policies
    print(f"Comparing {sampler.target_policies[0]} vs {sampler.target_policies[1]}:")
    print(
        f"  Difference: {comparison['difference']:.3f} ± {comparison['se_difference']:.3f}"
    )
    print(f"  P-value: {comparison['p_value']:.3f}")
    print(f"  Significant: {comparison['significant']}")

    # Weight diagnostics
    print("\n=== Weight Diagnostics ===\n")

    all_diagnostics = {}
    for policy in sampler.target_policies:
        weights = estimator.get_weights(policy)
        if weights is not None:
            # For pi_clone, expect weight ~1.0
            expected = 1.0 if policy == "pi_clone" else None
            diag = diagnose_weights(weights, policy, expected)
            all_diagnostics[policy] = diag
            print(diag.summary())
            print()

    # Summary table
    print("\n" + create_weight_summary_table(all_diagnostics))

    # Check for API non-determinism
    print("\n=== API Non-determinism Check ===\n")
    api_check = detect_api_nondeterminism(sampler, baseline_policy="pi_clone")
    print(f"Non-determinism detected: {api_check['detected']}")
    if "mean_weight" in api_check:
        print(
            f"Clone policy mean weight: {api_check['mean_weight']:.3f} "
            f"(deviation: {api_check['deviation']:.3f})"
        )
    print(f"Recommendation: {api_check['recommendation']}")


def demonstrate_judge_calibration() -> None:
    """Demonstrate judge score calibration with oracle labels."""

    print("\n=== Judge Score Calibration ===\n")

    # Example: Raw judge scores and oracle labels for subset
    raw_judge_scores = np.array([0.7, 0.8, 0.6, 0.9, 0.75, 0.85, 0.65, 0.95])
    oracle_labels = np.array(
        [0.8, 0.9, 0.5, 0.95]
    )  # First 4 samples have oracle labels

    # Calibrate judge scores
    calibrated_scores, stats = calibrate_judge_scores(
        judge_scores=raw_judge_scores,
        oracle_labels=oracle_labels,
        k_folds=2,  # Use 2 folds for small example
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


def main() -> None:
    """Run all examples."""

    # Create example data
    create_example_data()

    # Optionally compute new log probs (requires API key)
    if os.getenv("FIREWORKS_API_KEY"):
        print("\n=== Computing New Log Probabilities ===\n")
        compute_new_log_probs()
    else:
        print("\nSkipping log prob computation (no FIREWORKS_API_KEY)")

    # Prepare data (calibrate judge scores)
    prepare_data_for_cje()

    # Run CJE estimation
    run_cje_estimation()

    # Demonstrate judge calibration
    demonstrate_judge_calibration()

    # Cleanup
    for f in ["raw_data.jsonl", "cje_ready_data.jsonl"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
