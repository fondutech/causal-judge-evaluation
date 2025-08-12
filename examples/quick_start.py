#!/usr/bin/env python
"""
Quick Start Examples for CJE

This script demonstrates the main ways to use CJE:
1. Command-line interface
2. High-level Python API
3. Low-level Python API
"""

import json
from pathlib import Path

# Example data structure for reference
EXAMPLE_DATA = {
    "prompt": "What is machine learning?",
    "response": "Machine learning is a subset of artificial intelligence...",
    "base_policy_logprob": -35.704,
    "target_policy_logprobs": {"gpt4": -32.456, "claude": -33.789, "llama": -34.123},
    "metadata": {"judge_score": 0.85, "oracle_label": 0.90},
}


def example_cli() -> None:
    """Example: Using the command-line interface."""
    print("=" * 60)
    print("COMMAND LINE INTERFACE EXAMPLES")
    print("=" * 60)

    print("\n1. Validate dataset:")
    print("   $ python -m cje validate data.jsonl")

    print("\n2. Basic analysis:")
    print("   $ python -m cje analyze data.jsonl")

    print("\n3. Analysis with options:")
    print("   $ python -m cje analyze data.jsonl \\")
    print("       --estimator dr-cpo \\")
    print("       --oracle-coverage 0.5 \\")
    print("       --output results.json")

    print("\n4. With fresh draws for DR:")
    print("   $ python -m cje analyze data.jsonl \\")
    print("       --estimator dr-cpo \\")
    print("       --fresh-draws-dir ./responses \\")
    print("       --output dr_results.json")


def example_high_level_api() -> None:
    """Example: Using the high-level Python API."""
    print("\n" + "=" * 60)
    print("HIGH-LEVEL API EXAMPLE")
    print("=" * 60)

    from cje import analyze_dataset, export_results_json, export_results_csv

    # Assuming we have a data file
    data_path = "data.jsonl"

    if not Path(data_path).exists():
        print(f"\nNote: {data_path} not found. Using example code.")
        print("\nCode example:")
        print("-" * 40)

    print(
        """
    from cje import analyze_dataset, export_results_json
    
    # One-line analysis
    results = analyze_dataset(
        "data.jsonl",
        estimator="calibrated-ips",
        oracle_coverage=0.5  # Use 50% of oracle labels
    )
    
    # Display results
    print(f"Best policy: {results.best_policy()}")
    print(f"Estimates: {results.estimates}")
    print(f"Standard errors: {results.standard_errors}")
    
    # Get confidence intervals
    ci_lower, ci_upper = results.confidence_interval(alpha=0.05)
    print(f"95% CI: [{ci_lower}, {ci_upper}]")
    
    # Export to different formats
    export_results_json(results, "results.json")
    export_results_csv(results, "results.csv")
    """
    )


def example_low_level_api() -> None:
    """Example: Using the low-level Python API for more control."""
    print("\n" + "=" * 60)
    print("LOW-LEVEL API EXAMPLE")
    print("=" * 60)

    print(
        """
    from cje import (
        load_dataset_from_jsonl,
        calibrate_dataset,
        PrecomputedSampler,
        CalibratedIPS
    )
    
    # Step 1: Load data
    dataset = load_dataset_from_jsonl("data.jsonl")
    print(f"Loaded {dataset.n_samples} samples")
    
    # Step 2: Calibrate judge scores (if needed)
    calibrated_dataset, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
        oracle_coverage=0.5
    )
    print(f"Calibration RMSE: {cal_result.calibration_rmse:.3f}")
    
    # Step 3: Create sampler
    sampler = PrecomputedSampler(calibrated_dataset)
    print(f"Valid samples: {sampler.n_valid_samples}")
    
    # Step 4: Run estimation
    estimator = CalibratedIPS(sampler)
    results = estimator.fit_and_estimate()
    
    # Step 5: Analyze results
    for i, policy in enumerate(sampler.target_policies):
        estimate = results.estimates[i]
        se = results.standard_errors[i]
        print(f"{policy}: {estimate:.3f} ± {se:.3f}")
    """
    )


def example_dr_estimation() -> None:
    """Example: Using doubly robust estimation."""
    print("\n" + "=" * 60)
    print("DOUBLY ROBUST ESTIMATION EXAMPLE")
    print("=" * 60)

    print(
        """
    from cje import (
        load_dataset_from_jsonl,
        calibrate_dataset,
        PrecomputedSampler,
        DRCPOEstimator,
        load_fresh_draws_auto
    )
    
    # Load and calibrate data
    dataset = load_dataset_from_jsonl("data.jsonl")
    calibrated_dataset, cal_result = calibrate_dataset(
        dataset,
        enable_cross_fit=True,  # Important for DR
        n_folds=5
    )
    
    # Create sampler and DR estimator
    sampler = PrecomputedSampler(calibrated_dataset)
    dr_estimator = DRCPOEstimator(
        sampler,
        calibrator=cal_result.calibrator,  # Reuse calibration
        n_folds=5
    )
    
    # Add fresh draws for each policy
    for policy in sampler.target_policies:
        fresh_draws = load_fresh_draws_auto(
            data_dir=Path("./data"),
            policy=policy,
            fallback_synthetic=True  # Use synthetic if not found
        )
        dr_estimator.add_fresh_draws(policy, fresh_draws)
    
    # Run DR estimation
    results = dr_estimator.fit_and_estimate()
    print(f"DR estimates: {results.estimates}")
    
    # Check diagnostics
    if "dr_diagnostics" in results.metadata:
        for policy, diags in results.metadata["dr_diagnostics"].items():
            print(f"{policy} - Orthogonality: {diags['orthogonality']:.4f}")
    """
    )


def example_data_preparation() -> None:
    """Example: Preparing data for CJE."""
    print("\n" + "=" * 60)
    print("DATA PREPARATION EXAMPLE")
    print("=" * 60)

    print(
        """
    import json
    from cje import compute_teacher_forced_logprob
    
    # Step 1: Collect prompts and responses
    data = []
    
    for prompt, response in your_data:
        # Step 2: Compute log probabilities
        base_logprob = compute_teacher_forced_logprob(
            prompt, response, "base_model"
        ).logprob
        
        target_logprobs = {}
        for policy in ["gpt4", "claude", "llama"]:
            result = compute_teacher_forced_logprob(
                prompt, response, policy
            )
            target_logprobs[policy] = result.logprob
        
        # Step 3: Get judge scores
        judge_score = get_judge_score(prompt, response)
        
        # Step 4: (Optional) Get oracle labels for some samples
        oracle_label = None
        if should_get_oracle_label():
            oracle_label = get_oracle_label(prompt, response)
        
        # Step 5: Format for CJE
        data.append({
            "prompt": prompt,
            "response": response,
            "base_policy_logprob": base_logprob,
            "target_policy_logprobs": target_logprobs,
            "metadata": {
                "judge_score": judge_score,
                "oracle_label": oracle_label
            }
        })
    
    # Save to JSONL
    with open("data.jsonl", "w") as f:
        for record in data:
            f.write(json.dumps(record) + "\\n")
    """
    )


def main() -> None:
    """Run all examples."""
    print("\n" + "🚀 CJE QUICK START EXAMPLES 🚀".center(60))
    print("=" * 60)

    # Show example data structure
    print("\nEXAMPLE DATA STRUCTURE:")
    print("-" * 40)
    print(json.dumps(EXAMPLE_DATA, indent=2))

    # Run examples
    example_cli()
    example_high_level_api()
    example_low_level_api()
    example_dr_estimation()
    example_data_preparation()

    print("\n" + "=" * 60)
    print("For more examples, see:")
    print("- examples/arena_10k_simplified/ - Full pipeline example")
    print("- cje/tests/ - Test cases showing various use cases")
    print("- docs/ - Complete documentation")
    print("=" * 60)


if __name__ == "__main__":
    main()
