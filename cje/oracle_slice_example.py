#!/usr/bin/env python3
"""Example demonstrating oracle slice augmentation for honest confidence intervals.

This example shows how oracle slice augmentation accounts for the uncertainty
in learning the judge→oracle calibration map from a finite oracle slice,
providing more honest (wider) confidence intervals when the oracle slice is small.
"""

import numpy as np
from cje import (
    load_dataset_from_jsonl,
    calibrate_dataset,
    PrecomputedSampler,
    CalibratedIPS,
    DRCPOEstimator,
)
from cje.calibration.oracle_slice import OracleSliceConfig

# Fresh draws would be imported here if needed
# from cje.data.fresh_draws import load_fresh_draws_auto


def main() -> None:
    """Demonstrate oracle slice augmentation with different oracle coverage levels."""

    # Load your dataset (assumes judge scores and some oracle labels)
    # Replace with your actual data file
    dataset_path = "data/example_dataset.jsonl"

    try:
        dataset = load_dataset_from_jsonl(dataset_path)
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}")
        print("Creating synthetic example dataset...")
        dataset = create_synthetic_dataset()

    # Test with different oracle coverage levels
    oracle_coverages = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("Oracle Slice Augmentation Example")
    print("=" * 60)
    print("\nThis demonstrates how confidence intervals properly widen")
    print("when we have fewer oracle labels for calibration.\n")

    for coverage in oracle_coverages:
        print(f"\n--- Oracle Coverage: {coverage*100:.0f}% ---")

        # Calibrate dataset (manually mask for specified oracle coverage)
        # In practice, you'd have a dataset with partial oracle labels
        # Here we simulate by masking some labels
        for i, sample in enumerate(dataset.samples):
            if i % int(1 / coverage) != 0:
                if "oracle_label" in sample.metadata:
                    sample.metadata["oracle_label"] = None

        calibrated_dataset, cal_result = calibrate_dataset(
            dataset,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=5,
        )

        # Create sampler
        sampler = PrecomputedSampler(calibrated_dataset)

        # Run estimation WITHOUT augmentation
        print("\nWithout oracle augmentation:")
        estimator_no_aug = CalibratedIPS(
            sampler, oracle_slice_config=OracleSliceConfig(enable_augmentation=False)
        )
        result_no_aug = estimator_no_aug.fit_and_estimate()

        for i, policy in enumerate(
            sampler.target_policies[:2]
        ):  # Show first 2 policies
            est = result_no_aug.estimates[i]
            se = result_no_aug.standard_errors[i]
            ci_width = 2 * 1.96 * se
            print(f"  {policy}: {est:.4f} ± {se:.4f} (CI width: {ci_width:.4f})")

        # Run estimation WITH augmentation
        print("\nWith oracle augmentation (honest CIs):")
        estimator_with_aug = CalibratedIPS(
            sampler, oracle_slice_config=OracleSliceConfig(enable_augmentation=True)
        )
        result_with_aug = estimator_with_aug.fit_and_estimate()

        for i, policy in enumerate(
            sampler.target_policies[:2]
        ):  # Show first 2 policies
            est = result_with_aug.estimates[i]
            se = result_with_aug.standard_errors[i]
            ci_width = 2 * 1.96 * se

            # Get augmentation diagnostics
            aug_diag = result_with_aug.metadata.get("slice_augmentation", {}).get(
                policy, {}
            )
            slice_var_share = aug_diag.get("slice_variance_share", 0.0)

            print(f"  {policy}: {est:.4f} ± {se:.4f} (CI width: {ci_width:.4f})")
            print(f"    Slice variance contribution: {slice_var_share*100:.1f}%")

    print("\n" + "=" * 60)
    print("Key Observations:")
    print("1. With less oracle coverage, CIs are wider (more honest)")
    print("2. The augmentation term contributes more to variance with less data")
    print("3. Point estimates remain unbiased in both cases")
    print("4. The slice_variance_share shows the uncertainty from calibration")


def create_synthetic_dataset():  # type: ignore
    """Create a synthetic dataset for demonstration."""
    from cje.data import Dataset, Sample

    n_samples = 1000
    samples = []

    # Generate synthetic data
    np.random.seed(42)

    for i in range(n_samples):
        # Judge scores (0 to 1)
        judge_score = np.random.beta(2, 2)

        # Oracle label - correlated with judge but with noise
        oracle_label = judge_score + 0.1 * np.random.randn()
        oracle_label = np.clip(oracle_label, 0, 1)

        # Only include oracle label for some samples (will be masked later)
        include_oracle = np.random.rand() < 0.8

        # Create sample
        sample = Sample(
            prompt_id=str(i),
            prompt=f"prompt_{i}",
            response=f"response_{i}",
            reward=None,  # Will be calibrated
            base_policy_logprob=np.log(0.5),
            target_policy_logprobs={
                "policy_a": np.log(0.4 + 0.2 * judge_score),
                "policy_b": np.log(0.3 + 0.4 * judge_score),
            },
            metadata={
                "judge_score": judge_score,
                "oracle_label": oracle_label if include_oracle else None,
            },
        )
        samples.append(sample)

    return Dataset(samples=samples, target_policies=["policy_a", "policy_b"])


def demonstrate_dr_with_augmentation() -> None:
    """Demonstrate oracle augmentation with DR estimators."""
    print("\n" + "=" * 60)
    print("DR Estimation with Oracle Augmentation")
    print("=" * 60)

    # This would require fresh draws - skipping for basic example
    print("\nDR estimators can also use oracle augmentation.")
    print("The augmentation adjusts the IPS correction term in the DR formula.")
    print("This provides honest CIs that account for both:")
    print("  1. Uncertainty in the outcome model")
    print("  2. Uncertainty in the calibration map")

    # Example code (requires fresh draws):
    print("\nExample usage:")
    print(
        """
    estimator = DRCPOEstimator(
        sampler,
        calibrator=cal_result.calibrator,
        oracle_slice_config=OracleSliceConfig(enable_augmentation=True)
    )
    
    # Add fresh draws
    estimator.add_fresh_draws('policy_a', fresh_draws_a)
    
    # Estimate with honest CIs
    result = estimator.fit_and_estimate()
    """
    )


if __name__ == "__main__":
    main()
    demonstrate_dr_with_augmentation()
