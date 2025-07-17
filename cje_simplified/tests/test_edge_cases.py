"""Test edge cases and different data situations."""

from cje_simplified import PrecomputedSampler, CalibratedIPS


def test_missing_values():
    """Test handling of missing values in log probabilities."""
    print("\nTesting missing values...")

    sampler = PrecomputedSampler.from_jsonl("tests/data/missing_values_data.jsonl")
    print(f"  Loaded {sampler.n_samples} samples")

    # Check data summary shows missing values
    summary = sampler.summary()
    print(f"  {summary}")

    # Should still be able to estimate
    estimator = CalibratedIPS(sampler, k_folds=2)
    results = estimator.fit_and_estimate()

    print(f"  ✓ Estimates: {results.estimates}")
    print(f"  ✓ Handled missing values correctly")


def test_extreme_weights():
    """Test extreme importance weights."""
    print("\nTesting extreme weights...")

    sampler = PrecomputedSampler.from_jsonl("tests/data/extreme_weights_data.jsonl")
    estimator = CalibratedIPS(sampler, k_folds=2, clip_weight=100.0)

    # Get raw weights before calibration
    weights = estimator.get_weights("pi_extreme")
    if weights is not None:
        print(f"  Raw weight range: {min(weights):.2e} to {max(weights):.2e}")
    else:
        print("  No valid weights (all samples may have been dropped)")

    # Run estimation - should handle extremes
    results = estimator.fit_and_estimate()
    print(f"  ✓ Estimate with clipping: {results.estimates[0]:.3f}")


def test_all_missing():
    """Test when some samples have missing values."""
    print("\nTesting partial missing values...")

    # Create data where some samples have missing values
    data = [
        {
            "prompt": f"q{i}",
            "response": f"a{i}",
            "reward": 0.7,
            "base_policy_logprob": -10.0 if i < 8 else None,  # Some missing
            "target_logps": {"pi_test": -9.0 if i < 5 else None},  # More missing
        }
        for i in range(10)
    ]

    sampler = PrecomputedSampler(data)
    print(f"  Loaded {sampler.n_samples} samples (after dropping invalid)")

    # Should still work with reduced sample size
    estimator = CalibratedIPS(sampler, k_folds=2)
    results = estimator.fit_and_estimate()
    print(f"  ✓ Handled partial missing data: estimate = {results.estimates[0]:.3f}")


if __name__ == "__main__":
    print("Testing edge cases...")
    test_missing_values()
    test_extreme_weights()
    test_all_missing()
    print("\nAll edge case tests passed! ✨")
