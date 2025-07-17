"""Test edge cases and different data situations."""

from cje_simplified import PrecomputedSampler, CalibratedIPS


def test_missing_values() -> None:
    """Test when some samples have missing values."""
    print("\nTesting missing values...")

    # Create data where some samples have missing base_policy_logprob
    data = [
        {
            "prompt": f"q{i}",
            "response": f"a{i}",
            "reward": 0.7,
            "base_policy_logprob": -10.0 if i < 5 else None,  # Only first 5 have values
            "target_policy_logprobs": {"pi_test": -9.0},
        }
        for i in range(10)
    ]

    sampler = PrecomputedSampler(data)


def test_extreme_weights() -> None:
    """Test with extreme importance weights."""
    print("\nTesting extreme weights...")

    # Create data with very different log probs (extreme weights)
    data = [
        {
            "prompt": f"q{i}",
            "response": f"a{i}",
            "reward": 0.7,
            "base_policy_logprob": -50.0,  # Very low prob under base
            "target_policy_logprobs": {"pi_test": -5.0},  # Very high prob under target
        }
        for i in range(10)
    ]

    sampler = PrecomputedSampler(data)


def test_all_missing() -> None:
    """Test when some samples have missing values."""
    print("\nTesting partial missing values...")

    # Create data where some samples have missing values
    data = [
        {
            "prompt": f"q{i}",
            "response": f"a{i}",
            "reward": 0.7,
            "base_policy_logprob": -10.0 if i < 8 else None,  # Some missing
            "target_policy_logprobs": {
                "pi_test": -9.0 if i < 5 else None
            },  # More missing
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
