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
    # n_samples returns total (for backwards compatibility), n_valid_samples returns filtered
    assert sampler.n_samples == 10  # Total samples in dataset
    assert sampler.n_valid_samples == 5  # Valid samples after filtering
    print(
        f"  ✓ Correctly filtered: {sampler.n_samples} total, {sampler.n_valid_samples} valid"
    )


def test_extreme_weights() -> None:
    """Test with extreme importance weights."""
    print("\nTesting extreme weights...")

    # Create data with very different log probs (extreme weights)
    # log prob diff of 45 -> weight ratio of exp(45) ≈ 3.5e19!
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
    estimator = CalibratedIPS(sampler)
    results = estimator.fit_and_estimate()

    # With such extreme weights, estimate should still be computed
    assert results.estimates[0] is not None
    # Calibration should help stabilize these extreme weights
    print(f"  ✓ Handled extreme weights: estimate = {results.estimates[0]:.3f}")


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
    estimator = CalibratedIPS(sampler)
    results = estimator.fit_and_estimate()
    print(f"  ✓ Handled partial missing data: estimate = {results.estimates[0]:.3f}")


def test_all_invalid() -> None:
    """Test when ALL samples have missing values."""
    print("\nTesting all invalid samples...")

    # Create data where ALL samples have missing logprobs
    data = [
        {
            "prompt": f"q{i}",
            "response": f"a{i}",
            "reward": 0.7,
            "base_policy_logprob": None,  # All missing
            "target_policy_logprobs": {"pi_test": -9.0},
        }
        for i in range(5)
    ]

    try:
        sampler = PrecomputedSampler(data)
        print(f"  ✗ Should have raised ValueError but got {sampler.n_samples} samples")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No valid records after filtering" in str(e)
        print(f"  ✓ Correctly raised ValueError for all invalid samples")


if __name__ == "__main__":
    print("Testing edge cases...")
    test_missing_values()
    test_extreme_weights()
    test_all_missing()
    test_all_invalid()
    print("\nAll edge case tests passed! ✨")
