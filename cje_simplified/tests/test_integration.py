"""Integration test showing complete workflow."""

from cje_simplified import (
    PrecomputedSampler,
    CalibratedIPS,
    load_dataset_from_jsonl,
    calibrate_dataset,
    diagnose_weights,
)


def test_end_to_end_pipeline() -> None:
    """Run full integration test with existing test data."""

    # Use the judge calibration test data we created
    data_file = "tests/data/judge_calibration_data.jsonl"

    print("1. Loading data and calibrating judge scores...")
    # Load with judge scores as rewards initially
    dataset = load_dataset_from_jsonl(data_file, reward_field="judge_score")

    # Calibrate judge scores to oracle labels
    calibrated_dataset, cal_result = calibrate_dataset(
        dataset, judge_field="judge_score", oracle_field="oracle_label"
    )
    print(f"   ✓ Calibrated {calibrated_dataset.n_samples} samples")
    print(f"   ✓ Used {cal_result.n_oracle} oracle labels")

    print("\n2. Loading data for CJE...")
    # Load and estimate
    sampler = PrecomputedSampler(calibrated_dataset)
    print(f"   ✓ Loaded {sampler.n_samples} samples")
    print(f"   ✓ Target policies: {sampler.target_policies}")

    print("\n3. Running CJE estimation...")
    estimator = CalibratedIPS(sampler, k_folds=2)
    results = estimator.fit_and_estimate()
    print(f"   ✓ Estimate: {results.estimates[0]:.3f}")
    print(f"   ✓ Std Error: {results.standard_errors[0]:.3f}")

    print("\n4. Checking weight diagnostics...")
    weights = estimator.get_weights("pi_test")
    diag = diagnose_weights(weights, "pi_test")
    print(f"   ✓ Mean weight: {diag.mean_weight:.3f}")
    print(f"   ✓ ESS fraction: {diag.ess_fraction:.1%}")

    print("\n✅ Integration test passed!")


if __name__ == "__main__":
    test_end_to_end_pipeline()
