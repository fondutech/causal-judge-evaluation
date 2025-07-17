"""Integration test showing complete workflow."""

from cje_simplified import (
    PrecomputedSampler,
    CalibratedIPS,
    load_dataset_with_calibration,
    diagnose_weights,
)


def test_end_to_end_pipeline() -> None:
    """Run full integration test with existing test data."""

    # Use the judge calibration test data we created
    data_file = "tests/data/judge_calibration_data.jsonl"

    print("1. Calibrating judge scores to oracle labels...")
    dataset, cal_stats = load_dataset_with_calibration(
        data_file, judge_score_field="judge_score", oracle_label_field="oracle_label"
    )
    print(f"   ✓ Calibrated {dataset.n_samples} samples")
    print(f"   ✓ Used {cal_stats['n_oracle']} oracle labels")

    print("\n2. Loading data for CJE...")
    # Load and estimate
    sampler = PrecomputedSampler(dataset)
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
