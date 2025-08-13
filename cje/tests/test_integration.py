"""Integration test showing complete workflow."""

import numpy as np
from cje import (
    PrecomputedSampler,
    CalibratedIPS,
    load_dataset_from_jsonl,
    calibrate_dataset,
)
from cje.utils.diagnostics import compute_weight_diagnostics


def test_end_to_end_pipeline() -> None:
    """Run full integration test with existing test data."""

    # Use the judge calibration test data we created
    # Use absolute path resolution to work from any working directory
    import os
    from pathlib import Path

    test_dir = Path(__file__).parent
    data_file = test_dir / "data" / "judge_calibration_data.jsonl"

    print("1. Loading data and calibrating judge scores...")
    # Load data without rewards - they will be added through calibration
    dataset = load_dataset_from_jsonl(str(data_file))

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
    estimator = CalibratedIPS(sampler)
    results = estimator.fit_and_estimate()
    print(f"   ✓ Estimate: {results.estimates[0]:.3f}")
    print(f"   ✓ Std Error: {results.standard_errors[0]:.3f}")

    print("\n4. Checking weight diagnostics...")
    weights = estimator.get_weights("pi_test")
    if weights is not None:
        diag = compute_weight_diagnostics(weights, "pi_test")
        print(f"   ✓ Mean weight: {np.mean(weights):.3f}")
        print(f"   ✓ ESS fraction: {diag['ess_fraction']:.1%}")
    else:
        print("   ⚠ No weights available for pi_test")

    print("\n✅ Integration test passed!")


if __name__ == "__main__":
    test_end_to_end_pipeline()
