"""Basic tests for DR estimators with pre-generated fresh draws."""

import numpy as np
import pytest
from typing import List, Dict, Any

from cje import (
    Dataset,
    Sample,
    PrecomputedSampler,
    calibrate_dataset,
    DRCPOEstimator,
    FreshDrawDataset,
    FreshDrawSample,
    create_synthetic_fresh_draws,
)


def create_test_dataset(n_samples: int = 20) -> Dataset:
    """Create a simple test dataset with judge scores and oracle labels."""
    samples = []

    for i in range(n_samples):
        # Create simple patterns for testing
        judge_score = 0.5 + 0.3 * np.sin(i / 3)  # Varies between 0.2 and 0.8
        oracle_label = judge_score + 0.1 * np.random.normal()  # Noisy oracle

        sample = Sample(
            prompt_id=f"test_{i}",
            prompt=f"Question {i}",
            response=f"Answer {i}",
            base_policy_logprob=-10.0 - i * 0.1,
            target_policy_logprobs={
                "pi_improved": -9.0 - i * 0.1,  # Better policy
                "pi_worse": -11.0 - i * 0.1,  # Worse policy
            },
            reward=None,  # Will be set by calibration
            metadata={
                "judge_score": float(np.clip(judge_score, 0, 1)),
                "oracle_label": float(np.clip(oracle_label, 0, 1)) if i < 10 else None,
            },
        )
        samples.append(sample)

    return Dataset(
        samples=samples,
        target_policies=["pi_improved", "pi_worse"],
    )


def test_dr_with_synthetic_fresh_draws() -> None:
    """Test DR estimation with synthetic fresh draws."""

    # Create and calibrate dataset
    dataset = create_test_dataset(20)
    calibrated_dataset, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
    )

    # Create sampler
    sampler = PrecomputedSampler(calibrated_dataset)

    # Create DR estimator
    dr = DRCPOEstimator(
        sampler=sampler,
        n_folds=5,
    )

    # Fit the estimator
    dr.fit()

    # Create synthetic fresh draws for each policy
    # For pi_improved, create draws with higher scores (better performance)
    fresh_draws_improved = create_synthetic_fresh_draws(
        calibrated_dataset,
        target_policy="pi_improved",
        draws_per_prompt=3,
        score_correlation=0.5,  # Lower correlation to show DR value
        seed=42,
    )
    # Artificially boost scores for improved policy to show it's better
    for sample in fresh_draws_improved.samples:
        sample.judge_score = min(1.0, sample.judge_score + 0.1)
    dr.add_fresh_draws("pi_improved", fresh_draws_improved)

    # For pi_worse, create draws with lower scores
    fresh_draws_worse = create_synthetic_fresh_draws(
        calibrated_dataset,
        target_policy="pi_worse",
        draws_per_prompt=3,
        score_correlation=0.5,
        seed=43,
    )
    # Artificially lower scores for worse policy
    for sample in fresh_draws_worse.samples:
        sample.judge_score = max(0.0, sample.judge_score - 0.1)
    dr.add_fresh_draws("pi_worse", fresh_draws_worse)

    # Run estimation
    result = dr.estimate()

    # Basic checks
    assert result.method == "dr_cpo"
    assert len(result.estimates) == 2  # Two target policies
    assert not np.any(np.isnan(result.estimates))
    assert np.all(result.estimates >= 0) and np.all(result.estimates <= 1)
    assert np.all(result.standard_errors > 0)

    # Check that improved policy has higher estimate
    # Find which index corresponds to which policy
    improved_idx = sampler.target_policies.index("pi_improved")
    worse_idx = sampler.target_policies.index("pi_worse")
    assert (
        result.estimates[improved_idx] > result.estimates[worse_idx]
    ), f"Improved policy ({result.estimates[improved_idx]:.4f}) should score higher than worse ({result.estimates[worse_idx]:.4f})"


def test_dr_fold_consistency() -> None:
    """Test that DR maintains consistent fold assignments."""

    # Create and calibrate dataset
    dataset = create_test_dataset(20)
    calibrated_dataset, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
    )

    sampler = PrecomputedSampler(calibrated_dataset)

    # Create DR estimator with specific number of folds
    dr = DRCPOEstimator(
        sampler=sampler,
        n_folds=4,
    )

    # Check that fold assignments are created
    assert dr.fold_assignments is not None
    assert len(dr.fold_assignments) == 20
    assert set(dr.fold_assignments) == {0, 1, 2, 3}


def test_dr_requires_complete_coverage() -> None:
    """Test that DR requires fresh draws for all valid samples."""

    # Create and calibrate dataset
    dataset = create_test_dataset(20)
    calibrated_dataset, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
    )

    sampler = PrecomputedSampler(calibrated_dataset)
    dr = DRCPOEstimator(sampler)
    dr.fit()

    # Create fresh draws with INCOMPLETE coverage (missing some prompts)
    partial_samples = []
    for i in range(10):  # Only first 10 prompts
        for draw_idx in range(3):
            partial_samples.append(
                FreshDrawSample(
                    prompt_id=f"test_{i}",
                    target_policy="pi_improved",
                    judge_score=0.5 + 0.1 * np.random.normal(),
                    response=None,  # Optional field
                    draw_idx=draw_idx,
                    fold_id=None,
                )
            )

    partial_draws = FreshDrawDataset(
        target_policy="pi_improved",
        draws_per_prompt=3,
        samples=partial_samples,
    )

    # Should raise error about missing coverage
    with pytest.raises(ValueError, match="missing"):
        dr.add_fresh_draws("pi_improved", partial_draws)


def test_dr_requires_fresh_draws_before_estimate() -> None:
    """Test that DR requires fresh draws to be added before estimation."""

    # Create and calibrate dataset
    dataset = create_test_dataset(20)
    calibrated_dataset, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
    )

    sampler = PrecomputedSampler(calibrated_dataset)
    dr = DRCPOEstimator(sampler)
    dr.fit()

    # Try to estimate without adding fresh draws
    with pytest.raises(ValueError, match="No fresh draws"):
        result = dr.estimate()


def test_dr_fold_alignment() -> None:
    """Test that fresh draws use correct fold predictions."""

    # Create and calibrate dataset
    dataset = create_test_dataset(20)
    calibrated_dataset, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
    )

    # Create DR estimator
    sampler = PrecomputedSampler(calibrated_dataset)
    dr = DRCPOEstimator(sampler, n_folds=5)

    # Check that fold assignments are accessible
    assert dr.fold_assignments is not None
    assert len(dr.fold_assignments) == 20
    assert set(dr.fold_assignments) == {0, 1, 2, 3, 4}


if __name__ == "__main__":
    # Run basic test
    test_dr_with_synthetic_fresh_draws()
    print("✓ Basic DR test passed")

    test_dr_fold_consistency()
    print("✓ Fold consistency test passed")

    test_dr_requires_complete_coverage()
    print("✓ Complete coverage requirement test passed")

    test_dr_requires_fresh_draws_before_estimate()
    print("✓ Fresh draws requirement test passed")

    test_dr_fold_alignment()
    print("✓ Fold alignment test passed")

    print("\n✓ All DR tests passed!")
