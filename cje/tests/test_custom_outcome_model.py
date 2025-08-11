"""Test custom outcome model implementation."""

import numpy as np
from cje import (
    BaseOutcomeModel,
    LinearOutcomeModel,
    Dataset,
    Sample,
    PrecomputedSampler,
    calibrate_dataset,
    DRCPOEstimator,
    create_synthetic_fresh_draws,
)
from typing import List, Any


class SimpleAverageOutcomeModel(BaseOutcomeModel):
    """Example custom outcome model that just predicts average reward by judge score bin."""

    def __init__(self, n_folds: int = 5):
        """Initialize the model."""
        super().__init__(n_folds)

    def _fit_single_model(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: np.ndarray,
        judge_scores: np.ndarray,
    ) -> Any:
        """Learn average reward for each judge score decile."""
        # Create bins for judge scores
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_indices = np.digitize(judge_scores, bins) - 1
        bin_indices = np.clip(bin_indices, 0, 9)

        # Calculate mean reward for each bin
        bin_means = {}
        for i in range(10):
            mask = bin_indices == i
            if mask.any():
                bin_means[i] = rewards[mask].mean()
            else:
                bin_means[i] = rewards.mean()  # fallback

        return bin_means

    def _predict_single_model(
        self,
        model: Any,
        prompts: List[str],
        responses: List[str],
        judge_scores: np.ndarray,
    ) -> np.ndarray:
        """Predict using bin averages."""
        bin_means = model
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(judge_scores, bins) - 1
        bin_indices = np.clip(bin_indices, 0, 9)

        predictions = np.array([bin_means[i] for i in bin_indices])
        return predictions


def create_test_dataset(n_samples: int = 100) -> Dataset:
    """Create a test dataset."""
    np.random.seed(42)
    samples = []

    for i in range(n_samples):
        judge_score = np.random.beta(2, 2)  # Beta distribution for variety
        oracle_label = judge_score + 0.05 * np.random.normal()
        oracle_label = np.clip(oracle_label, 0, 1)

        sample = Sample(
            prompt=f"Question {i}",
            response=f"Answer {i}",
            base_policy_logprob=-10.0 - i * 0.05,
            target_policy_logprobs={
                "improved": -9.0 - i * 0.05,
            },
            reward=None,
            metadata={
                "prompt_id": f"test_{i}",
                "judge_score": float(judge_score),
                "oracle_label": float(oracle_label) if i < 50 else None,  # 50% coverage
            },
        )
        samples.append(sample)

    return Dataset(
        samples=samples,
        target_policies=["improved"],
    )


def test_custom_outcome_model() -> None:
    """Test using a custom outcome model in DR estimation."""

    # Create and calibrate dataset
    dataset = create_test_dataset(100)
    calibrated_dataset, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
    )

    # Create sampler
    sampler = PrecomputedSampler(calibrated_dataset)

    # Create DR estimator with custom outcome model
    custom_model = SimpleAverageOutcomeModel(n_folds=3)
    dr = DRCPOEstimator(
        sampler=sampler,
        outcome_model=custom_model,  # Use our custom model
        n_folds=3,
    )

    # Fit the estimator
    dr.fit()

    # Add fresh draws
    fresh_draws = create_synthetic_fresh_draws(
        calibrated_dataset,
        target_policy="improved",
        draws_per_prompt=5,
        seed=42,
    )
    dr.add_fresh_draws("improved", fresh_draws)

    # Get estimates
    result = dr.estimate()

    # Basic checks
    assert result.method == "dr_cpo"
    assert len(result.estimates) == 1
    assert not np.isnan(result.estimates[0])
    assert 0 <= result.estimates[0] <= 1
    assert result.standard_errors[0] > 0

    print(
        f"✓ Custom outcome model estimate: {result.estimates[0]:.4f} ± {result.standard_errors[0]:.4f}"
    )


def test_linear_outcome_model() -> None:
    """Test the provided LinearOutcomeModel example."""

    # Create and calibrate dataset
    dataset = create_test_dataset(100)
    calibrated_dataset, cal_result = calibrate_dataset(
        dataset,
        judge_field="judge_score",
        oracle_field="oracle_label",
    )

    # Create sampler
    sampler = PrecomputedSampler(calibrated_dataset)

    # Create DR estimator with LinearOutcomeModel
    linear_model = LinearOutcomeModel(n_folds=3, alpha=1.0)
    dr = DRCPOEstimator(
        sampler=sampler,
        outcome_model=linear_model,
        n_folds=3,
    )

    # Fit the estimator
    dr.fit()

    # Add fresh draws
    fresh_draws = create_synthetic_fresh_draws(
        calibrated_dataset,
        target_policy="improved",
        draws_per_prompt=5,
        seed=42,
    )
    dr.add_fresh_draws("improved", fresh_draws)

    # Get estimates
    result = dr.estimate()

    # Basic checks
    assert result.method == "dr_cpo"
    assert len(result.estimates) == 1
    assert not np.isnan(result.estimates[0])
    assert 0 <= result.estimates[0] <= 1
    assert result.standard_errors[0] > 0

    print(
        f"✓ Linear outcome model estimate: {result.estimates[0]:.4f} ± {result.standard_errors[0]:.4f}"
    )


if __name__ == "__main__":
    test_custom_outcome_model()
    test_linear_outcome_model()
    print("\n✓ All custom outcome model tests passed!")
