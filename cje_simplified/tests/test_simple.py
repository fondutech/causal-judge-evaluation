"""Minimal tests without file I/O."""

from cje_simplified import (
    PrecomputedSampler,
    CalibratedIPS,
    calibrate_judge_scores,
)
import numpy as np


def test_in_memory_pipeline():
    """Test pipeline with in-memory data only."""

    # Create simple data
    data = [
        {
            "prompt": f"q{i}",
            "response": f"a{i}",
            "reward": 0.7 + 0.01 * i,
            "p0_logprob": -10.0,
            "target_logps": {"pi_test": -9.0 + 0.1 * i},
        }
        for i in range(20)
    ]

    # Load and estimate
    sampler = PrecomputedSampler(data)
    estimator = CalibratedIPS(sampler, k_folds=2)
    results = estimator.fit_and_estimate()

    print(f"✓ Pipeline ran: estimate = {results.estimates[0]:.3f}")
    assert 0.6 < results.estimates[0] < 0.8


def test_judge_calibration():
    """Test judge calibration directly."""

    # Create synthetic scores
    judge_scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3)
    oracle_labels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    calibrated, stats = calibrate_judge_scores(judge_scores, oracle_labels)

    print(f"✓ Calibration ran: RMSE = {stats['rmse']:.3f}")
    assert len(calibrated) == len(judge_scores)
    assert all(0 <= s <= 1 for s in calibrated)


if __name__ == "__main__":
    print("Running simple tests...\n")
    test_in_memory_pipeline()
    test_judge_calibration()
    print("\nDone! ✨")
