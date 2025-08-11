"""Minimal tests without file I/O."""

from cje import calibrate_judge_scores
import numpy as np


def test_judge_calibration() -> None:
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
    test_judge_calibration()
    print("\nDone! ✨")
