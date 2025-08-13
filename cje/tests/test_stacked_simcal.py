"""Tests for stacked SIMCal implementation."""

import numpy as np
import pytest
from cje.calibration.simcal import SIMCalibrator, SimcalConfig


class TestStackedSIMCal:
    """Test suite for stacked SIMCal calibration."""

    def test_basic_stacking(self) -> None:
        """Test that stacking produces valid results."""
        np.random.seed(42)
        n = 100

        # Create synthetic data
        scores = np.random.uniform(0, 1, n)
        weights = np.exp(np.random.normal(0, 1, n))
        weights = weights / weights.mean()
        rewards = np.random.uniform(0, 1, n)

        # Run stacked calibration
        cfg = SimcalConfig(ess_floor=0.2, include_baseline=True)
        calibrator = SIMCalibrator(cfg)
        calibrated, info = calibrator.transform(weights, scores, rewards=rewards)

        # Check basic properties
        assert calibrated.shape == weights.shape
        assert np.all(calibrated > 0)
        assert np.abs(calibrated.mean() - 1.0) < 1e-10

        # Check info dict
        assert "mixture_weights" in info
        assert "candidates" in info
        assert "gamma" in info
        assert "var_before" in info
        assert "var_after" in info
        assert "ess_before" in info
        assert "ess_after" in info
        assert "oof_variance_reduction" in info

        # Check mixture weights
        mixture_weights = np.array(info["mixture_weights"])
        assert len(mixture_weights) == len(info["candidates"])
        assert np.abs(mixture_weights.sum() - 1.0) < 1e-10
        assert np.all(mixture_weights >= 0)

        # Check ESS constraint
        var_after = info["var_after"]
        ess_after = info["ess_after"]
        expected_ess = n / (1 + var_after)
        assert np.abs(ess_after - expected_ess) < 1e-10
        assert ess_after >= 0.2 * n - 1e-10  # ESS floor constraint

    def test_without_baseline(self) -> None:
        """Test stacking without baseline candidate."""
        np.random.seed(42)
        n = 100

        scores = np.random.uniform(0, 1, n)
        weights = np.exp(np.random.normal(0, 1, n))
        weights = weights / weights.mean()

        # Run without baseline
        cfg = SimcalConfig(include_baseline=False)
        calibrator = SIMCalibrator(cfg)
        calibrated, info = calibrator.transform(weights, scores)

        # Should only have increasing and decreasing
        assert len(info["candidates"]) == 2
        assert "baseline" not in info["candidates"]
        assert "increasing" in info["candidates"]
        assert "decreasing" in info["candidates"]

    def test_with_dr_residuals(self) -> None:
        """Test stacking with DR residuals."""
        np.random.seed(42)
        n = 100

        scores = np.random.uniform(0, 1, n)
        weights = np.exp(np.random.normal(0, 1, n))
        weights = weights / weights.mean()
        rewards = np.random.uniform(0, 1, n)
        residuals = rewards - 0.5  # Simple residuals

        # Run with residuals
        cfg = SimcalConfig()
        calibrator = SIMCalibrator(cfg)
        calibrated, info = calibrator.transform(weights, scores, residuals=residuals)

        # Check that DR mode was used
        assert info["if_type"] == "dr"

    def test_with_fold_ids(self) -> None:
        """Test stacking with pre-specified fold IDs."""
        np.random.seed(42)
        n = 100

        scores = np.random.uniform(0, 1, n)
        weights = np.exp(np.random.normal(0, 1, n))
        weights = weights / weights.mean()
        rewards = np.random.uniform(0, 1, n)
        fold_ids = np.array([i % 5 for i in range(n)])

        # Run with fold IDs
        cfg = SimcalConfig(n_folds=5)
        calibrator = SIMCalibrator(cfg)
        calibrated, info = calibrator.transform(
            weights, scores, rewards=rewards, fold_ids=fold_ids
        )

        # Check that correct number of folds was used
        assert info["n_folds"] == 5

    def test_baseline_shrinkage(self) -> None:
        """Test baseline shrinkage parameter."""
        np.random.seed(42)
        n = 100

        scores = np.random.uniform(0, 1, n)
        weights = np.exp(np.random.normal(0, 2, n))  # High variance
        weights = weights / weights.mean()

        # Run with and without shrinkage
        cfg_no_shrink = SimcalConfig(baseline_shrink=0.0)
        cfg_shrink = SimcalConfig(baseline_shrink=0.1)

        calibrator_no_shrink = SIMCalibrator(cfg_no_shrink)
        calibrator_shrink = SIMCalibrator(cfg_shrink)

        cal_no_shrink, _ = calibrator_no_shrink.transform(weights, scores)
        cal_shrink, _ = calibrator_shrink.transform(weights, scores)

        # With shrinkage should be closer to original weights
        dist_no_shrink = np.mean((cal_no_shrink - weights) ** 2)
        dist_shrink = np.mean((cal_shrink - weights) ** 2)
        assert dist_shrink < dist_no_shrink

    def test_variance_reduction(self) -> None:
        """Test that stacking reduces variance vs single candidates."""
        np.random.seed(42)
        n = 200

        # Create data where stacking should help
        scores = np.random.uniform(0, 1, n)
        # Make weights that are neither clearly increasing nor decreasing
        weights = np.exp(np.sin(scores * 2 * np.pi) + np.random.normal(0, 0.5, n))
        weights = weights / weights.mean()
        rewards = scores + np.random.normal(0, 0.1, n)

        cfg = SimcalConfig(include_baseline=True)
        calibrator = SIMCalibrator(cfg)
        calibrated, info = calibrator.transform(weights, scores, rewards=rewards)

        # Check variance reduction
        reduction = info["oof_variance_reduction"]
        # Stacked should not be worse than best single
        assert reduction <= 1.0 + 1e-10

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        np.random.seed(42)

        # Test with uniform weights
        n = 50
        scores = np.random.uniform(0, 1, n)
        weights = np.ones(n)

        cfg = SimcalConfig()
        calibrator = SIMCalibrator(cfg)
        calibrated, info = calibrator.transform(weights, scores)

        # Should remain uniform
        assert np.allclose(calibrated, 1.0)
        assert info["var_after"] < 1e-10
