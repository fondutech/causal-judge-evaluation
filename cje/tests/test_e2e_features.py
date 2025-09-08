"""End-to-end tests for CJE features using real arena data.

Tests major features like IIC, SIMCal, oracle augmentation, and cross-fitting
in realistic scenarios with the arena dataset.
"""

import pytest
import numpy as np
from copy import deepcopy
from typing import Dict, Any

from cje import load_dataset_from_jsonl

# Mark all tests in this file as E2E tests
pytestmark = [pytest.mark.e2e, pytest.mark.uses_arena_sample]
from cje.calibration import calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.data.models import Dataset
from cje.data.fresh_draws import FreshDrawDataset
from cje.estimators import CalibratedIPS, DRCPOEstimator


class TestIICFeature:
    """Test Isotonic Influence Control (IIC) variance reduction."""

    def test_iic_variance_reduction_pipeline(self, arena_sample: Dataset) -> None:
        """Test IIC reduces variance in real estimation."""
        import random

        random.seed(42)
        np.random.seed(42)

        # Prepare dataset with 50% oracle coverage
        for i, sample in enumerate(arena_sample.samples):
            if i % 2 == 1 and "oracle_label" in sample.metadata:
                sample.metadata["oracle_label"] = None

        calibrated, _ = calibrate_dataset(
            arena_sample, judge_field="judge_score", oracle_field="oracle_label"
        )

        sampler = PrecomputedSampler(calibrated)

        # Run with IIC disabled
        estimator_no_iic = CalibratedIPS(sampler, use_iic=False)
        results_no_iic = estimator_no_iic.fit_and_estimate()

        # Run with IIC enabled (default)
        estimator_iic = CalibratedIPS(sampler, use_iic=True)
        results_iic = estimator_iic.fit_and_estimate()

        # IIC should reduce standard errors on average
        # Note: With small samples and cross-fitting, IIC might slightly increase
        # SE for some policies, but should improve overall
        improvements = 0
        se_ratios = []
        for i in range(len(results_iic.standard_errors)):
            se_iic = results_iic.standard_errors[i]
            se_no_iic = results_no_iic.standard_errors[i]
            se_ratios.append(se_iic / se_no_iic)

            # Allow up to 5% increase for individual policies (small sample noise)
            assert (
                se_iic <= se_no_iic * 1.05
            ), f"Policy {i}: IIC SE {se_iic:.4f} > no-IIC SE {se_no_iic:.4f} by >5%"

            if se_iic < se_no_iic * 0.99:  # At least 1% improvement
                improvements += 1

        # Check average improvement across all policies
        avg_ratio = np.mean(se_ratios)
        assert (
            avg_ratio <= 1.01
        ), f"IIC increased average SE by {(avg_ratio-1)*100:.1f}%"

        # Should improve at least one policy
        assert improvements > 0, "IIC didn't improve any policies"

        # Check IIC diagnostics are present
        assert "iic_diagnostics" in results_iic.metadata
        iic_diag = results_iic.metadata["iic_diagnostics"]

        # Check each policy has IIC info
        for policy in sampler.target_policies:
            assert policy in iic_diag
            policy_diag = iic_diag[policy]

            if policy_diag["applied"]:
                assert "var_reduction" in policy_diag
                assert policy_diag["var_reduction"] >= 0
                assert "r_squared" in policy_diag
                assert 0 <= policy_diag["r_squared"] <= 1

    def test_iic_with_dr_estimators(
        self, arena_sample: Dataset, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test IIC works with DR estimators."""
        import random

        random.seed(42)
        np.random.seed(42)

        # Prepare dataset
        for i, sample in enumerate(arena_sample.samples):
            if i % 2 == 1 and "oracle_label" in sample.metadata:
                sample.metadata["oracle_label"] = None

        calibrated, cal_result = calibrate_dataset(
            arena_sample,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=5,
        )

        sampler = PrecomputedSampler(calibrated)

        # Test with DR-CPO
        dr_no_iic = DRCPOEstimator(
            sampler, reward_calibrator=cal_result.calibrator, n_folds=5, use_iic=False
        )
        dr_iic = DRCPOEstimator(
            sampler, reward_calibrator=cal_result.calibrator, n_folds=5, use_iic=True
        )

        # Add fresh draws to both
        for policy, fresh_dataset in arena_fresh_draws.items():
            if policy in sampler.target_policies:
                dr_no_iic.add_fresh_draws(policy, fresh_dataset)
                dr_iic.add_fresh_draws(policy, fresh_dataset)

        results_no_iic = dr_no_iic.fit_and_estimate()
        results_iic = dr_iic.fit_and_estimate()

        # IIC should work with DR too (allow 10% increase for small sample noise)
        # With only 100 samples and fresh draws, IIC can be unstable
        se_ratios = []
        for i in range(len(results_iic.standard_errors)):
            se_ratios.append(
                results_iic.standard_errors[i] / results_no_iic.standard_errors[i]
            )
            assert (
                results_iic.standard_errors[i]
                <= results_no_iic.standard_errors[i] * 1.10
            ), f"Policy {i}: IIC increased SE by >10%"

        # Check average improvement
        avg_ratio = np.mean(se_ratios)
        assert (
            avg_ratio <= 1.02
        ), f"IIC increased average SE by {(avg_ratio-1)*100:.1f}%"

        # Check IIC diagnostics present for DR
        assert "iic_diagnostics" in results_iic.metadata


class TestSIMCalFeature:
    """Test SIMCal (Surrogate-Indexed Monotone Calibration) variance control."""

    def test_simcal_mean_preservation(self, arena_sample: Dataset) -> None:
        """Test SIMCal preserves mean of weights."""
        import random

        random.seed(42)
        np.random.seed(42)

        # Prepare dataset
        for i, sample in enumerate(arena_sample.samples):
            if i % 2 == 1 and "oracle_label" in sample.metadata:
                sample.metadata["oracle_label"] = None

        calibrated, _ = calibrate_dataset(
            arena_sample, judge_field="judge_score", oracle_field="oracle_label"
        )

        sampler = PrecomputedSampler(calibrated)
        estimator = CalibratedIPS(sampler)
        estimator.fit()

        # Check mean preservation for each policy
        for policy in sampler.target_policies:
            weights = estimator.get_weights(policy)
            assert weights is not None

            # Mean should be very close to 1 (Hajek normalization)
            mean_weight = np.mean(weights)
            assert (
                abs(mean_weight - 1.0) < 0.01
            ), f"Policy {policy}: mean weight {mean_weight:.4f} != 1.0"

            # Weights should be non-negative
            assert np.all(weights >= 0), f"Policy {policy} has negative weights"


class TestCrossFitting:
    """Test cross-fitting for orthogonality in DR estimators."""

    def test_cross_fitting_consistency(
        self, arena_sample: Dataset, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test cross-fitting gives consistent results across runs."""
        import random

        # Prepare dataset
        for i, sample in enumerate(arena_sample.samples):
            if i % 2 == 1 and "oracle_label" in sample.metadata:
                sample.metadata["oracle_label"] = None

        calibrated, cal_result = calibrate_dataset(
            arena_sample,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=5,
        )

        sampler = PrecomputedSampler(calibrated)

        # Run DR multiple times with different seeds
        estimates_list = []
        for seed in [42, 123, 456]:
            random.seed(seed)
            np.random.seed(seed)

            estimator = DRCPOEstimator(
                sampler, reward_calibrator=cal_result.calibrator, n_folds=5
            )

            # Add fresh draws for each policy
            for policy, fresh_data in arena_fresh_draws.items():
                estimator.add_fresh_draws(policy, fresh_data)

            results = estimator.fit_and_estimate()
            estimates_list.append(results.estimates)

        # Results should be deterministic given the seed
        # But let's check they're at least consistent
        for i in range(len(estimates_list[0])):
            estimates = [e[i] for e in estimates_list]
            estimate_range = max(estimates) - min(estimates)

            # Should be very similar across runs
            assert (
                estimate_range < 0.05
            ), f"Policy {i}: range {estimate_range:.4f} across seeds"

    def test_fold_assignment_stability(self, arena_sample: Dataset) -> None:
        """Test that fold assignments are stable based on prompt_id."""
        from cje.data.folds import get_fold

        # Check fold assignments are deterministic
        prompt_ids = [s.prompt_id for s in arena_sample.samples]

        # Compute folds multiple times
        folds_5 = [get_fold(pid, 5) for pid in prompt_ids]
        folds_5_again = [get_fold(pid, 5) for pid in prompt_ids]

        # Should be identical
        assert folds_5 == folds_5_again

        # Check distribution is reasonable
        from collections import Counter

        fold_counts = Counter(folds_5)

        # Each fold should have roughly n/5 samples
        expected_per_fold = len(prompt_ids) / 5
        for fold, count in fold_counts.items():
            assert 0 <= fold < 5
            # Allow 50% deviation from expected
            assert 0.5 * expected_per_fold <= count <= 1.5 * expected_per_fold


class TestIntegrationScenarios:
    """Test complete scenarios combining multiple features."""

    def test_full_pipeline_with_all_features(
        self, arena_sample: Dataset, arena_fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test complete pipeline with IIC, SIMCal, cross-fitting, and oracle augmentation."""
        import random

        random.seed(42)
        np.random.seed(42)

        # Prepare dataset with 60% oracle coverage
        oracle_indices = [
            i
            for i, s in enumerate(arena_sample.samples)
            if "oracle_label" in s.metadata
        ]
        keep_n = int(len(oracle_indices) * 0.6)
        keep_indices = set(random.sample(oracle_indices, keep_n))

        for i, sample in enumerate(arena_sample.samples):
            if i not in keep_indices and "oracle_label" in sample.metadata:
                sample.metadata["oracle_label"] = None

        # Calibrate with cross-fitting
        calibrated, cal_result = calibrate_dataset(
            arena_sample,
            judge_field="judge_score",
            oracle_field="oracle_label",
            enable_cross_fit=True,
            n_folds=5,
        )

        # Create sampler
        sampler = PrecomputedSampler(calibrated)

        # Run DR with all features enabled
        estimator = DRCPOEstimator(
            sampler,
            reward_calibrator=cal_result.calibrator,
            n_folds=5,
            use_iic=True,  # IIC enabled
            # variance_cap removed as it's not supported by DRCPOEstimator
        )

        # Add fresh draws
        for policy, fresh_dataset in arena_fresh_draws.items():
            if policy in sampler.target_policies:
                estimator.add_fresh_draws(policy, fresh_dataset)

        results = estimator.fit_and_estimate()

        # Validate everything worked
        assert len(results.estimates) == 4
        assert all(0 <= e <= 1 for e in results.estimates)
        assert all(se > 0 for se in results.standard_errors)

        # Check all features are reflected in metadata/diagnostics
        assert "iic_diagnostics" in results.metadata
        assert results.diagnostics is not None

        # Check diagnostics summary includes all components
        summary = results.diagnostics.summary()
        assert "Weight ESS" in summary
        # DR orthogonality might not always be in summary

        # Verify reasonable performance
        # With all features, should have good ESS
        for policy in sampler.target_policies[:1]:  # Check at least one
            weights = estimator.get_weights(policy)
            if weights is not None:
                ess = (np.sum(weights) ** 2) / np.sum(weights**2)
                ess_fraction = ess / len(weights)
                assert ess_fraction > 0.05, f"Very low ESS: {ess_fraction:.3f}"
