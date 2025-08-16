"""Tests for oracle slice augmentation functionality."""
# mypy: ignore-errors

import numpy as np
import pytest
from unittest.mock import MagicMock
from cje.calibration.oracle_slice import (
    OracleSliceAugmentation,
    OracleSliceConfig,
)


class TestOracleSliceAugmentation:
    """Test suite for oracle slice augmentation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OracleSliceConfig()
        assert config.enable_augmentation is True
        assert config.enable_cross_fit is True
        assert config.min_pi == 0.01
        assert config.use_mar is False

    def test_disabled_augmentation(self):
        """Test that disabled augmentation returns zeros."""
        config = OracleSliceConfig(enable_augmentation=False)
        aug = OracleSliceAugmentation(config)

        # fit_m_hat should return ones when disabled
        weights = np.array([1.0, 2.0, 3.0])
        scores = np.array([0.1, 0.5, 0.9])
        m_hat = aug.fit_m_hat(weights, scores, "test_policy")
        np.testing.assert_array_equal(m_hat, np.ones_like(weights))

        # compute_augmentation should return zeros when disabled
        rewards = np.array([0.5, 0.6, 0.7])
        data = [{"prompt_id": i} for i in range(3)]
        aug_vector, diagnostics = aug.compute_augmentation(
            "test_policy", rewards, data, None
        )
        np.testing.assert_array_equal(aug_vector, np.zeros_like(rewards))
        assert diagnostics == {}

    def test_fit_m_hat_basic(self):
        """Test basic m̂(S) fitting without cross-fitting."""
        config = OracleSliceConfig(enable_cross_fit=False)
        aug = OracleSliceAugmentation(config)

        # Create test data with clear monotone pattern
        n = 100
        scores = np.linspace(0, 1, n)
        weights = 1 + 2 * scores  # Linear increasing

        m_hat = aug.fit_m_hat(weights, scores, "test_policy")

        # Check properties
        assert len(m_hat) == n
        assert np.abs(m_hat.mean() - 1.0) < 1e-10  # Should be normalized to mean 1

        # Check monotonicity (should be preserved by isotonic regression)
        diffs = np.diff(m_hat)
        assert np.all(diffs >= -1e-10)  # Allow small numerical errors

        # Check caching
        cached = aug._m_hat_cache.get("test_policy")
        assert cached is not None
        np.testing.assert_array_equal(cached, m_hat)

    def test_fit_m_hat_cross_fitted(self):
        """Test m̂(S) fitting with cross-fitting."""
        config = OracleSliceConfig(enable_cross_fit=True)
        aug = OracleSliceAugmentation(config)

        # Create test data
        n = 50
        scores = np.linspace(0, 1, n)
        weights = 1 + scores

        # Create fold assignments (5 folds)
        cv_folds = np.arange(n) % 5

        m_hat = aug.fit_m_hat(weights, scores, "test_policy", cv_folds)

        # Check properties
        assert len(m_hat) == n
        assert np.abs(m_hat.mean() - 1.0) < 1e-10  # Should be normalized

        # Each fold should have been predicted using model trained on other folds
        # This is hard to test directly, but we can check it ran without errors
        assert aug._m_hat_models.get("test_policy") is not None

    def test_compute_augmentation_no_oracle(self):
        """Test augmentation when no oracle labels are available."""
        config = OracleSliceConfig()
        aug = OracleSliceAugmentation(config)

        # Fit m̂(S) first
        weights = np.array([1.0, 2.0, 3.0])
        scores = np.array([0.1, 0.5, 0.9])
        aug.fit_m_hat(weights, scores, "test_policy")

        # Create data without oracle labels
        rewards = np.array([0.5, 0.6, 0.7])
        data = [{"prompt_id": i, "metadata": {}} for i in range(3)]

        aug_vector, diagnostics = aug.compute_augmentation(
            "test_policy", rewards, data, None
        )

        # Should return zeros when no oracle labels
        np.testing.assert_array_equal(aug_vector, np.zeros_like(rewards))
        assert diagnostics["p_oracle"] == 0.0
        assert diagnostics["n_oracle"] == 0

    def test_compute_augmentation_with_oracle(self):
        """Test augmentation with oracle labels available."""
        config = OracleSliceConfig()
        aug = OracleSliceAugmentation(config)

        # Fit m̂(S) first
        n = 6
        weights = np.ones(n) * 1.5  # Constant weights for simplicity
        scores = np.linspace(0, 1, n)
        aug.fit_m_hat(weights, scores, "test_policy")

        # Create data with oracle labels on half the samples
        rewards = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        oracle_labels = np.array([0.4, None, 0.8, None, 1.1, None])

        data = []
        for i in range(n):
            d = {"prompt_id": i, "metadata": {}}
            if oracle_labels[i] is not None:
                d["metadata"]["oracle_label"] = oracle_labels[i]
            data.append(d)

        aug_vector, diagnostics = aug.compute_augmentation(
            "test_policy", rewards, data, None
        )

        # Check augmentation properties
        assert len(aug_vector) == n
        assert diagnostics["p_oracle"] == 0.5  # 50% have oracle labels
        assert diagnostics["n_oracle"] == 3

        # Augmentation should be non-zero where we have oracle labels
        # and zero where we don't
        for i in range(n):
            if oracle_labels[i] is not None:
                # Should have non-zero augmentation (unless Y = f̂(S) exactly)
                if oracle_labels[i] != rewards[i]:
                    assert aug_vector[i] != 0.0
            else:
                # Should have zero augmentation
                assert aug_vector[i] == 0.0

        # Check that augmentation is unbiased (mean should be close to 0 in expectation)
        # Note: This is a weak test since we have few samples
        assert "aug_mean" in diagnostics
        assert "aug_var" in diagnostics

    def test_compute_augmentation_with_dataset_samples(self):
        """Test augmentation using dataset samples for oracle lookup."""
        config = OracleSliceConfig()
        aug = OracleSliceAugmentation(config)

        # Fit m̂(S)
        weights = np.ones(3)
        scores = np.array([0.1, 0.5, 0.9])
        aug.fit_m_hat(weights, scores, "test_policy")

        # Create mock dataset samples
        class MockSample:
            def __init__(self, prompt_id, oracle_label=None):
                self.prompt_id = prompt_id
                self.metadata = {}
                if oracle_label is not None:
                    self.metadata["oracle_label"] = oracle_label

        dataset_samples = [
            MockSample(0, oracle_label=0.3),
            MockSample(1),  # No oracle
            MockSample(2, oracle_label=0.9),
        ]

        # Create data pointing to these samples
        rewards = np.array([0.5, 0.6, 0.7])
        data = [{"prompt_id": i} for i in range(3)]

        aug_vector, diagnostics = aug.compute_augmentation(
            "test_policy", rewards, data, dataset_samples
        )

        # Check results
        assert diagnostics["n_oracle"] == 2  # Two samples have oracle labels
        assert aug_vector[1] == 0.0  # Middle sample has no oracle label

    def test_empty_data_handling(self):
        """Test handling of empty weights/scores."""
        config = OracleSliceConfig()
        aug = OracleSliceAugmentation(config)

        # Empty arrays
        weights = np.array([])
        scores = np.array([])

        m_hat = aug.fit_m_hat(weights, scores, "test_policy")
        assert len(m_hat) == 0

        # No m̂(S) fitted - compute_augmentation should handle gracefully
        rewards = np.array([0.5])
        data = [{"prompt_id": 0}]
        aug_vector, diagnostics = aug.compute_augmentation(
            "test_policy", rewards, data, None
        )
        np.testing.assert_array_equal(aug_vector, np.zeros_like(rewards))

    def test_get_diagnostics(self):
        """Test diagnostic retrieval."""
        config = OracleSliceConfig()
        aug = OracleSliceAugmentation(config)

        # Initially empty
        assert aug.get_diagnostics() == {}
        assert aug.get_diagnostics("nonexistent") == {}

        # After computation, diagnostics should be stored
        aug._diagnostics["policy1"] = {"test": 1}
        aug._diagnostics["policy2"] = {"test": 2}

        assert aug.get_diagnostics("policy1") == {"test": 1}
        assert len(aug.get_diagnostics()) == 2
        assert "policy1" in aug.get_diagnostics()
        assert "policy2" in aug.get_diagnostics()


class TestIntegrationWithEstimators:
    """Test integration with IPS and DR estimators."""

    def test_calibrated_ips_integration(self):
        """Test that CalibratedIPS can use oracle augmentation."""
        from cje.data import Dataset, Sample
        from cje.data.precomputed_sampler import PrecomputedSampler
        from cje.estimators.calibrated_ips import CalibratedIPS

        # Create simple dataset
        samples = []
        for i in range(10):
            sample = Sample(
                prompt_id=str(i),  # Must be string
                prompt=f"prompt_{i}",
                response=f"response_{i}",
                reward=0.5 + 0.05 * i,  # Increasing rewards
                base_policy_logprob=np.log(0.5),  # Use logprobs instead of probs
                target_policy_logprobs={"test_policy": np.log(0.6)},
                metadata={
                    "judge_score": 0.1 * i,  # Judge scores
                    "oracle_label": (
                        0.4 + 0.06 * i if i % 2 == 0 else None
                    ),  # 50% oracle
                },
            )
            samples.append(sample)

        dataset = Dataset(samples=samples, target_policies=["test_policy"])
        sampler = PrecomputedSampler(dataset)

        # Create estimator with oracle augmentation
        oracle_config = OracleSliceConfig(enable_augmentation=True)
        estimator = CalibratedIPS(sampler, oracle_slice_config=oracle_config)

        # Fit and estimate
        result = estimator.fit_and_estimate()

        # Check that augmentation was computed
        assert hasattr(estimator, "_aug_diagnostics")
        if "test_policy" in estimator._aug_diagnostics:
            diag = estimator._aug_diagnostics["test_policy"]
            assert diag.get("n_oracle", 0) == 5  # Half the samples
            assert diag.get("p_oracle", 0) == 0.5

        # Check that metadata includes augmentation info
        assert "slice_augmentation" in result.metadata
