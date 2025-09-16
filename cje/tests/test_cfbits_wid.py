"""Unit tests for CF-bits Wid (identification width) implementation."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import List

from cje.cfbits.identification import compute_identification_width


class TestWidPhase1Certificate:
    """Test the Phase-1 Wid certificate algorithm."""

    def create_mock_estimator(
        self,
        judge_scores: List[float],
        oracle_labels: List[float] = None,
        weights: np.ndarray = None,
    ) -> Mock:
        """Create a mock estimator with specified data."""
        estimator = Mock()
        sampler = Mock()

        # Create samples with judge scores
        samples = []
        for i, score in enumerate(judge_scores):
            metadata = Mock()
            metadata.judge_score = score

            # Add oracle label if provided
            if oracle_labels is not None and i < len(oracle_labels):
                metadata.oracle_label = oracle_labels[i]
            else:
                metadata.oracle_label = None

            sample = Mock()
            sample.metadata = metadata
            samples.append(sample)

        # Set up dataset with samples
        dataset = Mock()
        dataset.samples = samples
        sampler.dataset = dataset

        # Set up weights
        if weights is None:
            weights = np.ones(len(judge_scores))
        if len(judge_scores) > 0:
            weights = weights / np.mean(weights)  # Normalize to mean-1
        sampler.compute_importance_weights = Mock(return_value=weights)

        estimator.sampler = sampler
        return estimator

    def test_wid_with_no_samples(self):
        """Test Wid computation with no samples."""
        estimator = self.create_mock_estimator(judge_scores=[])

        wid, diagnostics = compute_identification_width(estimator, "test_policy")

        assert wid is None
        assert diagnostics["implemented"] is True
        assert diagnostics["reason"] == "no_samples"

    def test_wid_with_no_judge_scores(self):
        """Test Wid computation when samples lack judge scores."""
        estimator = Mock()
        sampler = Mock()

        # Create samples without judge scores
        samples = []
        for i in range(10):
            sample = Mock()
            sample.metadata = Mock()
            sample.metadata.judge_score = None
            samples.append(sample)

        dataset = Mock()
        dataset.samples = samples
        sampler.dataset = dataset
        estimator.sampler = sampler

        wid, diagnostics = compute_identification_width(estimator, "test_policy")

        assert wid is None
        assert diagnostics["reason"] == "no_judge_scores"

    def test_wid_with_no_oracle_labels(self):
        """Test Wid computation with no oracle labels."""
        judge_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        estimator = self.create_mock_estimator(
            judge_scores=judge_scores, oracle_labels=None  # No oracle labels
        )

        wid, diagnostics = compute_identification_width(estimator, "test_policy")

        assert wid is None
        assert diagnostics["reason"] == "no_oracle_labels"

    def test_wid_with_perfect_labels(self):
        """Test Wid with perfect oracle labels (no uncertainty)."""
        # Create monotone perfect labels
        judge_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        oracle_labels = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]

        estimator = self.create_mock_estimator(
            judge_scores=judge_scores, oracle_labels=oracle_labels
        )

        wid, diagnostics = compute_identification_width(
            estimator, "test_policy", n_bins=3
        )

        assert wid is not None
        assert wid >= 0  # Should be small but non-negative
        assert diagnostics["implemented"] is True
        assert diagnostics["n_oracle"] == len(oracle_labels)
        assert diagnostics["n_bins"] == 3

    def test_wid_increases_with_label_sparsity(self):
        """Test that Wid increases as oracle labels become sparser."""
        judge_scores = list(np.linspace(0, 1, 100))

        # Dense labels (all samples labeled)
        dense_labels = list(np.random.rand(100))
        estimator_dense = self.create_mock_estimator(
            judge_scores=judge_scores, oracle_labels=dense_labels
        )

        wid_dense, _ = compute_identification_width(
            estimator_dense, "test_policy", n_bins=10
        )

        # Sparse labels (only 10% labeled)
        sparse_labels = dense_labels[:10] + [None] * 90
        estimator_sparse = self.create_mock_estimator(
            judge_scores=judge_scores[:10] + judge_scores[10:],
            oracle_labels=sparse_labels,
        )

        wid_sparse, _ = compute_identification_width(
            estimator_sparse, "test_policy", n_bins=10
        )

        assert wid_dense is not None
        assert wid_sparse is not None
        assert wid_sparse > wid_dense  # Sparse should have more uncertainty

    def test_wid_monotone_correction(self):
        """Test monotone correction when labels violate isotonicity."""
        # Create non-monotone labels
        judge_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        oracle_labels = [0.2, 0.8, 0.1, 0.6, 0.9]  # Non-monotone

        estimator = self.create_mock_estimator(
            judge_scores=judge_scores, oracle_labels=oracle_labels
        )

        wid, diagnostics = compute_identification_width(
            estimator, "test_policy", n_bins=5
        )

        assert wid is not None
        assert wid > 0  # Should have uncertainty due to violations

        # Check that monotone correction was applied
        ell_up = diagnostics["ell_up"]
        u_down = diagnostics["u_down"]

        # Lower bounds should be non-decreasing
        for i in range(1, len(ell_up)):
            assert ell_up[i] >= ell_up[i - 1]

        # Upper bounds should be non-increasing
        for i in range(1, len(u_down)):
            assert u_down[i] <= u_down[i - 1]

    def test_wid_with_extreme_weights(self):
        """Test Wid computation with extreme importance weights."""
        judge_scores = list(np.linspace(0, 1, 20))
        oracle_labels = list(np.random.rand(20))

        # Create extreme weights (high variance)
        weights = np.ones(20)
        weights[0] = 100  # One sample has huge weight

        estimator = self.create_mock_estimator(
            judge_scores=judge_scores, oracle_labels=oracle_labels, weights=weights
        )

        wid, diagnostics = compute_identification_width(
            estimator, "test_policy", n_bins=5
        )

        assert wid is not None
        assert diagnostics["implemented"] is True

        # Check that target mass is heavily concentrated
        p_prime = diagnostics["p_prime_j"]
        assert max(p_prime) > 0.5  # Most mass in one bin due to extreme weight

    def test_wid_confidence_level(self):
        """Test that Wid changes with confidence level alpha."""
        judge_scores = list(np.linspace(0, 1, 50))
        oracle_labels = list(np.random.rand(50))

        estimator = self.create_mock_estimator(
            judge_scores=judge_scores, oracle_labels=oracle_labels
        )

        # Compute Wid at different confidence levels
        wid_95, _ = compute_identification_width(estimator, "test_policy", alpha=0.05)

        wid_99, _ = compute_identification_width(estimator, "test_policy", alpha=0.01)

        assert wid_95 is not None
        assert wid_99 is not None
        assert wid_99 >= wid_95  # More conservative CI should give wider Wid

    def test_wid_bin_adaptation(self):
        """Test that number of bins adapts to oracle sample size."""
        judge_scores = list(np.linspace(0, 1, 100))

        # Very few oracle labels
        few_labels = [0.5] * 5
        estimator_few = self.create_mock_estimator(
            judge_scores=judge_scores[:5] + judge_scores[5:],
            oracle_labels=few_labels + [None] * 95,
        )

        wid_few, diag_few = compute_identification_width(
            estimator_few, "test_policy", n_bins=20
        )

        # Many oracle labels
        many_labels = list(np.random.rand(100))
        estimator_many = self.create_mock_estimator(
            judge_scores=judge_scores, oracle_labels=many_labels
        )

        wid_many, diag_many = compute_identification_width(
            estimator_many, "test_policy", n_bins=20
        )

        assert wid_few is not None
        assert wid_many is not None

        # Should use fewer bins when labels are sparse
        assert diag_few["n_bins"] <= diag_many["n_bins"]
        assert diag_few["n_bins"] >= 6  # Minimum bins

    def test_wid_deterministic(self):
        """Test that Wid computation is deterministic with fixed inputs."""
        judge_scores = list(np.linspace(0, 1, 30))
        oracle_labels = [np.sin(s * np.pi) for s in judge_scores]
        weights = np.exp(-np.array(judge_scores))

        estimator = self.create_mock_estimator(
            judge_scores=judge_scores, oracle_labels=oracle_labels, weights=weights
        )

        # Compute multiple times
        results = []
        for _ in range(3):
            wid, _ = compute_identification_width(
                estimator, "test_policy", alpha=0.05, n_bins=10
            )
            results.append(wid)

        # All results should be identical
        assert all(w == results[0] for w in results)

    def test_wid_contributions_sum_to_total(self):
        """Test that per-bin contributions sum to total Wid."""
        judge_scores = list(np.linspace(0, 1, 40))
        oracle_labels = list(np.random.rand(40))

        estimator = self.create_mock_estimator(
            judge_scores=judge_scores, oracle_labels=oracle_labels
        )

        wid, diagnostics = compute_identification_width(
            estimator, "test_policy", n_bins=8
        )

        assert wid is not None

        # Contributions should sum to Wid
        contrib = diagnostics["contrib_j"]
        total_contrib = sum(contrib)

        assert np.isclose(total_contrib, wid, rtol=1e-10)

        # Each contribution should be non-negative
        assert all(c >= -1e-10 for c in contrib)  # Allow tiny numerical errors

    def test_wid_with_missing_judge_scores_in_oracle(self):
        """Test handling when oracle samples lack judge scores."""
        estimator = Mock()
        sampler = Mock()

        # Mix of samples with and without judge scores
        samples = []
        for i in range(20):
            sample = Mock()
            metadata = Mock()

            if i < 10:
                metadata.judge_score = i / 20
                metadata.oracle_label = np.random.rand()
            else:
                metadata.judge_score = i / 20
                metadata.oracle_label = None

            sample.metadata = metadata
            samples.append(sample)

        # Add oracle sample without judge score
        bad_sample = Mock()
        bad_metadata = Mock()
        bad_metadata.judge_score = None  # Missing score
        bad_metadata.oracle_label = 0.5
        bad_sample.metadata = bad_metadata
        samples.append(bad_sample)

        dataset = Mock()
        dataset.samples = samples
        sampler.dataset = dataset
        sampler.compute_importance_weights = Mock(return_value=np.ones(21))
        estimator.sampler = sampler

        wid, diagnostics = compute_identification_width(estimator, "test_policy")

        # Should handle missing scores gracefully
        assert wid is not None or diagnostics["reason"] == "oracle_missing_scores"

    def test_p_mass_unlabeled_diagnostic(self):
        """Test that p_mass_unlabeled is computed correctly."""
        judge_scores = list(np.linspace(0, 1, 100))
        # Only label first 10 samples (10% coverage)
        oracle_labels = list(np.random.rand(10)) + [None] * 90

        estimator = self.create_mock_estimator(
            judge_scores=judge_scores, oracle_labels=oracle_labels
        )

        wid, diagnostics = compute_identification_width(
            estimator, "test_policy", n_bins=10
        )

        assert "p_mass_unlabeled" in diagnostics
        assert diagnostics["p_mass_unlabeled"] > 0  # Should have mass on unlabeled bins
        assert diagnostics["p_mass_unlabeled"] <= 1.0  # Should be a valid probability

    def test_degenerate_scores(self):
        """Test handling when all judge scores are identical."""
        # All scores the same
        judge_scores = [0.5] * 50
        oracle_labels = [0.6] * 50

        estimator = self.create_mock_estimator(
            judge_scores=judge_scores, oracle_labels=oracle_labels
        )

        wid, diagnostics = compute_identification_width(
            estimator, "test_policy", n_bins=10
        )

        assert wid is not None
        assert wid >= 0  # Should handle degenerate case gracefully
        assert diagnostics["n_bins"] <= 2  # Should have very few bins


class TestWidGating:
    """Test Wid integration with gates."""

    def test_wid_triggers_warning_gate(self):
        """Test that large Wid triggers warning."""
        from cje.cfbits.core import apply_gates

        decision = apply_gates(
            wid=0.6,  # Above warning threshold
            wvar=0.2,
        )

        assert decision.state in ["WARNING", "CRITICAL"]
        assert any("identification" in r.lower() for r in decision.reasons)

    def test_wid_triggers_critical_gate(self):
        """Test that very large Wid triggers critical."""
        from cje.cfbits.core import apply_gates

        decision = apply_gates(
            wid=0.9,  # Above critical threshold
            wvar=0.2,
        )

        assert decision.state == "CRITICAL"
        assert any("dominates" in r.lower() for r in decision.reasons)

    def test_wmax_triggers_refuse(self):
        """Test that extreme Wmax triggers refuse."""
        from cje.cfbits.core import apply_gates

        decision = apply_gates(
            wid=1.5,  # Extreme value (above WMAX_THRESHOLD=1.0 for [0,1] KPI)
            wvar=0.8,
        )

        assert decision.state == "REFUSE"
        assert any("catastrophic" in r.lower() for r in decision.reasons)


class TestBudgetHelpers:
    """Test CF-bits budget helper functions."""

    def test_logs_for_delta_bits(self):
        """Test logs factor calculation for CF-bits improvement."""
        from cje.cfbits import logs_for_delta_bits

        # To gain 0.5 bits, need 2x logs
        factor = logs_for_delta_bits(0.5)
        assert factor == 2.0

        # To gain 1 bit, need 4x logs
        factor = logs_for_delta_bits(1.0)
        assert factor == 4.0

        # To gain 2 bits, need 16x logs
        factor = logs_for_delta_bits(2.0)
        assert factor == 16.0

    def test_bits_width_conversion(self):
        """Test conversion between bits and width."""
        from cje.cfbits import bits_to_width, width_to_bits

        # 0 bits = width of 1.0
        assert bits_to_width(0) == 1.0

        # 1 bit = width of 0.5
        assert bits_to_width(1.0) == 0.5

        # 2 bits = width of 0.25
        assert bits_to_width(2.0) == 0.25

        # Round trip
        width = 0.25
        bits = width_to_bits(width)
        assert bits_to_width(bits) == width

    def test_leaderboard_scoring_direction(self):
        """Test that higher bits lead to better scores."""
        import pandas as pd
        from cje.cfbits.aggregates import create_efficiency_leaderboard

        # Create test data
        data = pd.DataFrame(
            {
                "estimator": ["A", "B", "C"],
                "sample_size": [1000, 1000, 1000],
                "oracle_coverage": [0.1, 0.1, 0.1],
                "bits_tot_mean": [1.0, 2.0, 0.5],  # B has most bits
                "rmse": [0.1, 0.1, 0.1],  # Same RMSE
                "ifr_oua_gmean": [0.5, 0.5, 0.5],
            }
        )

        leaderboard = create_efficiency_leaderboard(data)

        # B should have best score (lowest) due to highest bits
        scores = leaderboard.sort_values("estimator")["combined_score"].values
        assert scores[1] < scores[0]  # B < A
        assert scores[1] < scores[2]  # B < C
