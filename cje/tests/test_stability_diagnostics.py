"""Tests for stability and drift detection diagnostics."""

import numpy as np
import pytest
from typing import Dict, Any

from cje.utils.diagnostics import (
    kendall_tau_drift,
    sequential_drift_detection,
    reliability_diagram,
    eif_qq_plot_data,
    compute_stability_diagnostics,
)


class TestKendallTauDrift:
    """Test Kendall tau rank drift detection."""

    def test_no_drift_identical(self) -> None:
        """Test with identical rankings (no drift)."""
        np.random.seed(42)
        scores = np.random.uniform(0, 1, 100)

        result = kendall_tau_drift(scores, scores)

        assert result["tau"] == 1.0, "Identical rankings should have tau=1"
        assert not result["drift_detected"], "No drift should be detected"
        assert "Excellent stability" in result["interpretation"]

    def test_complete_reversal(self) -> None:
        """Test with completely reversed rankings."""
        scores_1 = np.arange(50)
        scores_2 = scores_1[::-1]  # Reverse order

        result = kendall_tau_drift(scores_1, scores_2)

        assert result["tau"] == -1.0, "Reversed rankings should have tau=-1"
        assert result["drift_detected"], "Drift should be detected"
        assert "Severe drift" in result["interpretation"]

    def test_moderate_drift(self) -> None:
        """Test with moderate ranking changes."""
        np.random.seed(42)
        scores_1 = np.random.uniform(0, 1, 100)
        # Add noise to create moderate drift
        scores_2 = scores_1 + np.random.normal(0, 0.2, 100)

        result = kendall_tau_drift(scores_1, scores_2)

        assert 0.4 < result["tau"] < 0.8, f"Expected moderate tau, got {result['tau']}"
        assert "drift" in result["interpretation"].lower()

    def test_with_labels(self) -> None:
        """Test drift detection with oracle labels."""
        np.random.seed(42)
        n = 100

        scores_1 = np.random.uniform(0, 1, n)
        scores_2 = scores_1 + np.random.normal(0, 0.1, n)

        # Labels correlated with scores
        labels_1 = scores_1 + np.random.normal(0, 0.05, n)
        labels_2 = scores_2 * 0.5 + 0.25  # Different relationship

        result = kendall_tau_drift(scores_1, scores_2, labels_1, labels_2)

        assert "tau_with_labels_1" in result
        assert "tau_with_labels_2" in result
        assert "tau_change" in result
        assert "calibration_drift" in result


class TestSequentialDriftDetection:
    """Test sequential drift detection across batches."""

    def test_stable_sequence(self) -> None:
        """Test with stable score sequence."""
        np.random.seed(42)

        # Create stable batches with small variations
        batches = []
        base_scores = np.random.uniform(0, 1, 50)
        for i in range(5):
            batch = base_scores + np.random.normal(0, 0.01, 50)
            batches.append(batch)

        result = sequential_drift_detection(batches)

        assert len(result["tau_sequence"]) == 4  # n-1 comparisons
        assert all(tau > 0.9 for tau in result["tau_sequence"])
        assert len(result["drift_points"]) == 0
        assert result["overall_stability"] > 0.9

    def test_drift_at_midpoint(self) -> None:
        """Test with drift occurring at midpoint."""
        np.random.seed(42)

        batches = []
        # First half: stable
        base_scores = np.random.uniform(0, 1, 50)
        for i in range(3):
            batches.append(base_scores + np.random.normal(0, 0.01, 50))

        # Second half: completely different distribution to ensure drift detection
        new_scores = np.random.uniform(0, 1, 50) * 0.5 + 0.5  # Different range
        for i in range(2):
            batches.append(new_scores + np.random.normal(0, 0.01, 50))

        result = sequential_drift_detection(batches)

        # Check that drift is detected somewhere in the sequence
        assert result["overall_stability"] < 0.9, "Should have reduced stability"
        # The tau values should show a drop
        assert min(result["tau_sequence"]) < 0.8

    def test_insufficient_data(self) -> None:
        """Test with insufficient data."""
        result = sequential_drift_detection([np.array([1, 2, 3])])

        assert result["insufficient_data"]
        assert len(result["tau_sequence"]) == 0


class TestReliabilityDiagram:
    """Test reliability diagram and Brier score decomposition."""

    def test_perfect_calibration(self) -> None:
        """Test with perfectly calibrated predictions."""
        np.random.seed(42)
        n = 1000

        # Generate perfectly calibrated predictions
        predicted_probs = np.random.uniform(0, 1, n)
        true_binary = (np.random.uniform(0, 1, n) < predicted_probs).astype(int)

        result = reliability_diagram(predicted_probs, true_binary)

        assert result["ece"] < 0.05, "ECE should be low for calibrated predictions"
        assert result["mce"] < 0.1, "MCE should be low"
        assert result["is_calibrated"]

        # Check Brier score decomposition
        assert "brier_score" in result
        assert "brier_reliability" in result
        assert "brier_resolution" in result
        assert "brier_uncertainty" in result

    def test_overconfident_predictions(self) -> None:
        """Test with overconfident predictions."""
        np.random.seed(42)
        n = 1000

        # Overconfident: high probabilities but only 50% true
        predicted_probs = np.random.uniform(0.7, 0.95, n)
        true_binary = np.random.binomial(1, 0.5, n)

        result = reliability_diagram(predicted_probs, true_binary)

        assert result["ece"] > 0.1, "ECE should be high for miscalibrated"
        assert not result["is_calibrated"]
        assert result["brier_reliability"] > 0.05

    def test_underconfident_predictions(self) -> None:
        """Test with underconfident predictions."""
        np.random.seed(42)
        n = 1000

        # Underconfident: predictions around 0.5 but extreme outcomes
        predicted_probs = np.random.uniform(0.4, 0.6, n)
        true_binary = (np.random.uniform(0, 1, n) > 0.8).astype(int)

        result = reliability_diagram(predicted_probs, true_binary)

        assert not result["is_calibrated"]
        assert result["brier_resolution"] < 0.01  # Low resolution


class TestEIFQQPlot:
    """Test EIF Q-Q plot data generation."""

    def test_normal_data(self) -> None:
        """Test with normally distributed influence functions."""
        np.random.seed(42)
        ifs = np.random.normal(0, 1, 500)

        result = eif_qq_plot_data(ifs)

        assert result["is_normal"], "Normal data should pass normality test"
        assert result["shapiro_p"] > 0.05
        assert abs(result["skewness"]) < 0.5
        assert abs(result["kurtosis"]) < 0.5
        assert result["n_outliers"] < 5

    def test_heavy_tailed_data(self) -> None:
        """Test with heavy-tailed distribution."""
        np.random.seed(42)
        # t-distribution with df=3 (heavy tails)
        ifs = np.random.standard_t(df=3, size=500)

        result = eif_qq_plot_data(ifs)

        assert not result["is_normal"], "Heavy-tailed should fail normality"
        assert result["shapiro_p"] < 0.05
        assert abs(result["kurtosis"]) > 1  # Excess kurtosis

    def test_skewed_data(self) -> None:
        """Test with skewed distribution."""
        np.random.seed(42)
        # Exponential distribution (right-skewed)
        ifs = np.random.exponential(1, 500)

        result = eif_qq_plot_data(ifs, standardize=True)

        assert not result["is_normal"]
        assert result["skewness"] > 1  # Right-skewed
        assert len(result["theoretical_quantiles"]) == len(result["sample_quantiles"])

    def test_outlier_detection(self) -> None:
        """Test outlier detection in influence functions."""
        np.random.seed(42)
        ifs = np.random.normal(0, 1, 500)
        # Add outliers
        ifs[10] = 10
        ifs[20] = -8
        ifs[30] = 7

        result = eif_qq_plot_data(ifs)

        assert result["n_outliers"] >= 3
        assert 10 in result["outlier_indices"]
        assert 20 in result["outlier_indices"]


class TestStabilityDiagnosticsIntegration:
    """Test integrated stability diagnostics computation."""

    def test_compute_stability_diagnostics(self) -> None:
        """Test comprehensive stability diagnostics."""
        from cje.data import Dataset, Sample

        # Create mock dataset
        samples = []
        np.random.seed(42)

        # Create samples with gradual drift
        for i in range(200):
            drift = i / 200  # Gradual drift
            score = np.random.uniform(0, 1) + drift * 0.5
            oracle = score + np.random.normal(0, 0.1)

            sample = Sample(
                prompt_id=str(i),
                prompt=f"prompt_{i}",
                response=f"response_{i}",
                target_policy_logprobs={
                    "test_policy": -1.0
                },  # Required field with dummy value
                metadata={
                    "judge_score": float(score),
                    "oracle_label": float(oracle),
                },
            )
            samples.append(sample)

        dataset = Dataset(samples=samples, target_policies=["test_policy"])

        result = compute_stability_diagnostics(
            dataset,
            batch_size=50,
            judge_field="judge_score",
            oracle_field="oracle_label",
        )

        assert "n_samples" in result
        assert "drift_detection" in result
        assert "overall_tau_with_oracle" in result
        assert result["n_batches"] == 4

        # Should detect some drift due to gradual change
        drift = result["drift_detection"]
        assert isinstance(drift, dict), "drift_detection should be a dict"
        assert "tau_sequence" in drift
        tau_seq = drift["tau_sequence"]
        assert isinstance(tau_seq, list)
        assert len(tau_seq) == 3  # n_batches - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
