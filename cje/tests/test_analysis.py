"""Tests for the high-level analysis API."""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock, call
import numpy as np

from cje.analysis import analyze_dataset


class TestAnalyzeDataset:
    """Test the analyze_dataset high-level function."""

    def setup_method(self) -> None:
        """Use real test data from arena sample."""
        self.test_data_path = (
            Path(__file__).parent / "data" / "arena_sample" / "dataset.jsonl"
        )
        self.fresh_draws_dir = (
            Path(__file__).parent / "data" / "arena_sample" / "responses"
        )

        # For tests that need custom data
        self.mock_data = [
            {
                "prompt_id": f"id_{i}",
                "prompt": f"Question {i}",
                "response": f"Answer {i}",
                "base_policy_logprob": -10.0 - i,
                "target_policy_logprobs": {"policy_a": -8.0 - i, "policy_b": -12.0 - i},
                "metadata": {
                    "judge_score": 5.0 + (i % 5),
                    "oracle_label": 0.3 + 0.05 * i if i < 15 else None,
                },
            }
            for i in range(20)
        ]

    def create_test_file(self, data: Optional[list] = None) -> str:
        """Helper to create a test JSONL file."""
        if data is None:
            data = self.mock_data

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
            return f.name

    def test_analyze_basic_ips(self) -> None:
        """Test basic IPS analysis with real test data."""
        # Use real arena sample data
        result = analyze_dataset(
            str(self.test_data_path), estimator="calibrated-ips", oracle_coverage=1.0
        )

        # Check result structure
        assert result.method == "calibrated_ips"
        assert len(result.estimates) == 4  # Four policies in arena data
        assert len(result.standard_errors) == 4

        # Check policies are correct
        policies = result.metadata["target_policies"]
        assert "clone" in policies
        assert "premium" in policies
        assert "parallel_universe_prompt" in policies
        assert "unhelpful" in policies

        # Check metadata
        assert result.metadata["estimator"] == "calibrated-ips"
        assert result.metadata["dataset_path"] == str(self.test_data_path)

    def test_analyze_with_calibration(self) -> None:
        """Test analysis with judge calibration using real data."""
        result = analyze_dataset(
            str(self.test_data_path),
            estimator="calibrated-ips",
            oracle_coverage=0.5,  # Use 50% of oracle labels
            judge_field="judge_score",
            oracle_field="oracle_label",
        )

        # Check calibration happened
        assert result.method == "calibrated_ips"
        assert result.metadata["oracle_coverage"] == 0.5

        # Check we have results for all policies
        assert len(result.estimates) == 4

        # Check diagnostics exist
        result_dict = result.to_dict()
        assert "diagnostics" in result_dict
        assert "clone" in result_dict["diagnostics"]

    def test_analyze_dr_with_fresh_draws(self) -> None:
        """Test DR analysis with real fresh draws."""
        result = analyze_dataset(
            str(self.test_data_path),
            estimator="dr-cpo",
            oracle_coverage=1.0,
            fresh_draws_dir=str(self.fresh_draws_dir),
        )

        # Check DR was used
        assert result.method in ["dr_cpo", "dr", "doubly_robust"]

        # Check we got results for all policies
        assert len(result.estimates) == 4

        # DR should have diagnostics
        assert "dr_diagnostics" in result.metadata or "diagnostics" in result.to_dict()

    @patch("cje.analysis.load_fresh_draws_auto")
    def test_analyze_dr_fallback_synthetic(self, mock_load_fresh: MagicMock) -> None:
        """Test DR falls back to synthetic when fresh draws missing."""
        # Mock returns synthetic dataset
        mock_synthetic = MagicMock()
        mock_synthetic.n_samples = 20
        mock_load_fresh.return_value = mock_synthetic

        test_file = self.create_test_file()

        try:
            result = analyze_dataset(
                test_file,
                estimator="dr-cpo",
                oracle_coverage=1.0,
                # No fresh_draws_dir specified
            )

            # Should still work with synthetic
            assert result.method in ["dr_cpo", "dr", "doubly_robust"]

            # Check auto-loading was attempted
            mock_load_fresh.assert_called()

        finally:
            Path(test_file).unlink()

    def test_analyze_with_estimator_config(self) -> None:
        """Test passing estimator configuration."""
        test_file = self.create_test_file()

        try:
            result = analyze_dataset(
                test_file,
                estimator="calibrated-ips",
                estimator_config={"clip_weight": 10.0, "max_variance_ratio": 2.0},
            )

            # Config should be in metadata
            assert result.metadata.get("estimator_config") == {
                "clip_weight": 10.0,
                "max_variance_ratio": 2.0,
            }

        finally:
            Path(test_file).unlink()

    def test_analyze_invalid_estimator(self) -> None:
        """Test error handling for invalid estimator."""
        test_file = self.create_test_file()

        try:
            with pytest.raises(ValueError, match="Unknown estimator"):
                analyze_dataset(test_file, estimator="invalid-estimator")
        finally:
            Path(test_file).unlink()

    def test_analyze_missing_file(self) -> None:
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            analyze_dataset("nonexistent.jsonl", estimator="calibrated-ips")

    def test_analyze_insufficient_oracle_labels(self) -> None:
        """Test error when too few oracle labels for calibration."""
        # Create data with only 5 oracle labels (below minimum)
        limited_data = [
            {
                "prompt_id": f"id_{i}",
                "prompt": f"Question {i}",
                "response": f"Answer {i}",
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {"policy_a": -8.0},
                "metadata": {
                    "judge_score": 7.0,
                    "oracle_label": 0.8 if i < 5 else None,
                },
            }
            for i in range(20)
        ]

        test_file = self.create_test_file(limited_data)

        try:
            with pytest.raises(ValueError, match="Insufficient oracle"):
                analyze_dataset(
                    test_file, estimator="calibrated-ips", oracle_coverage=1.0
                )
        finally:
            Path(test_file).unlink()

    def test_analyze_mrdr_estimator(self) -> None:
        """Test MRDR estimator analysis."""
        # Add rewards for MRDR
        data_with_rewards = [
            {**record, "reward": 0.5 + 0.02 * i}
            for i, record in enumerate(self.mock_data)
        ]

        test_file = self.create_test_file(data_with_rewards)

        try:
            with patch("cje.analysis.load_fresh_draws_auto") as mock_fresh:
                # Setup minimal fresh draws
                mock_fresh.return_value = MagicMock(n_samples=20)

                result = analyze_dataset(
                    test_file, estimator="mrdr", estimator_config={"omega_mode": "w2"}
                )

                assert result.method in ["mrdr", "multiply_robust"]

        finally:
            Path(test_file).unlink()

    def test_analyze_tmle_estimator(self) -> None:
        """Test TMLE estimator analysis."""
        # Add rewards for TMLE
        data_with_rewards = [
            {**record, "reward": 0.5 + 0.02 * i}
            for i, record in enumerate(self.mock_data)
        ]

        test_file = self.create_test_file(data_with_rewards)

        try:
            with patch("cje.analysis.load_fresh_draws_auto") as mock_fresh:
                # Setup minimal fresh draws
                mock_fresh.return_value = MagicMock(n_samples=20)

                result = analyze_dataset(
                    test_file, estimator="tmle", estimator_config={"max_iterations": 5}
                )

                assert result.method in ["tmle", "targeted_maximum_likelihood"]

        finally:
            Path(test_file).unlink()

    def test_analyze_preserves_metadata(self) -> None:
        """Test that analyze preserves important metadata."""
        test_file = self.create_test_file()

        try:
            result = analyze_dataset(
                test_file,
                estimator="calibrated-ips",
                oracle_coverage=0.5,
                judge_field="judge_score",
                oracle_field="oracle_label",
            )

            # Check all expected metadata
            metadata = result.metadata
            assert metadata["dataset_path"] == test_file
            assert metadata["estimator"] == "calibrated-ips"
            assert metadata["oracle_coverage"] == 0.5
            assert metadata["judge_field"] == "judge_score"
            assert metadata["oracle_field"] == "oracle_label"
            assert "target_policies" in metadata
            assert metadata["target_policies"] == ["policy_a", "policy_b"]

        finally:
            Path(test_file).unlink()

    def test_analyze_result_structure(self) -> None:
        """Test the structure of the returned EstimationResult."""
        # Use data with rewards for simplicity
        data_with_rewards = [{**record, "reward": 0.7} for record in self.mock_data]

        test_file = self.create_test_file(data_with_rewards)

        try:
            result = analyze_dataset(test_file, estimator="raw-ips")

            # Check EstimationResult attributes
            assert hasattr(result, "estimates")
            assert hasattr(result, "standard_errors")
            assert hasattr(result, "method")
            assert hasattr(result, "n_samples_used")
            assert hasattr(result, "metadata")

            # Check array types
            assert isinstance(result.estimates, np.ndarray)
            assert isinstance(result.standard_errors, np.ndarray)

            # Check dict conversion
            result_dict = result.to_dict()
            assert "estimates" in result_dict
            assert "standard_errors" in result_dict
            assert "confidence_intervals" in result_dict
            assert "per_policy_results" in result_dict

        finally:
            Path(test_file).unlink()
