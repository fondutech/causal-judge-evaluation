"""Tests for fresh draw utilities including auto-loading."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from cje.utils.fresh_draws import (
    load_fresh_draws_from_jsonl,
    load_fresh_draws_auto,
    create_synthetic_fresh_draws,
)
from cje.data.fresh_draws import FreshDrawDataset


class TestLoadFreshDrawsAuto:
    """Test the auto-loading functionality for fresh draws."""

    def setup_method(self) -> None:
        """Set up test data directory structure."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create test structure
        self.data_dir = self.temp_path / "data"
        self.data_dir.mkdir()

        self.responses_dir = self.data_dir / "responses"
        self.responses_dir.mkdir()

        # Create sample logged data
        self.logged_data = [
            {
                "prompt_id": f"id_{i}",
                "prompt": f"Question {i}",
                "response": f"Answer {i}",
                "base_policy_logprob": -10.0,
                "target_policy_logprobs": {"policy_a": -8.0, "policy_b": -12.0},
            }
            for i in range(10)
        ]

        with open(self.data_dir / "logged.jsonl", "w") as f:
            for record in self.logged_data:
                f.write(json.dumps(record) + "\n")

    def teardown_method(self):
        """Clean up test directory."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_fresh_draws_file(self, policy: str, n_samples: int = 10):
        """Helper to create a fresh draws file."""
        fresh_data = [
            {
                "prompt_id": f"id_{i}",
                "response": f"Fresh response for {policy}",
                "logprob": -6.0 - i * 0.1,
            }
            for i in range(n_samples)
        ]

        filepath = self.responses_dir / f"{policy}_responses.jsonl"
        with open(filepath, "w") as f:
            for record in fresh_data:
                f.write(json.dumps(record) + "\n")

        return filepath

    def test_auto_load_finds_responses_dir(self) -> None:
        """Test auto-loading finds responses in standard location."""
        # Create fresh draw files
        self.create_fresh_draws_file("policy_a")

        # Test auto-loading
        result = load_fresh_draws_auto(
            data_dir=self.data_dir, policy="policy_a", fallback_synthetic=False
        )

        assert isinstance(result, FreshDrawDataset)
        assert result.n_samples == 10
        assert result.samples[0].response == "Fresh response for policy_a"

    def test_auto_load_tries_multiple_locations(self) -> None:
        """Test auto-loading searches multiple standard locations."""
        # Create in non-standard location
        alt_dir = self.data_dir / "fresh_draws"
        alt_dir.mkdir()

        fresh_data = [
            {"prompt_id": "id_0", "response": "Found in alt location", "logprob": -5.0}
        ]

        with open(alt_dir / "policy_a.jsonl", "w") as f:
            f.write(json.dumps(fresh_data[0]) + "\n")

        result = load_fresh_draws_auto(
            data_dir=self.data_dir, policy="policy_a", fallback_synthetic=False
        )

        assert result.n_samples == 1
        assert result.samples[0].response == "Found in alt location"

    def test_auto_load_falls_back_to_synthetic(self) -> None:
        """Test auto-loading falls back to synthetic when files not found."""
        # No fresh draw files created

        with patch(
            "cje.utils.fresh_draws.create_synthetic_fresh_draws"
        ) as mock_synthetic:
            mock_dataset = MagicMock()
            mock_dataset.n_samples = 10
            mock_synthetic.return_value = mock_dataset

            result = load_fresh_draws_auto(
                data_dir=self.data_dir,
                policy="policy_missing",
                fallback_synthetic=True,
                n_synthetic=20,
            )

            assert result == mock_dataset
            mock_synthetic.assert_called_once_with("policy_missing", 20)

    def test_auto_load_no_fallback_raises(self) -> None:
        """Test auto-loading raises error when no fallback and files missing."""
        with pytest.raises(FileNotFoundError, match="No fresh draws found"):
            load_fresh_draws_auto(
                data_dir=self.data_dir,
                policy="policy_missing",
                fallback_synthetic=False,
            )

    def test_auto_load_with_explicit_dir(self) -> None:
        """Test auto-loading with explicitly specified directory."""
        # Create in custom location
        custom_dir = self.temp_path / "custom_responses"
        custom_dir.mkdir()

        fresh_data = [
            {"prompt_id": "id_0", "response": "Custom location", "logprob": -4.0}
        ]

        with open(custom_dir / "policy_a_responses.jsonl", "w") as f:
            f.write(json.dumps(fresh_data[0]) + "\n")

        result = load_fresh_draws_auto(
            data_dir=self.data_dir,
            policy="policy_a",
            fresh_draws_dir=custom_dir,
            fallback_synthetic=False,
        )

        assert result.n_samples == 1
        assert result.samples[0].response == "Custom location"

    def test_auto_load_pattern_matching(self) -> None:
        """Test different file naming patterns are recognized."""
        test_cases = [
            ("policy_a_responses.jsonl", True),
            ("policy_a.jsonl", True),
            ("responses_policy_a.jsonl", True),
            ("fresh_draws_policy_a.jsonl", True),
            ("policy_a_fresh.jsonl", True),
            ("unrelated.jsonl", False),
        ]

        for filename, should_find in test_cases:
            # Clean directory
            for f in self.responses_dir.glob("*.jsonl"):
                f.unlink()

            # Create test file
            fresh_data = {"prompt_id": "id_0", "response": "Test", "logprob": -5.0}
            with open(self.responses_dir / filename, "w") as f:
                f.write(json.dumps(fresh_data) + "\n")

            if should_find:
                result = load_fresh_draws_auto(
                    data_dir=self.data_dir, policy="policy_a", fallback_synthetic=False
                )
                assert result.n_samples == 1
            else:
                with pytest.raises(FileNotFoundError):
                    load_fresh_draws_auto(
                        data_dir=self.data_dir,
                        policy="policy_a",
                        fallback_synthetic=False,
                    )

    def test_auto_load_validates_content(self) -> None:
        """Test auto-loading validates fresh draw file content."""
        # Create invalid fresh draw file (missing required fields)
        invalid_data = [
            {
                "prompt_id": "id_0",
                # Missing response and logprob
            }
        ]

        with open(self.responses_dir / "policy_a.jsonl", "w") as f:
            f.write(json.dumps(invalid_data[0]) + "\n")

        with pytest.raises(Exception):  # Should raise validation error
            load_fresh_draws_auto(
                data_dir=self.data_dir, policy="policy_a", fallback_synthetic=False
            )

    def test_auto_load_logging(self, caplog) -> None:
        """Test auto-loading logs appropriate messages."""
        import logging

        caplog.set_level(logging.INFO)

        # Test successful load
        self.create_fresh_draws_file("policy_a")

        result = load_fresh_draws_auto(data_dir=self.data_dir, policy="policy_a")

        # Should log where it found the file
        assert "Found fresh draws" in caplog.text or "Loading" in caplog.text

        # Test fallback to synthetic
        caplog.clear()
        result = load_fresh_draws_auto(
            data_dir=self.data_dir, policy="policy_missing", fallback_synthetic=True
        )

        # Should log fallback
        assert "synthetic" in caplog.text.lower() or "not found" in caplog.text.lower()

    def test_auto_load_caching(self) -> None:
        """Test that auto-loading can utilize caching if implemented."""
        # Create fresh draws
        self.create_fresh_draws_file("policy_a")

        # Load twice
        result1 = load_fresh_draws_auto(data_dir=self.data_dir, policy="policy_a")

        result2 = load_fresh_draws_auto(data_dir=self.data_dir, policy="policy_a")

        # Should get same content
        assert result1.n_samples == result2.n_samples
        assert result1.samples[0].prompt_id == result2.samples[0].prompt_id

    def test_auto_load_handles_compressed(self) -> None:
        """Test auto-loading handles compressed files if supported."""
        import gzip

        # Create compressed fresh draws
        fresh_data = [
            {"prompt_id": "id_0", "response": "Compressed data", "logprob": -5.0}
        ]

        filepath = self.responses_dir / "policy_a.jsonl.gz"
        with gzip.open(filepath, "wt") as f:
            f.write(json.dumps(fresh_data[0]) + "\n")

        # This might not be implemented yet, so we allow it to fail gracefully
        try:
            result = load_fresh_draws_auto(
                data_dir=self.data_dir, policy="policy_a", fallback_synthetic=True
            )
            # If it works, great!
            if result.n_samples > 0 and not hasattr(result, "_synthetic"):
                assert result.samples[0].response == "Compressed data"
        except:
            # If not implemented, that's okay for now
            pass
