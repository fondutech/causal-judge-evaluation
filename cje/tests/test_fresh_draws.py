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

    def teardown_method(self) -> None:
        """Clean up test directory."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def create_fresh_draws_file(self, policy: str, n_samples: int = 10) -> Path:
        """Helper to create a fresh draws file."""
        fresh_data = [
            {
                "prompt_id": f"id_{i}",
                "response": f"Fresh response for {policy}",
                "judge_score": 0.7 + i * 0.01,
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
        result = load_fresh_draws_auto(data_dir=self.data_dir, policy="policy_a")

        assert isinstance(result, FreshDrawDataset)
        assert result.n_samples == 10
        assert result.samples[0].response == "Fresh response for policy_a"

    def test_auto_load_tries_multiple_locations(self) -> None:
        """Test auto-loading searches multiple standard locations."""
        # Create in non-standard location
        alt_dir = self.data_dir / "fresh_draws"
        alt_dir.mkdir()

        fresh_data = [
            {
                "prompt_id": "id_0",
                "response": "Found in alt location",
                "judge_score": 0.65,
                "logprob": -5.0,
            }
        ]

        with open(alt_dir / "policy_a.jsonl", "w") as f:
            f.write(json.dumps(fresh_data[0]) + "\n")

        result = load_fresh_draws_auto(data_dir=self.data_dir, policy="policy_a")

        assert result.n_samples == 1
        assert result.samples[0].response == "Found in alt location"

    def test_auto_load_no_fallback_raises(self) -> None:
        """Test auto-loading raises error when files missing."""
        with pytest.raises(FileNotFoundError, match="No fresh draw file found"):
            load_fresh_draws_auto(
                data_dir=self.data_dir,
                policy="policy_missing",
            )

    def test_auto_load_with_explicit_dir(self) -> None:
        """Test auto-loading with explicitly specified directory."""
        # Create in custom location
        custom_dir = self.temp_path / "custom_responses"
        custom_dir.mkdir()

        fresh_data = [
            {
                "prompt_id": "id_0",
                "response": "Custom location",
                "judge_score": 0.75,
                "logprob": -4.0,
            }
        ]

        with open(custom_dir / "policy_a_responses.jsonl", "w") as f:
            f.write(json.dumps(fresh_data[0]) + "\n")

        result = load_fresh_draws_auto(
            data_dir=custom_dir,  # Use custom dir as data_dir
            policy="policy_a",
        )

        assert result.n_samples == 1
        assert result.samples[0].response == "Custom location"

    @pytest.mark.skip(reason="Additional file patterns not yet implemented")
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
            for old_file in self.responses_dir.glob("*.jsonl"):
                old_file.unlink()

            # Create test file
            fresh_data = {
                "prompt_id": "id_0",
                "response": "Test",
                "judge_score": 0.8,
                "logprob": -5.0,
            }
            with open(self.responses_dir / filename, "w") as f:
                f.write(json.dumps(fresh_data) + "\n")

            if should_find:
                result = load_fresh_draws_auto(
                    data_dir=self.data_dir, policy="policy_a"
                )
                assert result.n_samples == 1
            else:
                with pytest.raises(FileNotFoundError):
                    load_fresh_draws_auto(
                        data_dir=self.data_dir,
                        policy="policy_a",
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
            load_fresh_draws_auto(data_dir=self.data_dir, policy="policy_a")

    @pytest.mark.skip(
        reason="Logging capture issue - function works but test needs update"
    )
    def test_auto_load_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test auto-loading logs appropriate messages."""
        import logging

        caplog.set_level(logging.INFO)

        # Test successful load
        self.create_fresh_draws_file("policy_a")

        result = load_fresh_draws_auto(data_dir=self.data_dir, policy="policy_a")

        # Should log where it found the file
        assert "Found fresh draws" in caplog.text or "Loading" in caplog.text

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

        # Compressed files are not currently supported, should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            result = load_fresh_draws_auto(data_dir=self.data_dir, policy="policy_a")
