"""Tests for export utilities."""

import pytest
import json
import csv
import tempfile
from pathlib import Path
import numpy as np

from cje.utils.export import export_results_json, export_results_csv
from cje.data.models import EstimationResult


class TestExportJSON:
    """Test JSON export functionality."""

    def create_sample_result(self) -> EstimationResult:
        """Create a sample EstimationResult for testing."""
        return EstimationResult(
            estimates=np.array([0.7, 0.8, 0.6]),
            standard_errors=np.array([0.01, 0.02, 0.03]),
            method="calibrated_ips",
            n_samples_used={"policy_a": 100, "policy_b": 98, "policy_c": 95},
            metadata={
                "target_policies": ["policy_a", "policy_b", "policy_c"],
                "dataset_path": "test.jsonl",
                "estimator": "calibrated-ips",
                "oracle_coverage": 0.5,
                "clip_weight": None,
                "diagnostics": {
                    "policy_a": {
                        "weights": {"ess": 85.5, "cv": 0.45, "max": 5.2},
                        "status": "green",
                    },
                    "policy_b": {
                        "weights": {"ess": 45.2, "cv": 1.2, "max": 12.5},
                        "status": "yellow",
                    },
                    "policy_c": {
                        "weights": {"ess": 15.8, "cv": 2.5, "max": 35.0},
                        "status": "red",
                    },
                },
            },
        )

    def test_export_json_basic(self) -> None:
        """Test basic JSON export."""
        result = self.create_sample_result()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            export_results_json(result, output_path)

            # Read and verify JSON
            with open(output_path, "r") as f:
                data = json.load(f)

            # Check structure
            assert "timestamp" in data
            assert data["method"] == "calibrated_ips"
            assert data["estimates"] == [0.7, 0.8, 0.6]
            assert data["standard_errors"] == [0.01, 0.02, 0.03]

            # Check metadata preserved
            assert data["metadata"]["estimator"] == "calibrated-ips"
            assert data["metadata"]["oracle_coverage"] == 0.5

            # Check diagnostics preserved
            assert "diagnostics" in data
            assert data["diagnostics"]["policy_a"]["status"] == "green"
            assert data["diagnostics"]["policy_b"]["weights"]["cv"] == 1.2

            # Check per-policy results
            assert "per_policy_results" in data
            assert data["per_policy_results"]["policy_a"]["estimate"] == 0.7
            assert data["per_policy_results"]["policy_a"]["n_samples"] == 100

        finally:
            Path(output_path).unlink()

    def test_export_json_with_confidence_intervals(self) -> None:
        """Test JSON export includes confidence intervals."""
        result = self.create_sample_result()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            export_results_json(result, output_path)

            with open(output_path, "r") as f:
                data = json.load(f)

            # Check confidence intervals
            assert "confidence_intervals" in data
            assert data["confidence_intervals"]["alpha"] == 0.05
            assert len(data["confidence_intervals"]["lower"]) == 3
            assert len(data["confidence_intervals"]["upper"]) == 3

            # Verify CI calculation (estimate ± 1.96*SE)
            assert (
                abs(data["confidence_intervals"]["lower"][0] - (0.7 - 1.96 * 0.01))
                < 0.001
            )
            assert (
                abs(data["confidence_intervals"]["upper"][0] - (0.7 + 1.96 * 0.01))
                < 0.001
            )

        finally:
            Path(output_path).unlink()

    def test_export_json_handles_none_values(self) -> None:
        """Test JSON export handles None values properly."""
        result = EstimationResult(
            estimates=np.array([0.7]),
            standard_errors=np.array([0.01]),
            method="raw_ips",
            n_samples_used={"policy_a": 100},
            metadata={
                "target_policies": ["policy_a"],
                "clip_weight": None,  # None value
                "fresh_draws_dir": None,  # Another None
            },
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            export_results_json(result, output_path)

            with open(output_path, "r") as f:
                data = json.load(f)

            # None values should be preserved as null in JSON
            assert data["metadata"]["clip_weight"] is None
            assert data["metadata"]["fresh_draws_dir"] is None

        finally:
            Path(output_path).unlink()

    def test_export_json_pretty_format(self) -> None:
        """Test JSON is formatted for readability."""
        result = self.create_sample_result()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            export_results_json(result, output_path)

            with open(output_path, "r") as f:
                content = f.read()

            # Check formatting (indented, multi-line)
            assert "\n" in content
            assert "    " in content  # Indentation

            # Should be valid JSON
            json.loads(content)

        finally:
            Path(output_path).unlink()


class TestExportCSV:
    """Test CSV export functionality."""

    def create_sample_result(self) -> EstimationResult:
        """Create a sample EstimationResult for testing."""
        return EstimationResult(
            estimates=np.array([0.7, 0.8, 0.6]),
            standard_errors=np.array([0.01, 0.02, 0.03]),
            method="calibrated_ips",
            n_samples_used={"policy_a": 100, "policy_b": 98, "policy_c": 95},
            metadata={
                "target_policies": ["policy_a", "policy_b", "policy_c"],
                "dataset_path": "test.jsonl",
                "estimator": "calibrated-ips",
            },
        )

    def test_export_csv_basic(self) -> None:
        """Test basic CSV export."""
        result = self.create_sample_result()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            export_results_csv(result, output_path)

            # Read and verify CSV
            with open(output_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Should have 3 rows (one per policy)
            assert len(rows) == 3

            # Check first row
            row = rows[0]
            assert row["policy"] == "policy_a"
            assert float(row["estimate"]) == 0.7
            assert float(row["standard_error"]) == 0.01
            assert float(row["ci_lower"]) == pytest.approx(0.7 - 1.96 * 0.01, abs=0.001)
            assert float(row["ci_upper"]) == pytest.approx(0.7 + 1.96 * 0.01, abs=0.001)
            assert int(row["n_samples"]) == 100
            assert row["method"] == "calibrated_ips"

            # Check other policies
            assert rows[1]["policy"] == "policy_b"
            assert float(rows[1]["estimate"]) == 0.8
            assert rows[2]["policy"] == "policy_c"
            assert float(rows[2]["estimate"]) == 0.6

        finally:
            Path(output_path).unlink()

    def test_export_csv_missing_diagnostics(self) -> None:
        """Test CSV export handles missing diagnostics gracefully."""
        result = EstimationResult(
            estimates=np.array([0.7]),
            standard_errors=np.array([0.01]),
            method="raw_ips",
            n_samples_used={"policy_a": 100},
            metadata={"target_policies": ["policy_a"]},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            export_results_csv(result, output_path)

            with open(output_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Should still work without diagnostics
            assert len(rows) == 1
            assert rows[0]["policy"] == "policy_a"
            assert float(rows[0]["estimate"]) == 0.7

            # Diagnostic columns should not be present
            assert "ess" not in rows[0]
            assert "cv" not in rows[0]
            assert "status" not in rows[0]

        finally:
            Path(output_path).unlink()

    def test_export_csv_escaping(self) -> None:
        """Test CSV properly escapes special characters."""
        result = EstimationResult(
            estimates=np.array([0.7]),
            standard_errors=np.array([0.01]),
            method="calibrated_ips",
            n_samples_used={"policy,with,commas": 100},
            metadata={
                "target_policies": ["policy,with,commas"],
                "dataset_path": "path/with/quotes\"and'stuff.jsonl",
            },
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            export_results_csv(result, output_path)

            with open(output_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Check that special characters are handled
            assert rows[0]["policy"] == "policy,with,commas"

            # CSV should be valid and parseable
            assert len(rows) == 1

        finally:
            Path(output_path).unlink()

    def test_export_csv_header_order(self) -> None:
        """Test CSV has consistent column order."""
        result = self.create_sample_result()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            export_results_csv(result, output_path)

            with open(output_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)

            # Check expected column order
            expected_start = [
                "policy",
                "estimate",
                "standard_error",
                "ci_lower",
                "ci_upper",
                "n_samples",
                "method",
            ]
            assert header[: len(expected_start)] == expected_start

        finally:
            Path(output_path).unlink()
