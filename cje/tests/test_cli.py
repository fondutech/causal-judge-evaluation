"""Tests for the CJE command-line interface."""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from io import StringIO
import argparse

from cje.interface.cli import create_parser, run_analysis, validate_data, main


class TestCLIParser:
    """Test argument parsing for CLI commands."""

    def test_analyze_command_basic(self) -> None:
        """Test basic analyze command parsing."""
        parser = create_parser()
        args = parser.parse_args(["analyze", "data.jsonl"])

        assert args.command == "analyze"
        assert args.dataset == "data.jsonl"
        assert args.estimator == "calibrated-ips"  # default
        assert args.oracle_coverage == 1.0  # default
        assert args.output is None
        assert args.fresh_draws_dir is None

    def test_analyze_command_full(self) -> None:
        """Test analyze command with all options."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "analyze",
                "data.jsonl",
                "--estimator",
                "dr-cpo",
                "--oracle-coverage",
                "0.5",
                "--output",
                "results.json",
                "--fresh-draws-dir",
                "./responses",
                "--estimator-config",
                '{"n_folds": 10}',
                "--judge-field",
                "custom_judge",
                "--oracle-field",
                "custom_oracle",
                "--verbose",
            ]
        )

        assert args.command == "analyze"
        assert args.dataset == "data.jsonl"
        assert args.estimator == "dr-cpo"
        assert args.oracle_coverage == 0.5
        assert args.output == "results.json"
        assert args.fresh_draws_dir == "./responses"
        assert args.estimator_config == {"n_folds": 10}
        assert args.judge_field == "custom_judge"
        assert args.oracle_field == "custom_oracle"
        assert args.verbose is True
        assert args.quiet is False

    def test_validate_command(self) -> None:
        """Test validate command parsing."""
        parser = create_parser()
        args = parser.parse_args(["validate", "data.jsonl", "-v"])

        assert args.command == "validate"
        assert args.dataset == "data.jsonl"
        assert args.verbose is True

    def test_no_command(self) -> None:
        """Test parser with no command."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_invalid_estimator(self) -> None:
        """Test that invalid estimator is rejected by parser."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            with patch("sys.stderr", new=StringIO()):
                parser.parse_args(["analyze", "data.jsonl", "--estimator", "invalid"])


class TestAnalyzeCommand:
    """Test the analyze command execution."""

    @patch("cje.analysis.analyze_dataset")
    def test_run_analysis_basic(self, mock_analyze: MagicMock) -> None:
        """Test basic analysis run."""
        # Setup mock return value
        import numpy as np

        mock_result = MagicMock()
        mock_result.estimates = np.array([0.7, 0.8, 0.6])
        mock_result.standard_errors = np.array([0.01, 0.02, 0.03])
        mock_result.metadata = {"target_policies": ["policy_a", "policy_b", "policy_c"]}
        mock_analyze.return_value = mock_result

        # Create args
        args = argparse.Namespace(
            dataset="data.jsonl",
            estimator="calibrated-ips",
            oracle_coverage=1.0,
            judge_field="judge_score",
            oracle_field="oracle_label",
            estimator_config=None,
            fresh_draws_dir=None,
            output=None,
            verbose=False,
            quiet=False,
        )

        # Run command
        with patch("sys.stdout", new=StringIO()) as stdout:
            result = run_analysis(args)

        # Check success
        assert result == 0

        # Check analyze_dataset was called correctly
        mock_analyze.assert_called_once_with(
            "data.jsonl",
            estimator="calibrated-ips",
            oracle_coverage=1.0,
            judge_field="judge_score",
            oracle_field="oracle_label",
        )

        # Check output contains results
        output = stdout.getvalue()
        assert "policy_a: 0.700 ± 0.010" in output
        assert "policy_b: 0.800 ± 0.020" in output
        assert "Best policy: policy_b" in output

    @patch("cje.analysis.analyze_dataset")
    @patch("cje.utils.export.export_results_json")
    def test_run_analysis_with_output(
        self, mock_export: MagicMock, mock_analyze: MagicMock
    ) -> None:
        """Test analysis with output file."""
        # Setup mock
        import numpy as np

        mock_result = MagicMock()
        mock_result.estimates = np.array([0.7])
        mock_result.standard_errors = np.array([0.01])
        mock_result.metadata = {"target_policies": ["policy_a"]}
        mock_analyze.return_value = mock_result

        # Create args with output
        args = argparse.Namespace(
            dataset="data.jsonl",
            estimator="calibrated-ips",
            oracle_coverage=1.0,
            judge_field="judge_score",
            oracle_field="oracle_label",
            estimator_config=None,
            fresh_draws_dir=None,
            output="results.json",
            verbose=False,
            quiet=False,
        )

        # Run command
        with patch("sys.stdout", new=StringIO()):
            result = run_analysis(args)

        assert result == 0
        mock_export.assert_called_once_with(mock_result, "results.json")

    @patch("cje.analysis.analyze_dataset")
    def test_run_analysis_quiet_mode(self, mock_analyze: MagicMock) -> None:
        """Test analysis in quiet mode."""
        import numpy as np

        mock_result = MagicMock()
        mock_result.estimates = np.array([0.7])
        mock_result.standard_errors = np.array([0.01])
        mock_result.metadata = {"target_policies": ["policy_a"]}
        mock_analyze.return_value = mock_result

        args = argparse.Namespace(
            dataset="data.jsonl",
            estimator="calibrated-ips",
            oracle_coverage=1.0,
            judge_field="judge_score",
            oracle_field="oracle_label",
            estimator_config=None,
            fresh_draws_dir=None,
            output=None,
            verbose=False,
            quiet=True,
        )

        with patch("sys.stdout", new=StringIO()) as stdout:
            result = run_analysis(args)

        assert result == 0
        output = stdout.getvalue()
        # In quiet mode, should have minimal output
        assert "Running CJE analysis" not in output
        assert "Results:" not in output

    @patch("cje.analysis.analyze_dataset")
    def test_run_analysis_file_not_found(self, mock_analyze: MagicMock) -> None:
        """Test handling of missing dataset file."""
        mock_analyze.side_effect = FileNotFoundError("data.jsonl not found")

        args = argparse.Namespace(
            dataset="data.jsonl",
            estimator="calibrated-ips",
            oracle_coverage=1.0,
            judge_field="judge_score",
            oracle_field="oracle_label",
            estimator_config=None,
            fresh_draws_dir=None,
            output=None,
            verbose=False,
            quiet=False,
        )

        with patch("sys.stderr", new=StringIO()) as stderr:
            result = run_analysis(args)

        assert result == 1
        assert "Dataset file not found" in stderr.getvalue()

    @patch("cje.analysis.analyze_dataset")
    def test_run_analysis_value_error(self, mock_analyze: MagicMock) -> None:
        """Test handling of value errors."""
        mock_analyze.side_effect = ValueError("Invalid estimator config")

        args = argparse.Namespace(
            dataset="data.jsonl",
            estimator="calibrated-ips",
            oracle_coverage=1.0,
            judge_field="judge_score",
            oracle_field="oracle_label",
            estimator_config={"invalid": "config"},
            fresh_draws_dir=None,
            output=None,
            verbose=False,
            quiet=False,
        )

        with patch("sys.stderr", new=StringIO()) as stderr:
            result = run_analysis(args)

        assert result == 1
        assert "Invalid estimator config" in stderr.getvalue()


class TestValidateCommand:
    """Test the validate command execution."""

    @patch("cje.load_dataset_from_jsonl")
    @patch("cje.data.validation.validate_cje_data")
    def test_validate_valid_data(
        self, mock_validate: MagicMock, mock_load: MagicMock
    ) -> None:
        """Test validation of valid data."""
        # Setup mocks
        mock_dataset = MagicMock()
        mock_dataset.n_samples = 100
        mock_dataset.target_policies = ["policy_a", "policy_b"]
        mock_dataset.samples = [
            MagicMock(reward=0.8, metadata={"judge_score": 7}),
            MagicMock(reward=0.7, metadata={"judge_score": 6}),
        ]
        mock_load.return_value = mock_dataset
        mock_validate.return_value = (True, [])

        args = argparse.Namespace(dataset="data.jsonl", verbose=False)

        with patch("sys.stdout", new=StringIO()) as stdout:
            result = validate_data(args)

        assert result == 0
        output = stdout.getvalue()
        assert "✓ Loaded 100 samples" in output
        assert "✓ Target policies: policy_a, policy_b" in output
        assert "✓ Dataset is valid and ready for analysis" in output

    @patch("cje.load_dataset_from_jsonl")
    @patch("cje.data.validation.validate_cje_data")
    def test_validate_invalid_data(
        self, mock_validate: MagicMock, mock_load: MagicMock
    ) -> None:
        """Test validation of invalid data."""
        mock_dataset = MagicMock()
        mock_dataset.n_samples = 100
        mock_dataset.target_policies = ["policy_a"]
        mock_dataset.samples = []
        mock_load.return_value = mock_dataset
        mock_validate.return_value = (
            False,
            ["Missing required field: prompt_id", "No oracle labels found"],
        )

        args = argparse.Namespace(dataset="data.jsonl", verbose=False)

        with patch("sys.stdout", new=StringIO()) as stdout:
            result = validate_data(args)

        assert result == 1
        output = stdout.getvalue()
        assert "⚠️  Issues found:" in output
        assert "Missing required field: prompt_id" in output
        assert "No oracle labels found" in output

    @patch("cje.load_dataset_from_jsonl")
    @patch("cje.data.validation.validate_cje_data")
    def test_validate_verbose_mode(
        self, mock_validate: MagicMock, mock_load: MagicMock
    ) -> None:
        """Test validation in verbose mode."""
        # Setup dataset with judge scores and oracle labels
        mock_dataset = MagicMock()
        mock_dataset.n_samples = 3
        mock_dataset.target_policies = ["policy_a"]
        mock_dataset.samples = [
            MagicMock(
                reward=None,
                metadata={"judge_score": 7, "oracle_label": 0.8},
                base_policy_logprob=-10,
                target_policy_logprobs={"policy_a": -8},
            ),
            MagicMock(
                reward=None,
                metadata={"judge_score": 6, "oracle_label": 0.7},
                base_policy_logprob=-12,
                target_policy_logprobs={"policy_a": -9},
            ),
            MagicMock(
                reward=None,
                metadata={"judge_score": 8, "oracle_label": 0.9},
                base_policy_logprob=-11,
                target_policy_logprobs={"policy_a": -7},
            ),
        ]
        mock_load.return_value = mock_dataset
        mock_validate.return_value = (True, [])

        args = argparse.Namespace(dataset="data.jsonl", verbose=True)

        with patch("sys.stdout", new=StringIO()) as stdout:
            result = validate_data(args)

        assert result == 0
        output = stdout.getvalue()
        # Verbose mode shows detailed stats
        assert "Detailed Statistics:" in output
        assert "Judge scores: 3 samples" in output
        assert "Oracle labels: 3 samples" in output
        assert "Valid samples per policy:" in output

    @patch("cje.load_dataset_from_jsonl")
    def test_validate_file_not_found(self, mock_load: MagicMock) -> None:
        """Test validation with missing file."""
        mock_load.side_effect = FileNotFoundError("data.jsonl")

        args = argparse.Namespace(dataset="data.jsonl", verbose=False)

        with patch("sys.stderr", new=StringIO()) as stderr:
            result = validate_data(args)

        assert result == 1
        assert "Dataset file not found" in stderr.getvalue()


class TestMainEntry:
    """Test the main CLI entry point."""

    def test_main_no_args(self) -> None:
        """Test main with no arguments shows help."""
        with patch("sys.argv", ["cje"]):
            with patch("sys.stdout", new=StringIO()) as stdout:
                result = main()

        assert result == 0
        output = stdout.getvalue()
        assert "usage: cje" in output.lower()
        assert "available commands" in output.lower()

    @patch("cje.cli.run_analysis")
    def test_main_analyze_command(self, mock_run: MagicMock) -> None:
        """Test main routes to analyze command."""
        mock_run.return_value = 0

        with patch("sys.argv", ["cje", "analyze", "data.jsonl"]):
            result = main()

        assert result == 0
        mock_run.assert_called_once()

    @patch("cje.cli.validate_data")
    def test_main_validate_command(self, mock_validate: MagicMock) -> None:
        """Test main routes to validate command."""
        mock_validate.return_value = 0

        with patch("sys.argv", ["cje", "validate", "data.jsonl"]):
            result = main()

        assert result == 0
        mock_validate.assert_called_once()

    def test_main_unknown_command(self) -> None:
        """Test main with unknown command."""
        with patch("sys.argv", ["cje", "unknown", "data.jsonl"]):
            # Parser will exit with error
            with pytest.raises(SystemExit):
                with patch("sys.stderr", new=StringIO()):
                    main()
