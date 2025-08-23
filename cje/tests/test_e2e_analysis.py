"""End-to-end tests for user-facing analysis API and CLI.

Merged from test_analysis.py and test_cli.py to focus on complete workflows
using real arena data.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from cje import load_dataset_from_jsonl
from cje.interface.analysis import analyze_dataset
from cje.interface.cli import create_parser
from cje.utils.export import export_results_json, export_results_csv

# Mark all tests as E2E
pytestmark = [pytest.mark.e2e, pytest.mark.uses_arena_sample]


@pytest.mark.skip(reason="analyze_dataset API has changed - needs refactoring")
class TestE2EAnalysisAPI:
    """Test the high-level analysis API with real data."""

    def test_analyze_dataset_ips(self, arena_sample):
        """Test basic IPS analysis workflow."""
        # Run analysis with default IPS
        results = analyze_dataset(
            arena_sample, estimator="calibrated-ips", oracle_coverage=0.5
        )

        # Validate results structure
        assert results is not None
        assert len(results.estimates) == 4  # 4 policies
        assert len(results.standard_errors) == 4
        assert results.best_policy() in [0, 1, 2, 3]

        # Check confidence intervals
        cis = results.confidence_intervals()
        assert len(cis) == 4
        for ci in cis:
            assert len(ci) == 2  # lower, upper
            assert ci[0] <= ci[1]

    def test_analyze_dataset_dr(self, arena_sample, arena_fresh_draws):
        """Test DR analysis with fresh draws."""
        # Create a temporary directory for fresh draws
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write fresh draws to files
            fresh_dir = Path(tmpdir) / "fresh_draws"
            fresh_dir.mkdir()

            for policy, fresh_dataset in arena_fresh_draws.items():
                filepath = fresh_dir / f"{policy}_fresh.jsonl"
                with open(filepath, "w") as f:
                    for sample in fresh_dataset.samples:
                        json.dump(
                            {
                                "prompt_id": sample.prompt_id,
                                "judge_score": sample.judge_score,
                                "draw_idx": sample.draw_idx,
                            },
                            f,
                        )
                        f.write("\n")

            # Run DR analysis
            results = analyze_dataset(
                arena_sample,
                estimator="dr-cpo",
                oracle_coverage=0.5,
                fresh_draws_dir=str(fresh_dir),
            )

            # DR should work even without loading fresh draws from files
            # (auto-loading handles it)
            assert results is not None
            assert results.method == "dr_cpo"

    def test_analyze_with_export(self, arena_sample, tmp_path):
        """Test analysis with result export."""
        # Run analysis
        results = analyze_dataset(
            arena_sample, estimator="calibrated-ips", oracle_coverage=0.5
        )

        # Export to JSON
        json_path = tmp_path / "results.json"
        export_results_json(results, str(json_path))
        assert json_path.exists()

        # Verify JSON content
        with open(json_path) as f:
            data = json.load(f)
            assert "estimates" in data
            assert "standard_errors" in data
            assert "method" in data
            assert len(data["estimates"]) == 4

        # Export to CSV
        csv_path = tmp_path / "results.csv"
        export_results_csv(results, str(csv_path))
        assert csv_path.exists()

    def test_oracle_coverage_ablation(self, arena_sample):
        """Test different oracle coverage levels."""
        coverages = [0.1, 0.5, 1.0]
        results_by_coverage = []

        for coverage in coverages:
            results = analyze_dataset(
                arena_sample, estimator="calibrated-ips", oracle_coverage=coverage
            )
            results_by_coverage.append(results)

        # Higher coverage should generally give lower SEs
        # (though not strictly monotonic due to randomness)
        avg_ses = [np.mean(r.standard_errors) for r in results_by_coverage]
        assert avg_ses[2] <= avg_ses[0] * 1.5  # 100% coverage should be better than 10%


@pytest.mark.skip(reason="CLI has subcommands now - needs refactoring")
class TestE2ECLI:
    """Test command-line interface with real data."""

    def test_cli_basic_parsing(self):
        """Test CLI argument parsing."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "--data",
                "test.jsonl",
                "--estimator",
                "calibrated-ips",
                "--oracle-coverage",
                "0.5",
            ]
        )

        assert args.data == "test.jsonl"
        assert args.estimator == "calibrated-ips"
        assert args.oracle_coverage == 0.5

    def test_cli_estimator_choices(self):
        """Test all estimator choices are valid."""
        estimators = [
            "raw-ips",
            "calibrated-ips",
            "dr-cpo",
            "mrdr",
            "tmle",
            "stacked-dr",
        ]

        parser = create_parser()
        for est in estimators:
            args = parser.parse_args(["--data", "test.jsonl", "--estimator", est])
            assert args.estimator == est

    def test_cli_with_config(self):
        """Test CLI with estimator config."""
        parser = create_parser()
        args = parser.parse_args(
            [
                "--data",
                "test.jsonl",
                "--estimator",
                "mrdr",
                "--estimator-config",
                '{"omega_mode": "w2", "n_folds": 10}',
            ]
        )

        config = json.loads(args.estimator_config)
        assert config["omega_mode"] == "w2"
        assert config["n_folds"] == 10


@pytest.mark.skip(reason="analyze_dataset API has changed - needs refactoring")
class TestE2EWorkflows:
    """Test complete analysis workflows."""

    def test_full_comparison_workflow(self, arena_sample):
        """Test comparing multiple estimators."""
        estimators = ["calibrated-ips", "raw-ips"]
        results_dict = {}

        for est in estimators:
            results = analyze_dataset(arena_sample, estimator=est, oracle_coverage=0.5)
            results_dict[est] = results

        # Calibrated should have different (often better) estimates
        cal_ips = results_dict["calibrated-ips"]
        raw_ips = results_dict["raw-ips"]

        # They should give somewhat different results
        estimates_similar = np.allclose(cal_ips.estimates, raw_ips.estimates, rtol=0.01)
        assert not estimates_similar  # Should be different due to calibration

    def test_robustness_workflow(self, arena_sample):
        """Test robustness across different configurations."""
        configs = [{"variance_cap": 1.5}, {"variance_cap": 2.0}, {"variance_cap": 5.0}]

        results = []
        for config in configs:
            result = analyze_dataset(
                arena_sample,
                estimator="calibrated-ips",
                oracle_coverage=0.5,
                estimator_config=config,
            )
            results.append(result)

        # All should identify same or similar best policy
        best_policies = [r.best_policy() for r in results]
        # At least 2 out of 3 should agree
        from collections import Counter

        most_common = Counter(best_policies).most_common(1)[0][1]
        assert most_common >= 2


# Utility function for testing
def validate_results_structure(results):
    """Validate that results have expected structure."""
    assert hasattr(results, "estimates")
    assert hasattr(results, "standard_errors")
    assert hasattr(results, "method")
    assert hasattr(results, "best_policy")
    assert hasattr(results, "confidence_intervals")

    # Check data types
    assert isinstance(results.estimates, (list, np.ndarray))
    assert isinstance(results.standard_errors, (list, np.ndarray))
    assert isinstance(results.method, str)

    # Check callable methods
    assert callable(results.best_policy)
    assert callable(results.confidence_intervals)
