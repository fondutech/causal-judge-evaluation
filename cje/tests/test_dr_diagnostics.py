"""Tests for DR diagnostics functionality with new diagnostic system."""

import numpy as np
import pytest
from typing import Dict, Any

from cje.diagnostics import IPSDiagnostics, DRDiagnostics, Status
from cje.data.models import Sample, Dataset, EstimationResult
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample
from cje.estimators import CalibratedIPS
from cje.estimators.dr_base import DRCPOEstimator
from cje.estimators.tmle import TMLEEstimator
from cje.estimators.mrdr import MRDREstimator


class TestIPSDiagnostics:
    """Test IPS diagnostic functionality."""

    def test_ips_diagnostics_creation(self) -> None:
        """Test creation of IPSDiagnostics object."""
        diag = IPSDiagnostics(
            estimator_type="CalibratedIPS",
            method="calibrated_ips",
            n_samples_total=1000,
            n_samples_valid=950,
            n_policies=2,
            policies=["policy_a", "policy_b"],
            estimates={"policy_a": 0.5, "policy_b": 0.6},
            standard_errors={"policy_a": 0.05, "policy_b": 0.04},
            n_samples_used={"policy_a": 950, "policy_b": 950},
            weight_ess=0.85,
            weight_status=Status.GOOD,
            ess_per_policy={"policy_a": 0.85, "policy_b": 0.83},
            max_weight_per_policy={"policy_a": 5.2, "policy_b": 6.1},
            weight_tail_ratio_per_policy={"policy_a": 2.5, "policy_b": 3.0},
            calibration_rmse=0.15,
            calibration_r2=0.75,
            n_oracle_labels=500,
        )

        # Test basic properties
        assert diag.estimator_type == "CalibratedIPS"
        assert abs(diag.filter_rate - 0.05) < 1e-10  # (1000-950)/1000
        assert diag.best_policy == "policy_b"
        assert diag.worst_weight_tail_ratio == 3.0
        assert diag.is_calibrated is True
        assert diag.overall_status == Status.GOOD

        # Test validation
        issues = diag.validate()
        assert len(issues) == 0  # Should have no issues

        # Test serialization
        diag_dict = diag.to_dict()
        assert "estimator_type" in diag_dict
        assert "policies" in diag_dict

    def test_ips_diagnostics_with_issues(self) -> None:
        """Test IPSDiagnostics with problematic values."""
        diag = IPSDiagnostics(
            estimator_type="CalibratedIPS",
            method="raw_ips",
            n_samples_total=100,
            n_samples_valid=10,  # Very high filter rate
            n_policies=1,
            policies=["policy_a"],
            estimates={"policy_a": 0.5},
            standard_errors={"policy_a": 0.5},  # High SE
            n_samples_used={"policy_a": 10},
            weight_ess=0.05,  # Very low ESS
            weight_status=Status.CRITICAL,
            ess_per_policy={"policy_a": 0.05},
            max_weight_per_policy={"policy_a": 500.0},  # Extreme weight
            weight_tail_ratio_per_policy={"policy_a": 200.0},  # Heavy tail
        )

        # Test status detection
        assert diag.overall_status == Status.CRITICAL
        assert diag.filter_rate == 0.9  # 90% filtered

        # Test validation finds issues
        issues = diag.validate()
        assert len(issues) > 0
        assert any("filter rate" in issue.lower() for issue in issues)
        assert any("ess" in issue.lower() for issue in issues)

    def test_ips_diagnostics_summary(self) -> None:
        """Test summary string generation."""
        diag = IPSDiagnostics(
            estimator_type="CalibratedIPS",
            method="calibrated_ips",
            n_samples_total=1000,
            n_samples_valid=950,
            n_policies=2,
            policies=["policy_a", "policy_b"],
            estimates={"policy_a": 0.5, "policy_b": 0.6},
            standard_errors={"policy_a": 0.05, "policy_b": 0.04},
            n_samples_used={"policy_a": 950, "policy_b": 950},
            weight_ess=0.85,
            weight_status=Status.GOOD,
            ess_per_policy={"policy_a": 0.85, "policy_b": 0.83},
            max_weight_per_policy={"policy_a": 5.2, "policy_b": 6.1},
            weight_tail_ratio_per_policy={"policy_a": 2.5, "policy_b": 3.0},
        )

        summary = diag.summary()
        assert "CalibratedIPS" in summary
        assert "950/1000" in summary
        assert "policy_a" in summary
        assert "policy_b" in summary
        # The summary doesn't include individual estimates, just the best policy
        assert "85.0%" in summary  # Weight ESS


class TestDRDiagnostics:
    """Test DR diagnostic functionality."""

    def test_dr_diagnostics_creation(self) -> None:
        """Test creation of DRDiagnostics object."""
        diag = DRDiagnostics(
            # IPS fields
            estimator_type="DR_CalibratedIPS",
            method="dr_cpo",
            n_samples_total=1000,
            n_samples_valid=950,
            n_policies=2,
            policies=["policy_a", "policy_b"],
            estimates={"policy_a": 0.52, "policy_b": 0.61},
            standard_errors={"policy_a": 0.03, "policy_b": 0.025},
            n_samples_used={"policy_a": 950, "policy_b": 950},
            weight_ess=0.85,
            weight_status=Status.GOOD,
            ess_per_policy={"policy_a": 0.85, "policy_b": 0.83},
            max_weight_per_policy={"policy_a": 5.2, "policy_b": 6.1},
            weight_tail_ratio_per_policy={"policy_a": 2.5, "policy_b": 3.0},
            # DR-specific fields
            dr_cross_fitted=True,
            dr_n_folds=5,
            outcome_r2_range=(0.7, 0.85),
            outcome_rmse_mean=0.12,
            worst_if_tail_ratio=15.5,
            dr_diagnostics_per_policy={
                "policy_a": {
                    "dm_mean": 0.5,
                    "ips_corr_mean": 0.02,
                    "score_mean": 0.001,
                    "score_z": 0.5,
                },
                "policy_b": {
                    "dm_mean": 0.6,
                    "ips_corr_mean": 0.01,
                    "score_mean": -0.002,
                    "score_z": -0.8,
                },
            },
        )

        # Test inheritance
        assert isinstance(diag, IPSDiagnostics)
        assert isinstance(diag, DRDiagnostics)

        # Test DR-specific methods
        assert diag.dr_cross_fitted is True
        assert diag.dr_n_folds == 5
        assert diag.outcome_r2_range == (0.7, 0.85)

        # Test policy diagnostics access
        policy_a_diag = diag.get_policy_diagnostics("policy_a")
        assert policy_a_diag is not None
        assert policy_a_diag["dm_mean"] == 0.5

        # Test influence functions check
        assert diag.has_influence_functions() is False

    def test_dr_diagnostics_with_influence_functions(self) -> None:
        """Test DRDiagnostics with influence functions."""
        n = 100
        influence_funcs = {
            "policy_a": np.random.normal(0, 0.1, n),
            "policy_b": np.random.normal(0, 0.15, n),
        }

        diag = DRDiagnostics(
            estimator_type="DR_CalibratedIPS",
            method="tmle",
            n_samples_total=n,
            n_samples_valid=n,
            n_policies=2,
            policies=["policy_a", "policy_b"],
            estimates={"policy_a": 0.52, "policy_b": 0.61},
            standard_errors={"policy_a": 0.03, "policy_b": 0.025},
            n_samples_used={"policy_a": n, "policy_b": n},
            weight_ess=0.85,
            weight_status=Status.GOOD,
            ess_per_policy={"policy_a": 0.85, "policy_b": 0.83},
            max_weight_per_policy={"policy_a": 5.2, "policy_b": 6.1},
            weight_tail_ratio_per_policy={"policy_a": 2.5, "policy_b": 3.0},
            dr_cross_fitted=True,
            dr_n_folds=5,
            outcome_r2_range=(0.7, 0.85),
            outcome_rmse_mean=0.12,
            worst_if_tail_ratio=15.5,
            dr_diagnostics_per_policy={},
            influence_functions=influence_funcs,
        )

        assert diag.has_influence_functions() is True
        assert len(diag.influence_functions) == 2
        assert "policy_a" in diag.influence_functions
        assert len(diag.influence_functions["policy_a"]) == n

    def test_dr_diagnostics_validation(self) -> None:
        """Test DRDiagnostics validation."""
        diag = DRDiagnostics(
            estimator_type="DR_CalibratedIPS",
            method="dr_cpo",
            n_samples_total=100,
            n_samples_valid=100,
            n_policies=1,
            policies=["policy_a"],
            estimates={"policy_a": 0.5},
            standard_errors={"policy_a": 0.05},
            n_samples_used={"policy_a": 100},
            weight_ess=0.03,  # Very low ESS
            weight_status=Status.CRITICAL,
            ess_per_policy={"policy_a": 0.03},
            max_weight_per_policy={"policy_a": 1000.0},
            weight_tail_ratio_per_policy={"policy_a": 500.0},
            dr_cross_fitted=True,
            dr_n_folds=5,
            outcome_r2_range=(0.1, 0.2),  # Poor R²
            outcome_rmse_mean=0.5,
            worst_if_tail_ratio=200.0,  # Very heavy tail
            dr_diagnostics_per_policy={},
        )

        # Should detect issues
        issues = diag.validate()
        assert len(issues) > 0
        assert any("r²" in issue.lower() or "r2" in issue.lower() for issue in issues)
        assert any("tail" in issue.lower() for issue in issues)

        # Overall status should be critical
        assert diag.overall_status == Status.CRITICAL


class TestEstimatorDiagnosticIntegration:
    """Test that estimators produce correct diagnostic objects."""

    @pytest.fixture
    def sample_dataset(self) -> Dataset:
        """Create a sample dataset for testing."""
        np.random.seed(42)
        samples = []
        for i in range(100):
            sample = Sample(
                prompt_id=f"p{i}",
                prompt=f"prompt {i}",
                response=f"response {i}",
                base_policy_logprob=-10.0 + np.random.normal(0, 2),
                target_policy_logprobs={
                    "policy_a": -9.0 + np.random.normal(0, 2),
                    "policy_b": -11.0 + np.random.normal(0, 2),
                },
                reward=np.random.uniform(0, 1),
                metadata={"judge_score": np.random.uniform(0, 1)},
            )
            samples.append(sample)

        return Dataset(samples=samples, target_policies=["policy_a", "policy_b"])

    @pytest.fixture
    def fresh_draws(self) -> Dict[str, FreshDrawDataset]:
        """Create fresh draws for DR estimators."""
        fresh_draws = {}
        for policy in ["policy_a", "policy_b"]:
            fresh_samples = []
            for i in range(100):
                for j in range(5):  # 5 draws per prompt
                    fresh_sample = FreshDrawSample(
                        prompt_id=f"p{i}",
                        prompt=f"prompt {i}",
                        response=f"fresh {j}",
                        judge_score=np.random.uniform(0, 1),
                        target_policy=policy,
                        draw_idx=j,
                        fold_id=i % 5,
                    )
                    fresh_samples.append(fresh_sample)

            fresh_draws[policy] = FreshDrawDataset(
                samples=fresh_samples, target_policy=policy, draws_per_prompt=5
            )
        return fresh_draws

    def test_raw_ips_produces_diagnostics(self, sample_dataset: Dataset) -> None:
        """Test that CalibratedIPS with calibrate=False produces IPSDiagnostics."""
        sampler = PrecomputedSampler(sample_dataset)
        estimator = CalibratedIPS(sampler, calibrate=False)  # Raw mode
        estimator.fit()
        result = estimator.estimate()

        assert result.diagnostics is not None
        assert isinstance(result.diagnostics, IPSDiagnostics)
        assert not isinstance(result.diagnostics, DRDiagnostics)
        assert result.diagnostics.estimator_type == "CalibratedIPS"
        assert result.diagnostics.method == "raw_ips"
        assert result.diagnostics.calibration_rmse is None  # No calibration

    def test_calibrated_ips_produces_diagnostics(self, sample_dataset: Dataset) -> None:
        """Test that CalibratedIPS produces IPSDiagnostics with calibration info."""
        # Add calibration info to dataset
        sample_dataset.metadata["calibration_info"] = {
            "rmse": 0.15,
            "r2": 0.75,
            "n_oracle": 50,
        }

        sampler = PrecomputedSampler(sample_dataset)
        estimator = CalibratedIPS(sampler)
        estimator.fit()
        result = estimator.estimate()

        assert result.diagnostics is not None
        assert isinstance(result.diagnostics, IPSDiagnostics)
        assert not isinstance(result.diagnostics, DRDiagnostics)
        assert result.diagnostics.estimator_type == "CalibratedIPS"
        assert result.diagnostics.method == "calibrated_ips"
        assert result.diagnostics.calibration_rmse == 0.15
        assert result.diagnostics.calibration_r2 == 0.75

    def test_dr_cpo_produces_diagnostics(
        self, sample_dataset: Dataset, fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test that DRCPOEstimator produces DRDiagnostics."""
        sampler = PrecomputedSampler(sample_dataset)
        estimator = DRCPOEstimator(sampler, n_folds=5)
        estimator.fit()

        # Add fresh draws
        for policy, fresh_dataset in fresh_draws.items():
            estimator.add_fresh_draws(policy, fresh_dataset)

        result = estimator.estimate()

        assert result.diagnostics is not None
        assert isinstance(result.diagnostics, DRDiagnostics)
        assert isinstance(result.diagnostics, IPSDiagnostics)  # Also IPS
        assert result.diagnostics.estimator_type.startswith("DR")
        assert result.diagnostics.method == "drcpo"
        assert result.diagnostics.dr_cross_fitted is True
        assert result.diagnostics.dr_n_folds == 5
        assert result.diagnostics.dr_diagnostics_per_policy is not None

    def test_tmle_produces_diagnostics(
        self, sample_dataset: Dataset, fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test that TMLEEstimator produces DRDiagnostics."""
        sampler = PrecomputedSampler(sample_dataset)
        estimator = TMLEEstimator(sampler, n_folds=5)
        estimator.fit()

        # Add fresh draws
        for policy, fresh_dataset in fresh_draws.items():
            estimator.add_fresh_draws(policy, fresh_dataset)

        result = estimator.estimate()

        assert result.diagnostics is not None
        assert isinstance(result.diagnostics, DRDiagnostics)
        assert result.diagnostics.method == "tmle"
        assert result.diagnostics.dr_cross_fitted is True

        # Check TMLE-specific fields in per-policy diagnostics
        for policy in sample_dataset.target_policies:
            policy_diag = result.diagnostics.get_policy_diagnostics(policy)
            if policy_diag:
                assert "score_mean" in policy_diag or "dm_mean" in policy_diag

    def test_mrdr_produces_diagnostics(
        self, sample_dataset: Dataset, fresh_draws: Dict[str, FreshDrawDataset]
    ) -> None:
        """Test that MRDREstimator produces DRDiagnostics."""
        sampler = PrecomputedSampler(sample_dataset)
        estimator = MRDREstimator(sampler, n_folds=5, omega_mode="snips")
        estimator.fit()

        # Add fresh draws
        for policy, fresh_dataset in fresh_draws.items():
            estimator.add_fresh_draws(policy, fresh_dataset)

        result = estimator.estimate()

        assert result.diagnostics is not None
        assert isinstance(result.diagnostics, DRDiagnostics)
        assert result.diagnostics.method == "mrdr"
        assert result.diagnostics.dr_cross_fitted is True

    def test_diagnostic_serialization(self, sample_dataset: Dataset) -> None:
        """Test that diagnostics can be serialized to dict/JSON."""
        sampler = PrecomputedSampler(sample_dataset)
        estimator = CalibratedIPS(sampler)
        estimator.fit()
        result = estimator.estimate()

        # Test to_dict
        diag_dict = result.diagnostics.to_dict()
        assert isinstance(diag_dict, dict)
        assert "estimator_type" in diag_dict
        assert "policies" in diag_dict

        # Test JSON serialization
        import json

        json_str = json.dumps(diag_dict)
        loaded = json.loads(json_str)
        assert loaded["estimator_type"] == diag_dict["estimator_type"]
