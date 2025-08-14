"""Compatibility layer between DiagnosticSuite and legacy diagnostic formats."""

from typing import Dict, List, Optional, Any
import numpy as np

from .diagnostics import IPSDiagnostics, DRDiagnostics, Status
from ..diagnostics.suite import DiagnosticSuite


def create_ips_diagnostics_from_suite(
    suite: DiagnosticSuite,
    n_samples_used: Dict[str, int],
) -> IPSDiagnostics:
    """Create legacy IPSDiagnostics from DiagnosticSuite.

    This allows backward compatibility with code expecting the old format.

    Args:
        suite: The new unified diagnostic suite
        n_samples_used: Number of samples used per policy

    Returns:
        IPSDiagnostics in the legacy format
    """
    # Extract policy names
    policies = list(suite.weight_diagnostics.keys())

    # Convert weight diagnostics to legacy format
    ess_per_policy = {}
    max_weight_per_policy = {}
    tail_indices = {}

    for policy, metrics in suite.weight_diagnostics.items():
        ess_per_policy[policy] = metrics.ess / suite.estimation_summary.n_valid_samples
        max_weight_per_policy[policy] = metrics.max_weight
        if metrics.hill_index is not None:
            tail_indices[policy] = metrics.hill_index

    # Compute overall ESS
    total_ess = sum(metrics.ess for metrics in suite.weight_diagnostics.values())
    n_policies = len(policies)
    avg_ess = total_ess / n_policies if n_policies > 0 else 0
    weight_ess = avg_ess / suite.estimation_summary.n_valid_samples

    # Determine overall status
    min_ess = (
        min(m.ess for m in suite.weight_diagnostics.values())
        if suite.weight_diagnostics
        else 0
    )
    worst_tail = min(
        (
            m.hill_index
            for m in suite.weight_diagnostics.values()
            if m.hill_index is not None
        ),
        default=float("inf"),
    )

    if min_ess < 100 or worst_tail < 1.5:
        weight_status = Status.CRITICAL
    elif min_ess < 500 or worst_tail < 2.0:
        weight_status = Status.WARNING
    else:
        weight_status = Status.GOOD

    # Extract calibration info if available
    calibration_rmse = None
    calibration_r2 = None
    n_oracle_labels = None

    if suite.stability and suite.stability.ece is not None:
        # Use ECE as a proxy for RMSE (not exact but reasonable)
        calibration_rmse = suite.stability.ece

    return IPSDiagnostics(
        estimator_type=suite.estimator_type,
        method=suite.estimator_type.lower().replace("estimator", ""),
        n_samples_total=suite.estimation_summary.n_samples,
        n_samples_valid=suite.estimation_summary.n_valid_samples,
        n_policies=len(policies),
        policies=policies,
        estimates=suite.estimation_summary.estimates,
        standard_errors=suite.estimation_summary.standard_errors or {},
        n_samples_used=n_samples_used,
        weight_ess=weight_ess,
        weight_status=weight_status,
        ess_per_policy=ess_per_policy,
        max_weight_per_policy=max_weight_per_policy,
        tail_indices=tail_indices,  # Use new field
        calibration_rmse=calibration_rmse,
        calibration_r2=calibration_r2,
        n_oracle_labels=n_oracle_labels,
    )


def create_dr_diagnostics_from_suite(
    suite: DiagnosticSuite,
    n_samples_used: Dict[str, int],
    ips_diagnostics: Optional[IPSDiagnostics] = None,
) -> DRDiagnostics:
    """Create legacy DRDiagnostics from DiagnosticSuite.

    Args:
        suite: The new unified diagnostic suite
        n_samples_used: Number of samples used per policy
        ips_diagnostics: Optional IPS diagnostics to include

    Returns:
        DRDiagnostics in the legacy format
    """
    # Get weight diagnostics
    max_weight_overall = (
        max(m.max_weight for m in suite.weight_diagnostics.values())
        if suite.weight_diagnostics
        else 0.0
    )

    # Extract DR-specific metrics
    orthogonality_scores = {}
    dr_diagnostics_per_policy = {}

    if suite.dr_quality:
        orthogonality_scores = {
            policy: {"score": score, "p_value": None}
            for policy, score in suite.dr_quality.orthogonality_scores.items()
        }

        for policy in suite.dr_quality.orthogonality_scores:
            dr_diagnostics_per_policy[policy] = {
                "orthogonality_score": suite.dr_quality.orthogonality_scores.get(
                    policy, 0.0
                ),
                "dm_contribution": suite.dr_quality.dm_contributions.get(policy, 0.0),
                "ips_augmentation": suite.dr_quality.ips_contributions.get(policy, 0.0),
            }

    # Compute aggregates
    orthogonality_mean = (
        np.mean(list(suite.dr_quality.orthogonality_scores.values()))
        if suite.dr_quality and suite.dr_quality.orthogonality_scores
        else 0.0
    )

    orthogonality_max = (
        max(abs(s) for s in suite.dr_quality.orthogonality_scores.values())
        if suite.dr_quality and suite.dr_quality.orthogonality_scores
        else 0.0
    )

    # Extract influence function tail behavior (if available)
    worst_if_tail = 0.0  # Default value since we don't track this in suite

    # DRDiagnostics inherits from IPSDiagnostics, so we need to pass base fields
    # Get weight diagnostics from IPS diagnostics if available
    if ips_diagnostics:
        weight_ess = ips_diagnostics.weight_ess
        weight_status = ips_diagnostics.weight_status
        ess_per_policy = ips_diagnostics.ess_per_policy
        max_weight_per_policy = ips_diagnostics.max_weight_per_policy
        tail_indices = ips_diagnostics.tail_indices
    else:
        # Compute from suite
        weight_ess = 0.0
        weight_status = Status.GOOD
        ess_per_policy = {}
        max_weight_per_policy = {}
        tail_indices = {}
        for policy, metrics in suite.weight_diagnostics.items():
            ess_per_policy[policy] = (
                metrics.ess / suite.estimation_summary.n_valid_samples
            )
            max_weight_per_policy[policy] = metrics.max_weight
            if metrics.hill_index:
                tail_indices[policy] = metrics.hill_index

    return DRDiagnostics(
        # Base IPSDiagnostics fields
        estimator_type="DR",
        method="dr",
        n_samples_total=suite.estimation_summary.n_samples,
        n_samples_valid=suite.estimation_summary.n_valid_samples,
        n_policies=len(suite.weight_diagnostics),
        policies=list(suite.weight_diagnostics.keys()),
        estimates=suite.estimation_summary.estimates,
        standard_errors=suite.estimation_summary.standard_errors or {},
        n_samples_used=n_samples_used,
        weight_ess=weight_ess,
        weight_status=weight_status,
        ess_per_policy=ess_per_policy,
        max_weight_per_policy=max_weight_per_policy,
        tail_indices=tail_indices,
        # DR-specific fields
        dr_cross_fitted=True,
        dr_n_folds=5,
        outcome_r2_range=(0.0, 1.0),  # Default range
        outcome_rmse_mean=0.0,  # Not tracked in suite
        worst_if_tail_ratio=worst_if_tail,
        dr_diagnostics_per_policy=dr_diagnostics_per_policy,
        influence_functions={},  # Will be populated separately if needed
    )
