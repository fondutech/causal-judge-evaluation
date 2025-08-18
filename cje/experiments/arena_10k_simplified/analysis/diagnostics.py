"""Diagnostic computation and display for CJE analysis.

This module handles displaying weight diagnostics, DR diagnostics,
and generating diagnostic reports.

Following CLAUDE.md: Do one thing well - this module only handles diagnostics display.
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from cje.utils.extreme_weights_analysis import analyze_extreme_weights
from cje.diagnostics.weights import compute_weight_diagnostics
from cje.diagnostics.display import (
    create_weight_summary_table,
    format_dr_diagnostic_summary,
)
from cje.diagnostics.hera import (
    hera_audit,
    hera_drill_down,
    format_hera_drill_down,
    HERAMetrics,
)


# Diagnostic thresholds (following CLAUDE.md: explicit > magic values)
DIAGNOSTIC_THRESHOLDS = {
    "LOW_ESS": 0.1,
    "CRITICAL_ESS": 0.05,
    "EXTREME_CONCENTRATION": 0.9,
    "NEAR_ZERO_WEIGHT": 1e-10,
    "EXTREME_ESTIMATE_DIFF": 0.3,  # >30% difference from base is suspicious
}


def run_hera_preflight(
    dataset: Any,
    sampler: Any = None,
    target_ci_width: float = 0.03,
    verbose: bool = True,
) -> Dict[str, HERAMetrics]:
    """Run HERA pre-flight audit on all target policies.

    This provides early warning about overlap issues before estimation.

    Args:
        dataset: CJE dataset with samples
        sampler: Optional PrecomputedSampler (will extract policies from dataset if None)
        target_ci_width: Desired CI half-width for auto-tuning (default ±3%)
        verbose: Whether to print results

    Returns:
        Dictionary mapping policy names to HERAMetrics
    """
    if verbose:
        print("\n2. HERA Overlap Audit:")
        print("   " + "=" * 60)
        print(f"   {'Policy':<25} {'H':>8} {'E':>8}  {'Status':<12}")
        print("   " + "-" * 60)

    hera_results = {}
    critical_policies = []
    warning_policies = []

    # Get base policy log probabilities (handle None values)
    base_logprobs = []
    for s in dataset.samples:
        if s.base_policy_logprob is not None:
            base_logprobs.append(s.base_policy_logprob)
        else:
            base_logprobs.append(-np.inf)  # Treat None as impossible
    base_logprobs = np.array(base_logprobs)
    n_samples = len(base_logprobs)

    # Get target policies from dataset if sampler not provided
    if sampler is not None:
        target_policies = sampler.target_policies
    else:
        # Extract policies from dataset
        all_policies = set()
        for sample in dataset.samples:
            if sample.target_policy_logprobs:
                all_policies.update(sample.target_policy_logprobs.keys())
        target_policies = sorted(list(all_policies))

    # Audit each target policy
    for policy in target_policies:
        # Get target policy log probabilities (handle None values)
        target_logprobs = []
        for s in dataset.samples:
            val = s.target_policy_logprobs.get(policy)
            if val is not None:
                target_logprobs.append(val)
            else:
                target_logprobs.append(-np.inf)  # Treat None as impossible
        target_logprobs = np.array(target_logprobs)

        # Compute log-ratios (convert to numpy array for subtraction)
        delta_log = np.array(target_logprobs) - np.array(base_logprobs)

        # Run HERA audit
        hera = hera_audit(
            delta_log,
            n_samples=n_samples,
            target_ci_halfwidth=target_ci_width,
        )
        hera_results[policy] = hera

        # Track problematic policies
        if hera.hera_status == "critical":
            critical_policies.append(policy)
            status_symbol = "⚠️ CRITICAL"
        elif hera.hera_status == "warning":
            warning_policies.append(policy)
            status_symbol = "⚠ WARNING"
        else:
            status_symbol = "✓ OK"

        if verbose:
            print(
                f"   {policy:<25} {hera.hellinger_affinity:>7.1%} "
                f"{hera.ess_raw_fraction:>7.1%}  {status_symbol}"
            )

    if verbose:
        print("   " + "=" * 60)

        # Print recommendations
        if critical_policies:
            print("\n   ⚠️  CRITICAL overlap issues detected:")
            for policy in critical_policies:
                hera = hera_results[policy]
                print(
                    f"      '{policy}': H={hera.hellinger_affinity:.1%}, E={hera.ess_raw_fraction:.1%}"
                )
            print("      → IPS/Cal-IPS will be unreliable for these policies")
            print("      → Strongly recommend DR methods with fresh draws")

        if warning_policies:
            print("\n   ⚠  WARNING - marginal overlap:")
            for policy in warning_policies:
                hera = hera_results[policy]
                print(
                    f"      '{policy}': H={hera.hellinger_affinity:.1%}, E={hera.ess_raw_fraction:.1%}"
                )
            print("      → IPS methods may be less reliable")
            print("      → Consider DR methods for better accuracy")

        if not critical_policies and not warning_policies:
            print("\n   ✓ All policies have adequate overlap for IPS estimation")

    return hera_results


def display_hera_summary(
    hera_results: Dict[str, HERAMetrics],
    estimator_type: str,
) -> None:
    """Display HERA summary with estimator-specific warnings.

    Args:
        hera_results: Dictionary of HERA metrics per policy
        estimator_type: Type of estimator being used
    """
    # Check if any policies have issues
    critical = [p for p, h in hera_results.items() if h.hera_status == "critical"]
    warning = [p for p, h in hera_results.items() if h.hera_status == "warning"]

    if not critical and not warning:
        return  # No issues to report

    # IPS-based methods
    if estimator_type in ["calibrated-ips", "raw-ips"]:
        if critical:
            print("\n   ⚠️  HERA CRITICAL - Using IPS with severe overlap issues:")
            for policy in critical:
                h = hera_results[policy]
                print(
                    f"      {policy}: Results will be unreliable (H={h.hellinger_affinity:.1%})"
                )
            print("      Consider using DR methods instead")

    # DR methods
    elif estimator_type in ["dr-cpo", "mrdr", "tmle", "mrdr-tmle"]:
        if critical:
            print("\n   ⚠️  HERA CRITICAL - Even DR methods may struggle:")
            for policy in critical:
                h = hera_results[policy]
                print(
                    f"      {policy}: Extreme mismatch (H={h.hellinger_affinity:.1%})"
                )
            print("      Ensure you have sufficient fresh draws")


def run_hera_drill_down(
    dataset: Any,
    sampler: Any,  # Not used, kept for compatibility
    policy: str,
    n_bins: int = 10,
    verbose: bool = True,
) -> Optional[Dict]:
    """Run HERA drill-down analysis for a specific policy.

    This shows WHERE the overlap problems occur (e.g., which judge score ranges).

    Args:
        dataset: CJE dataset
        sampler: Not used (kept for compatibility)
        policy: Policy name to analyze
        n_bins: Number of bins for drill-down
        verbose: Whether to print results

    Returns:
        Drill-down results dictionary or None
    """
    # Get log probabilities (handle None values)
    base_logprobs = []
    target_logprobs = []
    for s in dataset.samples:
        # Base logprob
        if s.base_policy_logprob is not None:
            base_logprobs.append(s.base_policy_logprob)
        else:
            base_logprobs.append(-np.inf)

        # Target logprob
        val = s.target_policy_logprobs.get(policy)
        if val is not None:
            target_logprobs.append(val)
        else:
            target_logprobs.append(-np.inf)

    base_logprobs = np.array(base_logprobs)
    target_logprobs = np.array(target_logprobs)
    delta_log = np.array(target_logprobs) - np.array(base_logprobs)

    # Get judge scores as the index
    judge_scores = np.array(
        [s.metadata.get("judge_score", 0.0) for s in dataset.samples]
    )

    # Run drill-down
    drill_down = hera_drill_down(
        delta_log, judge_scores, n_bins=n_bins, index_name="judge_score"
    )

    if verbose:
        print(f"\n   HERA Drill-Down for '{policy}':")
        print(format_hera_drill_down(drill_down))

        # Analyze the pattern
        bins = drill_down["bins"]
        if bins:
            # Check if overlap degrades with judge score
            first_third = bins[: len(bins) // 3]
            last_third = bins[2 * len(bins) // 3 :]

            avg_h_first = np.mean([b["hellinger"] for b in first_third])
            avg_h_last = np.mean([b["hellinger"] for b in last_third])

            if avg_h_first > avg_h_last * 1.5:  # Significant degradation
                print("\n   💡 Insight: Overlap degrades for higher judge scores")
                print(
                    "      This suggests the policies differ more on high-quality responses"
                )
            elif avg_h_last > avg_h_first * 1.5:  # Opposite pattern
                print("\n   💡 Insight: Overlap degrades for lower judge scores")
                print(
                    "      This suggests the policies differ more on low-quality responses"
                )

    return drill_down


def display_weight_diagnostics(
    estimator: Any, sampler: Any, calibrated_dataset: Any, args: Any
) -> Dict[str, Any]:
    """Display weight diagnostics and return diagnostic data.

    Args:
        estimator: Fitted estimator
        sampler: PrecomputedSampler
        calibrated_dataset: Dataset with calibrated rewards
        args: Command-line arguments

    Returns:
        Dictionary of weight diagnostics per policy
    """
    print(f"\n5. Weight diagnostics:")

    # Try to use the estimator's diagnostics object directly if available
    if hasattr(estimator, "get_diagnostics"):
        diagnostics = estimator.get_diagnostics()
        # Check if it's an IPSDiagnostics object (not a plain dict from DR estimators)
        if diagnostics is not None and hasattr(diagnostics, "policies"):
            # Use the IPSDiagnostics object directly for display
            print("\n" + create_weight_summary_table(diagnostics))

            # Still need to return the dictionary format for downstream code
            all_weight_diagnostics = {}

            # Add base policy manually
            base_rewards = [
                s.reward for s in calibrated_dataset.samples if s.reward is not None
            ]
            base_diag = compute_weight_diagnostics(
                np.ones(len(base_rewards)),
                "base",
            )
            all_weight_diagnostics["base"] = base_diag

            # Add target policies from diagnostics
            for policy in diagnostics.policies:
                all_weight_diagnostics[policy] = {
                    "ess_fraction": diagnostics.ess_per_policy.get(policy, 0.0),
                    "max_weight": diagnostics.max_weight_per_policy.get(policy, 1.0),
                    "status": (
                        diagnostics.status_per_policy.get(policy)
                        if diagnostics.status_per_policy
                        else None
                    ),
                    "tail_index": (
                        diagnostics.tail_indices.get(policy)
                        if diagnostics.tail_indices
                        else None
                    ),
                }

            # Print warnings if issues found
            _print_weight_warnings(all_weight_diagnostics, sampler, estimator)

            return all_weight_diagnostics

    # Fallback: compute diagnostics manually
    base_rewards = [
        s.reward for s in calibrated_dataset.samples if s.reward is not None
    ]
    all_weight_diagnostics = {}

    # Base policy (uniform weights)
    base_diag = compute_weight_diagnostics(np.ones(len(base_rewards)), "base")
    all_weight_diagnostics["base"] = base_diag

    # Target policies
    for policy in sampler.target_policies:
        weights = estimator.get_weights(policy)
        if weights is not None:
            diag = compute_weight_diagnostics(weights, policy)
            all_weight_diagnostics[policy] = diag

    # Display table
    _display_weight_table(all_weight_diagnostics)

    # Print warnings
    _print_weight_warnings(all_weight_diagnostics, sampler, estimator)

    return all_weight_diagnostics


def display_dr_diagnostics(results: Any, args: Any) -> None:
    """Display DR diagnostics if available.

    Args:
        results: EstimationResult object
        args: Command-line arguments
    """
    # Check for DRDiagnostics object
    if hasattr(results, "diagnostics") and results.diagnostics is not None:
        from cje.diagnostics import DRDiagnostics

        if isinstance(results.diagnostics, DRDiagnostics):
            print(f"\n6. Doubly Robust diagnostics:")
            # Format the DR diagnostics
            summary = format_dr_diagnostic_summary(results.diagnostics)
            for line in summary.split("\n"):
                print(f"   {line}")

            # Check for issues
            if results.diagnostics.worst_if_tail_ratio > 100:
                print("\n   ⚠️  Warning: Heavy-tailed influence functions detected")
                print(
                    "      Consider using more fresh draws or checking policy overlap"
                )
            return

    # Fallback to legacy format
    if (
        args.estimator in ["dr-cpo", "mrdr", "tmle"]
        and "dr_diagnostics" in results.metadata
    ):
        print(f"\n6. Doubly Robust diagnostics:")
        # Use the dr_diagnostics directly from metadata
        dr_diagnostics = results.metadata["dr_diagnostics"]
        summary = format_dr_diagnostic_summary(dr_diagnostics)

        for line in summary.split("\n"):
            print(f"   {line}")

        # Check for issues
        if isinstance(dr_diagnostics, dict):
            worst_tail = (
                max(
                    d.get("if_tail_ratio_99_5", 0)
                    for d in dr_diagnostics.values()
                    if isinstance(d, dict)
                )
                if dr_diagnostics
                else 0
            )
            if worst_tail > 100:
                print("\n   ⚠️  Warning: Heavy-tailed influence functions detected")
                print(
                    "      Consider using more fresh draws or checking policy overlap"
                )

        if args.estimator == "tmle" and "tmle_max_score_z" in dr_diagnostics:
            if dr_diagnostics["tmle_max_score_z"] > 2:
                print("\n   ⚠️  Warning: TMLE orthogonality not achieved (|z| > 2)")
                print("      Targeting may not have fully converged")


def display_augmentation_diagnostics(
    estimator: Any, results: Any, oracle_coverage: float, args: Any
) -> None:
    """Display oracle augmentation diagnostics if available.

    Args:
        estimator: Fitted estimator (IPS or DR)
        results: EstimationResult object
        oracle_coverage: Fraction of oracle labels used
        args: Command-line arguments
    """
    # Check if augmentation diagnostics are available
    if not hasattr(estimator, "_aug_diagnostics"):
        return

    # Check if augmentation diagnostics are empty
    if not estimator._aug_diagnostics:
        # No augmentation data available
        if oracle_coverage < 1.0:
            print(f"\n7. Oracle Augmentation Impact:")
            print(f"   Oracle coverage used: {oracle_coverage:.1%}")
            print(f"   Note: Oracle augmentation not available for this configuration")
        return

    print(f"\n7. Oracle Augmentation Impact:")
    print(f"   Oracle coverage used: {oracle_coverage:.1%}")
    print(f"   " + "-" * 60)

    # Collect augmentation info for each policy
    for policy in estimator.sampler.target_policies:
        aug_diag = estimator._aug_diagnostics.get(policy, {})
        if not aug_diag:
            continue

        # Get the variance share
        slice_share = aug_diag.get("slice_variance_share", 0)

        # For DR estimators, calculate share of total variance
        if hasattr(estimator, "_influence_functions"):
            if_funcs = estimator._influence_functions.get(policy)
            if if_funcs is not None and len(if_funcs) > 1:
                total_var = np.var(if_funcs, ddof=1)
                aug_var = aug_diag.get("aug_var", 0)
                if total_var > 0:
                    aug_contribution = aug_var / total_var * 100
                    print(
                        f"   {policy}: {aug_contribution:.1f}% of uncertainty from calibration"
                    )
        else:
            # For IPS, use the stored slice_variance_share
            if slice_share > 0:
                print(
                    f"   {policy}: {slice_share:.1f}% of uncertainty from calibration"
                )

    print()
    print("   💡 Interpretation:")
    if oracle_coverage < 0.2:
        print("   - Low oracle coverage is inflating confidence intervals")
        print("   - Consider increasing oracle labels for tighter bounds")
    if oracle_coverage < 0.5:
        print("   - Augmentation accounts for calibration uncertainty")
        print("   - CIs are honest but could be tighter with more labels")


def analyze_extreme_weights_report(
    estimator: Any, sampler: Any, calibrated_dataset: Any, args: Any
) -> None:
    """Generate extreme weights analysis report.

    Args:
        estimator: Fitted estimator
        sampler: PrecomputedSampler
        calibrated_dataset: Dataset with calibrated rewards
        args: Command-line arguments
    """
    step_num = 7 if args.estimator in ["dr-cpo", "mrdr", "tmle"] else 6
    print(f"\n{step_num}. Analyzing extreme weights...")

    analysis_raw_weights = {}
    analysis_cal_weights = {}

    for policy in sampler.target_policies:
        raw_weights = estimator.get_raw_weights(policy)
        if raw_weights is not None:
            analysis_raw_weights[policy] = raw_weights

        cal_weights = estimator.get_weights(policy)
        if cal_weights is not None:
            analysis_cal_weights[policy] = cal_weights

    if analysis_raw_weights:
        try:
            report_dir = None
            if not args.no_plots:
                report_dir = (
                    Path(args.plot_dir)
                    if args.plot_dir
                    else Path(args.data).parent / "plots"
                )
                report_dir.mkdir(parents=True, exist_ok=True)

            json_report, text_report = analyze_extreme_weights(
                dataset=calibrated_dataset,
                sampler=sampler,
                raw_weights_dict=analysis_raw_weights,
                calibrated_weights_dict=analysis_cal_weights,
                n_extreme=5,
                output_dir=report_dir,
                near_zero_threshold=args.extreme_threshold_low,
            )

            # Print summary
            if "per_policy_analysis" in json_report:
                for policy, analysis in json_report["per_policy_analysis"].items():
                    if policy != "base":
                        stats = analysis.get("statistics", {})
                        n_high = stats.get("n_clipped_high", 0)
                        n_zero = stats.get("n_near_zero", 0)
                        print(f"   ✓ {policy}: {n_high} very high, {n_zero} near-zero")

            if report_dir:
                print(
                    f"   ✓ Saved detailed report to {report_dir}/extreme_weights_analysis.txt"
                )

        except Exception as e:
            print(f"   ⚠️  Could not generate extreme weights analysis: {e}")


def _display_weight_table(weight_diagnostics: Dict[str, Any]) -> None:
    """Display weight diagnostics in table format.

    Args:
        weight_diagnostics: Dictionary of diagnostics per policy
    """
    print("   " + "-" * 50)
    print("   Policy              ESS%    Max Weight    Status")
    print("   " + "-" * 50)

    for policy, diag in weight_diagnostics.items():
        ess_pct = diag.get("ess_fraction", 1.0) * 100
        max_weight = diag.get("max_weight", 1.0)
        status = diag.get("status", "OK")

        # Format status
        if isinstance(status, str):
            status_str = status
        elif hasattr(status, "name"):
            status_str = status.name
        else:
            status_str = str(status)

        print(f"   {policy:<18} {ess_pct:>5.1f}%  {max_weight:>10.2f}    {status_str}")

    print("   " + "-" * 50)


def _print_weight_warnings(
    weight_diagnostics: Dict[str, Any], sampler: Any, estimator: Any
) -> None:
    """Print warnings for weight diagnostic issues.

    Args:
        weight_diagnostics: Dictionary of diagnostics per policy
        sampler: PrecomputedSampler
        estimator: Fitted estimator
    """
    # Check for low ESS
    has_issues = any(
        d.get("ess_fraction", 1.0) < DIAGNOSTIC_THRESHOLDS["LOW_ESS"]
        for d in weight_diagnostics.values()
    )

    if has_issues:
        print("\n   ⚠️  Weight diagnostics warnings:")
        for policy, diag in weight_diagnostics.items():
            if diag.get("ess_fraction", 1.0) < DIAGNOSTIC_THRESHOLDS["LOW_ESS"]:
                print(f"   - {policy}: Low ESS ({diag['ess_fraction']:.1%})")

    # Check for extreme concentration
    for policy in sampler.target_policies:
        weights = estimator.get_weights(policy)
        if weights is not None and len(weights) > 0:
            near_zero = np.sum(
                weights < DIAGNOSTIC_THRESHOLDS["NEAR_ZERO_WEIGHT"]
            ) / len(weights)
            if near_zero > DIAGNOSTIC_THRESHOLDS["EXTREME_CONCENTRATION"]:
                print(f"\n   🔴 CRITICAL: {policy} has extreme weight concentration")
                print(f"      {near_zero:.1%} of samples have near-zero weight")
                print(
                    f"      Estimate based on only {len(weights) * (1-near_zero):.0f} effective samples"
                )
                print(
                    f"      Results may be unreliable - consider using DR with more fresh draws"
                )
