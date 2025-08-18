#!/usr/bin/env python3
"""Analysis utilities for ablation experiments.

This module provides functions to aggregate results across random seeds,
handle NaN estimates, and generate comprehensive reports.

Following CLAUDE.md: Simple, focused utilities for ablation analysis.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple


def load_ablation_results(results_path: str) -> List[Dict[str, Any]]:
    """Load ablation results from JSONL or JSON file.

    Args:
        results_path: Path to results file

    Returns:
        List of experiment results
    """
    path = Path(results_path)
    results = []

    if path.suffix == ".jsonl":
        # Load JSONL format
        with open(path, "r") as f:
            for line in f:
                results.append(json.loads(line))
    else:
        # Load JSON format
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                results = data
            else:
                results = [data]

    return results


def aggregate_results_by_config(
    results: List[Dict[str, Any]], confidence_level: float = 0.95
) -> List[Dict[str, Any]]:
    """Aggregate results across random seeds for each configuration.

    Args:
        results: List of experiment results
        confidence_level: Confidence level for intervals (default 0.95)

    Returns:
        List of aggregated results by configuration
    """
    from scipy import stats

    # Group results by configuration
    grouped = defaultdict(list)

    for r in results:
        # Create configuration key
        key = (
            r.get("estimator", "unknown"),
            r.get("oracle_coverage", 1.0),
            r.get("sample_fraction", 1.0),
        )
        grouped[key].append(r)

    # Aggregate each group
    aggregated = []
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    for (estimator, oracle_cov, sample_frac), group in grouped.items():
        # Extract successful results
        successful = [r for r in group if r.get("success", False)]

        if not successful:
            # No successful runs for this configuration
            aggregated.append(
                {
                    "estimator": estimator,
                    "oracle_coverage": oracle_cov,
                    "sample_fraction": sample_frac,
                    "n_seeds_total": len(group),
                    "n_seeds_successful": 0,
                    "all_failed": True,
                    "failure_reasons": [r.get("error", "Unknown") for r in group],
                }
            )
            continue

        # Collect estimates and SEs for each policy
        estimates_by_policy = defaultdict(list)
        ses_by_policy = defaultdict(list)

        for r in successful:
            if "estimates" in r:
                for policy, est in r["estimates"].items():
                    if est == est:  # Not NaN
                        estimates_by_policy[policy].append(est)

                        # Get SE if available
                        if "standard_errors" in r and r["standard_errors"]:
                            se = r["standard_errors"].get(policy)
                            if se is not None and se == se:  # Not NaN
                                ses_by_policy[policy].append(se)

        # Compute aggregated statistics
        agg_result = {
            "estimator": estimator,
            "oracle_coverage": oracle_cov,
            "sample_fraction": sample_frac,
            "n_seeds_total": len(group),
            "n_seeds_successful": len(successful),
            "estimates_mean": {},
            "estimates_se": {},
            "estimates_ci": {},
            "coverage_by_policy": {},  # Fraction of seeds that produced estimates
            "oracle_truths": (
                successful[0].get("oracle_truths", {}) if successful else {}
            ),
        }

        # Aggregate each policy
        for policy in estimates_by_policy:
            ests = estimates_by_policy[policy]
            ses = ses_by_policy[policy]

            if ests:
                # Mean estimate across seeds
                mean_est = np.mean(ests)

                # Combined standard error
                if ses and len(ses) == len(ests):
                    # Root mean square combination including between-seed variance
                    within_var = np.mean([s**2 for s in ses])
                    between_var = np.var(ests, ddof=1) if len(ests) > 1 else 0
                    combined_se = np.sqrt(within_var + between_var)
                else:
                    # Only between-seed variance
                    combined_se = (
                        np.std(ests, ddof=1) / np.sqrt(len(ests))
                        if len(ests) > 1
                        else 0
                    )

                agg_result["estimates_mean"][policy] = mean_est
                agg_result["estimates_se"][policy] = combined_se
                agg_result["estimates_ci"][policy] = (
                    mean_est - z_score * combined_se,
                    mean_est + z_score * combined_se,
                )
                agg_result["coverage_by_policy"][policy] = len(ests) / len(successful)

        aggregated.append(agg_result)

    return aggregated


def compute_rmse_vs_oracle(
    aggregated_results: List[Dict[str, Any]],
) -> Dict[str, Dict[float, float]]:
    """Compute RMSE vs oracle ground truth for each estimator.

    Args:
        aggregated_results: Aggregated results from aggregate_results_by_config

    Returns:
        Dictionary mapping estimator -> {coverage -> RMSE}
    """
    rmse_by_estimator: Dict[str, Dict[float, float]] = defaultdict(dict)

    for result in aggregated_results:
        if result.get("all_failed", False):
            continue

        estimator = result["estimator"]
        coverage = result["oracle_coverage"]

        oracle_truths = result.get("oracle_truths", {})
        estimates = result.get("estimates_mean", {})

        if oracle_truths and estimates:
            # Compute RMSE across policies
            squared_errors = []
            for policy in oracle_truths:
                if policy in estimates:
                    error = estimates[policy] - oracle_truths[policy]
                    squared_errors.append(error**2)

            if squared_errors:
                rmse = np.sqrt(np.mean(squared_errors))
                rmse_by_estimator[estimator][coverage] = rmse

    return dict(rmse_by_estimator)


def analyze_convergence(
    aggregated_results: List[Dict[str, Any]],
) -> Dict[str, Dict[float, float]]:
    """Analyze convergence of estimates as oracle coverage increases.

    Args:
        aggregated_results: Aggregated results

    Returns:
        Dictionary mapping estimator -> {coverage -> avg_se}
    """
    convergence: Dict[str, Dict[float, float]] = defaultdict(dict)

    for result in aggregated_results:
        if result.get("all_failed", False):
            continue

        estimator = result["estimator"]
        coverage = result["oracle_coverage"]
        ses = result.get("estimates_se", {})

        if ses:
            # Average SE across policies
            avg_se = np.mean(list(ses.values()))
            convergence[estimator][coverage] = avg_se

    return dict(convergence)


def generate_summary_report(
    aggregated_results: List[Dict[str, Any]], output_path: Optional[str] = None
) -> str:
    """Generate a markdown summary report.

    Args:
        aggregated_results: Aggregated results
        output_path: Optional path to save report

    Returns:
        Markdown-formatted report string
    """
    report = ["# Ablation Study Summary Report\n"]

    # Group by estimator
    by_estimator = defaultdict(list)
    for r in aggregated_results:
        by_estimator[r["estimator"]].append(r)

    # Report for each estimator
    for estimator in sorted(by_estimator.keys()):
        report.append(f"\n## {estimator}\n")

        results = by_estimator[estimator]

        # Create table
        report.append(
            "| Oracle Coverage | Sample Fraction | Success Rate | Avg RMSE | Avg SE |"
        )
        report.append(
            "|-----------------|-----------------|--------------|----------|---------|"
        )

        for r in sorted(
            results, key=lambda x: (x["oracle_coverage"], x["sample_fraction"])
        ):
            coverage = r["oracle_coverage"]
            fraction = r["sample_fraction"]
            success_rate = f"{r['n_seeds_successful']}/{r['n_seeds_total']}"

            # Compute average RMSE if available
            oracle_truths = r.get("oracle_truths", {})
            estimates = r.get("estimates_mean", {})

            if oracle_truths and estimates:
                squared_errors = []
                for policy in oracle_truths:
                    if policy in estimates:
                        error = estimates[policy] - oracle_truths[policy]
                        squared_errors.append(error**2)

                avg_rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else "N/A"
            else:
                avg_rmse = "N/A"

            # Average SE
            ses = r.get("estimates_se", {})
            avg_se = np.mean(list(ses.values())) if ses else "N/A"

            # Format values
            avg_rmse_str = (
                f"{avg_rmse:.4f}" if isinstance(avg_rmse, float) else str(avg_rmse)
            )
            avg_se_str = f"{avg_se:.4f}" if isinstance(avg_se, float) else str(avg_se)

            report.append(
                f"| {coverage:.1%} | {fraction:.1%} | {success_rate} | {avg_rmse_str} | {avg_se_str} |"
            )

        # Policy-specific coverage
        report.append(f"\n### Policy Coverage for {estimator}\n")

        # Get all policies
        all_policies = set()
        for r in results:
            all_policies.update(r.get("coverage_by_policy", {}).keys())

        if all_policies:
            report.append(
                "| Oracle Coverage | " + " | ".join(sorted(all_policies)) + " |"
            )
            report.append(
                "|-----------------|"
                + " | ".join(["--------"] * len(all_policies))
                + " |"
            )

            for r in sorted(results, key=lambda x: x["oracle_coverage"]):
                coverage = r["oracle_coverage"]
                row = [f"{coverage:.1%}"]

                for policy in sorted(all_policies):
                    cov = r.get("coverage_by_policy", {}).get(policy, 0)
                    row.append(f"{cov:.1%}")

                report.append("| " + " | ".join(row) + " |")

    # Add summary statistics
    report.append("\n## Overall Statistics\n")

    total_experiments = sum(r["n_seeds_total"] for r in aggregated_results)
    total_successful = sum(r["n_seeds_successful"] for r in aggregated_results)

    report.append(f"- Total experiments: {total_experiments}")
    report.append(
        f"- Successful experiments: {total_successful} ({total_successful/total_experiments:.1%})"
    )

    # Identify problematic configurations
    failed_configs = [r for r in aggregated_results if r.get("all_failed", False)]
    if failed_configs:
        report.append("\n### Failed Configurations\n")
        for r in failed_configs:
            report.append(
                f"- {r['estimator']} @ {r['oracle_coverage']:.1%} coverage: {r.get('failure_reasons', ['Unknown'])[0]}"
            )

    report_str = "\n".join(report)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report_str)

    return report_str


def main() -> None:
    """Main entry point for analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze ablation experiment results")
    parser.add_argument("results", help="Path to ablation results file")
    parser.add_argument("--output", help="Path to save summary report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load results
    results = load_ablation_results(args.results)
    print(f"Loaded {len(results)} experiment results")

    # Aggregate across seeds
    aggregated = aggregate_results_by_config(results)
    print(f"Aggregated into {len(aggregated)} configurations")

    # Generate report
    report = generate_summary_report(aggregated, args.output)

    if args.verbose:
        print("\n" + report)

    # Compute key metrics
    rmse_by_estimator = compute_rmse_vs_oracle(aggregated)
    convergence = analyze_convergence(aggregated)

    print("\n=== RMSE vs Oracle Ground Truth ===")
    for est_name in sorted(rmse_by_estimator.keys()):
        print(f"\n{est_name}:")
        for cov in sorted(rmse_by_estimator[est_name].keys()):
            print(f"  {cov:.1%} coverage: {rmse_by_estimator[est_name][cov]:.4f}")

    print("\n=== Standard Error Convergence ===")
    for est_name in sorted(convergence.keys()):
        print(f"\n{est_name}:")
        for cov in sorted(convergence[est_name].keys()):
            print(f"  {cov:.1%} coverage: {convergence[est_name][cov]:.4f}")

    if args.output:
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
