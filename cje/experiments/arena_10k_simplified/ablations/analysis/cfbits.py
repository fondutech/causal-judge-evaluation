"""CF-bits analysis helpers for ablation results."""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from cje.diagnostics import (
    CFBitsDiagnostics,
    format_cfbits_summary,
    format_cfbits_table,
)

logger = logging.getLogger(__name__)


def extract_cfbits_diagnostics(result: Dict[str, Any]) -> Dict[str, CFBitsDiagnostics]:
    """Extract CF-bits diagnostics from experiment result.

    Args:
        result: Experiment result dictionary

    Returns:
        Dictionary mapping policy names to CFBitsDiagnostics objects
    """
    cfbits_by_policy = {}

    # Check for structured diagnostics first (new format)
    if "cfbits_diagnostics" in result:
        for policy, diag in result["cfbits_diagnostics"].items():
            if isinstance(diag, CFBitsDiagnostics):
                cfbits_by_policy[policy] = diag
            elif isinstance(diag, dict):
                # Reconstruct from dict if needed
                try:
                    cfbits_by_policy[policy] = CFBitsDiagnostics(**diag)
                except Exception as e:
                    logger.debug(f"Could not reconstruct CFBitsDiagnostics: {e}")

    # Fall back to summary format (backward compatibility)
    elif "cfbits_summary" in result:
        estimator_name = result.get("spec", {}).get("estimator", "unknown")
        for policy, summary in result["cfbits_summary"].items():
            if summary and any(v is not None for v in summary.values()):
                try:
                    # Determine scenario from estimator type
                    fresh_draws_estimators = [
                        "dr-cpo",
                        "mrdr",
                        "tmle",
                        "tr-cpo-e",
                        "tr-cpo-e-anchored-orthogonal",
                        "stacked-dr",
                        "oc-dr-cpo",
                    ]
                    scenario = (
                        "fresh-draws"
                        if estimator_name in fresh_draws_estimators
                        else "logging-only"
                    )

                    cfbits_by_policy[policy] = CFBitsDiagnostics(
                        policy=policy,
                        estimator_type=estimator_name,
                        scenario=scenario,
                        wid=summary.get("wid"),
                        wvar=summary.get("wvar"),
                        w_tot=summary.get("w_tot"),
                        bits_tot=summary.get("bits_tot"),
                        ifr_oua=summary.get("ifr_oua"),
                        aess_oua=summary.get("aess_oua"),
                        aessf_sigmaS_lcb=summary.get("aessf_lcb"),
                        gate_state=summary.get("gate_state", "CRITICAL"),
                        gate_reasons=summary.get("gate_reasons", []),
                    )
                except Exception as e:
                    logger.debug(
                        f"Could not create CFBitsDiagnostics from summary: {e}"
                    )

    return cfbits_by_policy


def analyze_cfbits_from_file(results_file: Path) -> None:
    """Analyze CF-bits from a results file.

    Args:
        results_file: Path to JSONL results file
    """
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return

    all_cfbits = []

    with open(results_file, "r") as f:
        for line in f:
            try:
                result = json.loads(line)
                if not result.get("success", False):
                    continue

                cfbits_by_policy = extract_cfbits_diagnostics(result)
                for policy, cfbits_diag in cfbits_by_policy.items():
                    all_cfbits.append(cfbits_diag)

            except json.JSONDecodeError:
                continue

    if not all_cfbits:
        logger.info("No CF-bits diagnostics found in results")
        return

    # Group by estimator type
    by_estimator: Dict[str, List[CFBitsDiagnostics]] = {}
    for cfbits in all_cfbits:
        if cfbits.estimator_type not in by_estimator:
            by_estimator[cfbits.estimator_type] = []
        by_estimator[cfbits.estimator_type].append(cfbits)

    # Display summaries
    print("\n" + "=" * 80)
    print("CF-bits Analysis")
    print("=" * 80)

    for estimator, cfbits_list in by_estimator.items():
        print(f"\n{estimator}:")
        print("-" * 40)

        # Show individual summaries
        for cfbits in cfbits_list[:3]:  # Show first 3 policies
            print(f"  {cfbits.policy}: {format_cfbits_summary(cfbits)}")

        if len(cfbits_list) > 3:
            print(f"  ... and {len(cfbits_list) - 3} more policies")

    # Show comparative table if multiple estimators
    if len(by_estimator) > 1:
        print("\n" + "=" * 80)
        print("Comparative CF-bits Table")
        print("=" * 80)

        # Take first policy from each estimator for comparison
        comparison = []
        for estimator, cfbits_list in by_estimator.items():
            if cfbits_list:
                comparison.append(cfbits_list[0])

        if comparison:
            print(format_cfbits_table(comparison))


def get_cfbits_summary_stats(cfbits_list: List[CFBitsDiagnostics]) -> Dict[str, Any]:
    """Compute summary statistics across CF-bits diagnostics.

    Args:
        cfbits_list: List of CFBitsDiagnostics objects

    Returns:
        Dictionary with summary statistics
    """
    import numpy as np

    if not cfbits_list:
        return {}

    # Extract numeric values
    bits = [c.bits_tot for c in cfbits_list if c.bits_tot is not None]
    w_tot = [c.w_tot for c in cfbits_list if c.w_tot is not None]
    wid = [c.wid for c in cfbits_list if c.wid is not None]
    wvar = [c.wvar for c in cfbits_list if c.wvar is not None]

    # Gate distribution
    from cje.diagnostics import GateState

    gate_counts: Dict[str, int] = {}
    for c in cfbits_list:
        gate_key = (
            c.gate_state.value if hasattr(c.gate_state, "value") else str(c.gate_state)
        )
        gate_counts[gate_key] = gate_counts.get(gate_key, 0) + 1

    return {
        "n_diagnostics": len(cfbits_list),
        "bits_mean": float(np.mean(bits)) if bits else None,
        "bits_median": float(np.median(bits)) if bits else None,
        "bits_std": float(np.std(bits)) if bits else None,
        "w_tot_mean": float(np.mean(w_tot)) if w_tot else None,
        "wid_mean": float(np.mean(wid)) if wid else None,
        "wvar_mean": float(np.mean(wvar)) if wvar else None,
        "gate_distribution": gate_counts,
        "pct_good": 100.0 * gate_counts.get("good", 0) / len(cfbits_list),
        "pct_warning": 100.0 * gate_counts.get("warning", 0) / len(cfbits_list),
        "pct_critical": 100.0 * gate_counts.get("critical", 0) / len(cfbits_list),
        "pct_refuse": 100.0 * gate_counts.get("refuse", 0) / len(cfbits_list),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
        analyze_cfbits_from_file(results_file)
    else:
        print("Usage: python cfbits.py <results_file.jsonl>")
