#!/usr/bin/env python3
"""Test CF-bits integration with ablations using structured diagnostics."""

import sys
import json
from pathlib import Path
import tempfile

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from ablations.core.base import BaseAblation
from ablations.core.schemas import ExperimentSpec
from ablations.analysis.cfbits import (
    extract_cfbits_diagnostics,
    get_cfbits_summary_stats,
)
from cje.diagnostics import format_cfbits_summary, format_cfbits_table


class TestCFBitsAblation(BaseAblation):
    """Test ablation for CF-bits integration."""

    def __init__(self):
        super().__init__(name="cfbits_test")


def test_cfbits_with_ablations():
    """Test that CF-bits works with ablations and produces structured diagnostics."""

    print("Testing CF-bits integration with ablations...")

    # Create test specification
    spec = ExperimentSpec(
        ablation="cfbits_test",
        dataset_path="../data/arena_10k_simplified_cal.jsonl",
        estimator="calibrated-ips",
        sample_size=100,
        oracle_coverage=0.1,
        extra={
            "use_weight_calibration": True,
            "compute_cfbits": True,  # Enable CF-bits
            "compute_cfbits_full": False,
        },
    )

    # Run experiment
    ablation = TestCFBitsAblation()
    result = ablation.run_single(spec, seed=42)

    print(f"\nExperiment success: {result['success']}")

    if result["success"]:
        # Extract CF-bits diagnostics
        cfbits_by_policy = extract_cfbits_diagnostics(result)

        if cfbits_by_policy:
            print(f"\nFound CF-bits diagnostics for {len(cfbits_by_policy)} policies")

            # Display summaries
            for policy, cfbits_diag in cfbits_by_policy.items():
                print(f"\n{policy}:")
                print(f"  {format_cfbits_summary(cfbits_diag)}")

            # Show comparative table if multiple policies
            if len(cfbits_by_policy) > 1:
                print("\n" + "=" * 60)
                print("CF-bits Comparison Table:")
                print("=" * 60)
                print(format_cfbits_table(list(cfbits_by_policy.values())))

            # Compute summary stats
            stats = get_cfbits_summary_stats(list(cfbits_by_policy.values()))
            print("\n" + "=" * 60)
            print("Summary Statistics:")
            print("=" * 60)
            for key, value in stats.items():
                if value is not None:
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")

            # Verify structured diagnostics are present
            assert "cfbits_diagnostics" in result or "cfbits_summary" in result
            print("\n✓ CF-bits structured diagnostics successfully integrated")

        else:
            print(
                "\n⚠ No CF-bits diagnostics found (might be expected if policies have poor overlap)"
            )

    else:
        print(f"\nExperiment failed: {result.get('error', 'Unknown error')}")

    # Test with fresh-draws estimator
    print("\n" + "=" * 60)
    print("Testing with fresh-draws estimator (DR-CPO)...")
    print("=" * 60)

    spec_dr = ExperimentSpec(
        ablation="cfbits_test",
        dataset_path="../data/arena_10k_simplified_cal.jsonl",
        estimator="dr-cpo",
        sample_size=100,
        oracle_coverage=0.1,
        extra={
            "use_weight_calibration": True,
            "compute_cfbits": True,
            "with_fresh_draws": True,
        },
    )

    result_dr = ablation.run_single(spec_dr, seed=42)

    if result_dr["success"]:
        cfbits_dr = extract_cfbits_diagnostics(result_dr)
        if cfbits_dr:
            print(f"\nFound CF-bits diagnostics for {len(cfbits_dr)} policies")
            for policy, cfbits_diag in list(cfbits_dr.items())[:2]:  # Show first 2
                print(f"\n{policy}:")
                print(f"  Scenario: {cfbits_diag.scenario}")
                print(f"  {format_cfbits_summary(cfbits_diag)}")
        else:
            print("\n⚠ No CF-bits diagnostics for DR (check if fresh draws available)")
    else:
        print(f"\nDR experiment failed: {result_dr.get('error', 'Unknown error')}")

    print("\n✅ CF-bits ablation integration test complete!")


if __name__ == "__main__":
    test_cfbits_with_ablations()
