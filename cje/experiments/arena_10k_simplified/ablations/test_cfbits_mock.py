#!/usr/bin/env python3
"""Test CF-bits integration with mock data to verify structured diagnostics."""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.diagnostics import (
    CFBitsDiagnostics,
    GateState,
    format_cfbits_summary,
    format_cfbits_table,
)
from ablations.analysis.cfbits import get_cfbits_summary_stats


def test_structured_cfbits():
    """Test the structured CF-bits diagnostics work correctly."""

    print("Testing structured CF-bits diagnostics...")

    # Create mock CF-bits diagnostics for different scenarios
    cfbits_list = []

    # IPS scenario (logging-only)
    cfbits_ips = CFBitsDiagnostics(
        policy="pi_good",
        estimator_type="calibrated-ips",
        scenario="logging_only",
        wid=0.15,
        wvar=0.08,
        w_tot=0.23,
        bits_tot=2.1,
        ifr_oua=0.46,
        aess_oua=0.52,
        aessf_sigmaS=0.48,
        aessf_sigmaS_lcb=0.24,
        bc_sigmaS=0.69,
        gate_state=GateState.WARNING,
        gate_reasons=["Moderate efficiency loss"],
        labels_for_wid_reduction=1000,
    )
    cfbits_list.append(cfbits_ips)

    # DR scenario (fresh-draws)
    cfbits_dr = CFBitsDiagnostics(
        policy="pi_good",
        estimator_type="dr-cpo",
        scenario="fresh_draws",
        wid=0.12,
        wvar=0.05,
        w_tot=0.17,
        bits_tot=2.5,
        ifr_oua=0.68,
        aess_oua=0.74,
        aessf_sigmaS=0.48,
        aessf_sigmaS_lcb=0.24,
        bc_sigmaS=0.69,
        gate_state=GateState.GOOD,
        gate_reasons=[],
    )
    cfbits_list.append(cfbits_dr)

    # Poor overlap scenario
    cfbits_poor = CFBitsDiagnostics(
        policy="pi_bad",
        estimator_type="raw-ips",
        scenario="logging_only",
        wid=0.45,
        wvar=0.35,
        w_tot=0.80,
        bits_tot=0.3,
        ifr_oua=0.12,
        aess_oua=0.08,
        aessf_sigmaS=0.15,
        aessf_sigmaS_lcb=0.05,
        bc_sigmaS=0.28,
        gate_state=GateState.REFUSE,
        gate_reasons=["Catastrophic overlap", "A-ESSF < 5%"],
    )
    cfbits_list.append(cfbits_poor)

    # Display individual summaries
    print("\nIndividual CF-bits Summaries:")
    print("=" * 80)
    for cfbits in cfbits_list:
        print(f"\n{cfbits.estimator_type} / {cfbits.policy}:")
        print(f"  {format_cfbits_summary(cfbits)}")

    # Display comparative table
    print("\n\nComparative CF-bits Table:")
    print("=" * 80)
    print(format_cfbits_table(cfbits_list))

    # Compute summary statistics
    stats = get_cfbits_summary_stats(cfbits_list)
    print("\n\nSummary Statistics:")
    print("=" * 80)
    for key, value in stats.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # Test validation
    print("\n\nTesting Validation:")
    print("=" * 80)

    try:
        # Valid diagnostics
        cfbits_ips.validate()
        print("✓ IPS diagnostics valid")
        cfbits_dr.validate()
        print("✓ DR diagnostics valid")
        cfbits_poor.validate()
        print("✓ Poor overlap diagnostics valid")

        # Test invalid diagnostics
        invalid = CFBitsDiagnostics(
            policy="pi_test",
            estimator_type="test",
            scenario="invalid",  # Invalid scenario
            wid=0.5,
            wvar=0.5,
        )
        try:
            invalid.validate()
            print("✗ Should have failed validation")
        except ValueError as e:
            print(f"✓ Correctly rejected invalid scenario: {e}")

    except Exception as e:
        print(f"✗ Unexpected validation error: {e}")

    print("\n✅ Structured CF-bits diagnostics test complete!")


if __name__ == "__main__":
    test_structured_cfbits()
