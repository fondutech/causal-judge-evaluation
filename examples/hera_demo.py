#!/usr/bin/env python3
"""
HERA (Hellinger–ESS Raw Audit) Demonstration

HERA provides ungameable overlap diagnostics for importance sampling.
The two-number audit tells you whether to trust your estimates.
"""

import numpy as np
from cje.diagnostics.hera import (
    hera_audit,
    hera_drill_down,
    format_hera_drill_down,
    hera_summary_card,
    hera_threshold,
    HERAMetrics,
)


def demo_hera_gates() -> None:
    """Demonstrate HERA's audit gates."""

    print("=" * 70)
    print("  HERA: Hellinger–ESS Raw Audit")
    print("=" * 70)
    print("\nHERA uses two ungameable numbers to audit importance sampling:")
    print("  H = Hellinger affinity (structural overlap)")
    print("  E = ESS raw fraction (variance inflation)")
    print("\nHERA Gates:")
    print("  CRITICAL: H < 0.20 OR E < 0.10 → REFUSE IPS")
    print("  WARNING:  H < 0.35 OR E < 0.20 → Caution, prefer DR")
    print("  OK:       Otherwise → Proceed with IPS")
    print("=" * 70)

    # Test scenarios from your dataset
    scenarios = {
        "premium": {
            "description": "Extreme mismatch (your worst case)",
            "delta_log": np.concatenate(
                [
                    np.full(4850, -8.0),  # Most samples have tiny weights
                    np.full(50, 5.0),  # Few samples carry all weight
                ]
            ),
        },
        "unhelpful": {
            "description": "Poor overlap",
            "delta_log": np.concatenate([np.full(4000, -3.0), np.full(900, 1.5)]),
        },
        "parallel_universe": {
            "description": "Moderate overlap",
            "delta_log": np.random.RandomState(42).normal(0, 1.2, 4900),
        },
        "clone": {
            "description": "Good overlap",
            "delta_log": np.random.RandomState(42).normal(0, 0.5, 4900),
        },
    }

    hera_results = {}

    for policy, scenario in scenarios.items():
        print(f"\n### {policy}")
        print(f"    {scenario['description']}")
        print("-" * 60)

        # Run HERA audit with auto-tuning for ±3% CI
        metrics = hera_audit(
            scenario["delta_log"],
            n_samples=4900,
            target_ci_halfwidth=0.03,
        )
        hera_results[policy] = metrics

        # Display HERA results
        print(f"  H (Hellinger):     {metrics.hellinger_affinity:6.1%}")
        print(f"  E (Raw ESS):       {metrics.ess_raw_fraction:6.1%}")
        print(f"  HERA Status:       {metrics.hera_status.upper()}")

        if metrics.auto_tuned_threshold:
            meets = (
                "✓" if metrics.ess_raw_fraction >= metrics.auto_tuned_threshold else "✗"
            )
            print(f"  Auto-threshold:    {metrics.auto_tuned_threshold:6.1%} [{meets}]")

        print(f"\n  {metrics.recommendation}")

        # Show what happens for each estimator
        if metrics.hera_status == "critical":
            print("\n  Estimator behavior:")
            print("    IPS/Cal-IPS: ❌ REFUSED by HERA")
            print("    DR/TMLE:     ⚠️  Proceed with HERA warnings")
        elif metrics.hera_status == "warning":
            print("\n  Estimator behavior:")
            print("    IPS/Cal-IPS: ⚠️  Allowed with HERA warning")
            print("    DR/TMLE:     ✓ Preferred (HERA recommends)")

    # Show summary card
    print(hera_summary_card(hera_results, "HERA Audit Summary"))


def demo_hera_drill_down() -> None:
    """Demonstrate HERA's drill-down diagnostics."""

    print("\n" + "=" * 70)
    print("  HERA Drill-Down Analysis")
    print("=" * 70)
    print("\nHERA can analyze WHERE overlap problems occur:")

    np.random.seed(42)
    n = 4900

    # Create heterogeneous overlap: good for low judges, poor for high
    judge_scores = np.random.uniform(0, 10, n)

    # Overlap degrades with judge score (common pattern)
    delta_log = np.zeros(n)
    for i in range(n):
        if judge_scores[i] < 3:
            # Good overlap for low scores
            delta_log[i] = np.random.normal(0, 0.5)
        elif judge_scores[i] < 7:
            # Marginal overlap for medium scores
            delta_log[i] = np.random.normal(-1, 1.5)
        else:
            # Poor overlap for high scores
            delta_log[i] = np.random.normal(-3, 2.0)

    # Run overall HERA audit
    overall = hera_audit(delta_log, n_samples=n)
    print(f"\nOverall: {overall.summary()}")

    # Drill down by judge score deciles
    drill_down = hera_drill_down(
        delta_log, judge_scores, n_bins=10, index_name="judge_score"
    )

    print(format_hera_drill_down(drill_down))

    print("\n💡 Insight: HERA reveals problems concentrated in high judge scores!")
    print("   This tells you WHERE the overlap issues are, not just that they exist.")


def demo_hera_auto_tuning() -> None:
    """Show HERA's auto-tuning for different CI goals."""

    print("\n" + "=" * 70)
    print("  HERA Auto-Tuned Thresholds")
    print("=" * 70)
    print("\nHERA adapts ESS thresholds to your precision goals:")
    print("\nFormula: τ_ESS(δ) = 0.9604/(n·δ²)")
    print("-" * 40)

    # Your dataset size
    n = 4900
    print(f"Dataset size: n={n:,} samples\n")

    ci_targets = [0.01, 0.02, 0.03, 0.05, 0.10]

    print(f"{'Target CI':>12} | {'ESS Threshold':>15} | {'Interpretation'}")
    print("-" * 60)

    for delta in ci_targets:
        threshold = hera_threshold(n, delta)

        if threshold > 0.50:
            interpretation = "Very strict"
        elif threshold > 0.20:
            interpretation = "Strict"
        elif threshold > 0.10:
            interpretation = "Moderate"
        else:
            interpretation = "Lenient"

        print(f"±{delta:10.1%} | {threshold:14.1%} | {interpretation}")

    print("\nNote: HERA's fixed H thresholds (0.20/0.35) don't change.")
    print("      Only E thresholds adapt to your CI goals.")


def demo_hera_theory() -> None:
    """Show the theory behind HERA."""

    print("\n" + "=" * 70)
    print("  HERA Theory")
    print("=" * 70)

    print(
        """
HERA is based on two theoretical results:

1. Hellinger-TV Bound:
   TV(P_π', P_π₀) ≤ √(2(1-H))
   
   When H < 0.20, the distributions are nearly disjoint.
   No amount of calibration can fix this structural issue.

2. ESS-Variance Identity:
   Var(V_IPS) ≤ 1/(4n·E)
   
   Where E = ESS_raw/n = 1/(1 + χ²(P_π'||P_π₀))
   This directly controls confidence interval width.

HERA's genius: These are computed on RAW weights before
any calibration, making them ungameable. You can't hide
bad overlap from HERA.

Key Properties:
✓ Log-safe computation (no numerical overflow)
✓ Bounded metrics (H,E ∈ [0,1])
✓ Fast O(n) computation
✓ Works with any importance sampling setup
"""
    )


def main() -> None:
    """Run all HERA demos."""
    demo_hera_gates()
    demo_hera_drill_down()
    demo_hera_auto_tuning()
    demo_hera_theory()

    print("\n" + "=" * 70)
    print("  HERA Summary")
    print("=" * 70)
    print(
        """
HERA (Hellinger–ESS Raw Audit) provides:

1. Two ungameable numbers (H and E)
2. Clear gates (0.20/0.35 for H, 0.10/0.20 for E)  
3. Auto-tuning tied to CI goals
4. Drill-down to localize problems
5. Different behavior for IPS vs DR

Remember: HERA audits the FUNDAMENTAL overlap.
If HERA says no, listen.
"""
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
