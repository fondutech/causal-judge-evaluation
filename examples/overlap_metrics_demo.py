#!/usr/bin/env python3
"""
Demo of the new overlap metrics (Hellinger affinity and auto-tuned thresholds).

This shows how the metrics provide complementary information about policy overlap:
- Hellinger affinity: Structural compatibility (cannot be improved)
- ESS: Statistical efficiency (can be improved by calibration)
- Auto-tuned thresholds: Tied to desired CI width
"""

import numpy as np
from cje.diagnostics.overlap import (
    compute_overlap_metrics,
    diagnose_overlap_problems,
    compute_auto_tuned_threshold,
)


def demo_different_overlaps() -> None:
    """Demonstrate metrics for different overlap scenarios."""

    print("=" * 70)
    print("OVERLAP METRICS DEMONSTRATION")
    print("=" * 70)

    scenarios = [
        {
            "name": "Perfect Overlap",
            "weights": np.ones(1000),
            "description": "Uniform weights (policies identical)",
        },
        {
            "name": "Good Overlap",
            "weights": np.random.lognormal(-(0.5**2) / 2, 0.5, 1000),
            "description": "Mild log-normal (σ=0.5)",
        },
        {
            "name": "Marginal Overlap",
            "weights": np.random.lognormal(-(1.5**2) / 2, 1.5, 1000),
            "description": "Moderate log-normal (σ=1.5)",
        },
        {
            "name": "Poor Overlap",
            "weights": np.concatenate([np.full(900, 0.1), np.full(100, 9.1)]),
            "description": "90% low weights, 10% high",
        },
        {
            "name": "Catastrophic Overlap",
            "weights": np.concatenate([np.full(990, 0.001), np.full(10, 99.01)]),
            "description": "99% near-zero weights, 1% extreme",
        },
    ]

    for scenario in scenarios:
        weights = scenario["weights"]
        weights = weights / weights.mean()  # Normalize to mean 1

        print(f"\n### {scenario['name']}")
        print(f"    {scenario['description']}")
        print("-" * 60)

        # Compute metrics
        metrics = compute_overlap_metrics(
            weights, target_ci_halfwidth=0.01, auto_tune_threshold=True
        )

        # Display key metrics
        print(
            f"  Hellinger affinity:  {metrics.hellinger_affinity:6.1%} ({metrics.overlap_quality})"
        )
        print(f"  ESS fraction:        {metrics.ess_fraction:6.1%}")
        print(f"  Confidence penalty:  {metrics.confidence_penalty:6.1f}x wider CIs")

        if metrics.tail_index:
            print(f"  Tail index (Hill):   {metrics.tail_index:6.2f}")

        print(f"\n  Can calibrate help?  {'Yes' if metrics.can_calibrate else 'No'}")
        print(f"  Recommended method:  {metrics.recommended_method}")

        # Show auto-tuning
        if metrics.auto_tuned_threshold:
            meets = "✓" if metrics.ess_fraction >= metrics.auto_tuned_threshold else "✗"
            print(
                f"  Auto-tuned threshold: {metrics.auto_tuned_threshold:6.1%} [{meets}]"
            )


def demo_auto_tuning() -> None:
    """Demonstrate auto-tuning for different sample sizes and targets."""

    print("\n" + "=" * 70)
    print("AUTO-TUNED THRESHOLD DEMONSTRATION")
    print("=" * 70)

    print("\nESS threshold needed for target CI width:")
    print("-" * 60)
    print("Sample Size | Target CI | Critical ESS | Warning ESS")
    print("-" * 60)

    for n in [1000, 5000, 10000, 50000, 100000]:
        for target in [0.005, 0.01, 0.02]:
            critical = compute_auto_tuned_threshold(n, target, "critical")
            warning = compute_auto_tuned_threshold(n, target, "warning")
            print(f"{n:10,} |    ±{target:4.1%} | {critical:11.1%} | {warning:10.1%}")
        if n < 100000:
            print("-" * 60)

    print("\nInterpretation:")
    print("- Larger samples → lower ESS threshold needed")
    print("- Tighter CI target → higher ESS threshold needed")
    print("- Warning threshold = 50% of critical threshold")


def demo_diagnostic_flow() -> None:
    """Demonstrate the diagnostic decision flow."""

    print("\n" + "=" * 70)
    print("DIAGNOSTIC FLOW DEMONSTRATION")
    print("=" * 70)

    # Create challenging but not impossible weights
    np.random.seed(42)
    weights = np.random.lognormal(-(1.0**2) / 2, 1.0, 5000)
    weights = weights / weights.mean()

    print("\nAnalyzing weights from moderate distribution mismatch...")
    print("-" * 60)

    metrics = compute_overlap_metrics(
        weights, target_ci_halfwidth=0.015, auto_tune_threshold=True  # ±1.5% target
    )

    # Show the diagnostic cascade
    print("\n1. CHECK STRUCTURAL OVERLAP (Hellinger)")
    print(f"   Affinity = {metrics.hellinger_affinity:.1%}")
    if metrics.hellinger_affinity < 0.20:
        print("   ❌ STOP: Catastrophic mismatch, no method can help")
    elif metrics.hellinger_affinity < 0.35:
        print("   ⚠️  Poor overlap, but calibration might help")
    else:
        print("   ✓ Adequate structural overlap")

    print("\n2. CHECK STATISTICAL EFFICIENCY (ESS)")
    print(f"   ESS = {metrics.ess_fraction:.1%}")
    if metrics.ess_fraction < 0.10:
        print("   ❌ Very low efficiency, need DR or refuse")
    elif metrics.ess_fraction < 0.30:
        print("   ⚠️  Low efficiency, calibration recommended")
    else:
        print("   ✓ Acceptable efficiency")

    print("\n3. CHECK TAIL BEHAVIOR (Hill)")
    if metrics.tail_index:
        print(f"   Tail index = {metrics.tail_index:.2f}")
        if metrics.tail_index < 1.5:
            print("   ❌ Extremely heavy tails (infinite mean)")
        elif metrics.tail_index < 2.0:
            print("   ⚠️  Heavy tails (infinite variance)")
        else:
            print("   ✓ Acceptable tail behavior")

    print("\n4. RECOMMENDATION")
    should_proceed, explanation = diagnose_overlap_problems(metrics, verbose=False)
    print(explanation)


if __name__ == "__main__":
    demo_different_overlaps()
    demo_auto_tuning()
    demo_diagnostic_flow()

    print("\n" + "=" * 70)
    print("Key Insights:")
    print("- Hellinger measures what CAN'T be fixed (structural mismatch)")
    print("- ESS measures what CAN be improved (by calibration)")
    print("- Auto-tuning ties thresholds to your precision needs")
    print("- The metrics form a diagnostic cascade for decision-making")
    print("=" * 70)
