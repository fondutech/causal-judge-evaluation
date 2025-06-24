#!/usr/bin/env python3
"""
Test that a clone target policy (π_target = π_behavior) has all weights = 1.
This is a fundamental property of importance sampling.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from cje.loggers.precomputed_sampler import PrecomputedMultiTargetSampler
from rich.console import Console

console = Console()


def test_clone_policy_weights() -> bool:
    """Test that clone policy has unit weights."""

    console.print("\n[bold blue]Testing Clone Policy Weights[/bold blue]\n")

    # Create test data where one target policy is identical to behavior policy
    contexts = [f"Context {i}" for i in range(10)]
    responses = [f"Response {i}" for i in range(10)]

    # Behavior policy log probs
    logp_behavior = np.random.uniform(-15, -5, size=10).tolist()

    # Create lookup where:
    # - Policy 0: Clone of behavior policy (same log probs)
    # - Policy 1: Different policy
    # - Policy 2: Very different policy
    logp_lookup = {}

    for i, (ctx, resp) in enumerate(zip(contexts, responses)):
        logp_lookup[(ctx, resp)] = [
            logp_behavior[i],  # Clone policy - same as behavior
            logp_behavior[i] + np.random.uniform(-2, 2),  # Slightly different
            logp_behavior[i] + np.random.uniform(-5, 5),  # Very different
        ]

    # Create sampler
    sampler = PrecomputedMultiTargetSampler(logp_lookup=logp_lookup, n_policies=3)

    # Compute importance weights
    weights_matrix, stats = sampler.importance_weights_matrix(
        contexts=contexts,
        responses=responses,
        logp_behavior=logp_behavior,
        stabilize=False,  # No stabilization to see raw weights
        return_stats=True,
    )

    # Check clone policy weights
    clone_weights = weights_matrix[:, 0]

    console.print(f"📊 Clone Policy Weight Statistics:")
    console.print(f"   Min weight: {np.min(clone_weights):.6f}")
    console.print(f"   Max weight: {np.max(clone_weights):.6f}")
    console.print(f"   Mean weight: {np.mean(clone_weights):.6f}")
    console.print(f"   Std weight: {np.std(clone_weights):.6f}")

    # Check if all weights are approximately 1
    tolerance = 1e-10
    all_ones = np.allclose(clone_weights, 1.0, atol=tolerance)

    if all_ones:
        console.print(
            f"\n✅ [green]SUCCESS: All clone policy weights = 1.0 (within {tolerance})[/green]"
        )
    else:
        console.print(f"\n❌ [red]FAILURE: Clone policy weights deviate from 1.0[/red]")
        console.print(f"   Deviations: {clone_weights - 1.0}")

    # Also test with stabilization
    console.print(f"\n[bold]Testing with stabilization enabled:[/bold]")

    weights_stabilized, stats_stab = sampler.importance_weights_matrix(
        contexts=contexts,
        responses=responses,
        logp_behavior=logp_behavior,
        stabilize=True,
        return_stats=True,
    )

    clone_weights_stab = weights_stabilized[:, 0]
    console.print(f"   Mean weight (stabilized): {np.mean(clone_weights_stab):.6f}")
    console.print(f"   Std weight (stabilized): {np.std(clone_weights_stab):.6f}")

    # Check mathematical property: log(π/π) = 0 → exp(0) = 1
    console.print(f"\n[bold]Mathematical verification:[/bold]")
    for i in range(min(5, len(contexts))):  # Show first 5
        log_ratio = logp_lookup[(contexts[i], responses[i])][0] - logp_behavior[i]
        weight = np.exp(log_ratio)
        console.print(
            f"   Sample {i}: log(π_target/π_behavior) = {log_ratio:.6f}, "
            f"weight = exp({log_ratio:.6f}) = {weight:.6f}"
        )

    # Test extreme case with very negative log probs
    console.print(f"\n[bold]Testing extreme log probabilities:[/bold]")

    extreme_contexts = ["Extreme 1", "Extreme 2", "Extreme 3"]
    extreme_responses = ["Response 1", "Response 2", "Response 3"]
    extreme_logp_behavior = [-100.0, -200.0, -300.0]  # Very negative

    extreme_lookup = {}
    for i, (ctx, resp) in enumerate(zip(extreme_contexts, extreme_responses)):
        extreme_lookup[(ctx, resp)] = [
            extreme_logp_behavior[i],  # Clone
            extreme_logp_behavior[i] - 10,  # Much worse
            extreme_logp_behavior[i] + 10,  # Much better
        ]

    extreme_sampler = PrecomputedMultiTargetSampler(
        logp_lookup=extreme_lookup, n_policies=3
    )

    extreme_weights, _ = extreme_sampler.importance_weights_matrix(
        contexts=extreme_contexts,
        responses=extreme_responses,
        logp_behavior=extreme_logp_behavior,
        stabilize=False,
        return_stats=True,
    )

    extreme_clone = extreme_weights[:, 0]
    console.print(f"   Extreme clone weights: {extreme_clone}")
    console.print(f"   All ones? {np.allclose(extreme_clone, 1.0)}")

    return bool(all_ones)


if __name__ == "__main__":
    test_clone_policy_weights()
