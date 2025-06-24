#!/usr/bin/env python3
"""
Test that clone policy estimation returns the empirical mean reward.
When π_target = π_behavior, the IPS estimate should equal the sample mean.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from cje.loggers.precomputed_sampler import PrecomputedMultiTargetSampler
from cje.estimators import get_estimator
from rich.console import Console

console = Console()


def test_clone_policy_estimation() -> None:
    """Test that clone policy IPS returns empirical mean."""

    console.print("\n[bold blue]Testing Clone Policy Estimation[/bold blue]\n")

    # Create test data
    n_samples = 100
    contexts = [f"Context {i}" for i in range(n_samples)]
    responses = [f"Response {i}" for i in range(n_samples)]

    # Generate rewards and log probs
    np.random.seed(42)
    rewards = np.random.uniform(0, 1, size=n_samples)
    logp_behavior = np.random.uniform(-15, -5, size=n_samples).tolist()

    # Empirical mean reward
    empirical_mean = np.mean(rewards)
    console.print(f"📊 Empirical mean reward: {empirical_mean:.6f}")

    # Create lookup with clone policy
    logp_lookup = {}
    for i, (ctx, resp) in enumerate(zip(contexts, responses)):
        logp_lookup[(ctx, resp)] = [
            logp_behavior[i],  # Policy 0: Clone
            logp_behavior[i] + np.random.uniform(-2, 2),  # Policy 1: Different
            logp_behavior[i] + np.random.uniform(-5, 5),  # Policy 2: Very different
        ]

    # Create sampler
    sampler = PrecomputedMultiTargetSampler(logp_lookup=logp_lookup, n_policies=3)

    # Prepare logs
    logs = []
    for i in range(n_samples):
        logs.append(
            {
                "context": contexts[i],
                "response": responses[i],
                "reward": rewards[i],
                "logp": logp_behavior[i],
            }
        )

    # Test with different estimators
    estimators_to_test = ["IPS", "SNIPS", "CalibratedIPS"]

    console.print("\n[bold]Testing estimators:[/bold]")

    for est_name in estimators_to_test:
        estimator = get_estimator(est_name, sampler=sampler)
        estimator.fit(logs)
        result = estimator.estimate()

        clone_estimate = result.v_hat[0]  # Policy 0 is the clone
        difference = abs(clone_estimate - empirical_mean)

        console.print(f"\n{est_name}:")
        console.print(f"   Clone policy estimate: {clone_estimate:.6f}")
        console.print(f"   Empirical mean: {empirical_mean:.6f}")
        console.print(f"   Difference: {difference:.6f}")

        # Check if close (within numerical precision)
        if difference < 1e-10:
            console.print(f"   ✅ [green]EXACT match (difference < 1e-10)[/green]")
        elif difference < 1e-6:
            console.print(f"   ✅ [green]Very close match (difference < 1e-6)[/green]")
        elif difference < 0.01:
            console.print(f"   ⚠️  [yellow]Close match (difference < 0.01)[/yellow]")
        else:
            console.print(f"   ❌ [red]Large difference![/red]")

    # Also test doubly-robust estimators
    console.print("\n[bold]Testing doubly-robust estimators:[/bold]")
    console.print(
        "[dim]Note: DR estimators use outcome models, so may differ slightly[/dim]"
    )

    for est_name in ["DRCPO", "MRDR"]:
        try:
            estimator = get_estimator(est_name, sampler=sampler)
            estimator.fit(logs)
            result = estimator.estimate()

            clone_estimate = result.v_hat[0]
            difference = abs(clone_estimate - empirical_mean)

            console.print(f"\n{est_name}:")
            console.print(f"   Clone policy estimate: {clone_estimate:.6f}")
            console.print(f"   Difference from empirical: {difference:.6f}")
        except Exception as e:
            console.print(f"\n{est_name}: [red]Error - {str(e)}[/red]")

    # Test with extreme weights to verify numerical stability
    console.print("\n[bold]Testing numerical stability with extreme log probs:[/bold]")

    # Create data with very large log prob differences
    extreme_lookup = {}
    extreme_logp_behavior = []

    for i in range(n_samples):
        # Some samples have very negative log probs
        if i % 10 == 0:
            behavior_logp = -100.0
        else:
            behavior_logp = -10.0

        extreme_logp_behavior.append(behavior_logp)
        extreme_lookup[(contexts[i], responses[i])] = [
            behavior_logp,  # Clone
            behavior_logp + 50,  # Much higher probability
            behavior_logp - 50,  # Much lower probability
        ]

    extreme_sampler = PrecomputedMultiTargetSampler(
        logp_lookup=extreme_lookup, n_policies=3
    )

    # Update logs with extreme behavior log probs
    extreme_logs = []
    for i in range(n_samples):
        extreme_logs.append(
            {
                "context": contexts[i],
                "response": responses[i],
                "reward": rewards[i],
                "logp": extreme_logp_behavior[i],
            }
        )

    # Test IPS with extreme values
    ips_extreme = get_estimator("IPS", sampler=extreme_sampler)
    ips_extreme.fit(extreme_logs)
    result_extreme = ips_extreme.estimate()

    console.print(f"\nIPS with extreme log probs:")
    console.print(f"   Clone estimate: {result_extreme.v_hat[0]:.6f}")
    console.print(f"   Empirical mean: {empirical_mean:.6f}")
    console.print(f"   Difference: {abs(result_extreme.v_hat[0] - empirical_mean):.6f}")


if __name__ == "__main__":
    test_clone_policy_estimation()
