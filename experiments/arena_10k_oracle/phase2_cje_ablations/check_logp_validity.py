#!/usr/bin/env python3
"""Check validity of log probability computations."""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.table import Table

console = Console()


def analyze_logp_distributions():
    """Analyze the distribution of log probabilities to check for validity."""

    # Load data
    tf_file = Path("../data/p0_with_target_logps_fixed.jsonl")

    # Store all logps
    p0_logps = []
    target_logps = defaultdict(list)

    # Suspicious patterns
    suspicious_p0 = []
    suspicious_targets = defaultdict(list)

    with open(tf_file, "r") as f:
        for line in f:
            item = json.loads(line)

            # P0 log prob
            p0_logp = item["total_logprob"]
            p0_logps.append(p0_logp)

            # Check for suspicious P0 values
            if p0_logp == -50.0 or p0_logp == 0.0 or p0_logp > -1.0:
                suspicious_p0.append(
                    {
                        "prompt_id": item["prompt_id"],
                        "logp": p0_logp,
                        "response_len": len(item["response"]),
                    }
                )

            # Target log probs
            for policy, logp in item.get("target_logps", {}).items():
                target_logps[policy].append(logp)

                # Check for suspicious target values
                if logp == 0.0 or logp > -0.1:
                    suspicious_targets[policy].append(
                        {
                            "prompt_id": item["prompt_id"],
                            "logp": logp,
                            "response_len": len(item["response"]),
                        }
                    )

    # Display P0 statistics
    console.print("\n[bold]P0 Log Probability Statistics[/bold]")
    console.print(f"Total samples: {len(p0_logps)}")
    console.print(f"Mean: {np.mean(p0_logps):.2f}")
    console.print(f"Median: {np.median(p0_logps):.2f}")
    console.print(f"Std: {np.std(p0_logps):.2f}")
    console.print(f"Min: {np.min(p0_logps):.2f}")
    console.print(f"Max: {np.max(p0_logps):.2f}")
    console.print(f"Count of -50.0: {sum(1 for x in p0_logps if x == -50.0)}")
    console.print(f"Count of 0.0: {sum(1 for x in p0_logps if x == 0.0)}")

    # Display target statistics
    console.print("\n[bold]Target Policy Log Probability Statistics[/bold]")
    table = Table(title="Target Policy Statistics")
    table.add_column("Policy", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Median", style="green")
    table.add_column("Std", style="yellow")
    table.add_column("Min", style="red")
    table.add_column("Max", style="red")
    table.add_column("Zeros", style="magenta")

    for policy in sorted(target_logps.keys()):
        logps = target_logps[policy]
        table.add_row(
            policy,
            f"{np.mean(logps):.2f}",
            f"{np.median(logps):.2f}",
            f"{np.std(logps):.2f}",
            f"{np.min(logps):.2f}",
            f"{np.max(logps):.2f}",
            str(sum(1 for x in logps if x == 0.0)),
        )

    console.print(table)

    # Show suspicious cases
    console.print("\n[bold red]Suspicious P0 Cases[/bold red]")
    console.print(f"Found {len(suspicious_p0)} suspicious P0 log probs")
    if suspicious_p0:
        for case in suspicious_p0[:5]:
            console.print(
                f"  {case['prompt_id']}: logp={case['logp']}, len={case['response_len']}"
            )

    console.print("\n[bold red]Suspicious Target Cases[/bold red]")
    for policy, cases in suspicious_targets.items():
        if cases:
            console.print(f"\n{policy}: {len(cases)} suspicious cases")
            for case in cases[:3]:
                console.print(
                    f"  {case['prompt_id']}: logp={case['logp']}, len={case['response_len']}"
                )


def check_template_consistency():
    """Check if templates are being applied consistently."""

    console.print("\n[bold]Checking Template Consistency[/bold]")

    # The issue: P0 uses llama4 template, but target policies might not
    # Also, target policies have different system prompts and user templates

    console.print("\n[yellow]P0 Configuration:[/yellow]")
    console.print("- Template format: llama4")
    console.print("- System prompt: None")
    console.print("- User template: {context}")

    console.print("\n[yellow]Target Policy Configurations:[/yellow]")
    console.print("pi_cot:")
    console.print(
        "  - System: 'You are a helpful assistant. Always think step by step...'"
    )
    console.print(
        "  - User template: 'Let's work through this step by step...\\n\\n{context}\\n\\n...'"
    )
    console.print("pi_bigger_model:")
    console.print("  - System: None")
    console.print("  - User template: {context}")
    console.print("pi_bad:")
    console.print("  - System: 'You are learning to be helpful. Keep responses brief.'")
    console.print("  - User template: '{context}\\n\\n(Reply briefly):'")

    console.print("\n[bold red]⚠️  Critical Issues:[/bold red]")
    console.print("1. Target policies don't specify completions_template_format")
    console.print("2. Different system prompts mean different tokenization")
    console.print(
        "3. Different user templates mean P0 response doesn't start where expected"
    )
    console.print("4. This makes log probabilities incomparable!")


def analyze_weight_anomalies():
    """Look for patterns in extreme weights."""

    tf_file = Path("../data/p0_with_target_logps_fixed.jsonl")

    # Count anomalies
    weight_anomalies = defaultdict(int)

    with open(tf_file, "r") as f:
        for line in f:
            item = json.loads(line)
            p0_logp = item["total_logprob"]

            for policy, target_logp in item.get("target_logps", {}).items():
                # Calculate weight
                log_diff = target_logp - p0_logp

                # Categorize anomalies
                if log_diff > 20:
                    weight_anomalies[f"{policy}_extreme_high"] += 1
                elif log_diff < -20:
                    weight_anomalies[f"{policy}_extreme_low"] += 1
                elif abs(target_logp - p0_logp) < 0.01:
                    weight_anomalies[f"{policy}_suspiciously_close"] += 1

                # Check if target is much better than P0
                if target_logp > p0_logp + 10:
                    weight_anomalies[f"{policy}_much_better_than_p0"] += 1

    console.print("\n[bold]Weight Anomaly Analysis[/bold]")
    table = Table(title="Anomalous Weight Patterns")
    table.add_column("Pattern", style="cyan")
    table.add_column("Count", style="red")

    for pattern, count in sorted(weight_anomalies.items()):
        if count > 0:
            table.add_row(pattern, str(count))

    console.print(table)


def main():
    """Run all validity checks."""
    console.print("[bold blue]🔍 Checking Log Probability Validity[/bold blue]")

    # Analyze distributions
    analyze_logp_distributions()

    # Check template consistency
    check_template_consistency()

    # Analyze weight anomalies
    analyze_weight_anomalies()

    console.print("\n[bold]📋 Recommendations:[/bold]")
    console.print("1. Re-run teacher forcing with consistent templates")
    console.print("2. Use same system prompt and user template for all policies")
    console.print("3. Validate that P0 log probs are computed correctly")
    console.print("4. Consider using raw completions API for all policies")


if __name__ == "__main__":
    main()
