#!/usr/bin/env python3
"""Analyze why pi_bad has high scores."""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from rich.console import Console
from rich.table import Table

console = Console()


def analyze_log_probs():
    """Analyze the log probability distributions."""
    tf_file = Path("../data/p0_with_target_logps.checkpoint.jsonl")

    # Collect log probs
    p0_logps = []
    target_logps = defaultdict(list)
    log_diffs = defaultdict(list)

    with open(tf_file, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                p0_logp = item["total_logprob"]
                p0_logps.append(p0_logp)

                for policy_name, target_logp in item.get("target_logps", {}).items():
                    target_logps[policy_name].append(target_logp)
                    # Log difference = log P(response | π_target) - log P(response | π_0)
                    log_diff = target_logp - p0_logp
                    log_diffs[policy_name].append(log_diff)

            except json.JSONDecodeError:
                continue

    console.print(f"\n📊 Analyzed {len(p0_logps)} samples")

    # Create summary table
    table = Table(title="Log Probability Analysis")
    table.add_column("Policy", style="cyan")
    table.add_column("Mean Target LogP", style="green")
    table.add_column("Mean Log Diff", style="yellow")
    table.add_column("% Better than P0", style="magenta")
    table.add_column("Extreme Values", style="red")

    for policy_name in sorted(target_logps.keys()):
        logps = np.array(target_logps[policy_name])
        diffs = np.array(log_diffs[policy_name])

        # Count how often target is better than P0
        better_count = np.sum(diffs > 0)
        better_pct = (better_count / len(diffs)) * 100

        # Find extreme values
        extreme_high = np.sum(diffs > 10)  # exp(10) ≈ 22,000x more likely
        extreme_low = np.sum(diffs < -10)  # exp(-10) ≈ 0.000045x as likely

        table.add_row(
            policy_name,
            f"{np.mean(logps):.2f}",
            f"{np.mean(diffs):.2f}",
            f"{better_pct:.1f}%",
            f"↑{extreme_high} ↓{extreme_low}",
        )

    console.print(table)

    # Look for suspicious patterns
    console.print("\n🔍 Checking for suspicious patterns...")

    # Check for zero log probs
    for policy_name, logps in target_logps.items():
        zero_count = sum(1 for lp in logps if lp == 0.0)
        if zero_count > 0:
            console.print(
                f"[red]⚠️  {policy_name}: Found {zero_count} samples with logp=0.0 (probability=1.0)![/red]"
            )

    # Compare pi_bad vs others
    if "pi_bad" in log_diffs and "pi_bigger_model" in log_diffs:
        bad_diffs = np.array(log_diffs["pi_bad"])
        bigger_diffs = np.array(log_diffs["pi_bigger_model"])

        # Find cases where pi_bad >> pi_bigger_model
        suspicious = np.where((bad_diffs - bigger_diffs) > 20)[0]
        if len(suspicious) > 0:
            console.print(
                f"\n[red]🚨 Found {len(suspicious)} cases where pi_bad is >10^20x better than pi_bigger_model![/red]"
            )

    return target_logps, log_diffs


def analyze_scores():
    """Analyze judge scores for different responses."""
    # Load P0 scores
    score_file = Path("../data/p0_scored_deterministic.jsonl")
    p0_scores = {}

    if score_file.exists():
        with open(score_file, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    prompt_id = item.get("prompt_id")
                    if prompt_id:
                        p0_scores[prompt_id] = item.get("judge_score", {}).get(
                            "mean", 0.5
                        )
                except:
                    continue

    console.print(f"\n📈 Loaded {len(p0_scores)} P0 scores")
    console.print(f"   Mean P0 score: {np.mean(list(p0_scores.values())):.3f}")

    # TODO: Load target scores when available
    # This would help us understand if pi_bad responses actually get high scores


def check_specific_examples():
    """Look at specific examples to understand what's happening."""
    tf_file = Path("../data/p0_with_target_logps.checkpoint.jsonl")

    console.print("\n📝 Examining specific examples where pi_bad wins...")

    examples = []
    with open(tf_file, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                logps = item.get("target_logps", {})

                if "pi_bad" in logps and "pi_bigger_model" in logps:
                    bad_logp = logps["pi_bad"]
                    bigger_logp = logps["pi_bigger_model"]

                    # Find cases where pi_bad significantly beats pi_bigger_model
                    if bad_logp - bigger_logp > 20:  # pi_bad is >10^20x more likely
                        examples.append(
                            {
                                "prompt_id": item["prompt_id"],
                                "prompt": item["prompt"][:100] + "...",
                                "response_preview": item["response"][:100] + "...",
                                "p0_logp": item["total_logprob"],
                                "bad_logp": bad_logp,
                                "bigger_logp": bigger_logp,
                                "diff": bad_logp - bigger_logp,
                            }
                        )

                        if len(examples) >= 5:
                            break
            except:
                continue

    # Display examples
    for i, ex in enumerate(examples, 1):
        console.print(f"\n[yellow]Example {i}:[/yellow]")
        console.print(f"  Prompt: {ex['prompt']}")
        console.print(f"  Response: {ex['response_preview']}")
        console.print(f"  P0 logp: {ex['p0_logp']:.2f}")
        console.print(f"  pi_bad logp: {ex['bad_logp']:.2f}")
        console.print(f"  pi_bigger logp: {ex['bigger_logp']:.2f}")
        console.print(
            f"  [red]Difference: {ex['diff']:.2f} (pi_bad is {np.exp(ex['diff']):.0e}x more likely!)[/red]"
        )


def main():
    """Run the analysis."""
    console.print("[bold blue]🔍 Analyzing Why pi_bad Has High Estimates[/bold blue]")

    # Analyze log probabilities
    target_logps, log_diffs = analyze_log_probs()

    # Analyze scores
    analyze_scores()

    # Look at specific examples
    check_specific_examples()

    console.print("\n[bold]🤔 Possible Explanations:[/bold]")
    console.print("1. **API/Implementation Bug**: Zero log probs suggest an error")
    console.print(
        "2. **Prompt Engineering**: The 'unhelpful' system prompt might trigger unexpected behavior"
    )
    console.print(
        "3. **Model Quirk**: The model might assign high probability to certain 'bad' patterns"
    )
    console.print("4. **Data Issue**: Some responses might be corrupted or truncated")


if __name__ == "__main__":
    main()
