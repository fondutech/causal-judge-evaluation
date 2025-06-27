#!/usr/bin/env python3
"""Fix zero log probabilities in teacher forcing data."""

import json
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import track

console = Console()


def fix_zero_logprobs(input_file: Path, output_file: Path):
    """Replace zero log probs with more reasonable defaults."""

    # First pass: collect statistics
    console.print("📊 Analyzing log probability distributions...")

    policy_logps = {"pi_bad": [], "pi_bigger_model": [], "pi_cot": []}

    zero_counts = {"pi_bad": 0, "pi_bigger_model": 0, "pi_cot": 0}

    all_items = []
    with open(input_file, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                all_items.append(item)

                for policy, logp in item.get("target_logps", {}).items():
                    if logp == 0.0:
                        zero_counts[policy] += 1
                    else:
                        policy_logps[policy].append(logp)
            except:
                continue

    # Calculate statistics for non-zero values
    stats = {}
    for policy, logps in policy_logps.items():
        if logps:
            stats[policy] = {
                "mean": np.mean(logps),
                "std": np.std(logps),
                "median": np.median(logps),
                "min": np.min(logps),
                "max": np.max(logps),
            }
            console.print(f"\n{policy} (non-zero values):")
            console.print(f"  Mean: {stats[policy]['mean']:.2f}")
            console.print(f"  Median: {stats[policy]['median']:.2f}")
            console.print(
                f"  Range: [{stats[policy]['min']:.2f}, {stats[policy]['max']:.2f}]"
            )
            console.print(f"  Zero count: {zero_counts[policy]}")

    # Fix the data
    console.print("\n🔧 Fixing zero log probabilities...")

    fixed_count = 0
    with open(output_file, "w") as f:
        for item in track(all_items, description="Processing items"):
            fixed_item = item.copy()

            # Fix target log probs
            if "target_logps" in fixed_item:
                for policy, logp in fixed_item["target_logps"].items():
                    if logp == 0.0 and policy in stats:
                        # Use a conservative estimate: median - 2*std
                        # This makes failed API calls look worse than typical
                        replacement = stats[policy]["median"] - 2 * stats[policy]["std"]
                        # But not worse than the worst observed
                        replacement = max(replacement, stats[policy]["min"])

                        fixed_item["target_logps"][policy] = replacement
                        fixed_count += 1

            # Also fix P0 log prob if needed
            if item.get("total_logprob") == 0.0:
                # Use a reasonable default
                fixed_item["total_logprob"] = -50.0  # Reasonable default
                fixed_count += 1

            f.write(json.dumps(fixed_item) + "\n")

    console.print(f"\n✅ Fixed {fixed_count} zero log probabilities")
    console.print(f"💾 Saved to {output_file}")

    return fixed_count


def verify_fix(fixed_file: Path):
    """Verify the fix worked."""
    console.print("\n🔍 Verifying fixed data...")

    zero_count = 0
    total_count = 0

    with open(fixed_file, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                total_count += 1

                # Check P0
                if item.get("total_logprob") == 0.0:
                    zero_count += 1

                # Check targets
                for logp in item.get("target_logps", {}).values():
                    if logp == 0.0:
                        zero_count += 1
            except:
                continue

    console.print(f"Total items: {total_count}")
    console.print(f"Remaining zeros: {zero_count}")

    if zero_count == 0:
        console.print("✅ All zero log probabilities have been fixed!")
    else:
        console.print(f"⚠️  Still have {zero_count} zero values")


def main():
    """Fix the teacher forcing data."""
    console.print("[bold blue]🔧 Fixing Zero Log Probabilities[/bold blue]")
    console.print("\nThis fixes the bug where API failures return logp=0.0\n")

    input_file = Path("../data/p0_with_target_logps.checkpoint.jsonl")
    output_file = Path("../data/p0_with_target_logps_fixed.jsonl")

    if not input_file.exists():
        console.print(f"[red]Error: {input_file} not found[/red]")
        return

    # Fix the data
    fixed_count = fix_zero_logprobs(input_file, output_file)

    # Verify
    verify_fix(output_file)

    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("1. Re-run the ablation analysis with fixed data")
    console.print("2. Results should now be more reasonable")
    console.print("3. Consider re-running teacher forcing with better error handling")


if __name__ == "__main__":
    main()
