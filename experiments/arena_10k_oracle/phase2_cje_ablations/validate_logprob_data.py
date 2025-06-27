#!/usr/bin/env python3
"""
Validate log probability data before running any analysis.

This ensures we never use invalid log probabilities that would corrupt importance weights.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from rich.console import Console
from rich.table import Table

console = Console()


def validate_logprob_data(data: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data into valid and invalid samples.

    Returns:
        valid_data: Samples with all valid log probabilities
        invalid_data: Samples with any invalid log probabilities
    """
    valid_data = []
    invalid_data = []

    for item in data:
        is_valid = True
        issues = []

        # Check P0 log prob
        p0_logp = item.get("total_logprob")
        if p0_logp is None:
            is_valid = False
            issues.append("P0 logprob missing")
        elif not isinstance(p0_logp, (int, float)):
            is_valid = False
            issues.append(f"P0 logprob invalid type: {type(p0_logp)}")
        elif not np.isfinite(p0_logp):
            is_valid = False
            issues.append(f"P0 logprob not finite: {p0_logp}")
        elif p0_logp > 0:
            is_valid = False
            issues.append(f"P0 logprob positive: {p0_logp}")
        elif p0_logp == 0.0:
            is_valid = False
            issues.append("P0 logprob is exactly 0.0 (suspicious)")
        elif p0_logp == -50.0:
            is_valid = False
            issues.append("P0 logprob is -50.0 (replacement value)")

        # Check target log probs
        for policy, logp in item.get("target_logps", {}).items():
            if logp is None:
                is_valid = False
                issues.append(f"{policy} logprob missing")
            elif not isinstance(logp, (int, float)):
                is_valid = False
                issues.append(f"{policy} logprob invalid type: {type(logp)}")
            elif not np.isfinite(logp):
                is_valid = False
                issues.append(f"{policy} logprob not finite: {logp}")
            elif logp > 0:
                is_valid = False
                issues.append(f"{policy} logprob positive: {logp}")
            elif logp == 0.0:
                is_valid = False
                issues.append(f"{policy} logprob is exactly 0.0 (suspicious)")

        if is_valid:
            valid_data.append(item)
        else:
            item["validation_issues"] = issues
            invalid_data.append(item)

    return valid_data, invalid_data


def analyze_validation_results(
    valid_data: List[Dict], invalid_data: List[Dict]
) -> None:
    """Display detailed analysis of validation results."""

    total = len(valid_data) + len(invalid_data)

    console.print(f"\n[bold]Validation Summary:[/bold]")
    console.print(f"Total samples: {total}")
    console.print(
        f"Valid samples: {len(valid_data)} ({len(valid_data)/total*100:.1f}%)"
    )
    console.print(
        f"Invalid samples: {len(invalid_data)} ({len(invalid_data)/total*100:.1f}%)"
    )

    if invalid_data:
        # Analyze issue types
        issue_counts = defaultdict(int)
        for item in invalid_data:
            for issue in item.get("validation_issues", []):
                issue_counts[issue] += 1

        # Display issue breakdown
        console.print("\n[bold]Issue Breakdown:[/bold]")
        table = Table(title="Validation Issues")
        table.add_column("Issue Type", style="cyan")
        table.add_column("Count", style="red")
        table.add_column("Percentage", style="yellow")

        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            percentage = count / len(invalid_data) * 100
            table.add_row(issue, str(count), f"{percentage:.1f}%")

        console.print(table)

        # Show sample invalid entries
        console.print("\n[bold]Sample Invalid Entries:[/bold]")
        for i, item in enumerate(invalid_data[:3]):
            console.print(f"\n{i+1}. ID: {item.get('prompt_id', 'unknown')}")
            console.print(f"   Issues: {', '.join(item['validation_issues'])}")
            console.print(f"   Response length: {len(item.get('response', ''))}")


def main():
    """Validate the current log probability data."""
    console.print("[bold blue]🔍 Validating Log Probability Data[/bold blue]\n")

    # Check multiple data files
    data_files = [
        Path("../data/p0_with_target_logps.jsonl"),
        Path("../data/p0_with_target_logps_fixed.jsonl"),
        Path("../data/p0_with_target_logps_valid_only.jsonl"),
    ]

    for data_file in data_files:
        if not data_file.exists():
            continue

        console.print(f"\n[cyan]Checking: {data_file}[/cyan]")

        # Load data
        data = []
        with open(data_file, "r") as f:
            for line in f:
                data.append(json.loads(line))

        # Validate
        valid_data, invalid_data = validate_logprob_data(data)

        # Analyze results
        analyze_validation_results(valid_data, invalid_data)

        # Save validation results
        if invalid_data:
            output_file = data_file.parent / f"{data_file.stem}_validation_issues.json"
            with open(output_file, "w") as f:
                json.dump(
                    {
                        "file": str(data_file),
                        "total_invalid": len(invalid_data),
                        "invalid_samples": invalid_data[:10],  # First 10 for inspection
                    },
                    f,
                    indent=2,
                )
            console.print(f"\n💾 Validation issues saved to: {output_file}")

    console.print("\n[bold]Recommendations:[/bold]")
    console.print("1. Use only validated data for analysis")
    console.print("2. Re-score any samples with invalid log probabilities")
    console.print("3. Never use replacement values like 0.0 or -50.0")
    console.print("4. Fail explicitly when log prob computation fails")


if __name__ == "__main__":
    main()
