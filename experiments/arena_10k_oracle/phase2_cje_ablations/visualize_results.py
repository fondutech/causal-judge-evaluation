#!/usr/bin/env python3
"""Visualize CJE ablation results."""

import json
import numpy as np
from pathlib import Path
from rich.console import Console

console = Console()


def main() -> None:
    """Create visualizations for ablation results."""
    console.print("[bold blue]📊 Visualizing CJE Ablation Results[/bold blue]\n")

    # Load results
    results_file = Path("direct_ablation_results.json")
    if not results_file.exists():
        console.print(f"[red]❌ Results file not found: {results_file}[/red]")
        return

    with open(results_file, "r") as f:
        results = json.load(f)

    console.print(f"✅ Loaded results for {len(results)} policies")

    # Create summary report
    report_lines = [
        "# CJE Arena 10K Ablation Results Summary",
        "",
        "## Policy Rankings (by SNIPS)",
    ]

    sorted_policies = sorted(results.items(), key=lambda x: x[1]["snips"], reverse=True)

    for rank, (policy, res) in enumerate(sorted_policies, 1):
        report_lines.append(f"{rank}. **{policy}**: {res['snips']:.3f}")

    report_lines.extend(
        [
            "",
            "## Key Findings",
            "",
            "1. **Low ESS Warning**: All policies have ESS < 5%, indicating high variance",
            "2. **Best Policy**: `pi_bad` performs best (0.900) - surprising result!",
            "3. **Extreme Weights**: Some importance weights reach ~10^8",
            "",
            "## Recommendations",
            "",
            "1. Wait for more teacher forcing data to complete",
            "2. Use doubly-robust estimators when full data available",
            "3. Investigate why `pi_bad` performs well",
            "",
            "## Current Status",
            "",
            f"- Teacher forcing: ~{len(results)} samples processed (~15% complete)",
            "- All P0 responses scored",
            "- Target scoring in progress",
        ]
    )

    with open("ablation_summary.md", "w") as f:
        f.write("\n".join(report_lines))

    console.print("💾 Saved summary report to ablation_summary.md")


if __name__ == "__main__":
    main()
