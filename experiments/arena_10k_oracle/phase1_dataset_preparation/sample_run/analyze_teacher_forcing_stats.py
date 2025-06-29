#!/usr/bin/env python3
"""
Analyze teacher forcing statistics from the sample run.
This helps verify the fix is working correctly.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt

console = Console()


class TeacherForcingAnalyzer:
    """Analyze teacher forcing results in detail."""

    def __init__(self, sample_mode: bool = True):
        self.sample_mode = sample_mode
        self.data_dir = Path(__file__).parent.parent / "data"
        if sample_mode:
            self.data_dir = self.data_dir / "sample_1pct"

        self.stats = {
            "total_computations": 0,
            "successful": 0,
            "failed": 0,
            "zero_values": 0,
            "suspicious_zeros": [],
            "log_prob_distribution": [],
            "response_lengths": [],
            "by_policy": defaultdict(
                lambda: {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "zeros": 0,
                    "log_probs": [],
                }
            ),
        }

    def load_teacher_forcing_data(self) -> List[Dict[str, Any]]:
        """Load teacher forcing results."""
        filename = (
            "p0_with_target_logps_sample.jsonl"
            if self.sample_mode
            else "p0_with_target_logps.jsonl"
        )
        filepath = self.data_dir / filename

        if not filepath.exists():
            console.print(f"[red]Error: {filepath} not found[/red]")
            return []

        data = []
        with open(filepath) as f:
            for line in f:
                data.append(json.loads(line))

        return data

    def analyze_data(self, data: List[Dict[str, Any]]) -> None:
        """Analyze teacher forcing results."""
        for item in data:
            prompt_id = item.get("prompt_id", "unknown")
            response = item.get("response", "")
            response_len = len(response)

            if "target_logps" not in item:
                continue

            self.stats["response_lengths"].append(response_len)

            for policy, logp in item["target_logps"].items():
                self.stats["total_computations"] += 1
                policy_stats = self.stats["by_policy"][policy]
                policy_stats["total"] += 1

                if logp is None:
                    self.stats["failed"] += 1
                    policy_stats["failed"] += 1
                else:
                    self.stats["successful"] += 1
                    policy_stats["successful"] += 1
                    self.stats["log_prob_distribution"].append(logp)
                    policy_stats["log_probs"].append(logp)

                    # Check for suspicious zeros
                    if logp == 0.0:
                        self.stats["zero_values"] += 1
                        policy_stats["zeros"] += 1

                        if response:  # Non-empty response with 0.0
                            self.stats["suspicious_zeros"].append(
                                {
                                    "prompt_id": prompt_id,
                                    "policy": policy,
                                    "response": response,
                                    "response_length": response_len,
                                }
                            )

    def generate_report(self) -> None:
        """Generate comprehensive analysis report."""
        console.print(
            Panel.fit("Teacher Forcing Analysis Report", title="📊 Detailed Statistics")
        )

        # Overall statistics
        console.print("\n🔍 Overall Statistics:")
        success_rate = (
            (self.stats["successful"] / self.stats["total_computations"] * 100)
            if self.stats["total_computations"] > 0
            else 0
        )

        table = Table(title="Summary Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow", justify="right")
        table.add_column("Percentage", style="green", justify="right")

        table.add_row(
            "Total Computations", f"{self.stats['total_computations']:,}", "-"
        )
        table.add_row(
            "Successful", f"{self.stats['successful']:,}", f"{success_rate:.1f}%"
        )
        table.add_row(
            "Failed (null)", f"{self.stats['failed']:,}", f"{100 - success_rate:.1f}%"
        )
        table.add_row(
            "Zero Values",
            f"{self.stats['zero_values']:,}",
            f"{self.stats['zero_values'] / self.stats['total_computations'] * 100:.1f}%",
        )
        table.add_row(
            "Suspicious Zeros",
            f"{len(self.stats['suspicious_zeros'])}",
            f"{len(self.stats['suspicious_zeros']) / self.stats['total_computations'] * 100:.1f}%",
            style="bold red" if self.stats["suspicious_zeros"] else "green",
        )

        console.print(table)

        # Policy-specific statistics
        console.print("\n📋 By Policy:")
        policy_table = Table(title="Policy Breakdown")
        policy_table.add_column("Policy", style="cyan")
        policy_table.add_column("Total", justify="right")
        policy_table.add_column("Success Rate", justify="right")
        policy_table.add_column("Zeros", justify="right")
        policy_table.add_column("Avg Log Prob", justify="right")

        for policy, stats in self.stats["by_policy"].items():
            if stats["total"] > 0:
                success_rate = stats["successful"] / stats["total"] * 100
                avg_logp = np.mean(stats["log_probs"]) if stats["log_probs"] else 0

                policy_table.add_row(
                    policy,
                    f"{stats['total']:,}",
                    f"{success_rate:.1f}%",
                    f"{stats['zeros']:,}",
                    f"{avg_logp:.2f}",
                )

        console.print(policy_table)

        # Log probability distribution
        if self.stats["log_prob_distribution"]:
            logps = np.array(self.stats["log_prob_distribution"])
            console.print("\n📈 Log Probability Distribution:")
            console.print(f"  Min: {logps.min():.2f}")
            console.print(f"  Max: {logps.max():.2f}")
            console.print(f"  Mean: {logps.mean():.2f}")
            console.print(f"  Median: {np.median(logps):.2f}")
            console.print(f"  Std Dev: {logps.std():.2f}")

            # Check for reasonable ranges
            reasonable_min = -100
            reasonable_max = 0
            out_of_range = np.sum((logps < reasonable_min) | (logps > reasonable_max))
            if out_of_range > 0:
                console.print(
                    f"  [yellow]⚠️  {out_of_range} values outside reasonable range [{reasonable_min}, {reasonable_max}][/yellow]"
                )

        # Response length analysis
        if self.stats["response_lengths"]:
            lengths = np.array(self.stats["response_lengths"])
            console.print("\n📏 Response Length Statistics:")
            console.print(f"  Min: {lengths.min()} chars")
            console.print(f"  Max: {lengths.max()} chars")
            console.print(f"  Mean: {lengths.mean():.1f} chars")
            console.print(f"  Median: {np.median(lengths):.1f} chars")

        # Critical issues
        if self.stats["suspicious_zeros"]:
            console.print("\n[red]❌ CRITICAL: Suspicious Zero Values Found![/red]")
            console.print(
                f"Found {len(self.stats['suspicious_zeros'])} responses with 0.0 log probability:"
            )

            for i, item in enumerate(
                self.stats["suspicious_zeros"][:5]
            ):  # Show first 5
                console.print(f"\n  {i+1}. Prompt ID: {item['prompt_id']}")
                console.print(f"     Policy: {item['policy']}")
                console.print(f"     Response length: {item['response_length']} chars")
                console.print(f"     Response preview: '{item['response'][:100]}...'")

            if len(self.stats["suspicious_zeros"]) > 5:
                console.print(
                    f"\n  ... and {len(self.stats['suspicious_zeros']) - 5} more"
                )
        else:
            console.print("\n[green]✅ No suspicious zero values found![/green]")

    def plot_distribution(self) -> None:
        """Create distribution plots (optional)."""
        if not self.stats["log_prob_distribution"]:
            return

        try:
            plt.figure(figsize=(12, 6))

            # Log probability histogram
            plt.subplot(1, 2, 1)
            logps = self.stats["log_prob_distribution"]
            plt.hist(logps, bins=50, edgecolor="black", alpha=0.7)
            plt.xlabel("Log Probability")
            plt.ylabel("Count")
            plt.title("Teacher Forcing Log Probability Distribution")
            plt.axvline(x=0, color="red", linestyle="--", label="Zero line")
            if min(logps) < -50:
                plt.xlim(-50, 0)
            plt.legend()

            # Response length vs log prob scatter
            plt.subplot(1, 2, 2)
            if self.stats["response_lengths"] and len(
                self.stats["response_lengths"]
            ) == len(logps):
                plt.scatter(self.stats["response_lengths"], logps, alpha=0.5)
                plt.xlabel("Response Length (chars)")
                plt.ylabel("Log Probability")
                plt.title("Response Length vs Log Probability")

                # Highlight any zeros
                zero_indices = [i for i, lp in enumerate(logps) if lp == 0.0]
                if zero_indices:
                    zero_lengths = [
                        self.stats["response_lengths"][i] for i in zero_indices
                    ]
                    zero_logps = [logps[i] for i in zero_indices]
                    plt.scatter(
                        zero_lengths,
                        zero_logps,
                        color="red",
                        s=100,
                        marker="x",
                        label="Zero values",
                    )
                    plt.legend()

            plt.tight_layout()
            plot_path = self.data_dir / "teacher_forcing_analysis.png"
            plt.savefig(plot_path)
            console.print(f"\n📊 Saved distribution plots to: {plot_path}")
        except Exception as e:
            console.print(f"[yellow]Could not create plots: {e}[/yellow]")

    def save_detailed_report(self) -> None:
        """Save detailed report to file."""
        report_path = self.data_dir / "teacher_forcing_analysis.json"

        # Prepare serializable stats
        report_data = {
            "summary": {
                "total_computations": self.stats["total_computations"],
                "successful": self.stats["successful"],
                "failed": self.stats["failed"],
                "zero_values": self.stats["zero_values"],
                "suspicious_zero_count": len(self.stats["suspicious_zeros"]),
                "success_rate": (
                    (self.stats["successful"] / self.stats["total_computations"] * 100)
                    if self.stats["total_computations"] > 0
                    else 0
                ),
            },
            "by_policy": dict(self.stats["by_policy"]),
            "suspicious_zeros": self.stats["suspicious_zeros"][
                :20
            ],  # Limit to 20 examples
            "log_prob_stats": {
                "min": (
                    min(self.stats["log_prob_distribution"])
                    if self.stats["log_prob_distribution"]
                    else None
                ),
                "max": (
                    max(self.stats["log_prob_distribution"])
                    if self.stats["log_prob_distribution"]
                    else None
                ),
                "mean": (
                    np.mean(self.stats["log_prob_distribution"])
                    if self.stats["log_prob_distribution"]
                    else None
                ),
                "median": (
                    np.median(self.stats["log_prob_distribution"])
                    if self.stats["log_prob_distribution"]
                    else None
                ),
            },
        }

        with open(report_path, "w") as f:
            json.dumps(report_data, f, indent=2)

        console.print(f"\n💾 Saved detailed report to: {report_path}")

    def run(self) -> bool:
        """Run the analysis and return success status."""
        # Load data
        data = self.load_teacher_forcing_data()
        if not data:
            return False

        console.print(f"📊 Analyzing {len(data)} samples...")

        # Analyze
        self.analyze_data(data)

        # Generate report
        self.generate_report()

        # Create plots
        self.plot_distribution()

        # Save detailed report
        self.save_detailed_report()

        # Return success based on suspicious zeros
        return len(self.stats["suspicious_zeros"]) == 0


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze teacher forcing statistics")
    parser.add_argument(
        "--full", action="store_true", help="Analyze full dataset (default: sample)"
    )
    args = parser.parse_args()

    analyzer = TeacherForcingAnalyzer(sample_mode=not args.full)
    success = analyzer.run()

    if not success:
        console.print("\n[red]⚠️  Analysis found critical issues![/red]")
        sys.exit(1)
    else:
        console.print("\n[green]✅ Analysis passed all checks![/green]")
        sys.exit(0)


if __name__ == "__main__":
    import sys

    main()
