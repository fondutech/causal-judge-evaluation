#!/usr/bin/env python3
"""
Monitor the progress of the 1% sample run in real-time.
Shows API calls, costs, and validates outputs as they're generated.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import threading

console = Console()


class SampleRunMonitor:
    """Real-time monitoring of sample run progress."""

    def __init__(self):
        self.sample_dir = Path(__file__).parent.parent / "data" / "sample_1pct"
        self.running = True
        self.stats: Dict[str, Any] = {
            "start_time": time.time(),
            "current_step": "Starting...",
            "files": {},
            "api_calls": 0,
            "estimated_cost": 0.0,
            "issues": [],
        }

        # Expected files and their steps
        self.expected_files = {
            "arena_questions_base_sample.jsonl": "01_prepare_data",
            "p0_replies_sample.jsonl": "02a_generate_p0",
            "target_responses_sample.jsonl": "02b_generate_targets",
            "p0_with_target_logps_sample.jsonl": "02c_teacher_forcing",
            "oracle_labels_sample.jsonl": "03_oracle_labels",
            "p0_scored_sample.jsonl": "04_judge_scores",
        }

        # Cost estimates per API call
        self.costs_per_call = {
            "p0_replies": 0.0002,  # $0.20/1k
            "target_responses": 0.0002,
            "teacher_forcing": 0.0003,
            "oracle_labels": 0.0025,
            "judge_scores": 0.0002,
        }

    def check_file_status(self, filename: str) -> Dict[str, Any]:
        """Check status of a specific file."""
        filepath = self.sample_dir / filename

        if not filepath.exists():
            return {"exists": False, "count": 0, "last_modified": None}

        try:
            count = 0
            has_errors = False
            zero_logps = 0

            with open(filepath) as f:
                for line in f:
                    count += 1
                    data = json.loads(line)

                    # Check for teacher forcing issues
                    if "target_logps" in data:
                        for policy, logp in data["target_logps"].items():
                            if logp == 0.0 and data.get("response"):
                                zero_logps += 1

            return {
                "exists": True,
                "count": count,
                "last_modified": filepath.stat().st_mtime,
                "has_errors": has_errors,
                "zero_logps": zero_logps,
            }
        except:
            return {
                "exists": True,
                "count": 0,
                "last_modified": filepath.stat().st_mtime,
                "error": True,
            }

    def update_stats(self):
        """Update current statistics."""
        # Check each expected file
        for filename, step in self.expected_files.items():
            status = self.check_file_status(filename)
            self.stats["files"][filename] = status

            # Estimate current step
            if status["exists"] and status["count"] > 0:
                self.stats["current_step"] = step

        # Calculate API calls and costs
        total_calls = 0
        total_cost = 0.0

        if "p0_replies_sample.jsonl" in self.stats["files"]:
            count = self.stats["files"]["p0_replies_sample.jsonl"].get("count", 0)
            total_calls += count
            total_cost += count * self.costs_per_call["p0_replies"]

        if "target_responses_sample.jsonl" in self.stats["files"]:
            count = self.stats["files"]["target_responses_sample.jsonl"].get("count", 0)
            total_calls += count
            total_cost += count * self.costs_per_call["target_responses"]

        # Add teacher forcing calls
        if "p0_with_target_logps_sample.jsonl" in self.stats["files"]:
            count = self.stats["files"]["p0_with_target_logps_sample.jsonl"].get(
                "count", 0
            )
            # 4 policies per prompt (including pi_clone)
            total_calls += count * 4
            total_cost += count * 4 * self.costs_per_call["teacher_forcing"]

        self.stats["api_calls"] = total_calls
        self.stats["estimated_cost"] = total_cost

        # Check for issues
        self.stats["issues"] = []
        for filename, status in self.stats["files"].items():
            if status.get("zero_logps", 0) > 0:
                self.stats["issues"].append(
                    f"⚠️ {filename}: {status['zero_logps']} suspicious 0.0 log probs"
                )

    def create_display(self) -> Table:
        """Create the monitoring display."""
        # Update stats first
        self.update_stats()

        # Main table
        table = Table(title="1% Sample Run Monitor", show_header=True)
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="yellow")

        # Runtime
        elapsed = time.time() - self.stats["start_time"]
        table.add_row("Runtime", f"{elapsed/60:.1f} minutes")
        table.add_row("Current Step", self.stats["current_step"])
        table.add_row("API Calls", f"{self.stats['api_calls']:,}")
        table.add_row("Estimated Cost", f"${self.stats['estimated_cost']:.2f}")

        # File status
        table.add_row("", "")  # Spacer
        table.add_row("FILE STATUS", "", style="bold")

        for filename, status in self.stats["files"].items():
            if status["exists"]:
                status_str = f"✅ {status['count']} entries"
                if status.get("zero_logps", 0) > 0:
                    status_str += f" ⚠️ {status['zero_logps']} zeros"
            else:
                status_str = "⏳ Waiting..."

            table.add_row(f"  {filename}", status_str)

        # Issues
        if self.stats["issues"]:
            table.add_row("", "")
            table.add_row("ISSUES", "", style="bold red")
            for issue in self.stats["issues"]:
                table.add_row("", issue)

        return table

    def monitor_loop(self):
        """Main monitoring loop."""
        with Live(self.create_display(), refresh_per_second=1) as live:
            while self.running:
                time.sleep(2)
                live.update(self.create_display())

    def stop(self):
        """Stop monitoring."""
        self.running = False

    def run(self):
        """Start monitoring in a separate thread."""
        monitor_thread = threading.Thread(target=self.monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()

        console.print("\n🔍 Monitoring sample run... Press Ctrl+C to stop\n")

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
            console.print("\n\n✋ Monitoring stopped")

            # Final summary
            if self.stats["issues"]:
                console.print(
                    Panel.fit(
                        f"[bold yellow]⚠️ Found {len(self.stats['issues'])} issues[/bold yellow]\n"
                        + "\n".join(self.stats["issues"]),
                        title="Issues Detected",
                    )
                )
            else:
                console.print(
                    Panel.fit(
                        "[bold green]✅ No issues detected[/bold green]\n"
                        f"Total API calls: {self.stats['api_calls']:,}\n"
                        f"Estimated cost: ${self.stats['estimated_cost']:.2f}",
                        title="Run Complete",
                    )
                )


def main():
    """Main entry point."""
    monitor = SampleRunMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
