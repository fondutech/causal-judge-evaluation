#!/usr/bin/env python3
"""
Monitor and run Phase 2 analysis when ready.
"""

import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table

console = Console()


def count_lines(filepath: Path) -> int:
    """Count lines in a file."""
    if not filepath.exists():
        return 0
    return sum(1 for _ in open(filepath))


def check_data_status():
    """Check the status of all data files."""
    data_dir = Path("data")

    files = {
        "p0_deterministic": data_dir / "p0_scored_deterministic.jsonl",
        "p0_uncertainty": data_dir / "p0_scored_uncertainty.jsonl",
        "targets_deterministic": data_dir / "targets_scored_deterministic.jsonl",
        "targets_uncertainty": data_dir / "targets_scored_uncertainty.jsonl",
        "targets_checkpoint": data_dir / "targets_scored_uncertainty.checkpoint.jsonl",
    }

    status = {}
    for name, filepath in files.items():
        count = count_lines(filepath)
        status[name] = {"path": filepath, "count": count, "exists": filepath.exists()}

    return status


def run_uncertainty_scoring_with_monitoring():
    """Run uncertainty scoring with progress monitoring."""
    script_path = Path("phase1_dataset_preparation/04d_score_targets_uncertainty.py")
    checkpoint_path = Path("data/targets_scored_uncertainty.checkpoint.jsonl")

    console.print("[cyan]Starting uncertainty scoring with monitoring...[/cyan]")

    # Start the process
    process = subprocess.Popen(
        ["python", str(script_path), "--batch-size", "8"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    last_count = 0
    stall_counter = 0

    while process.poll() is None:
        time.sleep(30)  # Check every 30 seconds

        current_count = count_lines(checkpoint_path)
        if current_count > last_count:
            rate = (current_count - last_count) / 30
            console.print(
                f"[green]Progress: {current_count:,}/30,000 ({current_count/300:.1f}%) - {rate:.1f} items/sec[/green]"
            )
            last_count = current_count
            stall_counter = 0
        else:
            stall_counter += 1
            console.print(
                f"[yellow]No progress for {stall_counter * 30} seconds...[/yellow]"
            )

            if stall_counter > 10:  # 5 minutes without progress
                console.print("[red]Process appears stalled. Terminating...[/red]")
                process.terminate()
                time.sleep(5)
                if process.poll() is None:
                    process.kill()
                break

    # Check final result
    if process.returncode == 0:
        console.print("[green]✅ Scoring completed successfully![/green]")
    else:
        console.print(f"[red]❌ Process exited with code {process.returncode}[/red]")

        # Check if we have enough data to continue
        final_count = count_lines(checkpoint_path)
        if final_count > 10000:
            console.print(
                f"[yellow]But we have {final_count:,} entries - continuing with partial data[/yellow]"
            )


def main():
    """Monitor and manage the experiment."""
    console.print("[bold cyan]Arena 10K Oracle Experiment Monitor[/bold cyan]")

    # Check current status
    status = check_data_status()

    table = Table(title="Current Data Status")
    table.add_column("Dataset", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Status", justify="center")

    expected = {
        "p0_deterministic": 10000,
        "p0_uncertainty": 10000,
        "targets_deterministic": 30000,
        "targets_uncertainty": 30000,
        "targets_checkpoint": 30000,
    }

    for name, info in status.items():
        count = info["count"]
        exp = expected[name]
        status_icon = "✅" if count >= exp else "❌" if count == 0 else "⏳"
        table.add_row(name, f"{count:,}", f"{exp:,}", status_icon)

    console.print(table)

    # If uncertainty scoring is incomplete, run it
    if status["targets_uncertainty"]["count"] < 30000:
        console.print(
            "\n[yellow]Uncertainty scoring is incomplete. Starting monitored run...[/yellow]"
        )
        run_uncertainty_scoring_with_monitoring()

    # Run quick analysis to see what we have
    console.print("\n[cyan]Running quick analysis with available data...[/cyan]")
    subprocess.run(["python", "quick_ablation_analysis.py"])


if __name__ == "__main__":
    main()
