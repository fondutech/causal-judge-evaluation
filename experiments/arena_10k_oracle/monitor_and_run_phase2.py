#!/usr/bin/env python3
"""
Monitor P0 scoring completion and automatically run Phase 2 analysis.
"""

import time
import subprocess
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import json

console = Console()


def check_p0_completion(data_dir: Path) -> dict:
    """Check if P0 scoring is complete."""
    status = {
        "deterministic": {"checkpoint": 0, "final": False},
        "uncertainty": {"checkpoint": 0, "final": False},
    }

    # Check checkpoint files
    for judge_type in ["deterministic", "uncertainty"]:
        checkpoint_file = data_dir / f"p0_scored_{judge_type}.checkpoint.jsonl"
        final_file = data_dir / f"p0_scored_{judge_type}.jsonl"

        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                status[judge_type]["checkpoint"] = sum(1 for _ in f)

        if final_file.exists():
            # Check if it's the new file (not the old one from 10:40 AM)
            import os

            mod_time = os.path.getmtime(final_file)
            # If modified after noon today, it's the new file
            if mod_time > time.mktime(
                datetime.now().replace(hour=12, minute=0).timetuple()
            ):
                status[judge_type]["final"] = True

    return status


def monitor_p0_completion(data_dir: Path, check_interval: int = 30):
    """Monitor P0 completion and return when both are done."""
    console.print("\n[bold cyan]🔍 Monitoring P0 Scoring Progress[/bold cyan]")
    console.print("=" * 60)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        det_task = progress.add_task("P0 Deterministic", total=10000)
        unc_task = progress.add_task("P0 Uncertainty", total=10000)

        while True:
            status = check_p0_completion(data_dir)

            # Update progress bars
            progress.update(det_task, completed=status["deterministic"]["checkpoint"])
            progress.update(unc_task, completed=status["uncertainty"]["checkpoint"])

            # Check if both are complete
            if (
                status["deterministic"]["checkpoint"] >= 10000
                and status["uncertainty"]["checkpoint"] >= 10000
            ):
                console.print(
                    "\n✅ [bold green]Both P0 scorings complete![/bold green]"
                )
                return True

            # Check if deterministic is complete and we can start with just that
            if status["deterministic"]["checkpoint"] >= 10000:
                console.print(
                    "\n✅ [bold green]P0 Deterministic complete![/bold green]"
                )
                console.print(
                    "[yellow]P0 Uncertainty still running, but we can start Phase 2 with deterministic only[/yellow]"
                )
                return True

            time.sleep(check_interval)


def run_phase2_analysis():
    """Run the Phase 2 direct ablation analysis."""
    console.print("\n[bold cyan]🚀 Starting Phase 2 Analysis[/bold cyan]")
    console.print("=" * 60)

    # Determine which judge types are available
    data_dir = Path("data")
    available_judges = []

    # Check for completed files
    for judge_type in ["deterministic", "uncertainty"]:
        checkpoint_file = data_dir / f"p0_scored_{judge_type}.checkpoint.jsonl"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                if sum(1 for _ in f) >= 10000:
                    available_judges.append(judge_type)

    if not available_judges:
        console.print("[red]❌ No completed judge scores found![/red]")
        return

    console.print(f"📊 Running analysis with: {', '.join(available_judges)}")

    # Run the direct ablation analysis
    cmd = (
        ["python", "phase2_cje_ablations/run_direct_ablations.py", "--judge-types"]
        + available_judges
        + [
            "--n-bootstrap",
            "200",
            "--output",
            "phase2_cje_ablations/results/p0_only_results.json",
        ]
    )

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        console.print("\n✅ [bold green]Phase 2 analysis complete![/bold green]")
        console.print(
            "📊 Results saved to: phase2_cje_ablations/results/p0_only_results.json"
        )
    else:
        console.print(f"\n[red]❌ Error running analysis:[/red]")
        console.print(result.stderr)


def main():
    """Monitor and run Phase 2 when ready."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-interval", type=int, default=30, help="Check interval in seconds"
    )
    parser.add_argument(
        "--no-wait", action="store_true", help="Run immediately if P0 is ready"
    )
    args = parser.parse_args()

    data_dir = Path("data")

    # Check current status
    status = check_p0_completion(data_dir)

    # If already complete or --no-wait, run immediately
    if args.no_wait and (
        status["deterministic"]["checkpoint"] >= 10000
        or status["uncertainty"]["checkpoint"] >= 10000
    ):
        run_phase2_analysis()
    else:
        # Monitor until completion
        if monitor_p0_completion(data_dir, args.check_interval):
            # Wait a bit for files to be written
            console.print("\n⏳ Waiting 30 seconds for files to finalize...")
            time.sleep(30)

            # Run Phase 2
            run_phase2_analysis()


if __name__ == "__main__":
    main()
