#!/usr/bin/env python3
"""Monitor progress of running ablation experiments."""

import time
import json
from pathlib import Path
from datetime import datetime, timedelta


def count_completed_experiments():
    """Count completed experiments from results directories."""

    results_dir = Path("ablations/results")
    counts = {}

    for ablation in ["oracle_coverage", "sample_size", "interaction"]:
        ablation_dir = results_dir / ablation
        results_file = ablation_dir / "results.jsonl"

        if results_file.exists():
            with open(results_file) as f:
                results = [json.loads(line) for line in f if line.strip()]
                successful = sum(1 for r in results if r.get("success", False))
                total = len(results)
                counts[ablation] = {"successful": successful, "total": total}
        else:
            counts[ablation] = {"successful": 0, "total": 0}

    return counts


def check_process_status():
    """Check if the experiment process is still running."""
    import subprocess

    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_full_scale.py"], capture_output=True, text=True
        )
        return len(result.stdout.strip()) > 0
    except:
        return False


def tail_log(log_file="full_scale_run2.log", n_lines=10):
    """Get last n lines from log file."""
    log_path = Path(log_file)

    if not log_path.exists():
        return []

    with open(log_path) as f:
        lines = f.readlines()
        return lines[-n_lines:]


def main():
    """Monitor experiment progress."""

    print("=" * 70)
    print("ABLATION EXPERIMENT MONITOR")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = datetime.now()

    while True:
        # Check if process is running
        is_running = check_process_status()

        # Count completed experiments
        counts = count_completed_experiments()

        # Calculate totals
        total_successful = sum(c["successful"] for c in counts.values())
        total_experiments = sum(c["total"] for c in counts.values())

        # Clear screen (works on Unix-like systems)
        print("\033[2J\033[H")

        # Print header
        print("=" * 70)
        print("ABLATION EXPERIMENT MONITOR")
        print("=" * 70)

        # Print status
        runtime = datetime.now() - start_time
        print(f"Status: {'🟢 RUNNING' if is_running else '🔴 STOPPED'}")
        print(f"Runtime: {str(runtime).split('.')[0]}")
        print(f"Total Progress: {total_successful}/{total_experiments} experiments")
        print()

        # Print per-ablation progress
        print("Progress by Ablation:")
        print("-" * 40)

        expected = {
            "oracle_coverage": 7 * 5,  # 7 coverage levels × 5 seeds
            "sample_size": 6 * 2 * 5,  # 6 sizes × 2 estimators × 5 seeds
            "interaction": 4 * 5 * 3,  # 4 oracle × 5 sizes × 3 seeds
        }

        for ablation, count in counts.items():
            pct = (
                (count["successful"] / expected[ablation] * 100)
                if expected[ablation] > 0
                else 0
            )
            bar_length = int(pct / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(
                f"{ablation:20s}: [{bar}] {count['successful']:3d}/{expected[ablation]:3d} ({pct:5.1f}%)"
            )

        print()

        # Show recent log lines
        print("Recent Activity:")
        print("-" * 40)
        recent_lines = tail_log(n_lines=5)
        for line in recent_lines:
            line = line.strip()
            if line:
                # Truncate long lines
                if len(line) > 65:
                    line = line[:62] + "..."
                print(f"  {line}")

        print()

        # Estimate completion
        if total_successful > 0 and is_running:
            total_expected = sum(expected.values())
            rate = total_successful / (
                runtime.total_seconds() / 60
            )  # experiments per minute
            remaining = total_expected - total_successful
            eta_minutes = remaining / rate if rate > 0 else 0
            eta = datetime.now() + timedelta(minutes=eta_minutes)
            print(
                f"Estimated completion: {eta.strftime('%H:%M:%S')} ({eta_minutes:.0f} minutes)"
            )

        # Check if complete
        if not is_running and total_experiments > 0:
            print()
            print("✅ EXPERIMENTS COMPLETE!")
            print(f"Total successful: {total_successful}")
            print(f"Total runtime: {str(runtime).split('.')[0]}")
            print()
            print("Run analyze_results.py to generate summary tables and figures")
            break

        # Wait before next update
        time.sleep(10)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
