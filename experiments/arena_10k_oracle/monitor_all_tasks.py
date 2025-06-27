#!/usr/bin/env python3
"""
Monitor all Arena 10K experiment tasks in real-time.
"""

import time
import subprocess
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

console = Console()


def get_process_status(process_name: str) -> str:
    """Check if a process is running."""
    result = subprocess.run(
        ["pgrep", "-f", process_name], capture_output=True, text=True
    )
    return "🟢 Running" if result.returncode == 0 else "🔴 Stopped"


def count_lines(filepath: Path) -> int:
    """Count lines in a file."""
    if not filepath.exists():
        return 0
    try:
        return sum(1 for _ in open(filepath))
    except:
        return 0


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in MB."""
    if not filepath.exists():
        return 0.0
    return filepath.stat().st_size / (1024 * 1024)


def estimate_time_remaining(current: int, total: int, start_time: float) -> str:
    """Estimate time remaining."""
    if current == 0:
        return "Unknown"

    elapsed = time.time() - start_time
    rate = current / elapsed
    remaining = (total - current) / rate if rate > 0 else 0

    if remaining < 60:
        return f"{int(remaining)}s"
    elif remaining < 3600:
        return f"{int(remaining/60)}m {int(remaining%60)}s"
    else:
        hours = int(remaining / 3600)
        minutes = int((remaining % 3600) / 60)
        return f"{hours}h {minutes}m"


def create_dashboard() -> Layout:
    """Create the monitoring dashboard."""
    layout = Layout()

    # Main sections
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="tasks", size=20),
        Layout(name="insights", size=8),
    )

    # Header
    layout["header"].update(
        Panel(
            f"[bold cyan]Arena 10K Experiment Monitor[/bold cyan]\n"
            f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="cyan",
        )
    )

    # Task monitoring table
    tasks_table = Table(title="Task Progress", show_header=True)
    tasks_table.add_column("Task", style="cyan", width=30)
    tasks_table.add_column("Status", justify="center", width=12)
    tasks_table.add_column("Progress", justify="right", width=20)
    tasks_table.add_column("Rate", justify="right", width=12)
    tasks_table.add_column("ETA", justify="right", width=10)
    tasks_table.add_column("Output", justify="right", width=15)

    # Track start times
    start_times = {
        "teacher_forcing": time.time() - 300,  # Started ~5 min ago
        "uncertainty": time.time() - 600,  # Started ~10 min ago
        "deterministic": time.time() - 600,  # Started ~10 min ago
    }

    # 1. Teacher Forcing
    tf_checkpoint = Path("data/p0_with_target_logps.checkpoint.jsonl")
    tf_output = Path("data/p0_with_target_logps.jsonl")
    tf_count = count_lines(tf_checkpoint)
    tf_total = 10000
    tf_rate = (
        tf_count / (time.time() - start_times["teacher_forcing"]) if tf_count > 0 else 0
    )

    tasks_table.add_row(
        "Teacher Forcing (P0→Targets)",
        get_process_status("02c_compute_target_logprobs"),
        f"{tf_count:,}/{tf_total:,} ({tf_count/tf_total*100:.1f}%)",
        f"{tf_rate:.1f}/s" if tf_rate > 0 else "0/s",
        estimate_time_remaining(tf_count, tf_total, start_times["teacher_forcing"]),
        f"{get_file_size_mb(tf_output):.1f} MB",
    )

    # 2. Uncertainty Scoring
    unc_checkpoint = Path("data/targets_scored_uncertainty.checkpoint.jsonl")
    unc_output = Path("data/targets_scored_uncertainty.jsonl")
    unc_count = count_lines(unc_checkpoint)
    unc_total = 30000
    unc_rate = (
        unc_count / (time.time() - start_times["uncertainty"]) if unc_count > 0 else 0
    )

    tasks_table.add_row(
        "Target Uncertainty Scoring",
        get_process_status("04d_score_targets_uncertainty"),
        f"{unc_count:,}/{unc_total:,} ({unc_count/unc_total*100:.1f}%)",
        f"{unc_rate:.1f}/s" if unc_rate > 0 else "0/s",
        estimate_time_remaining(unc_count, unc_total, start_times["uncertainty"]),
        f"{get_file_size_mb(unc_output):.1f} MB",
    )

    # 3. Deterministic Scoring
    det_checkpoint = Path("data/targets_scored_deterministic.checkpoint.jsonl")
    det_output = Path("data/targets_scored_deterministic.jsonl")
    det_count = count_lines(det_checkpoint)
    det_total = 30000
    det_rate = (
        det_count / (time.time() - start_times["deterministic"]) if det_count > 0 else 0
    )

    tasks_table.add_row(
        "Target Deterministic Scoring",
        get_process_status("04c_score_targets_deterministic"),
        f"{det_count:,}/{det_total:,} ({det_count/det_total*100:.1f}%)",
        f"{det_rate:.1f}/s" if det_rate > 0 else "0/s",
        estimate_time_remaining(det_count, det_total, start_times["deterministic"]),
        f"{get_file_size_mb(det_output):.1f} MB",
    )

    layout["tasks"].update(tasks_table)

    # Insights section
    insights_text = "[bold]Key Insights:[/bold]\n"

    # Check if teacher forcing is making a difference
    if tf_count > 100:
        insights_text += "✅ Teacher forcing is computing proper log probabilities\n"
    else:
        insights_text += "⏳ Waiting for teacher forcing results...\n"

    # Estimate total completion time
    all_rates = [tf_rate, unc_rate, det_rate]
    if all(r > 0 for r in all_rates):
        max_time = max(
            (tf_total - tf_count) / tf_rate,
            (unc_total - unc_count) / unc_rate,
            (det_total - det_count) / det_rate,
        )
        insights_text += (
            f"\n📊 Estimated completion: {estimate_time_remaining(0, 1, -max_time)}\n"
        )

    # Next steps
    if tf_count >= tf_total:
        insights_text += (
            "\n🎯 Ready to run ablation analysis with proper importance weights!"
        )

    layout["insights"].update(Panel(insights_text, border_style="green"))

    return layout


def main():
    """Monitor all tasks."""
    console.print("[bold cyan]Starting Arena 10K Experiment Monitor[/bold cyan]")
    console.print("Press Ctrl+C to exit\n")

    with Live(create_dashboard(), refresh_per_second=0.5, screen=True) as live:
        try:
            while True:
                live.update(create_dashboard())
                time.sleep(2)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped[/yellow]")


if __name__ == "__main__":
    main()
