#!/usr/bin/env python3
"""Monitor the progress of scoring jobs."""

import time
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live

console = Console()


def count_lines(filepath: Path) -> int:
    """Count lines in a file."""
    if not filepath.exists():
        return 0
    return sum(1 for _ in open(filepath))


def estimate_eta(current: int, total: int, start_time: float) -> str:
    """Estimate time remaining."""
    if current == 0:
        return "Unknown"

    elapsed = time.time() - start_time
    rate = current / elapsed
    remaining = (total - current) / rate if rate > 0 else 0

    if remaining < 60:
        return f"{int(remaining)}s"
    elif remaining < 3600:
        return f"{int(remaining/60)}m"
    else:
        return f"{remaining/3600:.1f}h"


def main():
    """Monitor scoring progress."""
    data_dir = Path("data")

    # Files to monitor
    files = {
        "Uncertainty Scoring": {
            "checkpoint": data_dir / "targets_scored_uncertainty.checkpoint.jsonl",
            "output": data_dir / "targets_scored_uncertainty.jsonl",
            "total": 30000,
            "start_count": 0,
            "start_time": time.time(),
        },
        "Deterministic Scoring": {
            "checkpoint": data_dir / "targets_scored_deterministic.checkpoint.jsonl",
            "output": data_dir / "targets_scored_deterministic.jsonl",
            "total": 30000,
            "start_count": 0,
            "start_time": time.time(),
        },
    }

    # Get initial counts
    for name, info in files.items():
        info["start_count"] = count_lines(info["checkpoint"])

    console.print("[bold cyan]Arena 10K Scoring Progress Monitor[/bold cyan]")
    console.print("Press Ctrl+C to exit\n")

    with Live(auto_refresh=True, refresh_per_second=0.5) as live:
        while True:
            try:
                # Create status table
                table = Table(title=f"Status at {datetime.now().strftime('%H:%M:%S')}")
                table.add_column("Task", style="cyan")
                table.add_column("Progress", justify="right")
                table.add_column("Rate", justify="right")
                table.add_column("ETA", justify="right")
                table.add_column("Status", justify="center")

                all_complete = True

                for name, info in files.items():
                    current = count_lines(info["checkpoint"])
                    total = info["total"]

                    # Calculate rate
                    elapsed = time.time() - info["start_time"]
                    new_items = current - info["start_count"]
                    rate = new_items / elapsed if elapsed > 0 else 0

                    # Progress
                    pct = current / total * 100
                    progress = f"{current:,}/{total:,} ({pct:.1f}%)"

                    # Status
                    if current >= total:
                        status = "✅ Complete"
                    elif rate > 0:
                        status = "🔄 Running"
                        all_complete = False
                    else:
                        status = "⏸️  Paused"
                        all_complete = False

                    # ETA
                    eta = (
                        estimate_eta(current, total, info["start_time"])
                        if rate > 0
                        else "N/A"
                    )

                    table.add_row(
                        name,
                        progress,
                        f"{rate:.1f}/s" if rate > 0 else "0/s",
                        eta,
                        status,
                    )

                # Add summary
                table.add_row("", "", "", "", "")

                # Check for output files
                unc_output = count_lines(files["Uncertainty Scoring"]["output"])
                det_output = count_lines(files["Deterministic Scoring"]["output"])

                if unc_output > 0 or det_output > 0:
                    table.add_row(
                        "Output Files",
                        f"Unc: {unc_output:,}, Det: {det_output:,}",
                        "",
                        "",
                        "📄",
                    )

                live.update(table)

                if all_complete:
                    live.stop()
                    console.print("\n[bold green]🎉 All scoring complete![/bold green]")

                    # Final summary
                    console.print("\nFinal counts:")
                    for name, info in files.items():
                        final = count_lines(info["output"])
                        console.print(f"  {name}: {final:,} entries")
                    break

                time.sleep(10)  # Update every 10 seconds

            except KeyboardInterrupt:
                live.stop()
                console.print("\n[yellow]Monitoring stopped[/yellow]")
                break
            except Exception as e:
                live.stop()
                console.print(f"\n[red]Error: {e}[/red]")
                break


if __name__ == "__main__":
    main()
