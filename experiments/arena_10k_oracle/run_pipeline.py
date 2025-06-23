#!/usr/bin/env python3
"""
Orchestrate the complete Arena 10K Oracle experiment.

This script provides a unified interface for both phases:
- Phase 1: Dataset preparation (data generation, oracle labeling, judge scoring)
- Phase 2: CJE pipeline ablations (different estimators and configurations)
"""

import subprocess
import sys
from pathlib import Path
import json
import time
from rich.console import Console
from rich.table import Table

console = Console()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))


def run_command(cmd: list[str], description: str, check: bool = True) -> bool:
    """Run a command and display status."""
    console.print(f"\n[bold blue]→ {description}[/bold blue]")
    console.print(f"  Command: {' '.join(cmd)}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=check)
        elapsed = time.time() - start_time
        if result.returncode == 0:
            console.print(f"  [green]✓ Success[/green] ({elapsed:.1f}s)")
            return True
        else:
            console.print(f"  [red]✗ Failed[/red] (exit code: {result.returncode})")
            return False
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        console.print(f"  [red]✗ Error: {e}[/red] ({elapsed:.1f}s)")
        return False
    except KeyboardInterrupt:
        console.print(f"  [yellow]⚠ Interrupted by user[/yellow]")
        raise


def check_data_files() -> bool:
    """Check that required data files exist."""
    console.print("\n[bold]Checking data files...[/bold]")

    required_files = {
        "Prompts": "data/arena_prompts_10k.jsonl",
        "Logging policy responses": "data/p0_replies.jsonl",
        "Oracle calibration labels": "data/labeling/oracle_labels_calibration_detailed.jsonl",
        "Oracle validation labels": "data/labeling/oracle_labels_validation_detailed.jsonl",
    }

    all_exist = True
    for name, path in required_files.items():
        if Path(path).exists():
            size = Path(path).stat().st_size / 1024 / 1024  # MB
            console.print(f"  ✓ {name}: {path} ({size:.1f} MB)")
        else:
            console.print(f"  ✗ {name}: {path} [red]MISSING[/red]")
            all_exist = False

    return all_exist


def check_phase1_complete() -> bool:
    """Check if Phase 1 dataset preparation is complete."""
    dataset_info = Path("data/dataset_info.json")
    if dataset_info.exists():
        console.print("[green]✅ Phase 1 complete - dataset ready[/green]")
        return True
    else:
        console.print("[yellow]⚠️  Phase 1 incomplete - dataset not finalized[/yellow]")
        return False


def run_phase1_finalization() -> bool:
    """Run Phase 1 dataset finalization."""
    cmd = ["python", "phase1_dataset_preparation/06_finalize_dataset.py"]
    return run_command(cmd, "Finalizing Phase 1 dataset")


def run_phase2_ablations() -> bool:
    """Run Phase 2 CJE ablations."""
    cmd = ["python", "phase2_cje_ablations/run_ablations.py"]
    return run_command(cmd, "Running Phase 2 CJE ablations")


def main() -> int:
    """Run the complete pipeline."""
    console.print("[bold cyan]Arena 10K Oracle Experiment Pipeline[/bold cyan]")
    console.print("=" * 50)

    # Check experiment phase
    phase1_complete = check_phase1_complete()

    console.print("\n[bold]Experiment Phases:[/bold]")
    console.print("1. Phase 1: Dataset Preparation")
    console.print("2. Phase 2: CJE Pipeline Ablations")
    console.print("3. Run complete pipeline (Phase 1 + Phase 2)")

    if not phase1_complete:
        console.print(
            "\n[yellow]Note: Phase 1 must be completed before Phase 2[/yellow]"
        )

    choice = console.input("\nSelect option [1-3]: ")

    # Execute based on choice
    if choice == "1" or (choice == "3" and not phase1_complete):
        console.print("\n[bold cyan]Phase 1: Dataset Preparation[/bold cyan]")
        console.print("Please run the following scripts in order:")
        console.print("1. cd phase1_dataset_preparation")
        console.print("2. python 01_prepare_data.py")
        console.print("3. python 02_generate_logs.py")
        console.print("4. python 02b_generate_target_ground_truth.py")
        console.print("5. python 04_generate_oracle_labels.py")
        console.print("6. python 05a_deterministic_judge_scores.py")
        console.print("7. python 05b_uncertainty_judge_scores.py")
        console.print("8. python 06_finalize_dataset.py")

        finalize = console.input("\nRun dataset finalization now? [y/N]: ").lower()
        if finalize == "y":
            if not run_phase1_finalization():
                console.print("[red]Dataset finalization failed![/red]")
                return 1

    if choice in ["2", "3"]:
        if not phase1_complete and choice == "2":
            console.print("\n[red]Cannot run Phase 2 - Phase 1 not complete![/red]")
            return 1

        console.print("\n[bold cyan]Phase 2: CJE Pipeline Ablations[/bold cyan]")
        if not run_phase2_ablations():
            console.print("[red]CJE ablations failed![/red]")
            return 1

    console.print("\n[bold green]Pipeline completed successfully![/bold green]")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        sys.exit(130)
