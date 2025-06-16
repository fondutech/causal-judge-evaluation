#!/usr/bin/env python3
"""
Check the status of the Arena 10K Oracle experiment.

Shows what data has been generated and what steps remain.
"""

import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils.progress import console
from typing import Dict, Any
from rich.table import Table


def check_file_status(file_path: Path) -> tuple[bool, int]:
    """Check if file exists and return line count."""
    if file_path.exists():
        with open(file_path, "r") as f:
            count = sum(1 for _ in f)
        return True, count
    return False, 0


def analyze_data_completeness(file_path: Path) -> Dict[str, Any]:
    """Analyze the completeness of generated data."""
    if not file_path.exists():
        return {}

    # Check what fields are present
    with open(file_path, "r") as f:
        first_line = f.readline()
        if not first_line:
            return {}

        sample = json.loads(first_line)

    # Count how many samples have each field
    field_counts: Dict[str, int] = {}
    total_samples = 0

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            total_samples += 1

            for field in [
                "response",
                "total_logprob",
                "judge_score_raw",
                "pi_clone_response",
                "pi_cot_response",
                "pi_bigger_model_response",
            ]:
                if field in data and data[field] is not None and data[field] != "":
                    field_counts[field] = field_counts.get(field, 0) + 1

    return {"total": total_samples, "fields": field_counts}


def main() -> None:
    console.print("[bold blue]🔬 Arena 10K Oracle Experiment Status[/bold blue]\n")

    data_dir = Path("../data")
    scripts_dir = Path(".")

    # Check each step's output
    steps = [
        ("Step 1: Sample Prompts", data_dir / "prompts.jsonl", 10000),
        ("Step 2: Generate π₀ Logs", data_dir / "p0_replies.jsonl", 72),
        ("Step 3: Judge Scoring", data_dir / "p0_scored.jsonl", 72),
        ("Step 4: Target Policies", data_dir / "all_policies.jsonl", 72),
        (
            "Step 5: Export for Labeling",
            data_dir / "labeling" / "calibration_export_surge.csv",
            18,
        ),
    ]

    # Create status table
    table = Table(title="Pipeline Status")
    table.add_column("Step", style="cyan")
    table.add_column("File", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Count", justify="right")
    table.add_column("Expected", justify="right")

    all_complete = True

    for step_name, file_path, expected in steps:
        exists, count = check_file_status(file_path)
        status = "✓" if exists else "✗"
        if exists and count < expected:
            status = "⚠️"
            all_complete = False
        elif not exists:
            all_complete = False

        table.add_row(
            step_name,
            file_path.name,
            status,
            str(count) if exists else "-",
            str(expected),
        )

    console.print(table)

    # Detailed analysis of the main data file
    main_data = data_dir / "all_policies.jsonl"
    if main_data.exists():
        console.print("\n[bold]Data Completeness Analysis:[/bold]")
        analysis = analyze_data_completeness(main_data)

        if analysis:
            total = analysis["total"]
            fields = analysis["fields"]

            field_table = Table()
            field_table.add_column("Field", style="cyan")
            field_table.add_column("Count", justify="right")
            field_table.add_column("Percentage", justify="right")

            for field in [
                "response",
                "total_logprob",
                "judge_score_raw",
                "pi_clone_response",
                "pi_cot_response",
                "pi_bigger_model_response",
            ]:
                count = fields.get(field, 0)
                pct = (count / total * 100) if total > 0 else 0
                field_table.add_row(field, str(count), f"{pct:.1f}%")

            console.print(field_table)

    # Check for checkpoint files
    console.print("\n[bold]Checkpoint Files:[/bold]")
    checkpoint_files = list(data_dir.glob("*.checkpoint.*")) + list(
        scripts_dir.glob("*.checkpoint.*")
    )
    if checkpoint_files:
        for cf in checkpoint_files:
            console.print(f"  ⚠️  {cf}")
        console.print(
            "\n[yellow]Note: Checkpoint files found. Some processes may have been interrupted.[/yellow]"
        )
    else:
        console.print("  ✓ No checkpoint files (all processes completed cleanly)")

    # Next steps
    console.print("\n[bold]Next Steps:[/bold]")
    if all_complete:
        console.print("  ✓ All data generation complete!")
        console.print(
            "  1. Upload calibration_export_surge.csv to crowdsourcing platform"
        )
        console.print("  2. Collect human labels (54 total)")
        console.print("  3. Run: python 06_import_labels.py")
        console.print("  4. Run CJE estimation")
    else:
        if not (data_dir / "prompts.jsonl").exists():
            console.print("  1. Run: python 01_sample_prompts.py")
        elif not (data_dir / "p0_replies.jsonl").exists():
            console.print("  1. Run: python 02_generate_logs.py")
        elif not (data_dir / "p0_scored.jsonl").exists():
            console.print("  1. Run: python 03_add_judge_scores.py")
        elif not (data_dir / "all_policies.jsonl").exists():
            console.print("  1. Run: python 04_generate_target_policies.py")
        elif not (data_dir / "labeling" / "calibration_export_surge.csv").exists():
            console.print("  1. Run: python 05_export_for_labeling.py")

    # Cost summary
    console.print("\n[bold]Cost Summary:[/bold]")
    console.print("  π₀ generation: ~$0.03 (72 samples)")
    console.print("  Judge scoring: ~$0.01 (72 samples)")
    console.print("  Target policies: ~$0.15 (includes Maverick model)")
    console.print("  Human labeling: ~$4.32 (54 labels)")
    console.print("  [bold]Total: ~$4.51[/bold]")


if __name__ == "__main__":
    main()
