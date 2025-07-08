#!/usr/bin/env python3
"""
Simple Phase 1 Pipeline for Arena 10K Oracle Dataset

Usage:
    python run_phase1_pipeline.py 5      # Run with 5 samples
    python run_phase1_pipeline.py 10000  # Run with 10000 samples
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import shutil
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils.progress import console
from config_loader import load_arena_config


def run_command(cmd: str, description: str):
    """Run a command and check for success."""
    console.print(f"\n[bold cyan]Running: {description}[/bold cyan]")
    console.print(f"Command: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"[red]❌ Failed: {description}[/red]")
        console.print(f"[red]Error: {result.stderr}[/red]")
        sys.exit(1)

    console.print(f"[green]✅ Success: {description}[/green]")
    return result.stdout


def clean_data_directory():
    """Clean the data directory."""
    data_dir = Path("data")

    # Clean directory
    if data_dir.exists():
        shutil.rmtree(data_dir)

    # Recreate directory structure
    data_dir.mkdir(exist_ok=True)
    (data_dir / "labeling").mkdir(exist_ok=True)
    (data_dir / "labeling" / "checkpoints").mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 1 pipeline for Arena 10K Oracle dataset"
    )

    parser.add_argument(
        "n_samples",
        type=int,
        help="Number of samples to process (e.g., 5 for testing, 10000 for full run)",
    )

    parser.add_argument(
        "--with-oracle",
        action="store_true",
        help="Generate oracle labels (requires OPENAI_API_KEY)",
    )

    args = parser.parse_args()

    # Hard-coded settings
    SEED = 42
    JUDGE_PROVIDER = "fireworks"

    # Load config
    config = load_arena_config()

    console.print(f"\n[bold cyan]🚀 Arena 10K Oracle Phase 1 Pipeline[/bold cyan]")
    console.print(f"Samples: {args.n_samples}")
    console.print(f"Seed: {SEED}")
    console.print(f"Judge Provider: {JUDGE_PROVIDER}")
    console.print(f"Config: {config.experiment_name}")

    # Check API keys
    if not os.environ.get("FIREWORKS_API_KEY"):
        console.print("[red]❌ Error: FIREWORKS_API_KEY not set[/red]")
        console.print("Please run: source /path/to/set_secrets.sh")
        sys.exit(1)

    # Always clean data directory
    console.print("\n[bold]Cleaning data directory...[/bold]")
    clean_data_directory()

    # Create timestamp for logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"phase1_run_{timestamp}.log"

    # Pipeline steps
    steps = [
        (
            "Step 1: Prepare prompts",
            f"python 01_prepare_data.py --samples {args.n_samples} --seed {SEED}",
        ),
        ("Step 2: Generate responses", f"python 02_generate_responses.py"),
        ("Step 2b: Compute log probabilities", f"python 02b_compute_logprobs.py"),
        (
            "Step 3a: Deterministic judge scoring",
            f"python 03_judge_scores_deterministic.py --provider {JUDGE_PROVIDER}",
        ),
        (
            "Step 3b: Uncertainty judge scoring",
            f"python 03b_judge_scores_uncertainty.py --provider {JUDGE_PROVIDER}",
        ),
    ]

    # Add oracle step if requested
    if args.with_oracle:
        if not os.environ.get("OPENAI_API_KEY"):
            console.print(
                "[red]❌ Error: OPENAI_API_KEY not set for oracle labels[/red]"
            )
            sys.exit(1)
        steps.append(
            ("Step 4: Generate oracle labels", f"python 04_generate_oracle_labels.py")
        )

    steps.append(("Step 5: Finalize dataset", f"python 05_finalize_dataset.py"))

    # Run pipeline
    console.print(f"\n[bold]Starting pipeline with {len(steps)} steps...[/bold]")

    for step_name, cmd in steps:
        run_command(f"{cmd} 2>&1 | tee -a {log_file}", step_name)

    # Summary
    console.print(f"\n[bold green]✅ Pipeline completed successfully![/bold green]")
    console.print(f"Log file: {log_file}")

    # Check output files
    expected_files = [
        "data/arena_prompts_10k.jsonl",
        "data/all_responses.jsonl",
        "data/logprobs.jsonl",
        "data/responses_scored_deterministic.jsonl",
        "data/responses_scored_uncertainty.jsonl",
        "data/dataset_info.json",
    ]

    console.print("\n[bold]Output files:[/bold]")
    for file_path in expected_files:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / 1024 / 1024
            console.print(f"  ✅ {file_path} ({size_mb:.2f} MB)")
        else:
            console.print(f"  ❌ {file_path} (missing)")

    # Print quick stats
    if Path("data/dataset_info.json").exists():
        with open("data/dataset_info.json") as f:
            info = json.load(f)

        console.print("\n[bold]Dataset Summary:[/bold]")
        if "components" in info:
            console.print(
                f"  Prompts: {info['components'].get('prompts', {}).get('total', 'N/A')}"
            )
            console.print(
                f"  P0 responses: {info['components'].get('p0_policy', {}).get('responses', 'N/A')}"
            )
            console.print(
                f"  Target responses: {info['components'].get('target_policies', {}).get('total_responses', 'N/A')}"
            )


if __name__ == "__main__":
    main()
