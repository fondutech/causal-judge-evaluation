#!/usr/bin/env python3
"""
Phase 1 Pipeline for Arena 10K Oracle Dataset

Automatically resumes from checkpoint if interrupted.
Defaults to 10,000 samples.
Always generates oracle labels for calibration.

Usage:
    python run_phase1_pipeline.py         # Run with 10,000 samples (default)
    python run_phase1_pipeline.py 5       # Run with 5 samples for testing

To start fresh:
    rm -rf data/ .pipeline_checkpoint.pkl
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import shutil
from datetime import datetime
import pickle

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils.progress import console
from config_loader import load_arena_config


class PipelineCheckpoint:
    """Manages pipeline state for resumable runs."""

    def __init__(self, checkpoint_file: str = ".pipeline_checkpoint.pkl"):
        self.checkpoint_file = checkpoint_file
        self.state = self._load_checkpoint()

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint if it exists."""
        if Path(self.checkpoint_file).exists():
            try:
                with open(self.checkpoint_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not load checkpoint: {e}[/yellow]"
                )

        return {
            "completed_steps": [],
            "n_samples": None,
            "seed": None,
            "with_oracle": False,
            "timestamp": None,
            "log_file": None,
        }

    def save(self):
        """Save current state."""
        with open(self.checkpoint_file, "wb") as f:
            pickle.dump(self.state, f)

    def mark_step_complete(self, step_name: str):
        """Mark a step as completed."""
        if step_name not in self.state["completed_steps"]:
            self.state["completed_steps"].append(step_name)
            self.save()

    def is_step_complete(self, step_name: str) -> bool:
        """Check if a step is already completed."""
        return step_name in self.state["completed_steps"]

    def clear(self):
        """Clear checkpoint file."""
        if Path(self.checkpoint_file).exists():
            Path(self.checkpoint_file).unlink()
        self.state = self._load_checkpoint()

    def matches_current_run(self, n_samples: int, seed: int, with_oracle: bool) -> bool:
        """Check if checkpoint matches current run parameters."""
        return (
            self.state.get("n_samples") == n_samples
            and self.state.get("seed") == seed
            and self.state.get("with_oracle") == with_oracle
        )

    def initialize_run(
        self,
        n_samples: int,
        seed: int,
        with_oracle: bool,
        timestamp: str,
        log_file: str,
    ):
        """Initialize checkpoint for new run."""
        self.state.update(
            {
                "n_samples": n_samples,
                "seed": seed,
                "with_oracle": with_oracle,
                "timestamp": timestamp,
                "log_file": log_file,
            }
        )
        self.save()


def check_data_integrity(step_name: str) -> Tuple[bool, Optional[str]]:
    """Check if data from a step is valid and complete."""
    # Map steps to their expected output files
    step_outputs = {
        "Step 1: Prepare prompts": ["data/arena_prompts_10k.jsonl"],
        "Step 2: Generate responses": ["data/all_responses.jsonl"],
        "Step 2b: Compute log probabilities": ["data/logprobs.jsonl"],
        "Step 3a: Deterministic judge scoring": [
            "../data/p0_scored_deterministic.jsonl",
            "../data/targets_scored_deterministic.jsonl",
        ],
        "Step 3b: Uncertainty judge scoring": [
            "../data/p0_scored_uncertainty.jsonl",
            "../data/targets_scored_uncertainty.jsonl",
        ],
        "Step 4: Generate oracle labels": [
            "data/oracle_labels_calibration.jsonl",
            "data/oracle_labels_validation.jsonl",
        ],
        "Step 5: Validate and summarize": ["data/dataset_info.json"],
    }

    expected_files = step_outputs.get(step_name, [])

    for file_path in expected_files:
        if not Path(file_path).exists():
            return False, f"Missing file: {file_path}"

        # Check if file is non-empty
        if Path(file_path).stat().st_size == 0:
            return False, f"Empty file: {file_path}"

        # For JSONL files, check if they're valid
        if file_path.endswith(".jsonl"):
            try:
                with open(file_path) as f:
                    # Just check first line
                    first_line = f.readline()
                    if first_line:
                        json.loads(first_line)
            except Exception as e:
                return False, f"Invalid JSON in {file_path}: {e}"

    return True, None


def run_command(
    cmd: str,
    description: str,
    checkpoint: Optional[PipelineCheckpoint] = None,
    force: bool = False,
    log_file: str = "phase1_run.log",
) -> str:
    """Run a command with optional checkpoint support."""

    # Check if step is already complete
    if checkpoint and checkpoint.is_step_complete(description) and not force:
        # Verify data integrity
        is_valid, error_msg = check_data_integrity(description)
        if is_valid:
            console.print(
                f"\n[green]✓ Skipping: {description} (already completed)[/green]"
            )
            return ""
        else:
            console.print(
                f"\n[yellow]⚠️  {description} marked complete but data issue found: {error_msg}[/yellow]"
            )
            console.print("[yellow]Re-running this step...[/yellow]")

    console.print(f"\n[bold cyan]Running: {description}[/bold cyan]")
    console.print(f"Command: {cmd}")

    # Add logging to command
    cmd_with_log = f"{cmd} 2>&1 | tee -a {log_file}"

    result = subprocess.run(cmd_with_log, shell=True, capture_output=False, text=True)

    if result.returncode != 0:
        console.print(f"[red]❌ Failed: {description}[/red]")
        console.print(f"[red]Check log file: {log_file}[/red]")
        sys.exit(1)

    # Mark step as complete
    if checkpoint:
        checkpoint.mark_step_complete(description)

    console.print(f"[green]✅ Success: {description}[/green]")
    return ""


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
        description="Run Phase 1 pipeline for Arena 10K Oracle dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_phase1_pipeline.py             # Resume or start with 10,000 samples (default)
  python run_phase1_pipeline.py 5           # Resume or start with 5 samples
  
The pipeline automatically resumes from the last checkpoint if interrupted.
To start fresh, manually delete: rm -rf data/ .pipeline_checkpoint.pkl
        """,
    )

    parser.add_argument(
        "n_samples",
        type=int,
        nargs="?",
        default=10000,
        help="Number of samples to process (default: 10000, use 5 for testing)",
    )

    parser.add_argument(
        "--force-step",
        type=str,
        help="Force re-run of a specific step (e.g., 'Step 2b: Compute log probabilities')",
    )

    args = parser.parse_args()

    # Hard-coded settings
    SEED = 42

    # Load config
    config = load_arena_config()

    # Always use checkpoint
    checkpoint = PipelineCheckpoint()

    # Check if we have an existing checkpoint
    has_existing_checkpoint = bool(checkpoint.state["completed_steps"])

    # Check if resuming from existing checkpoint
    if has_existing_checkpoint:
        if not checkpoint.matches_current_run(args.n_samples, SEED, True):
            console.print(
                "[yellow]⚠️  Existing checkpoint has different parameters[/yellow]"
            )
            console.print("Current run:")
            console.print(f"  Samples: {args.n_samples}")
            console.print("Checkpoint:")
            console.print(f"  Samples: {checkpoint.state.get('n_samples')}")
            console.print(
                "\n[bold]Continuing anyway to preserve existing work...[/bold]"
            )
            console.print(
                "[dim]To start fresh, manually delete data/ and .pipeline_checkpoint.pkl[/dim]"
            )
            # Update checkpoint to match current parameters
            checkpoint.state["n_samples"] = args.n_samples
            checkpoint.state["seed"] = SEED
            checkpoint.save_checkpoint()

        console.print(f"[green]✅ Resuming from checkpoint[/green]")
        console.print(f"Completed steps: {len(checkpoint.state['completed_steps'])}")

    console.print(f"\n[bold cyan]🚀 Arena 10K Oracle Phase 1 Pipeline[/bold cyan]")
    console.print(f"Samples: {args.n_samples}")
    console.print(f"Seed: {SEED}")
    console.print(f"Config: {config.experiment_name}")
    console.print(f"Mode: {'Resuming' if has_existing_checkpoint else 'Fresh start'}")

    # Check API keys
    if not os.environ.get("FIREWORKS_API_KEY"):
        console.print("[red]❌ Error: FIREWORKS_API_KEY not set[/red]")
        console.print(
            "Please run: source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh"
        )
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]❌ Error: OPENAI_API_KEY not set for oracle labels[/red]")
        console.print(
            "Please run: source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh"
        )
        sys.exit(1)

    # Initialize or resume
    if not has_existing_checkpoint:
        # Check if data directory exists
        data_dir = Path("data")
        if data_dir.exists() and any(data_dir.iterdir()):
            console.print(
                "\n[yellow]⚠️  Data directory exists but no checkpoint found[/yellow]"
            )
            console.print(
                "[bold]Creating checkpoint to preserve existing data...[/bold]"
            )
            console.print("[dim]To start fresh, manually delete data/ directory[/dim]")
        else:
            # Only clean if directory doesn't exist or is empty
            console.print("\n[bold]Initializing data directory...[/bold]")
            data_dir.mkdir(exist_ok=True)
            (data_dir / "labeling").mkdir(exist_ok=True)
            (data_dir / "labeling" / "checkpoints").mkdir(exist_ok=True)

        # Create timestamp for logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"phase1_run_{timestamp}.log"

        # Initialize checkpoint for new run
        checkpoint.initialize_run(args.n_samples, SEED, True, timestamp, log_file)
    else:
        # Use existing log file from checkpoint
        log_file = checkpoint.state.get("log_file", "phase1_run.log")

    # Pipeline steps
    steps = [
        (
            "Step 1: Prepare prompts",
            f"python 01_prepare_data.py --samples {args.n_samples} --seed {SEED}",
        ),
        ("Step 2: Generate responses", "python 02_generate_responses.py"),
        ("Step 2b: Compute log probabilities", "python 02b_compute_logprobs.py"),
        (
            "Step 3a: Deterministic judge scoring",
            "python 03_judge_scores_deterministic.py",
        ),
        (
            "Step 3b: Uncertainty judge scoring",
            "python 03b_judge_scores_uncertainty.py",
        ),
    ]

    # Add oracle step (always included)
    steps.append(
        ("Step 4: Generate oracle labels", "python 04_generate_oracle_labels.py")
    )

    steps.append(
        ("Step 5: Validate and summarize", "python 05_validate_and_summarize.py")
    )

    # Run pipeline
    total_steps = len(steps)
    completed_before = len(checkpoint.state["completed_steps"])

    console.print(
        f"\n[bold]Pipeline progress: {completed_before}/{total_steps} steps completed[/bold]"
    )

    for step_name, cmd in steps:
        force = args.force_step == step_name
        run_command(cmd, step_name, checkpoint, force=force, log_file=log_file)

    # Summary
    console.print(f"\n[bold green]✅ Pipeline completed successfully![/bold green]")
    console.print(f"Log file: {log_file}")

    # Check output files
    expected_files = [
        "data/arena_prompts_10k.jsonl",
        "data/all_responses.jsonl",
        "data/logprobs.jsonl",
        "data/dataset_info.json",
    ]

    # Oracle files are always expected
    expected_files.extend(
        [
            "data/oracle_labels_calibration.jsonl",
            "data/oracle_labels_validation.jsonl",
            "../data/p0_scored_deterministic.jsonl",
            "../data/p0_scored_uncertainty.jsonl",
            "../data/targets_scored_deterministic.jsonl",
            "../data/targets_scored_uncertainty.jsonl",
        ]
    )

    console.print("\n[bold]Output files:[/bold]")
    all_present = True
    for file_path in expected_files:
        if Path(file_path).exists():
            size_mb = Path(file_path).stat().st_size / 1024 / 1024
            console.print(f"  ✅ {file_path} ({size_mb:.2f} MB)")
        else:
            console.print(f"  ❌ {file_path} (missing)")
            all_present = False

    # Print quick stats
    if Path("data/dataset_info.json").exists():
        with open("data/dataset_info.json") as f:
            info = json.load(f)

        console.print("\n[bold]Dataset Summary:[/bold]")
        if "components" in info:
            prompts = info["components"].get("prompts", {})
            p0 = info["components"].get("p0_policy", {})
            targets = info["components"].get("target_policies", {})
            oracle = info["components"].get("oracle_labels", {})

            console.print(f"  Prompts: {prompts.get('count', 'N/A')}")
            console.print(f"  P0 responses: {p0.get('responses', 'N/A')}")
            console.print(
                f"  Target responses: {targets.get('total_responses', 'N/A')}"
            )
            console.print(f"  Oracle labels: {oracle.get('total', 'N/A')}")

    # Clean up checkpoint if successful
    if all_present:
        console.print("\n[green]✅ All files generated successfully![/green]")
        checkpoint.clear()
        console.print("[dim]Checkpoint cleared.[/dim]")
    else:
        console.print(
            "\n[yellow]⚠️  Some files are missing. Checkpoint preserved for debugging.[/yellow]"
        )


if __name__ == "__main__":
    main()
