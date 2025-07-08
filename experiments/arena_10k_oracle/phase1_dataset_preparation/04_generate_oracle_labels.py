#!/usr/bin/env python3
"""
Step 4: Generate oracle labels for calibration and validation.

This script generates oracle labels using OpenAI GPT-4 for:
- 25% of P0 responses (calibration)
- 5% of target responses (validation)

All settings are fixed to ensure consistency.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import random

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.oracle_labeling import add_oracle_labels
from cje.utils.progress import console


def load_responses_for_oracle(
    responses_file: str, policy_filter: List[str], fraction: float, seed: int
) -> List[Dict[str, Any]]:
    """Load and sample responses for oracle labeling."""

    # Load all responses
    all_responses = []
    with open(responses_file) as f:
        for line in f:
            data = json.loads(line)
            prompt_id = data["prompt_id"]
            prompt = data["prompt"]

            # Extract responses for specified policies
            for policy in policy_filter:
                if policy in data["responses"]:
                    resp_data = data["responses"][policy]
                    all_responses.append(
                        {
                            "context": prompt,
                            "response": resp_data["response"],
                            "prompt_id": prompt_id,
                            "policy": policy,
                            "model": resp_data["model"],
                        }
                    )

    # Sample fraction
    random.seed(seed)
    n_samples = int(len(all_responses) * fraction)
    sampled = random.sample(all_responses, n_samples)
    return sampled


def main():
    # No arguments - everything fixed for consistency
    SEED = 42  # Fixed seed for reproducibility

    # Fixed settings
    INPUT_FILE = "data/all_responses.jsonl"
    CALIBRATION_FRACTION = 0.25
    VALIDATION_FRACTION = 0.25
    ORACLE_MODEL = "gpt-4o"  # Using GPT-4 instead of o3 for cost

    console.print("[bold cyan]Step 4: Generate Oracle Labels[/bold cyan]")

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY not set![/red]")
        console.print("Oracle labeling requires OpenAI API access.")
        sys.exit(1)

    # Check input exists
    if not Path(INPUT_FILE).exists():
        console.print(f"[red]Error: {INPUT_FILE} not found![/red]")
        sys.exit(1)

    # Create checkpoint directory
    checkpoint_dir = "data/oracle_checkpoints"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Load calibration data (P0 responses)
    console.print(f"\n[bold]Loading calibration data (P0 responses)...[/bold]")
    calibration_data = load_responses_for_oracle(
        INPUT_FILE, policy_filter=["p0"], fraction=CALIBRATION_FRACTION, seed=SEED
    )
    console.print(
        f"  Sampled {len(calibration_data)} P0 responses ({CALIBRATION_FRACTION:.0%})"
    )

    # Load validation data (target policy responses)
    console.print(f"\n[bold]Loading validation data (target responses)...[/bold]")
    target_policies = ["pi_clone", "pi_cot", "pi_bigger_model", "pi_bad"]
    validation_data = load_responses_for_oracle(
        INPUT_FILE,
        policy_filter=target_policies,
        fraction=VALIDATION_FRACTION,
        seed=SEED + 1,  # Different seed
    )
    console.print(
        f"  Sampled {len(validation_data)} target responses ({VALIDATION_FRACTION:.0%})"
    )

    # Generate oracle labels for calibration
    if calibration_data:
        console.print(f"\n[bold]Generating calibration oracle labels...[/bold]")
        console.print(f"  Using {ORACLE_MODEL} as oracle judge")

        calibration_labeled = add_oracle_labels(
            calibration_data,
            provider="openai",
            model_name=ORACLE_MODEL,
            fraction=1.0,  # Label all sampled data
            seed=SEED,
            template="deterministic",
            checkpoint_dir=str(checkpoint_dir),
        )

        # Save calibration labels in both locations
        output_file = "data/oracle_labels_calibration.jsonl"
        with open(output_file, "w") as f:
            for item in calibration_labeled:
                f.write(json.dumps(item) + "\n")

        # Also save to Phase 2 location
        phase2_dir = Path("../data/labeling")
        phase2_dir.mkdir(parents=True, exist_ok=True)
        phase2_file = phase2_dir / "oracle_labels_calibration_detailed.jsonl"
        with open(phase2_file, "w") as f:
            for item in calibration_labeled:
                f.write(json.dumps(item) + "\n")

        console.print(f"✅ Saved {len(calibration_labeled)} calibration labels")

    # Generate oracle labels for validation
    if validation_data:
        console.print(f"\n[bold]Generating validation oracle labels...[/bold]")

        validation_labeled = add_oracle_labels(
            validation_data,
            provider="openai",
            model_name=ORACLE_MODEL,
            fraction=1.0,  # Label all sampled data
            seed=SEED + 2,  # Different seed from calibration
            template="deterministic",
            checkpoint_dir=str(checkpoint_dir),
        )

        # Save validation labels in both locations
        output_file = "data/oracle_labels_validation.jsonl"
        with open(output_file, "w") as f:
            for item in validation_labeled:
                f.write(json.dumps(item) + "\n")

        # Also save to Phase 2 location
        phase2_dir = Path("../data/labeling")
        phase2_dir.mkdir(parents=True, exist_ok=True)
        phase2_file = phase2_dir / "oracle_labels_validation_detailed.jsonl"
        with open(phase2_file, "w") as f:
            for item in validation_labeled:
                f.write(json.dumps(item) + "\n")

        console.print(f"✅ Saved {len(validation_labeled)} validation labels")

    # Clean up checkpoint directory if empty
    try:
        Path(checkpoint_dir).rmdir()
    except:
        pass  # Directory not empty, that's fine

    console.print(f"\n[bold green]Oracle labeling complete![/bold green]")


if __name__ == "__main__":
    main()
