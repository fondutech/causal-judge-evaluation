#!/usr/bin/env python3
"""
Prepare data for CJE pipeline by ensuring compatibility with expected formats.

This script:
1. Converts oracle labels from JSONL to CSV format if needed
2. Ensures judge scores are in the expected format
3. Validates data consistency
"""

import json
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils.progress import console


def convert_oracle_labels_to_csv() -> None:
    """Convert oracle labels from JSONL to CSV format if needed."""

    # Check if CSV already exists
    csv_path = Path("data/labeling/oracle_labels.csv")
    if csv_path.exists():
        console.print(f"✅ Oracle labels CSV already exists: {csv_path}")
        return

    console.print("Converting oracle labels from JSONL to CSV...")

    # Load calibration labels
    cal_labels = []
    cal_file = Path("data/labeling/oracle_labels_calibration_detailed.jsonl")
    if cal_file.exists():
        with open(cal_file) as f:
            for line in f:
                item = json.loads(line)
                cal_labels.append(
                    {
                        "uid": item["uid"],
                        "prompt_id": item["prompt_id"],
                        "policy": "pi_0",  # Calibration is always π₀
                        "oracle_score": item["oracle_label"]["score"]
                        / 10.0,  # Convert to 0-1
                        "dataset_type": "calibration",
                        "model": item["oracle_label"].get("model", "unknown"),
                        "task_id": f"cal_{item['uid']}",  # Create unique task ID
                    }
                )

    # Load validation labels
    val_labels = []
    val_file = Path("data/labeling/oracle_labels_validation_detailed.jsonl")
    if val_file.exists():
        with open(val_file) as f:
            for line in f:
                item = json.loads(line)
                val_labels.append(
                    {
                        "uid": item["uid"],
                        "prompt_id": item["prompt_id"],
                        "policy": item["policy"],
                        "oracle_score": item["oracle_label"]["score"]
                        / 10.0,  # Convert to 0-1
                        "dataset_type": "validation",
                        "model": item["oracle_label"].get("model", "unknown"),
                        "task_id": f"val_{item['uid']}",  # Create unique task ID
                    }
                )

    # Combine and save
    all_labels = cal_labels + val_labels
    df = pd.DataFrame(all_labels)

    # Add rating column (0-10 scale) for compatibility with import script
    df["rating"] = df["oracle_score"] * 10

    df.to_csv(csv_path, index=False)
    console.print(f"✅ Saved {len(df)} oracle labels to {csv_path}")

    # Print summary
    console.print("\nOracle Label Summary:")
    console.print(f"  Calibration labels: {len(cal_labels)}")
    console.print(f"  Validation labels: {len(val_labels)}")
    console.print(f"  Policies: {sorted(df['policy'].unique())}")
    console.print(
        f"  Score range: [{df['oracle_score'].min():.2f}, {df['oracle_score'].max():.2f}]"
    )


def validate_judge_scores() -> None:
    """Validate judge score files are ready for CJE."""

    files_to_check = [
        ("Deterministic scores", "data/p0_scored_deterministic.jsonl"),
        ("Uncertainty scores", "data/p0_scored_uncertainty.jsonl"),
    ]

    console.print("\nValidating judge score files...")

    for name, path in files_to_check:
        if not Path(path).exists():
            console.print(f"❌ {name} not found: {path}")
            console.print(
                f"   Run: python scripts/04{'b' if 'uncertainty' in path else 'a'}_add_judge_scores_*.py"
            )
            continue

        # Check format
        with open(path) as f:
            first_line = json.loads(f.readline())

        required_fields = ["prompt", "response", "judge_score_raw", "judge_variance"]
        missing = [f for f in required_fields if f not in first_line]

        if missing:
            console.print(f"⚠️ {name} missing fields: {missing}")
        else:
            # Count records
            line_count = sum(1 for _ in open(path))
            console.print(f"✅ {name}: {line_count} records with all required fields")


def prepare_cje_data() -> None:
    """Prepare final data file for CJE pipeline."""

    # For CJE, we need scored responses with optional oracle labels
    # The main input is the judge-scored π₀ responses

    scored_file = Path("data/p0_scored_deterministic.jsonl")
    if not scored_file.exists():
        console.print("❌ Judge scores not found. Run scoring first.")
        return

    console.print("\nPreparing CJE input data...")

    # Load scored responses
    responses = []
    with open(scored_file) as f:
        for line in f:
            responses.append(json.loads(line))

    console.print(f"✅ Loaded {len(responses)} scored responses")

    # The CJE pipeline will handle matching with oracle labels internally
    # based on the configuration oracle.calibration_file and oracle.validation_file

    console.print("\nData is ready for CJE pipeline!")
    console.print("Run: cje run --cfg-path configs --cfg-name arena_10k_oracle")


def main() -> None:
    """Run all preparation steps."""
    console.print("[bold]Preparing Arena 10K Oracle data for CJE pipeline[/bold]\n")

    # 1. Convert oracle labels if needed
    convert_oracle_labels_to_csv()

    # 2. Validate judge scores
    validate_judge_scores()

    # 3. Prepare final data
    prepare_cje_data()


if __name__ == "__main__":
    main()
