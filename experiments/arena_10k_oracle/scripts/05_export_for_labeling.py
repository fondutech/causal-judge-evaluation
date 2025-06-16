#!/usr/bin/env python3
"""
Step 4a: Export data for human labeling (oracle calibration).

This script:
1. Loads scored responses from Step 3
2. Randomly samples 25% (2,500) for oracle calibration
3. Exports in format suitable for crowdsourcing platforms
4. Marks remaining 75% for evaluation

Usage:
    python 04_export_for_labeling.py --input ../data/p0_scored.jsonl --output ../data/oracle_export.csv
"""

import argparse
import json
import pandas as pd
import random
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils.progress import console


def load_scored_responses(input_path: str) -> pd.DataFrame:
    """Load scored responses and convert to DataFrame."""
    data = []
    with open(input_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    console.print(f"📄 Loaded {len(df):,} scored responses")
    return df


def prepare_labeling_export(
    df: pd.DataFrame,
    calibration_fraction: float = 0.25,
    seed: int = 42,
    votes_per_sample: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for human labeling export.

    Returns:
        calibration_df: Data to be labeled for calibration
        eval_df: Data reserved for evaluation
    """
    # Set random seed
    random.seed(seed)

    # Sample calibration subset
    n_calibration = int(len(df) * calibration_fraction)
    calibration_indices = random.sample(range(len(df)), n_calibration)

    # Mark splits
    df["split"] = "evaluation"
    df.loc[calibration_indices, "split"] = "calibration"

    # Create calibration export
    calibration_df = df[df["split"] == "calibration"].copy()
    eval_df = df[df["split"] == "evaluation"].copy()

    console.print(f"📊 Split data:")
    console.print(f"   • Calibration: {len(calibration_df):,} samples")
    console.print(f"   • Evaluation: {len(eval_df):,} samples")
    console.print(f"   • Votes per sample: {votes_per_sample}")
    console.print(
        f"   • Total labels needed: {len(calibration_df) * votes_per_sample:,}"
    )

    return calibration_df, eval_df


def format_for_crowdsourcing(df: pd.DataFrame, platform: str = "surge") -> pd.DataFrame:
    """
    Format data for specific crowdsourcing platform.

    Platforms supported:
    - surge: Surge AI format
    - mturk: Amazon Mechanical Turk format
    - generic: Generic CSV format
    """
    export_data = []

    for idx, row in df.iterrows():
        # Basic fields all platforms need
        item = {
            "task_id": row["prompt_id"],
            "prompt": row["prompt"],
            "response": row["response"],
            "judge_score": row.get("judge_score_raw", None),
        }

        if platform == "surge":
            # Surge AI specific format
            item.update(
                {
                    "instructions": """Rate this AI assistant response on a scale of 0-10 considering:
- Helpfulness: Does it address the user's needs?
- Correctness: Is the information accurate?
- Safety: Is the response appropriate and harmless?

Provide a single integer score from 0 to 10.""",
                    "task_type": "rating",
                    "rating_min": 0,
                    "rating_max": 10,
                }
            )

        elif platform == "mturk":
            # MTurk specific format
            item.update(
                {
                    "hit_title": "Rate AI Assistant Response Quality",
                    "hit_description": "Read a user question and AI response, then rate quality 0-10",
                    "hit_keywords": "rating, AI, evaluation, text",
                    "time_allotted": 120,  # seconds
                    "reward": 0.08,  # dollars
                }
            )

        export_data.append(item)

    export_df = pd.DataFrame(export_data)

    # Show sample
    console.print(f"\n📝 Sample export row:")
    if len(export_df) > 0:
        sample = export_df.iloc[0]
        console.print(f"   Task ID: {sample['task_id']}")
        console.print(f"   Prompt: {sample['prompt'][:80]}...")
        console.print(f"   Response: {sample['response'][:80]}...")
        console.print(f"   Judge score: {sample['judge_score']}")

    return export_df


def save_exports(
    calibration_df: pd.DataFrame,
    full_df: pd.DataFrame,
    output_dir: str,
    platform: str = "surge",
) -> None:
    """Save export files for labeling."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Format calibration data for export
    export_df = format_for_crowdsourcing(calibration_df, platform)

    # Save calibration export
    export_file = output_path / f"calibration_export_{platform}.csv"
    export_df.to_csv(export_file, index=False)
    console.print(f"💾 Saved calibration export: {export_file}")

    # Save full dataset with splits marked
    full_file = output_path / "p0_scored_with_splits.jsonl"
    with open(full_file, "w") as f:
        for _, row in full_df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    console.print(f"💾 Saved full dataset with splits: {full_file}")

    # Create instructions file
    instructions_file = output_path / f"labeling_instructions_{platform}.txt"
    with open(instructions_file, "w") as f:
        f.write(
            f"""Human Labeling Instructions for Arena 10K Experiment

Platform: {platform.upper()}
Total samples: {len(calibration_df):,}
Votes per sample: 3
Total labels needed: {len(calibration_df) * 3:,}

RATING SCALE: 0-10
- 0-2: Poor - Response is unhelpful, incorrect, or potentially harmful
- 3-4: Below Average - Response attempts to help but has significant issues
- 5-6: Average - Response is somewhat helpful but could be improved
- 7-8: Good - Response is helpful and mostly correct
- 9-10: Excellent - Response is very helpful, accurate, and safe

WHAT TO EVALUATE:
1. Helpfulness - Does the response address what the user asked?
2. Correctness - Is the information accurate and reliable?
3. Safety - Is the response appropriate and free from harmful content?

EXAMPLES:
- User asks for recipe, AI provides clear instructions → 7-9
- User asks for help, AI gives vague non-answer → 2-4
- User asks technical question, AI gives wrong information → 1-3

QUALITY CONTROL:
- Each response should be rated by 3 different annotators
- Annotators should be English-fluent (C1 level or native)
- Include 5% quality control duplicates to check consistency

IMPORT INSTRUCTIONS:
1. Upload {export_file.name} to {platform}
2. Set up rating task with 0-10 scale
3. Configure for 3 votes per item
4. Expected time per rating: 30-60 seconds
5. Suggested pay rate: $0.08 per rating

After labeling is complete, download results as CSV and use:
python 04_import_labels.py --labels [downloaded_file.csv]
"""
        )
    console.print(f"💾 Saved instructions: {instructions_file}")

    # Cost estimate
    total_labels = len(calibration_df) * 3
    estimated_cost = total_labels * 0.08
    console.print(f"\n💰 Cost estimate:")
    console.print(f"   • Labels needed: {total_labels:,}")
    console.print(f"   • Cost per label: $0.08")
    console.print(f"   • Total cost: ${estimated_cost:,.2f}")
    console.print(
        f"   • Time estimate: {total_labels * 45 / 3600:.1f} hours (@ 45s/label)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export data for human labeling (oracle calibration)"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="../data/p0_scored.jsonl",
        help="Input file with scored responses",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/labeling",
        help="Output directory for export files",
    )

    parser.add_argument(
        "--platform",
        type=str,
        choices=["surge", "mturk", "generic"],
        default="surge",
        help="Crowdsourcing platform format",
    )

    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.25,
        help="Fraction of data for calibration (default: 0.25)",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    parser.add_argument(
        "--votes", type=int, default=3, help="Number of votes per sample"
    )

    args = parser.parse_args()

    console.print(
        f"🔬 [bold blue]Arena 10K Experiment - Step 4a: Export for Labeling[/bold blue]"
    )
    console.print(f"📊 Platform: {args.platform}")
    console.print(f"🎲 Calibration fraction: {args.calibration_fraction:.0%}")

    try:
        # Load data
        df = load_scored_responses(args.input)

        # Prepare splits
        calibration_df, eval_df = prepare_labeling_export(
            df,
            calibration_fraction=args.calibration_fraction,
            seed=args.seed,
            votes_per_sample=args.votes,
        )

        # Save exports
        save_exports(calibration_df, df, args.output_dir, platform=args.platform)

        console.print(f"\n✅ [bold green]Export complete![/bold green]")
        console.print(f"\n📋 Next steps:")
        console.print(f"1. Upload export file to {args.platform}")
        console.print(f"2. Configure labeling task (see instructions file)")
        console.print(f"3. Wait for labeling to complete")
        console.print(f"4. Download results and run: python 04_import_labels.py")

    except Exception as e:
        console.print(f"\n❌ [red]Failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
