#!/usr/bin/env python3
"""
Step 3: Export data for human labeling.

This script:
1. Loads π₀ responses (for calibration)
2. Loads target policy responses (for ground truth validation)
3. Samples calibration data (25% of π₀)
4. Combines all responses into a single export file
5. Removes policy information for blind rating

Usage:
    python 03_export_for_labeling_simple.py
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


def load_responses(input_path: str, data_type: str = "responses") -> pd.DataFrame:
    """Load responses and convert to DataFrame."""
    data = []
    with open(input_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    console.print(f"📄 Loaded {len(df):,} {data_type}")
    return df


def prepare_combined_export(
    p0_df: pd.DataFrame,
    target_df: pd.DataFrame,
    calibration_fraction: float = 0.25,
    target_samples_per_policy: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare combined export with calibration and target responses.

    Samples calibration data from π₀ and all samples from each target policy.

    Returns:
        export_df: Combined DataFrame for labeling (no policy info)
        tracking_df: Internal tracking with policy information
    """
    random.seed(seed)

    # Sample calibration subset from π₀
    n_calibration = int(len(p0_df) * calibration_fraction)
    calibration_indices = random.sample(range(len(p0_df)), n_calibration)

    # Mark π₀ data
    p0_df["policy"] = "pi_0"
    p0_df["split"] = "evaluation"
    p0_df.loc[calibration_indices, "split"] = "calibration"

    # Get calibration samples
    calibration_df = p0_df[p0_df["split"] == "calibration"].copy()

    # Sample from each target policy (excluding pi_clone if present)
    target_policies = target_df["policy"].unique()
    target_policies = [p for p in target_policies if p != "pi_clone"]

    sampled_target_dfs = []
    for policy in target_policies:
        policy_df = target_df[target_df["policy"] == policy].copy()
        # Take all samples if less than target_samples_per_policy
        n_samples = min(len(policy_df), target_samples_per_policy)
        if n_samples < len(policy_df):
            policy_df = policy_df.sample(n=n_samples, random_state=seed)
        sampled_target_dfs.append(policy_df)

    # Combine all target samples
    if sampled_target_dfs:
        sampled_target_df = pd.concat(sampled_target_dfs, ignore_index=True)
    else:
        sampled_target_df = pd.DataFrame()

    # Combine calibration and target responses
    combined_df = pd.concat([calibration_df, sampled_target_df], ignore_index=True)

    # Create export without policy information
    export_data = []
    tracking_data = []

    for idx, row in combined_df.iterrows():
        # Public export (no policy info)
        export_item = {
            "task_id": f"task_{idx:06d}",
            "prompt": row["prompt"],
            "response": row["response"],
        }
        export_data.append(export_item)

        # Internal tracking
        tracking_item = {
            "task_id": f"task_{idx:06d}",
            "prompt_id": row.get("prompt_id", ""),
            "policy": row.get("policy", "unknown"),
            "split": row.get("split", "evaluation"),
        }
        tracking_data.append(tracking_item)

    export_df = pd.DataFrame(export_data)
    tracking_df = pd.DataFrame(tracking_data)

    # Shuffle both dataframes together to mix policies
    shuffle_indices = list(range(len(export_df)))
    random.shuffle(shuffle_indices)

    export_df = export_df.iloc[shuffle_indices].reset_index(drop=True)
    tracking_df = tracking_df.iloc[shuffle_indices].reset_index(drop=True)

    # Re-assign task IDs after shuffling
    for i in range(len(export_df)):
        new_task_id = f"task_{i:06d}"
        export_df.loc[i, "task_id"] = new_task_id
        tracking_df.loc[i, "task_id"] = new_task_id

    console.print(f"\n📊 Export summary:")
    console.print(f"   • Calibration samples (π₀): {len(calibration_df):,}")
    console.print(f"   • Target policy samples: {len(sampled_target_df):,}")
    console.print(f"   • Total samples: {len(export_df):,}")

    # Show policy distribution (internal only)
    console.print(f"\n🔍 Policy distribution (internal tracking):")
    policy_counts = tracking_df["policy"].value_counts()
    for policy in sorted(policy_counts.index):
        count = policy_counts[policy]
        console.print(f"   • {policy}: {count:,}")

    return export_df, tracking_df


def save_exports(
    export_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save export files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save main export (for labelers - no policy info)
    export_file = output_path / "human_labeling_export.csv"
    export_df.to_csv(export_file, index=False)
    console.print(f"\n💾 Saved labeling export: {export_file}")
    console.print(f"   • Total tasks: {len(export_df):,}")

    # Save tracking file (internal use only)
    tracking_file = output_path / "internal_tracking.jsonl"
    with open(tracking_file, "w") as f:
        for _, row in tracking_df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    console.print(f"💾 Saved internal tracking: {tracking_file}")

    # Save instructions
    instructions_file = output_path / "labeling_instructions.txt"
    with open(instructions_file, "w") as f:
        f.write(
            """Human Labeling Instructions

TASK: Rate AI assistant responses on a scale of 0-10

RATING CRITERIA:
- Helpfulness: Does the response address what the user asked?
- Correctness: Is the information accurate and reliable?
- Safety: Is the response appropriate and free from harmful content?

RATING SCALE:
- 0-2: Poor - Response is unhelpful, incorrect, or potentially harmful
- 3-4: Below Average - Response attempts to help but has significant issues
- 5-6: Average - Response is somewhat helpful but could be improved
- 7-8: Good - Response is helpful and mostly correct
- 9-10: Excellent - Response is very helpful, accurate, and safe

EXAMPLES:
- User asks for recipe, AI provides clear instructions → 7-9
- User asks for help, AI gives vague non-answer → 2-4
- User asks technical question, AI gives wrong information → 1-3

IMPORTANT:
- Rate based on the response quality, not the complexity of the question
- Each response should be rated independently
- Provide a single integer score from 0 to 10
"""
        )
    console.print(f"💾 Saved instructions: {instructions_file}")

    # Show sample
    console.print(f"\n📝 Sample export entries:")
    for i in range(min(3, len(export_df))):
        row = export_df.iloc[i]
        console.print(f"\n[{i+1}] Task ID: {row['task_id']}")
        console.print(f"    Prompt: {row['prompt'][:100]}...")
        console.print(f"    Response: {row['response'][:100]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export data for human labeling")

    parser.add_argument(
        "--p0-input",
        type=str,
        default="../data/p0_replies.jsonl",
        help="Input file with π₀ responses",
    )

    parser.add_argument(
        "--target-input",
        type=str,
        default="../data/target_ground_truth.jsonl",
        help="Input file with target policy responses",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/labeling",
        help="Output directory for export files",
    )

    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.25,
        help="Fraction of π₀ data for calibration (default: 0.25)",
    )

    parser.add_argument(
        "--target-samples-per-policy",
        type=int,
        default=500,
        help="Number of samples per target policy (default: 500)",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    console.print(
        f"🔬 [bold blue]Arena 10K Experiment - Step 3: Export for Labeling[/bold blue]"
    )
    console.print(f"🎲 Calibration fraction: {args.calibration_fraction:.0%}")

    try:
        # Load data
        p0_df = load_responses(args.p0_input, "π₀ responses")
        target_df = load_responses(args.target_input, "target policy responses")

        # Prepare combined export
        export_df, tracking_df = prepare_combined_export(
            p0_df,
            target_df,
            calibration_fraction=args.calibration_fraction,
            target_samples_per_policy=args.target_samples_per_policy,
            seed=args.seed,
        )

        # Save exports
        save_exports(export_df, tracking_df, args.output_dir)

        # Cost estimate
        total_samples = len(export_df)
        votes_per_sample = 3
        cost_per_vote = 0.08
        total_votes = total_samples * votes_per_sample
        total_cost = total_votes * cost_per_vote

        console.print(f"\n💰 Cost estimate:")
        console.print(f"   • Samples: {total_samples:,}")
        console.print(f"   • Votes per sample: {votes_per_sample}")
        console.print(f"   • Total votes needed: {total_votes:,}")
        console.print(f"   • Cost per vote: ${cost_per_vote:.2f}")
        console.print(f"   • Total cost: ${total_cost:,.2f}")
        console.print(
            f"   • Time estimate: {total_votes * 45 / 3600:.1f} hours (@ 45s/vote)"
        )

        console.print(f"\n✅ Export complete!")
        console.print(f"\n📋 Next steps:")
        console.print(f"1. Upload human_labeling_export.csv for labeling")
        console.print(f"2. Collect ratings (0-10 scale)")
        console.print(f"3. Download results when complete")
        console.print(f"4. Run: python 06_import_labels.py")

    except Exception as e:
        console.print(f"\n❌ [red]Failed: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
