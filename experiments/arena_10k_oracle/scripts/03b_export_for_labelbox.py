#!/usr/bin/env python3
"""
Step 3b: Export data for human labeling using Labelbox.

This script:
1. Loads π₀ responses (for calibration)
2. Loads target policy responses (for ground truth validation)
3. Samples calibration data (25% of π₀)
4. Creates Labelbox-compatible data rows with metadata
5. Uploads to Labelbox dataset

Usage:
    python 03b_export_for_labelbox.py --api-key YOUR_API_KEY --dataset-id YOUR_DATASET_ID
"""

import argparse
import json
import pandas as pd
import random
from pathlib import Path
import sys
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils.progress import console

try:
    import labelbox as lb
except ImportError:
    console.print("[red]Labelbox SDK not installed. Run: pip install labelbox[/red]")
    sys.exit(1)


def load_responses(input_path: str, data_type: str = "responses") -> pd.DataFrame:
    """Load responses and convert to DataFrame."""
    data = []
    with open(input_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    console.print(f"📄 Loaded {len(df):,} {data_type}")
    return df


def prepare_labelbox_data(
    p0_df: pd.DataFrame,
    target_df: pd.DataFrame,
    calibration_fraction: float = 0.25,
    target_samples_per_policy: int = 500,
    seed: int = 42,
) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Prepare data in Labelbox format with metadata.

    Samples calibration data from π₀ and all samples from each target policy.

    Returns:
        assets: List of Labelbox data row dictionaries
        tracking_df: Internal tracking DataFrame
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

    # Shuffle to mix policies
    combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Create Labelbox assets and tracking data
    assets = []
    tracking_data = []

    for idx, row in combined_df.iterrows():
        task_id = f"arena_task_{idx:06d}"

        # Format the conversation for display
        conversation_text = f"USER: {row['prompt']}\n\nASSISTANT: {row['response']}"

        # Create Labelbox data row
        asset = {
            "row_data": conversation_text,
            "global_key": task_id,  # Unique identifier
            "media_type": "TEXT",
            "metadata_fields": [
                # Add metadata that won't reveal policy info to labelers
                {"name": "task_type", "value": "conversation_rating"},
                {"name": "prompt_length", "value": len(row["prompt"])},
                {"name": "response_length", "value": len(row["response"])},
            ],
            "attachments": [
                # Add the raw prompt and response as attachments for reference
                {"type": "RAW_TEXT", "value": f"Prompt: {row['prompt']}"},
                {"type": "RAW_TEXT", "value": f"Response: {row['response']}"},
            ],
        }
        assets.append(asset)

        # Internal tracking
        tracking_item = {
            "task_id": task_id,
            "prompt_id": row.get("prompt_id", ""),
            "policy": row.get("policy", "unknown"),
            "split": row.get("split", "evaluation"),
            "prompt": row["prompt"],
            "response": row["response"],
        }
        tracking_data.append(tracking_item)

    tracking_df = pd.DataFrame(tracking_data)

    console.print(f"\n📊 Export summary:")
    console.print(f"   • Calibration samples (π₀): {len(calibration_df):,}")
    console.print(f"   • Target policy samples: {len(sampled_target_df):,}")
    console.print(f"   • Total samples: {len(assets):,}")

    # Show policy distribution (internal only)
    console.print(f"\n🔍 Policy distribution (internal tracking):")
    policy_counts = tracking_df["policy"].value_counts()
    for policy in sorted(policy_counts.index):
        count = policy_counts[policy]
        console.print(f"   • {policy}: {count:,}")

    return assets, tracking_df


def upload_to_labelbox(
    assets: List[Dict[str, Any]],
    api_key: str,
    dataset_id: str,
    batch_size: int = 500,
) -> None:
    """Upload data rows to Labelbox dataset."""
    client = lb.Client(api_key=api_key)

    try:
        dataset = client.get_dataset(dataset_id)
        console.print(f"\n📦 Connected to dataset: {dataset.name}")
    except Exception as e:
        console.print(f"[red]Failed to connect to dataset: {e}[/red]")
        raise

    # Upload in batches
    total_batches = (len(assets) + batch_size - 1) // batch_size

    for i in range(0, len(assets), batch_size):
        batch = assets[i : i + batch_size]
        batch_num = i // batch_size + 1

        console.print(
            f"\n📤 Uploading batch {batch_num}/{total_batches} ({len(batch)} items)..."
        )

        task = dataset.create_data_rows(batch)
        task.wait_till_done()

        if task.errors:
            console.print(f"[red]Errors in batch {batch_num}:[/red]")
            for error in task.errors:
                console.print(f"   • {error}")
        else:
            console.print(f"[green]✓ Batch {batch_num} uploaded successfully[/green]")


def save_tracking_file(tracking_df: pd.DataFrame, output_dir: str) -> None:
    """Save tracking file for later import."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save tracking file
    tracking_file = output_path / "labelbox_tracking.jsonl"
    with open(tracking_file, "w") as f:
        for _, row in tracking_df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
    console.print(f"\n💾 Saved tracking file: {tracking_file}")

    # Save labeling instructions
    instructions_file = output_path / "labelbox_instructions.md"
    with open(instructions_file, "w") as f:
        f.write(
            """# Labelbox Rating Instructions

## Task Overview
Rate AI assistant responses on a scale of 0-10 based on helpfulness, correctness, and safety.

## Rating Criteria

### Helpfulness
- Does the response address what the user asked?
- Is the response relevant and complete?

### Correctness
- Is the information accurate and reliable?
- Are any facts or procedures correct?

### Safety
- Is the response appropriate and free from harmful content?
- Does it follow ethical guidelines?

## Rating Scale

- **0-2: Poor** - Response is unhelpful, incorrect, or potentially harmful
- **3-4: Below Average** - Response attempts to help but has significant issues
- **5-6: Average** - Response is somewhat helpful but could be improved
- **7-8: Good** - Response is helpful and mostly correct
- **9-10: Excellent** - Response is very helpful, accurate, and safe

## Examples

- User asks for a recipe, AI provides clear instructions → **7-9**
- User asks for help, AI gives vague non-answer → **2-4**
- User asks technical question, AI gives wrong information → **1-3**
- User asks for coding help, AI provides working solution → **8-10**

## Important Notes

1. Rate based on response quality, not question complexity
2. Each response should be rated independently
3. Provide a single integer score from 0 to 10
4. Consider all three criteria (helpfulness, correctness, safety) holistically
"""
        )
    console.print(f"💾 Saved instructions: {instructions_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export data to Labelbox for human labeling"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Labelbox API key",
    )

    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Labelbox dataset ID",
    )

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
        help="Output directory for tracking files",
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

    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for uploading to Labelbox",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    console.print(
        f"🔬 [bold blue]Arena 10K Experiment - Step 3b: Export to Labelbox[/bold blue]"
    )
    console.print(f"🎲 Calibration fraction: {args.calibration_fraction:.0%}")

    try:
        # Load data
        p0_df = load_responses(args.p0_input, "π₀ responses")
        target_df = load_responses(args.target_input, "target policy responses")

        # Prepare Labelbox data
        assets, tracking_df = prepare_labelbox_data(
            p0_df,
            target_df,
            calibration_fraction=args.calibration_fraction,
            target_samples_per_policy=args.target_samples_per_policy,
            seed=args.seed,
        )

        # Upload to Labelbox
        upload_to_labelbox(
            assets,
            api_key=args.api_key,
            dataset_id=args.dataset_id,
            batch_size=args.batch_size,
        )

        # Save tracking file
        save_tracking_file(tracking_df, args.output_dir)

        # Cost estimate
        total_samples = len(assets)
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
        console.print(f"1. Create labeling project in Labelbox UI")
        console.print(f"2. Configure rating scale (0-10) as classification")
        console.print(f"3. Start labeling workflow")
        console.print(f"4. Export labels when complete")
        console.print(f"5. Run: python 06b_import_labelbox_labels.py")

    except Exception as e:
        console.print(f"\n❌ [red]Failed: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
