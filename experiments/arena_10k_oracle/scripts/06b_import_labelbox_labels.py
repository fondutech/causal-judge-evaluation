#!/usr/bin/env python3
"""
Step 6b: Import human labels from Labelbox.

This script:
1. Exports labels from Labelbox project
2. Matches labels with tracking file
3. Computes aggregated scores (mean/median)
4. Adds policy and split information
5. Saves enriched labeled data

Usage:
    python 06b_import_labelbox_labels.py --api-key YOUR_API_KEY --project-id YOUR_PROJECT_ID
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Any
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils.progress import console

try:
    import labelbox as lb
except ImportError:
    console.print("[red]Labelbox SDK not installed. Run: pip install labelbox[/red]")
    sys.exit(1)


def download_labelbox_labels(api_key: str, project_id: str) -> List[Dict[str, Any]]:
    """Download labels from Labelbox project."""
    client = lb.Client(api_key=api_key)

    try:
        project = client.get_project(project_id)
        console.print(f"📦 Connected to project: {project.name}")
    except Exception as e:
        console.print(f"[red]Failed to connect to project: {e}[/red]")
        raise

    # Export labels
    console.print("📥 Exporting labels from Labelbox...")

    # Configure export params - adjust based on your project setup
    export_params = {
        "data_row_details": True,
        "metadata_fields": True,
        "attachments": True,
        "project_details": True,
        "label_details": True,
    }

    export_task = project.export_labels(**export_params)
    export_task.wait_till_done()

    if export_task.errors:
        console.print("[red]Export errors:[/red]")
        for error in export_task.errors:
            console.print(f"   • {error}")
        raise Exception("Failed to export labels")

    # Get the export result
    labels = list(export_task.result)
    console.print(f"✅ Downloaded {len(labels):,} labeled items")

    return labels


def load_tracking_data(tracking_file: str) -> pd.DataFrame:
    """Load tracking data from export step."""
    data = []
    with open(tracking_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    console.print(f"📄 Loaded {len(df):,} tracking entries")
    return df


def extract_ratings(label_data: Dict[str, Any]) -> List[float]:
    """Extract rating scores from Labelbox label format."""
    ratings = []

    # The exact structure depends on your Labelbox project configuration
    # This assumes a classification task with numeric ratings 0-10

    if "Label" in label_data and "classifications" in label_data["Label"]:
        for classification in label_data["Label"]["classifications"]:
            # Assuming the classification has a "value" field with the rating
            if "value" in classification:
                value = classification["value"]
                # Handle different possible formats
                if isinstance(value, dict) and "answer" in value:
                    try:
                        rating = float(value["answer"])
                        ratings.append(rating)
                    except (ValueError, TypeError):
                        console.print(
                            f"[yellow]Warning: Could not parse rating: {value}[/yellow]"
                        )
                elif isinstance(value, (int, float)):
                    ratings.append(float(value))

    # Alternative: check for "labels" field (different Labelbox versions)
    elif "labels" in label_data:
        for label in label_data["labels"]:
            if "score" in label or "rating" in label:
                try:
                    rating = float(label.get("score", label.get("rating")))
                    ratings.append(rating)
                except (ValueError, TypeError):
                    pass

    return ratings


def process_labels(
    labels: List[Dict[str, Any]],
    tracking_df: pd.DataFrame,
) -> pd.DataFrame:
    """Process Labelbox labels and merge with tracking data."""

    # Create a mapping from global_key to tracking data
    tracking_map = {row["task_id"]: row for _, row in tracking_df.iterrows()}

    processed_data = []
    missing_count = 0

    for label_item in labels:
        # Get the global key (task_id)
        global_key = label_item.get("data_row", {}).get("global_key", "")

        if not global_key:
            console.print(f"[yellow]Warning: Missing global_key in label item[/yellow]")
            continue

        # Look up tracking data
        if global_key not in tracking_map:
            missing_count += 1
            continue

        tracking_data = tracking_map[global_key]

        # Extract ratings
        ratings = extract_ratings(label_item)

        if not ratings:
            console.print(
                f"[yellow]Warning: No ratings found for {global_key}[/yellow]"
            )
            continue

        # Create processed entry
        processed_entry = {
            "task_id": global_key,
            "prompt_id": tracking_data["prompt_id"],
            "prompt": tracking_data["prompt"],
            "response": tracking_data["response"],
            "policy": tracking_data["policy"],
            "split": tracking_data["split"],
            "ratings": ratings,
            "rating_count": len(ratings),
            "mean_rating": np.mean(ratings),
            "median_rating": np.median(ratings),
            "std_rating": np.std(ratings) if len(ratings) > 1 else 0.0,
            "min_rating": min(ratings),
            "max_rating": max(ratings),
        }

        processed_data.append(processed_entry)

    if missing_count > 0:
        console.print(
            f"[yellow]Warning: {missing_count} labels had no matching tracking data[/yellow]"
        )

    df = pd.DataFrame(processed_data)
    console.print(f"\n📊 Processed {len(df):,} labeled items")

    # Show summary statistics
    console.print(f"\n📈 Rating statistics:")
    console.print(f"   • Mean rating: {df['mean_rating'].mean():.2f}")
    console.print(f"   • Std rating: {df['mean_rating'].std():.2f}")
    console.print(f"   • Ratings per item: {df['rating_count'].mean():.1f}")

    # Show by policy
    console.print(f"\n🔍 Ratings by policy:")
    for policy in df["policy"].unique():
        policy_df = df[df["policy"] == policy]
        console.print(
            f"   • {policy}: {policy_df['mean_rating'].mean():.2f} ± {policy_df['mean_rating'].std():.2f}"
        )

    return df


def save_labeled_data(df: pd.DataFrame, output_dir: str) -> None:
    """Save processed labeled data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as JSONL for compatibility with CJE pipeline
    output_file = output_path / "human_labeled_scores.jsonl"
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            # Format for CJE pipeline
            entry = {
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "response": row["response"],
                "policy": row["policy"],
                "split": row["split"],
                "human_score": row["mean_rating"] / 10.0,  # Normalize to [0, 1]
                "human_ratings": row["ratings"],
                "rating_stats": {
                    "mean": row["mean_rating"],
                    "median": row["median_rating"],
                    "std": row["std_rating"],
                    "count": row["rating_count"],
                },
            }
            f.write(json.dumps(entry) + "\n")

    console.print(f"\n💾 Saved labeled data: {output_file}")

    # Also save as CSV for easy inspection
    csv_file = output_path / "human_labeled_scores.csv"
    df.to_csv(csv_file, index=False)
    console.print(f"💾 Saved CSV: {csv_file}")

    # Save summary report
    report_file = output_path / "labeling_report.txt"
    with open(report_file, "w") as f:
        f.write("Labelbox Human Labeling Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total labeled items: {len(df):,}\n")
        f.write(f"Mean rating: {df['mean_rating'].mean():.2f}\n")
        f.write(f"Std rating: {df['mean_rating'].std():.2f}\n\n")

        f.write("By Policy:\n")
        for policy in df["policy"].unique():
            policy_df = df[df["policy"] == policy]
            f.write(
                f"  {policy}: {len(policy_df):,} items, mean={policy_df['mean_rating'].mean():.2f}\n"
            )

        f.write("\nBy Split:\n")
        for split in df["split"].unique():
            split_df = df[df["split"] == split]
            f.write(
                f"  {split}: {len(split_df):,} items, mean={split_df['mean_rating'].mean():.2f}\n"
            )

    console.print(f"💾 Saved report: {report_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import human labels from Labelbox")

    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Labelbox API key",
    )

    parser.add_argument(
        "--project-id",
        type=str,
        required=True,
        help="Labelbox project ID",
    )

    parser.add_argument(
        "--tracking-file",
        type=str,
        default="../data/labeling/labelbox_tracking.jsonl",
        help="Tracking file from export step",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/labeling",
        help="Output directory for labeled data",
    )

    args = parser.parse_args()

    console.print(
        f"🔬 [bold blue]Arena 10K Experiment - Step 6b: Import Labelbox Labels[/bold blue]"
    )

    try:
        # Download labels from Labelbox
        labels = download_labelbox_labels(args.api_key, args.project_id)

        # Load tracking data
        tracking_df = load_tracking_data(args.tracking_file)

        # Process and merge labels
        labeled_df = process_labels(labels, tracking_df)

        # Save results
        save_labeled_data(labeled_df, args.output_dir)

        console.print(f"\n✅ Import complete!")
        console.print(f"\n📋 Next steps:")
        console.print(f"1. Review human_labeled_scores.jsonl")
        console.print(f"2. Check labeling_report.txt for quality metrics")
        console.print(f"3. Use labeled data for CJE calibration and validation")

    except Exception as e:
        console.print(f"\n❌ [red]Failed: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
