#!/usr/bin/env python3
"""
Step 6: Import human labels and run calibration.

This script:
1. Imports human labels from crowdsourcing platform
2. Aggregates multiple votes per sample
3. Fits isotonic calibration from judge scores to human labels
4. Applies calibration to all data
5. Saves calibrated dataset for next steps

Usage:
    python 04_import_labels.py --labels ../data/labeling/human_labels.csv --data ../data/labeling/p0_scored_with_splits.jsonl
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.utils.progress import console


def load_human_labels(labels_path: str, platform: str = "surge") -> pd.DataFrame:
    """Load and validate human labels from crowdsourcing platform."""
    console.print(f"📄 Loading human labels from {labels_path}")

    # Load labels
    labels_df = pd.read_csv(labels_path)

    # Platform-specific parsing
    if platform == "surge":
        # Expected columns: task_id, worker_id, rating
        required_cols = ["task_id", "rating"]
    elif platform == "mturk":
        # Expected columns: HITId, WorkerId, Answer.rating
        labels_df = labels_df.rename(
            columns={"HITId": "task_id", "Answer.rating": "rating"}
        )
        required_cols = ["task_id", "rating"]
    else:
        # Generic format
        required_cols = ["task_id", "rating"]

    # Validate columns
    missing_cols = set(required_cols) - set(labels_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert rating to numeric
    labels_df["rating"] = pd.to_numeric(labels_df["rating"], errors="coerce")

    # Remove invalid ratings
    valid_mask = (labels_df["rating"] >= 0) & (labels_df["rating"] <= 10)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        console.print(f"⚠️  [yellow]Removing {invalid_count} invalid ratings[/yellow]")
        labels_df = labels_df[valid_mask]

    console.print(f"✅ Loaded {len(labels_df):,} valid ratings")
    console.print(f"📊 Unique tasks rated: {labels_df['task_id'].nunique():,}")

    return labels_df


def aggregate_votes(labels_df: pd.DataFrame, min_votes: int = 2) -> pd.DataFrame:
    """Aggregate multiple votes per task."""
    console.print(f"📊 Aggregating votes...")

    # Group by task and aggregate
    agg_df = (
        labels_df.groupby("task_id")
        .agg({"rating": ["mean", "std", "count"]})
        .reset_index()
    )

    # Flatten column names
    agg_df.columns = ["task_id", "human_score", "human_score_std", "vote_count"]

    # Filter by minimum votes
    before_filter = len(agg_df)
    agg_df = agg_df[agg_df["vote_count"] >= min_votes]
    after_filter = len(agg_df)

    if before_filter > after_filter:
        console.print(
            f"⚠️  [yellow]Filtered {before_filter - after_filter} tasks with < {min_votes} votes[/yellow]"
        )

    console.print(f"✅ Aggregated to {len(agg_df):,} tasks")
    console.print(f"📊 Average votes per task: {agg_df['vote_count'].mean():.1f}")
    console.print(f"📊 Average inter-rater std: {agg_df['human_score_std'].mean():.2f}")

    return agg_df


def load_full_dataset(data_path: str) -> pd.DataFrame:
    """Load the full dataset with splits."""
    data = []
    with open(data_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    console.print(f"📄 Loaded {len(df):,} total samples")
    console.print(f"   • Calibration: {(df['split'] == 'calibration').sum():,}")
    console.print(f"   • Evaluation: {(df['split'] == 'evaluation').sum():,}")

    return df


def fit_calibration(
    df: pd.DataFrame,
    human_scores_df: pd.DataFrame,
    judge_col: str = "judge_score_raw",
    plot: bool = True,
) -> IsotonicRegression:
    """Fit isotonic calibration from judge scores to human scores."""
    console.print(f"🔬 Fitting isotonic calibration...")

    # Merge human scores with calibration data
    cal_df = df[df["split"] == "calibration"].copy()
    cal_df = cal_df.merge(
        human_scores_df[["task_id", "human_score"]],
        left_on="prompt_id",
        right_on="task_id",
        how="inner",
    )

    console.print(f"📊 Calibration samples with human labels: {len(cal_df):,}")

    # Remove any samples with missing judge scores
    cal_df = cal_df.dropna(subset=[judge_col, "human_score"])

    # Fit isotonic regression
    iso_model = IsotonicRegression(out_of_bounds="clip")
    iso_model.fit(cal_df[judge_col], cal_df["human_score"])

    # Calculate calibration metrics
    cal_df["calibrated_score"] = iso_model.predict(cal_df[judge_col])

    # Metrics
    from scipy.stats import spearmanr, pearsonr

    spearman_before = spearmanr(cal_df[judge_col], cal_df["human_score"])[0]
    spearman_after = spearmanr(cal_df["calibrated_score"], cal_df["human_score"])[0]
    pearson_before = pearsonr(cal_df[judge_col], cal_df["human_score"])[0]
    pearson_after = pearsonr(cal_df["calibrated_score"], cal_df["human_score"])[0]

    mae_before = np.abs(cal_df[judge_col] - cal_df["human_score"]).mean()
    mae_after = np.abs(cal_df["calibrated_score"] - cal_df["human_score"]).mean()

    console.print(f"\n📊 Calibration metrics:")
    console.print(f"   • Spearman ρ: {spearman_before:.3f} → {spearman_after:.3f}")
    console.print(f"   • Pearson r: {pearson_before:.3f} → {pearson_after:.3f}")
    console.print(f"   • MAE: {mae_before:.2f} → {mae_after:.2f}")

    if plot:
        # Create calibration plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Before calibration
        ax1.scatter(cal_df[judge_col], cal_df["human_score"], alpha=0.5)
        ax1.plot([0, 10], [0, 10], "k--", alpha=0.5)
        ax1.set_xlabel("Judge Score (Raw)")
        ax1.set_ylabel("Human Score")
        ax1.set_title(f"Before Calibration (ρ={spearman_before:.3f})")
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)

        # After calibration with isotonic curve
        ax2.scatter(cal_df[judge_col], cal_df["human_score"], alpha=0.5, label="Data")

        # Plot isotonic curve
        x_range = np.linspace(0, 10, 100)
        y_isotonic = iso_model.predict(x_range)
        ax2.plot(x_range, y_isotonic, "r-", linewidth=2, label="Isotonic fit")
        ax2.plot([0, 10], [0, 10], "k--", alpha=0.5, label="Perfect calibration")

        ax2.set_xlabel("Judge Score (Raw)")
        ax2.set_ylabel("Calibrated Score")
        ax2.set_title(f"After Calibration (ρ={spearman_after:.3f})")
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.legend()

        plt.tight_layout()
        plot_path = (
            Path(cal_df["prompt_id"].iloc[0]).parent.parent / "calibration_plot.png"
        )
        plt.savefig(plot_path, dpi=150)
        plt.close()

        console.print(f"📊 Saved calibration plot: {plot_path}")

    return iso_model


def apply_calibration(df: pd.DataFrame, iso_model: IsotonicRegression) -> pd.DataFrame:
    """Apply calibration to all data."""
    console.print(f"🔧 Applying calibration to all {len(df):,} samples...")

    # Apply calibration
    df["reward_calibrated"] = iso_model.predict(df["judge_score_raw"])

    # Add calibration metadata
    df["calibration"] = {
        "method": "isotonic",
        "source": "human_oracle",
        "out_of_bounds": "clip",
    }

    # Summary statistics
    console.print(f"📊 Calibrated score statistics:")
    console.print(f"   • Mean: {df['reward_calibrated'].mean():.2f}")
    console.print(f"   • Std: {df['reward_calibrated'].std():.2f}")
    console.print(
        f"   • Range: [{df['reward_calibrated'].min():.2f}, {df['reward_calibrated'].max():.2f}]"
    )

    return df


def save_calibrated_data(
    df: pd.DataFrame, output_path: str, iso_model: IsotonicRegression
) -> None:
    """Save calibrated data and model."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save calibrated data
    with open(output_file, "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")

    console.print(f"💾 Saved calibrated data: {output_file}")

    # Save calibration model
    import joblib

    model_path = output_file.parent / "judge_calibration_model.pkl"
    joblib.dump(iso_model, model_path)
    console.print(f"💾 Saved calibration model: {model_path}")

    # Save summary statistics
    summary = {
        "total_samples": len(df),
        "calibration_samples": len(df[df["split"] == "calibration"]),
        "evaluation_samples": len(df[df["split"] == "evaluation"]),
        "calibrated_score_mean": float(df["reward_calibrated"].mean()),
        "calibrated_score_std": float(df["reward_calibrated"].std()),
        "calibrated_score_min": float(df["reward_calibrated"].min()),
        "calibrated_score_max": float(df["reward_calibrated"].max()),
    }

    summary_path = output_file.parent / "calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"💾 Saved summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import human labels and calibrate judge scores"
    )

    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to human labels CSV from crowdsourcing",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="../data/labeling/p0_scored_with_splits.jsonl",
        help="Path to full dataset with splits",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../data/p0_calibrated.jsonl",
        help="Output path for calibrated data",
    )

    parser.add_argument(
        "--platform",
        type=str,
        choices=["surge", "mturk", "generic"],
        default="surge",
        help="Crowdsourcing platform format",
    )

    parser.add_argument(
        "--min-votes", type=int, default=2, help="Minimum votes per sample to include"
    )

    parser.add_argument(
        "--no-plot", action="store_true", help="Skip creating calibration plots"
    )

    args = parser.parse_args()

    console.print(
        f"🔬 [bold blue]Arena 10K Experiment - Step 6: Import Labels & Calibrate[/bold blue]"
    )

    try:
        # Load human labels
        labels_df = load_human_labels(args.labels, platform=args.platform)

        # Aggregate votes
        human_scores_df = aggregate_votes(labels_df, min_votes=args.min_votes)

        # Load full dataset
        full_df = load_full_dataset(args.data)

        # Fit calibration
        iso_model = fit_calibration(full_df, human_scores_df, plot=not args.no_plot)

        # Apply calibration
        calibrated_df = apply_calibration(full_df, iso_model)

        # Save results
        save_calibrated_data(calibrated_df, args.output, iso_model)

        console.print(f"\n✅ [bold green]Calibration complete![/bold green]")
        console.print(f"\n📋 Next steps:")
        console.print(f"1. Generate target policy responses (Step 5)")
        console.print(f"2. Run CJE estimation (Step 7)")
        console.print(f"3. Validate against ground truth (Step 7)")

    except Exception as e:
        console.print(f"\n❌ [red]Failed: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
