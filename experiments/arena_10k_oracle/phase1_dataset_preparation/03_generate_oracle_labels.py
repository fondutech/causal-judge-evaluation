#!/usr/bin/env python3
"""
Step 5: Generate oracle labels for Arena 10K experiment using CJE's oracle labeling.

This script generates oracle labels for TWO distinct datasets:
1. CALIBRATION SET: 25% of pi_0 (logging policy) responses from p0_replies.jsonl
   - Used to calibrate judge scores to human preferences
   - These are the baseline responses that CJE uses for calibration

2. VALIDATION SET: All target policy responses from target_ground_truth.jsonl
   - Used to validate CJE's predictions against ground truth
   - Includes pi_cot, pi_bigger_model, and pi_bad policies

Total oracle labels: ~4,000 (2,500 calibration + 1,500 validation)

Usage:
    # Generate all oracle labels with o3
    export OPENAI_API_KEY="your-api-key"
    python 05_generate_oracle_labels.py --model o3-2025-04-16 --temperature 1.0

    # Test with small sample
    python 05_generate_oracle_labels.py --model gpt-4o --calibration-fraction 0.01 --validation-fraction 0.1

    # Resume from checkpoint (automatic)
    python 05_generate_oracle_labels.py --model o3-2025-04-16 --temperature 1.0

Notes:
    - Requires OPENAI_API_KEY environment variable
    - Saves checkpoints every 10 items for resuming
    - Outputs separate detailed files for calibration and validation
    - Progress bars show real-time status
"""

import json
import pandas as pd
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple
import random

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.oracle_labeling import add_oracle_labels
from cje.utils.progress import console


def load_calibration_data(
    fraction: float = 0.25, seed: int = 42
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Load pi_0 responses for calibration oracle labels.

    Args:
        fraction: Fraction of pi_0 responses to label (default 25%)
        seed: Random seed for reproducible sampling

    Returns:
        Tuple of (rows for labeling, total pi_0 count)
    """
    console.print(f"\n📊 [bold]Loading Calibration Data (pi_0 responses):[/bold]")
    console.print(f"   Source: ../data/p0_replies.jsonl")

    all_rows = []
    with open("../data/p0_replies.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            all_rows.append(
                {
                    "context": data["prompt"],
                    "response": data["response"],
                    "prompt_id": data["prompt_id"],
                    "policy": "pi_0",  # Logging policy
                    "dataset_type": "calibration",
                    "total_logprob": data.get("total_logprob", None),
                    "model_info": data.get("logging_policy", {}),
                }
            )

    total_count = len(all_rows)

    # Sample fraction for oracle labeling
    rng = random.Random(seed)
    sample_size = int(total_count * fraction)
    sampled_indices = rng.sample(range(total_count), sample_size)
    sampled_rows = [all_rows[i] for i in sampled_indices]

    console.print(f"   Total pi_0 responses: {total_count}")
    console.print(f"   Sampling {fraction:.0%} for oracle labels: {sample_size}")
    console.print(f"   Purpose: Judge calibration to human preferences")

    return sampled_rows, total_count


def load_validation_data(
    fraction: float = 0.05, seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load and sample target policy responses for validation oracle labels.

    Args:
        fraction: Fraction of target responses to label (default 5% = 500 prompts × 3 policies = 1,500 labels)
        seed: Random seed for reproducible sampling

    Returns:
        List of rows for labeling
    """
    console.print(
        f"\n📊 [bold]Loading Validation Data (target policy responses):[/bold]"
    )
    console.print(f"   Source: ../data/target_responses.jsonl")

    all_rows = []
    with open("../data/target_responses.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            all_rows.append(
                {
                    "context": data["prompt"],
                    "response": data["response"],
                    "prompt_id": data["prompt_id"],
                    "policy": data["policy"],
                    "dataset_type": "validation",
                    "model_name": data.get("model_name", ""),
                    "temperature": data.get("temperature", 0.0),
                    "description": data.get("description", ""),
                }
            )

    # Sample by prompt_id to ensure balanced policies
    prompt_ids = list(set(row["prompt_id"] for row in all_rows))
    rng = random.Random(seed)
    sample_size = int(len(prompt_ids) * fraction)
    sampled_prompt_ids = set(rng.sample(prompt_ids, sample_size))

    # Get all responses for sampled prompts
    sampled_rows = [row for row in all_rows if row["prompt_id"] in sampled_prompt_ids]

    # Count by policy
    policy_counts: dict[str, int] = {}
    for row in sampled_rows:
        policy = row["policy"]
        policy_counts[policy] = policy_counts.get(policy, 0) + 1

    console.print(
        f"   Total target responses: {len(all_rows)} (30,000 = 10,000 prompts × 3 policies)"
    )
    console.print(
        f"   Sampling {fraction:.0%} of prompts: {sample_size} prompts × 3 policies = {len(sampled_rows)} labels"
    )
    console.print(f"   Purpose: Validate CJE predictions against ground truth")
    console.print(f"   Policy breakdown:")
    for policy, count in sorted(policy_counts.items()):
        console.print(f"     {policy}: {count}")

    return sampled_rows


def save_oracle_results(
    rows_with_oracle: List[Dict[str, Any]], output_path: str
) -> None:
    """Save oracle results with clear separation of calibration and validation data."""

    # Separate calibration and validation
    calibration_rows = [
        r for r in rows_with_oracle if r.get("dataset_type") == "calibration"
    ]
    validation_rows = [
        r for r in rows_with_oracle if r.get("dataset_type") == "validation"
    ]

    # Prepare data for saving
    all_ratings = []

    for row in rows_with_oracle:
        if "y_true" in row and row["y_true"] is not None:
            all_ratings.append(
                {
                    "prompt_id": row["prompt_id"],
                    "policy": row["policy"],
                    "oracle_score": row["y_true"],  # 0-1 range
                    "oracle_score_10": row["y_true"] * 10,  # 1-10 scale
                    "dataset_type": row["dataset_type"],
                }
            )

    # Save combined CSV
    ratings_df = pd.DataFrame(all_ratings)
    ratings_df.to_csv(output_path, index=False)

    console.print(f"\n💾 [bold]Output Summary:[/bold]")
    console.print(f"   Total oracle labels saved: {len(all_ratings)}")
    console.print(
        f"     - Calibration (pi_0): {len([r for r in all_ratings if r['dataset_type'] == 'calibration'])}"
    )
    console.print(
        f"     - Validation (target): {len([r for r in all_ratings if r['dataset_type'] == 'validation'])}"
    )
    console.print(f"   Main output: {output_path}")
    console.print(f"   Columns: {', '.join(ratings_df.columns)}")

    # Save detailed results with separate files for clarity
    base_path = Path(output_path)

    # Calibration details
    calibration_detailed_path = (
        base_path.parent / f"{base_path.stem}_calibration_detailed.jsonl"
    )
    with open(calibration_detailed_path, "w") as f:
        for row in calibration_rows:
            if "y_true" in row and row["y_true"] is not None:
                f.write(
                    json.dumps(
                        {
                            "prompt_id": row["prompt_id"],
                            "policy": row["policy"],
                            "oracle_score": row["y_true"],
                            "oracle_score_10": row["y_true"] * 10,
                            "prompt_snippet": (
                                row["context"][:100] + "..."
                                if len(row["context"]) > 100
                                else row["context"]
                            ),
                            "response_snippet": (
                                row["response"][:200] + "..."
                                if len(row["response"]) > 200
                                else row["response"]
                            ),
                        }
                    )
                    + "\n"
                )
    console.print(f"   Calibration details: {calibration_detailed_path}")

    # Validation details
    validation_detailed_path = (
        base_path.parent / f"{base_path.stem}_validation_detailed.jsonl"
    )
    with open(validation_detailed_path, "w") as f:
        for row in validation_rows:
            if "y_true" in row and row["y_true"] is not None:
                f.write(
                    json.dumps(
                        {
                            "prompt_id": row["prompt_id"],
                            "policy": row["policy"],
                            "oracle_score": row["y_true"],
                            "oracle_score_10": row["y_true"] * 10,
                            "prompt_snippet": (
                                row["context"][:100] + "..."
                                if len(row["context"]) > 100
                                else row["context"]
                            ),
                            "response_snippet": (
                                row["response"][:200] + "..."
                                if len(row["response"]) > 200
                                else row["response"]
                            ),
                        }
                    )
                    + "\n"
                )
    console.print(f"   Validation details: {validation_detailed_path}")

    # Print statistics by dataset type and policy
    console.print(f"\n📊 [bold]Oracle Rating Statistics:[/bold]")

    if calibration_rows:
        cal_scores = [
            r["y_true"]
            for r in calibration_rows
            if "y_true" in r and r["y_true"] is not None
        ]
        if cal_scores:
            console.print(f"\n   Calibration Set (pi_0):")
            console.print(f"     Count: {len(cal_scores)}")
            console.print(
                f"     Mean: {sum(cal_scores)/len(cal_scores):.2f} ({sum(cal_scores)/len(cal_scores)*10:.1f}/10)"
            )
            console.print(f"     Range: [{min(cal_scores):.2f}, {max(cal_scores):.2f}]")

    if validation_rows:
        console.print(f"\n   Validation Set by Policy:")
        for policy in ["pi_cot", "pi_bigger_model", "pi_bad"]:
            policy_scores = [
                r["y_true"]
                for r in validation_rows
                if r.get("policy") == policy
                and "y_true" in r
                and r["y_true"] is not None
            ]
            if policy_scores:
                console.print(f"     {policy}:")
                console.print(f"       Count: {len(policy_scores)}")
                console.print(
                    f"       Mean: {sum(policy_scores)/len(policy_scores):.2f} ({sum(policy_scores)/len(policy_scores)*10:.1f}/10)"
                )


def analyze_results_by_policy(rows_with_oracle: List[Dict[str, Any]]) -> None:
    """Analyze oracle ratings by policy with clear separation of datasets."""
    from collections import defaultdict
    import numpy as np

    # Separate by dataset type
    calibration_scores = defaultdict(list)
    validation_scores = defaultdict(list)

    for row in rows_with_oracle:
        if "y_true" in row and row["y_true"] is not None:
            policy = row.get("policy", "unknown")
            if row.get("dataset_type") == "calibration":
                calibration_scores[policy].append(row["y_true"])
            else:
                validation_scores[policy].append(row["y_true"])

    console.print(f"\n📊 [bold]Detailed Analysis by Dataset and Policy:[/bold]")
    console.print("=" * 60)

    # Calibration analysis
    if calibration_scores:
        console.print(f"\n[bold]CALIBRATION SET:[/bold]")
        for policy in sorted(calibration_scores.keys()):
            scores = calibration_scores[policy]
            if scores:
                console.print(f"\n{policy} (n={len(scores)}):")
                console.print(f"  Mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
                console.print(
                    f"  10-scale: {np.mean(scores)*10:.1f} ± {np.std(scores)*10:.1f}"
                )
                console.print(f"  Median: {np.median(scores):.2f}")
                console.print(f"  Range: [{min(scores):.2f}, {max(scores):.2f}]")

    # Validation analysis
    if validation_scores:
        console.print(f"\n[bold]VALIDATION SET:[/bold]")
        for policy in sorted(validation_scores.keys()):
            scores = validation_scores[policy]
            if scores:
                console.print(f"\n{policy} (n={len(scores)}):")
                console.print(f"  Mean: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
                console.print(
                    f"  10-scale: {np.mean(scores)*10:.1f} ± {np.std(scores)*10:.1f}"
                )
                console.print(f"  Median: {np.median(scores):.2f}")
                console.print(f"  Range: [{min(scores):.2f}, {max(scores):.2f}]")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate oracle labels for Arena 10K experiment (calibration + validation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script generates oracle labels for two purposes:
1. CALIBRATION: Labels for pi_0 responses to calibrate judge scores
2. VALIDATION: Labels for target policies to validate CJE predictions

Example usage:
  # Generate all oracle labels with o3
  python scripts/05_generate_oracle_labels.py --model o3-2025-04-16 --temperature 1.0
  
  # Generate only 10% for testing
  python scripts/05_generate_oracle_labels.py --calibration-fraction 0.025 --validation-fraction 0.1
""",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../data/labeling/oracle_labels.csv",
        help="Output path for oracle labels",
    )

    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.25,
        help="Fraction of pi_0 responses to label for calibration (default: 0.25)",
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.05,
        help="Fraction of prompts to sample for validation (default: 0.05 = 500 prompts × 3 policies = 1,500 labels)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Provider for oracle judge (openai, anthropic, etc.)",
    )

    parser.add_argument(
        "--model", type=str, default="o3-2025-04-16", help="Model name for oracle judge"
    )

    parser.add_argument(
        "--template",
        type=str,
        default="comprehensive_judge",
        help="Judge template (defaults to comprehensive_judge for better criteria)",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (o3 models require 1.0)",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="../data/labeling/checkpoints",
        help="Directory for saving checkpoints",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output showing judge inputs and outputs",
    )

    args = parser.parse_args()

    console.print(f"\n🔬 [bold blue]Arena 10K Oracle Label Generation[/bold blue]")
    console.print(f"\n📥 [bold]Data Sources:[/bold]")
    console.print(f"   Calibration: ../data/p0_replies.jsonl (pi_0 responses)")
    console.print(f"   Validation: ../data/target_responses.jsonl (target policies)")
    console.print(f"   Output: {args.output}")
    console.print(f"   Checkpoints: {args.checkpoint_dir}/")

    console.print(f"\n🤖 [bold]Oracle Configuration:[/bold]")
    console.print(f"   Provider: {args.provider}")
    console.print(f"   Model: {args.model}")
    console.print(f"   Temperature: {args.temperature}")
    console.print(f"   Calibration fraction: {args.calibration_fraction:.1%}")
    console.print(f"   Validation fraction: {args.validation_fraction:.1%}")

    try:
        # Load both datasets
        calibration_rows, total_pi0 = load_calibration_data(
            fraction=args.calibration_fraction, seed=args.seed
        )
        validation_rows = load_validation_data(
            fraction=args.validation_fraction, seed=args.seed
        )

        # Combine all rows for oracle labeling
        all_rows = calibration_rows + validation_rows

        console.print(f"\n📊 [bold]Total Items for Oracle Labeling:[/bold]")
        console.print(f"   Calibration (pi_0): {len(calibration_rows)}")
        console.print(f"   Validation (targets): {len(validation_rows)}")
        console.print(f"   TOTAL: {len(all_rows)}")

        # Generate oracle labels
        console.print(f"\n🏷️  Generating oracle labels...")

        # Show temperature setting
        if args.temperature != 0.0:
            console.print(f"[yellow]Using temperature={args.temperature}[/yellow]")

        # Create checkpoint directory if needed
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Note: We're labeling ALL rows (fraction=1.0) since we already sampled above
        rows_with_oracle = add_oracle_labels(
            all_rows,
            provider=args.provider,
            model_name=args.model,
            fraction=1.0,  # Label all rows we've selected
            seed=args.seed,
            template=args.template,
            temperature=args.temperature,
            max_tokens=16,
            score_field="y_true",
        )

        # Save results
        save_oracle_results(rows_with_oracle, args.output)

        # Analyze by policy
        analyze_results_by_policy(rows_with_oracle)

        console.print(f"\n✅ [bold green]Oracle labeling complete![/bold green]")
        console.print(f"\n📋 [bold]Next Steps:[/bold]")
        console.print(f"1. Review the oracle labels:")
        console.print(
            f"   - Check calibration scores: {args.output.replace('.csv', '_calibration_detailed.jsonl')}"
        )
        console.print(
            f"   - Check validation scores: {args.output.replace('.csv', '_validation_detailed.jsonl')}"
        )
        console.print(
            f"2. Import labels for experiment: python scripts/06_import_labels.py --labels {args.output}"
        )
        console.print(f"3. Run CJE pipeline to get predictions")
        console.print(f"4. Compare CJE predictions to validation oracle labels")

    except Exception as e:
        console.print(f"\n❌ [red]Failed: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
