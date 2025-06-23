#!/usr/bin/env python3
"""
Finalize the Arena 10K Oracle dataset by combining all generated data.

This script creates a complete dataset containing:
- Prompts
- Responses (π₀ and target policies)
- Oracle labels (calibration and validation)
- Judge scores (deterministic and uncertainty)

The output is a unified dataset ready for CJE ablations.
"""

import json
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table

console = Console()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def create_dataset_summary() -> Dict[str, Any]:
    """Create a comprehensive dataset ready for CJE ablations."""

    console.print("[bold cyan]Arena 10K Oracle Dataset Finalization[/bold cyan]\n")

    # Check all required files exist
    required_files = {
        "Prompts": "../data/arena_prompts_10k.jsonl",
        "π₀ responses": "../data/p0_replies.jsonl",
        "Target responses": "../data/target_ground_truth.jsonl",
        "Deterministic scores": "../data/p0_scored_deterministic.jsonl",
        "Uncertainty scores": "../data/p0_scored_uncertainty.jsonl",
        "Calibration oracle labels": "../data/labeling/oracle_labels_calibration_detailed.jsonl",
        "Validation oracle labels": "../data/labeling/oracle_labels_validation_detailed.jsonl",
    }

    missing = []
    for name, path in required_files.items():
        if not Path(path).exists():
            console.print(f"❌ Missing: {name} at {path}")
            missing.append(name)
        else:
            size_mb = Path(path).stat().st_size / 1024 / 1024
            console.print(f"✅ Found: {name} ({size_mb:.1f} MB)")

    if missing:
        console.print(f"\n[red]Cannot proceed - missing files: {missing}[/red]")
        return {}

    # Create dataset summary
    dataset_info = {
        "dataset_name": "Arena 10K Oracle",
        "created_date": pd.Timestamp.now().isoformat(),
        "components": {},
        "statistics": {},
        "file_paths": {},
    }

    # Load and summarize each component
    console.print("\n[bold]Dataset Components:[/bold]")

    # 1. Prompts
    prompts = load_jsonl(Path("../data/arena_prompts_10k.jsonl"))
    dataset_info["components"]["prompts"] = {
        "count": len(prompts),
        "description": "ChatBot Arena prompts",
    }
    console.print(f"\n1. Prompts: {len(prompts)} total")

    # 2. Responses
    p0_responses = load_jsonl(Path("../data/p0_replies.jsonl"))
    target_responses = load_jsonl(Path("../data/target_ground_truth.jsonl"))

    dataset_info["components"]["responses"] = {
        "p0_count": len(p0_responses),
        "target_count": len(target_responses),
        "policies": ["pi_0", "pi_cot", "pi_bigger_model", "pi_bad"],
    }
    console.print(f"\n2. Responses:")
    console.print(f"   - π₀ (calibration): {len(p0_responses)}")
    console.print(f"   - Target policies: {len(target_responses)}")

    # 3. Judge Scores
    det_scores = load_jsonl(Path("../data/p0_scored_deterministic.jsonl"))
    unc_scores = load_jsonl(Path("../data/p0_scored_uncertainty.jsonl"))

    dataset_info["components"]["judge_scores"] = {
        "deterministic_count": len(det_scores),
        "uncertainty_count": len(unc_scores),
        "methods": ["deterministic", "confidence_interval"],
    }
    console.print(f"\n3. Judge Scores:")
    console.print(f"   - Deterministic: {len(det_scores)}")
    console.print(f"   - Uncertainty: {len(unc_scores)}")

    # 4. Oracle Labels
    cal_labels = load_jsonl(
        Path("../data/labeling/oracle_labels_calibration_detailed.jsonl")
    )
    val_labels = load_jsonl(
        Path("../data/labeling/oracle_labels_validation_detailed.jsonl")
    )

    dataset_info["components"]["oracle_labels"] = {
        "calibration_count": len(cal_labels),
        "validation_count": len(val_labels),
        "total_count": len(cal_labels) + len(val_labels),
    }
    console.print(f"\n4. Oracle Labels:")
    console.print(f"   - Calibration: {len(cal_labels)}")
    console.print(f"   - Validation: {len(val_labels)}")
    console.print(f"   - Total: {len(cal_labels) + len(val_labels)}")

    # Statistics
    console.print("\n[bold]Dataset Statistics:[/bold]")

    # Score distributions
    det_scores_values = [item["judge_score_raw"] for item in det_scores]
    unc_scores_values = [item["judge_score_raw"] for item in unc_scores]
    unc_variances = [item["judge_variance"] for item in unc_scores]

    dataset_info["statistics"]["judge_scores"] = {
        "deterministic": {
            "mean": float(pd.Series(det_scores_values).mean()),
            "std": float(pd.Series(det_scores_values).std()),
            "min": float(pd.Series(det_scores_values).min()),
            "max": float(pd.Series(det_scores_values).max()),
        },
        "uncertainty": {
            "mean_score": float(pd.Series(unc_scores_values).mean()),
            "mean_variance": float(pd.Series(unc_variances).mean()),
            "std_variance": float(pd.Series(unc_variances).std()),
        },
    }

    console.print(
        f"\nDeterministic scores: mean={dataset_info['statistics']['judge_scores']['deterministic']['mean']:.3f}, "
        f"std={dataset_info['statistics']['judge_scores']['deterministic']['std']:.3f}"
    )
    console.print(
        f"Uncertainty scores: mean={dataset_info['statistics']['judge_scores']['uncertainty']['mean_score']:.3f}, "
        f"mean_var={dataset_info['statistics']['judge_scores']['uncertainty']['mean_variance']:.4f}"
    )

    # Oracle label distributions by policy
    val_labels_df = pd.DataFrame(val_labels)
    oracle_by_policy = (
        val_labels_df.groupby("policy")["oracle_label"]
        .apply(lambda x: pd.DataFrame([item for item in x])["score"].mean() / 10.0)
        .to_dict()
    )

    dataset_info["statistics"]["oracle_scores_by_policy"] = {
        k: float(v) for k, v in oracle_by_policy.items()
    }

    console.print("\nOracle scores by policy:")
    for policy, score in sorted(oracle_by_policy.items()):
        console.print(f"   {policy}: {score:.3f}")

    # File paths for easy reference
    dataset_info["file_paths"] = {
        "prompts": "data/arena_prompts_10k.jsonl",
        "p0_responses": "data/p0_replies.jsonl",
        "target_responses": "data/target_ground_truth.jsonl",
        "p0_scored_deterministic": "data/p0_scored_deterministic.jsonl",
        "p0_scored_uncertainty": "data/p0_scored_uncertainty.jsonl",
        "oracle_labels_calibration": "data/labeling/oracle_labels_calibration_detailed.jsonl",
        "oracle_labels_validation": "data/labeling/oracle_labels_validation_detailed.jsonl",
        "oracle_labels_csv": "data/labeling/oracle_labels.csv",
    }

    # Save dataset info
    output_path = Path("../data/dataset_info.json")
    with open(output_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    console.print(f"\n✅ Dataset info saved to: {output_path}")

    # Create summary table
    table = Table(title="Arena 10K Oracle Dataset Summary")
    table.add_column("Component", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Status", justify="center")

    table.add_row("Prompts", f"{len(prompts):,}", "✅")
    table.add_row("π₀ Responses", f"{len(p0_responses):,}", "✅")
    table.add_row("Target Responses", f"{len(target_responses):,}", "✅")
    table.add_row("Deterministic Scores", f"{len(det_scores):,}", "✅")
    table.add_row("Uncertainty Scores", f"{len(unc_scores):,}", "✅")
    table.add_row("Oracle Labels", f"{len(cal_labels) + len(val_labels):,}", "✅")

    console.print("\n")
    console.print(table)

    console.print(
        "\n[bold green]Dataset is ready for Phase 2: CJE Ablations![/bold green]"
    )
    console.print("\nNext steps:")
    console.print("1. cd ../phase2_cje_ablations")
    console.print("2. Run different CJE configurations")
    console.print("3. Compare results across ablations")

    return dataset_info


if __name__ == "__main__":
    create_dataset_summary()
