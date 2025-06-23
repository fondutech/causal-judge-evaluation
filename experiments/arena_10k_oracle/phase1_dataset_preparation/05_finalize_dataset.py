#!/usr/bin/env python3
"""
Finalize the Arena 10K Oracle dataset by combining all generated data.

This script creates a complete dataset containing:
- Prompts
- Responses (π₀ and target policies)
- Oracle labels (calibration and validation)
- Judge scores (deterministic and uncertainty) for ALL policies

The output is a unified dataset ready for CJE ablations.
"""

import json
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table
import numpy as np

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
        "π₀ deterministic scores": "../data/p0_scored_deterministic.jsonl",
        "π₀ uncertainty scores": "../data/p0_scored_uncertainty.jsonl",
        "Target deterministic scores": "../data/targets_scored_deterministic.jsonl",
        "Target uncertainty scores": "../data/targets_scored_uncertainty.jsonl",
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

    # 2. π₀ Responses and Scores
    p0_responses = load_jsonl(Path("../data/p0_replies.jsonl"))
    det_scores = load_jsonl(Path("../data/p0_scored_deterministic.jsonl"))
    unc_scores = load_jsonl(Path("../data/p0_scored_uncertainty.jsonl"))

    dataset_info["components"]["p0_policy"] = {
        "responses": len(p0_responses),
        "deterministic_scores": len(det_scores),
        "uncertainty_scores": len(unc_scores),
        "description": "Logging policy (π₀) data",
    }
    console.print(f"\n2. π₀ Policy:")
    console.print(f"   - Responses: {len(p0_responses)}")
    console.print(f"   - Deterministic scores: {len(det_scores)}")
    console.print(f"   - Uncertainty scores: {len(unc_scores)}")

    # 3. Target Policy Responses and Scores
    target_responses = load_jsonl(Path("../data/target_ground_truth.jsonl"))
    target_det_scores = load_jsonl(Path("../data/targets_scored_deterministic.jsonl"))
    target_unc_scores = load_jsonl(Path("../data/targets_scored_uncertainty.jsonl"))

    # Group by policy
    target_policies = {}
    for resp in target_responses:
        policy = resp.get("policy", "unknown")
        if policy not in target_policies:
            target_policies[policy] = 0
        target_policies[policy] += 1

    dataset_info["components"]["target_policies"] = {
        "total_responses": len(target_responses),
        "deterministic_scores": len(target_det_scores),
        "uncertainty_scores": len(target_unc_scores),
        "policies": target_policies,
        "description": "Target policy data",
    }

    console.print(f"\n3. Target Policies:")
    for policy, count in sorted(target_policies.items()):
        console.print(f"   - {policy}: {count} responses")
    console.print(f"   - Total deterministic scores: {len(target_det_scores)}")
    console.print(f"   - Total uncertainty scores: {len(target_unc_scores)}")

    # 4. Oracle Labels
    cal_labels = load_jsonl(
        Path("../data/labeling/oracle_labels_calibration_detailed.jsonl")
    )
    val_labels = load_jsonl(
        Path("../data/labeling/oracle_labels_validation_detailed.jsonl")
    )

    dataset_info["components"]["oracle_labels"] = {
        "calibration": len(cal_labels),
        "validation": len(val_labels),
        "total": len(cal_labels) + len(val_labels),
        "description": "Ground truth labels for calibration and validation",
    }

    console.print(f"\n4. Oracle Labels:")
    console.print(f"   - Calibration: {len(cal_labels)} (for judge calibration)")
    console.print(f"   - Validation: {len(val_labels)} (for CJE evaluation)")
    console.print(f"   - Total: {len(cal_labels) + len(val_labels)}")

    # 5. Analyze judge scoring consistency
    console.print("\n[bold]Judge Scoring Analysis:[/bold]")

    # π₀ scores
    if det_scores and unc_scores:
        det_means = [r["judge_score"] for r in det_scores]
        unc_means = [r["judge_score"] for r in unc_scores]
        unc_vars = [r.get("judge_score_variance", 0.0) for r in unc_scores]

        console.print(f"\nπ₀ Judge Scores:")
        console.print(
            f"  Deterministic: mean={np.mean(det_means):.3f}, std={np.std(det_means):.3f}"
        )
        console.print(
            f"  Uncertainty: mean={np.mean(unc_means):.3f}, std={np.std(unc_means):.3f}"
        )
        console.print(f"  Mean variance: {np.mean(unc_vars):.4f}")

        dataset_info["statistics"]["p0_judge_scores"] = {
            "deterministic": {
                "mean": float(np.mean(det_means)),
                "std": float(np.std(det_means)),
            },
            "uncertainty": {
                "mean": float(np.mean(unc_means)),
                "std": float(np.std(unc_means)),
                "mean_variance": float(np.mean(unc_vars)),
            },
        }

    # Target policy scores
    if target_det_scores and target_unc_scores:
        console.print(f"\nTarget Policy Judge Scores:")

        policy_stats = {}
        for policy in target_policies.keys():
            det_policy = [r for r in target_det_scores if r.get("policy") == policy]
            unc_policy = [r for r in target_unc_scores if r.get("policy") == policy]

            if det_policy and unc_policy:
                det_means = [r["judge_score"] for r in det_policy]
                unc_means = [r["judge_score"] for r in unc_policy]
                unc_vars = [r.get("judge_score_variance", 0.0) for r in unc_policy]

                console.print(f"\n  {policy}:")
                console.print(
                    f"    Deterministic: mean={np.mean(det_means):.3f}, std={np.std(det_means):.3f}"
                )
                console.print(
                    f"    Uncertainty: mean={np.mean(unc_means):.3f}, std={np.std(unc_means):.3f}"
                )
                console.print(f"    Mean variance: {np.mean(unc_vars):.4f}")

                policy_stats[policy] = {
                    "deterministic": {
                        "mean": float(np.mean(det_means)),
                        "std": float(np.std(det_means)),
                    },
                    "uncertainty": {
                        "mean": float(np.mean(unc_means)),
                        "std": float(np.std(unc_means)),
                        "mean_variance": float(np.mean(unc_vars)),
                    },
                }

        dataset_info["statistics"]["target_judge_scores"] = policy_stats

    # Save dataset info
    info_path = Path("../data/dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    console.print(f"\n[bold green]Dataset info saved to: {info_path}[/bold green]")

    # Create summary table
    table = Table(title="Dataset Summary", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Status", justify="center")

    table.add_row("Arena Prompts", f"{len(prompts):,}", "✅")
    table.add_row("π₀ Responses", f"{len(p0_responses):,}", "✅")
    table.add_row("Target Responses", f"{len(target_responses):,}", "✅")
    table.add_row("π₀ Det. Scores", f"{len(det_scores):,}", "✅")
    table.add_row("π₀ Unc. Scores", f"{len(unc_scores):,}", "✅")
    table.add_row("Target Det. Scores", f"{len(target_det_scores):,}", "✅")
    table.add_row("Target Unc. Scores", f"{len(target_unc_scores):,}", "✅")
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

    # Recommendations
    console.print("\n[bold yellow]Recommendations:[/bold yellow]")
    console.print("• Compare deterministic vs uncertainty scoring impact")
    console.print("• Analyze if uncertainty varies by policy quality")
    console.print("• Test if uncertainty improves calibration")
    console.print("• Examine judge consistency across policies")

    return dataset_info


if __name__ == "__main__":
    create_dataset_summary()
