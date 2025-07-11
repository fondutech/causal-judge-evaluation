#!/usr/bin/env python3
"""
Validate and summarize the Arena 10K Oracle dataset.

This script:
1. Validates all required data files are present
2. Verifies Phase 2 format data was created correctly
3. Creates a comprehensive dataset summary with statistics
4. Provides guidance for next steps
"""

import json
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional
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


def verify_phase2_data():
    """Verify Phase 2 format data was created by previous steps."""

    console.print("\n[bold]Verifying Phase 2 data:[/bold]")

    # Check Phase 2 data directory
    phase2_data_dir = Path("../data")

    # Expected files from judge scoring scripts
    expected_files = [
        ("p0_scored_deterministic.jsonl", "P0 deterministic scores"),
        ("p0_scored_uncertainty.jsonl", "P0 uncertainty scores"),
        ("targets_scored_deterministic.jsonl", "Target deterministic scores"),
        ("targets_scored_uncertainty.jsonl", "Target uncertainty scores"),
    ]

    file_counts = {}
    all_found = True

    for filename, description in expected_files:
        filepath = phase2_data_dir / filename
        if filepath.exists():
            # Count records
            count = 0
            with open(filepath) as f:
                for line in f:
                    count += 1
            file_counts[filename] = count
            console.print(f"  ✅ {filename}: {count} records")
        else:
            console.print(f"  ❌ Missing: {filename} ({description})")
            all_found = False
            file_counts[filename] = 0

    # Check oracle labels
    console.print("\n[bold]Verifying oracle labels:[/bold]")
    labeling_dir = phase2_data_dir / "labeling"

    oracle_files = [
        ("oracle_labels_calibration_detailed.jsonl", "Calibration labels"),
        ("oracle_labels_validation_detailed.jsonl", "Validation labels"),
    ]

    for filename, description in oracle_files:
        filepath = labeling_dir / filename
        if filepath.exists():
            count = 0
            with open(filepath) as f:
                for line in f:
                    count += 1
            console.print(f"  ✅ {filename}: {count} records")
        else:
            console.print(f"  ⚠️  Optional: {filename} ({description}) not found")

    return all_found, file_counts


def create_dataset_summary() -> Dict[str, Any]:
    """Create a comprehensive dataset ready for CJE ablations."""

    console.print("[bold cyan]Arena 10K Oracle Dataset Finalization[/bold cyan]\n")

    # Check all required files exist
    required_files = {
        "Prompts": "data/arena_prompts_10k.jsonl",
        "All responses": "data/all_responses.jsonl",
        "Log probabilities": "data/logprobs.jsonl",
    }

    optional_files = {
        "Calibration oracle labels": "data/oracle_labels_calibration.jsonl",
        "Validation oracle labels": "data/oracle_labels_validation.jsonl",
    }

    missing = []
    for name, path in required_files.items():
        if not Path(path).exists():
            console.print(f"❌ Missing: {name} at {path}")
            missing.append(name)
        else:
            size_mb = Path(path).stat().st_size / 1024 / 1024
            console.print(f"✅ Found: {name} ({size_mb:.1f} MB)")

    # Check optional files
    for name, path in optional_files.items():
        if not Path(path).exists():
            console.print(f"⚠️  Optional file missing: {name} at {path}")
        else:
            size_mb = Path(path).stat().st_size / 1024 / 1024
            console.print(f"✅ Found: {name} ({size_mb:.1f} MB)")

    if missing:
        console.print(
            f"\n[red]Cannot proceed - missing required files: {missing}[/red]"
        )
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
    prompts = load_jsonl(Path("data/arena_prompts_10k.jsonl"))
    dataset_info["components"]["prompts"] = {
        "count": len(prompts),
        "description": "ChatBot Arena prompts",
    }
    console.print(f"\n1. Prompts: {len(prompts)} total")

    # 2. All Responses and Log Probs
    all_responses = load_jsonl(Path("data/all_responses.jsonl"))
    logprob_data = load_jsonl(Path("data/logprobs.jsonl"))

    # Load Phase 2 format scores for analysis
    phase2_dir = Path("../data")
    p0_det_scores = []
    p0_unc_scores = []
    target_det_scores = []
    target_unc_scores = []

    if (phase2_dir / "p0_scored_deterministic.jsonl").exists():
        p0_det_scores = load_jsonl(phase2_dir / "p0_scored_deterministic.jsonl")
    if (phase2_dir / "p0_scored_uncertainty.jsonl").exists():
        p0_unc_scores = load_jsonl(phase2_dir / "p0_scored_uncertainty.jsonl")
    if (phase2_dir / "targets_scored_deterministic.jsonl").exists():
        target_det_scores = load_jsonl(
            phase2_dir / "targets_scored_deterministic.jsonl"
        )
    if (phase2_dir / "targets_scored_uncertainty.jsonl").exists():
        target_unc_scores = load_jsonl(phase2_dir / "targets_scored_uncertainty.jsonl")

    # Count P0 responses
    p0_response_count = sum(1 for entry in all_responses if "p0" in entry["responses"])

    dataset_info["components"]["p0_policy"] = {
        "responses": p0_response_count,
        "with_logprobs": len(logprob_data),
        "deterministic_scores": len(p0_det_scores),
        "uncertainty_scores": len(p0_unc_scores),
        "description": "Logging policy (π₀) data",
    }
    console.print(f"\n2. π₀ Policy:")
    console.print(f"   - Responses: {p0_response_count}")
    console.print(f"   - With log probs: {len(logprob_data)}")
    console.print(f"   - Deterministic scores (Phase 2): {len(p0_det_scores)}")
    console.print(f"   - Uncertainty scores (Phase 2): {len(p0_unc_scores)}")

    # 3. Target Policy Responses

    # Group by policy from all_responses
    target_policies = {}
    total_target_responses = 0
    for entry in all_responses:
        for policy_name in entry["responses"]:
            if policy_name != "p0":
                if policy_name not in target_policies:
                    target_policies[policy_name] = 0
                target_policies[policy_name] += 1
                total_target_responses += 1

    dataset_info["components"]["target_policies"] = {
        "total_responses": total_target_responses,
        "deterministic_scores": len(target_det_scores),
        "uncertainty_scores": len(target_unc_scores),
        "policies": target_policies,
        "description": "Target policy data",
    }

    console.print(f"\n3. Target Policies:")
    for policy, count in sorted(target_policies.items()):
        console.print(f"   - {policy}: {count} responses")
    console.print(
        f"   - Total deterministic scores (Phase 2): {len(target_det_scores)}"
    )
    console.print(f"   - Total uncertainty scores (Phase 2): {len(target_unc_scores)}")

    # 4. Oracle Labels (optional)
    cal_labels_path = Path("data/oracle_labels_calibration.jsonl")
    val_labels_path = Path("data/oracle_labels_validation.jsonl")

    cal_labels = []
    val_labels = []

    if cal_labels_path.exists():
        cal_labels = load_jsonl(cal_labels_path)
    if val_labels_path.exists():
        val_labels = load_jsonl(val_labels_path)

    if cal_labels or val_labels:
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
    else:
        console.print(f"\n4. Oracle Labels: [yellow]Not generated[/yellow]")

    # Verify Phase 2 data exists
    phase2_found, phase2_counts = verify_phase2_data()

    if not phase2_found:
        console.print("\n[red]Error: Phase 2 data not found![/red]")
        console.print("Make sure to run judge scoring scripts (03_*.py) first.")
        return {}

    # 5. Analyze judge scoring consistency
    console.print("\n[bold]Judge Scoring Analysis:[/bold]")

    # π₀ scores
    if p0_det_scores and p0_unc_scores:
        det_means = [r["judge_score"] for r in p0_det_scores]
        unc_means = [r["judge_score"] for r in p0_unc_scores]
        unc_vars = [r.get("judge_score_variance", 0.0) for r in p0_unc_scores]

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
    info_path = Path("data/dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    console.print(f"\n[bold green]Dataset info saved to: {info_path}[/bold green]")

    # Create summary table
    table = Table(title="Dataset Summary", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Status", justify="center")

    table.add_row("Arena Prompts", f"{len(prompts):,}", "✅")
    table.add_row("π₀ Responses", f"{p0_response_count:,}", "✅")
    table.add_row("Target Responses", f"{total_target_responses:,}", "✅")
    table.add_row("π₀ Det. Scores", f"{len(p0_det_scores):,}", "✅")
    table.add_row("π₀ Unc. Scores", f"{len(p0_unc_scores):,}", "✅")
    table.add_row("Target Det. Scores", f"{len(target_det_scores):,}", "✅")
    table.add_row("Target Unc. Scores", f"{len(target_unc_scores):,}", "✅")
    table.add_row("Oracle Labels", f"{len(cal_labels) + len(val_labels):,}", "✅")

    console.print("\n")
    console.print(table)

    console.print("\n[bold green]Dataset validation complete![/bold green]")

    # Phase 2 specific instructions
    console.print("\n[bold]Phase 2 Data Location:[/bold]")
    phase2_dir = Path("../data").absolute()
    console.print(f"  {phase2_dir}")
    console.print("\n[bold]Phase 2 Files Available:[/bold]")
    console.print(
        f"  • p0_scored_deterministic.jsonl ({phase2_counts.get('p0_scored_deterministic.jsonl', 0)} records)"
    )
    console.print(
        f"  • p0_scored_uncertainty.jsonl ({phase2_counts.get('p0_scored_uncertainty.jsonl', 0)} records)"
    )
    console.print(
        f"  • targets_scored_deterministic.jsonl ({phase2_counts.get('targets_scored_deterministic.jsonl', 0)} records)"
    )
    console.print(
        f"  • targets_scored_uncertainty.jsonl ({phase2_counts.get('targets_scored_uncertainty.jsonl', 0)} records)"
    )
    console.print(f"  • labeling/oracle_labels_*_detailed.jsonl")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. cd ../phase2_cje_ablations")
    console.print("2. python run_phase2_ablations.py")
    console.print("3. Analyze estimator performance across different configurations")

    return dataset_info


if __name__ == "__main__":
    create_dataset_summary()
