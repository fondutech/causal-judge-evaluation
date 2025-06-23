#!/usr/bin/env python3
"""
Phase 2: Run CJE pipeline ablations on the prepared dataset.

This script runs different CJE configurations to explore:
1. Judge scoring methods (deterministic vs uncertainty)
2. Different estimators
3. Calibration methods
4. Other pipeline variations
"""

import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table
import sys

console = Console()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


ABLATION_CONFIGS = {
    "ipw_deterministic": {
        "description": "IPW with deterministic judge scores",
        "judge_scores": "p0_scored_deterministic.jsonl",
        "estimator": "ipw",
        "calibrator": "isotonic",
    },
    "ipw_uncertainty": {
        "description": "IPW with uncertainty-aware judge scores",
        "judge_scores": "p0_scored_uncertainty.jsonl",
        "estimator": "ipw",
        "calibrator": "isotonic",
    },
    "snipw_deterministic": {
        "description": "Self-normalized IPW with deterministic scores",
        "judge_scores": "p0_scored_deterministic.jsonl",
        "estimator": "snipw",
        "calibrator": "isotonic",
    },
    "snipw_uncertainty": {
        "description": "Self-normalized IPW with uncertainty scores",
        "judge_scores": "p0_scored_uncertainty.jsonl",
        "estimator": "snipw",
        "calibrator": "isotonic",
    },
    # Note: DR estimators require target policy samples which are not in current dataset
    # Uncomment below when target samples are available
    # "dr_deterministic": {
    #     "description": "Doubly robust estimator with deterministic scores",
    #     "judge_scores": "p0_scored_deterministic.jsonl",
    #     "estimator": "dr",
    #     "calibrator": "isotonic",
    #     "requires_target_samples": True,
    # },
    # "dr_uncertainty": {
    #     "description": "Doubly robust estimator with uncertainty scores",
    #     "judge_scores": "p0_scored_uncertainty.jsonl",
    #     "estimator": "dr",
    #     "calibrator": "isotonic",
    #     "requires_target_samples": True,
    # },
}


def create_config_file(ablation_name: str, config: Dict[str, Any]) -> Path:
    """Create a Hydra config file for this ablation."""

    # Base config structure
    hydra_config = {
        "defaults": ["base"],
        "experiment_name": f"arena_10k_{ablation_name}",
        "data": {
            "input_path": f"../data/{config['judge_scores']}",
            "output_dir": f"../../outputs/arena_10k_{ablation_name}",
        },
        "estimator": {
            "_target_": f"cje.estimators.{config['estimator']}.{config['estimator'].upper()}Estimator",
        },
        "calibrator": {
            "_target_": f"cje.calibration.{config['calibrator']}.{config['calibrator'].capitalize()}Calibrator",
            "cv_folds": 5,
        },
        "oracle": {
            "enabled": True,
            "calibration_file": "../data/labeling/oracle_labels_calibration_detailed.jsonl",
            "validation_file": "../data/labeling/oracle_labels_validation_detailed.jsonl",
        },
        "policies": ["pi_0", "pi_cot", "pi_bigger_model", "pi_bad"],
        "logging": {
            "level": "INFO",
            "rich": True,
        },
    }

    # Save config
    config_path = Path(f"configs/{ablation_name}.yaml")
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(hydra_config, f, default_flow_style=False)

    return config_path


def run_ablation(ablation_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single CJE ablation."""

    console.print(f"\n[bold blue]Running ablation: {ablation_name}[/bold blue]")
    console.print(f"Description: {config['description']}")

    # Create config file
    config_path = create_config_file(ablation_name, config)
    console.print(f"Config: {config_path}")

    # Run CJE
    cmd = [
        "cje",
        "run",
        "--cfg-path",
        str(config_path.parent),
        "--cfg-name",
        ablation_name,
    ]

    # Change to parent directory for proper paths
    import os

    original_dir = os.getcwd()
    os.chdir("../..")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            console.print("[green]✅ Success[/green]")

            # Load results
            results_path = Path(f"outputs/arena_10k_{ablation_name}/results.json")
            if results_path.exists():
                with open(results_path) as f:
                    return json.load(f)
            else:
                console.print("[yellow]⚠️  Results file not found[/yellow]")
                return {}
        else:
            console.print(f"[red]❌ Failed with code {result.returncode}[/red]")
            console.print(f"Error: {result.stderr}")
            return {}

    finally:
        os.chdir(original_dir)


def compare_results(all_results: Dict[str, Dict[str, Any]]):
    """Compare results across all ablations."""

    console.print("\n[bold cyan]Results Comparison[/bold cyan]\n")

    # Create comparison table
    table = Table(title="CJE Ablation Results")
    table.add_column("Ablation", style="cyan")
    table.add_column("Judge Method", style="yellow")
    table.add_column("Estimator", style="green")

    # Add columns for each policy
    policies = ["pi_cot", "pi_bigger_model", "pi_bad"]
    for policy in policies:
        table.add_column(f"{policy} Score", justify="right")

    # Add rows
    for ablation_name, results in all_results.items():
        config = ABLATION_CONFIGS[ablation_name]

        row = [
            ablation_name,
            "Deterministic" if "deterministic" in ablation_name else "Uncertainty",
            config["estimator"].upper(),
        ]

        # Add policy scores
        for policy in policies:
            score = (
                results.get("policy_values", {}).get(policy, {}).get("estimate", "N/A")
            )
            if isinstance(score, (int, float)):
                row.append(f"{score:.3f}")
            else:
                row.append(str(score))

        table.add_row(*row)

    console.print(table)

    # Save comparison
    comparison_path = Path("results/ablation_comparison.json")
    comparison_path.parent.mkdir(exist_ok=True)

    with open(comparison_path, "w") as f:
        json.dump(
            {
                "ablations": all_results,
                "summary": {
                    ablation: {
                        "config": ABLATION_CONFIGS[ablation],
                        "policy_scores": {
                            policy: results.get("policy_values", {})
                            .get(policy, {})
                            .get("estimate")
                            for policy in policies
                        },
                    }
                    for ablation, results in all_results.items()
                },
            },
            f,
            indent=2,
        )

    console.print(f"\n✅ Comparison saved to: {comparison_path}")


def main():
    """Run all CJE ablations."""

    console.print("[bold cyan]Phase 2: CJE Pipeline Ablations[/bold cyan]")
    console.print("=" * 50)

    # Check dataset is ready
    dataset_info_path = Path("../data/dataset_info.json")
    if not dataset_info_path.exists():
        console.print(
            "[red]Dataset not finalized! Run phase1_dataset_preparation/06_finalize_dataset.py first[/red]"
        )
        return

    # Load dataset info
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

    console.print(f"\nDataset: {dataset_info['dataset_name']}")
    console.print(f"Created: {dataset_info['created_date']}")

    # Ask which ablations to run
    console.print(f"\nAvailable ablations: {len(ABLATION_CONFIGS)}")
    for name, config in ABLATION_CONFIGS.items():
        console.print(f"  - {name}: {config['description']}")

    response = console.input("\nRun all ablations? [Y/n]: ").lower()

    if response == "n":
        # Let user select specific ablations
        selected = console.input("Enter ablation names (comma-separated): ").split(",")
        selected = [s.strip() for s in selected if s.strip() in ABLATION_CONFIGS]
    else:
        selected = list(ABLATION_CONFIGS.keys())

    # Run selected ablations
    all_results = {}
    for ablation_name in selected:
        results = run_ablation(ablation_name, ABLATION_CONFIGS[ablation_name])
        all_results[ablation_name] = results

    # Compare results
    if len(all_results) > 1:
        compare_results(all_results)

    console.print("\n[bold green]All ablations complete![/bold green]")


if __name__ == "__main__":
    main()
