#!/usr/bin/env python3
"""
Run Phase 2 CJE ablations on Arena 10K data.

This script runs multiple CJE estimators with different configurations
to compare their performance on the prepared dataset.
"""

import sys
from pathlib import Path
import yaml
import json
from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje.loggers import PrecomputedSampler
from cje.estimators import (
    MultiIPSEstimator,
    CalibratedIPSEstimator,
    MultiDRCPOEstimator,
    MultiMRDREstimator,
    MultiSNIPSEstimator,
)

console = Console()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single_ablation(config_path: Path) -> Dict[str, Any]:
    """Run a single ablation experiment."""
    config = load_config(config_path)
    experiment_name = config["experiment_name"]

    console.print(f"\n[bold cyan]Running: {experiment_name}[/bold cyan]")

    # Load data using PrecomputedSampler
    console.print("Loading data...")

    # Load P0 data (has log probabilities)
    p0_data = []
    with open(config["data"]["logging_policy_file"]) as f:
        for line in f:
            p0_data.append(json.loads(line))

    # Create sampler from P0 data (which contains target log probs)
    sampler = PrecomputedSampler(
        data=p0_data, target_policies=config["policies"]["target"]
    )

    console.print(f"Loaded {len(p0_data)} P0 samples with log probabilities")

    # Initialize estimator based on config
    estimator_config = config["estimator"]
    estimator_class = estimator_config["_target_"]

    if "MultiIPSEstimator" in estimator_class:
        estimator = MultiIPSEstimator(sampler)
    elif "CalibratedIPS" in estimator_class:
        # CalibratedIPSEstimator handles calibration internally
        estimator = CalibratedIPSEstimator(sampler)
    elif "MultiDRCPO" in estimator_class:
        estimator = MultiDRCPOEstimator(sampler)
    elif "MultiMRDR" in estimator_class:
        estimator = MultiMRDREstimator(sampler)
    elif "MultiSNIPS" in estimator_class:
        estimator = MultiSNIPSEstimator(sampler)
    else:
        raise ValueError(f"Unknown estimator: {estimator_class}")

    # Extract contexts and judge scores for fitting
    contexts = [item["prompt"] for item in p0_data]
    judge_scores = [item["judge_score"] for item in p0_data]

    # Fit and estimate
    console.print(f"Fitting {estimator.__class__.__name__}...")
    estimator.fit(contexts, judge_scores)
    results = estimator.estimate()

    # Load oracle labels if available
    oracle_results = None
    if config.get("oracle", {}).get("enabled", False):
        console.print("Loading oracle labels for validation...")
        validation_file = config["oracle"]["validation_file"]
        if Path(validation_file).exists():
            oracle_data = []
            with open(validation_file) as f:
                for line in f:
                    oracle_data.append(json.loads(line))

            # Compute oracle estimates
            oracle_scores = {}
            for policy in config["policies"]["target"]:
                policy_scores = [
                    d["y_true"] for d in oracle_data if d["policy"] == policy
                ]
                if policy_scores:
                    oracle_scores[policy] = sum(policy_scores) / len(policy_scores)

            oracle_results = oracle_scores

    return {
        "experiment": experiment_name,
        "estimator": estimator.__class__.__name__,
        "results": results,
        "oracle": oracle_results,
        "n_samples": len(logs),
    }


def main():
    """Run all ablation experiments."""
    console.print("[bold cyan]Phase 2: CJE Ablations on Arena 10K Data[/bold cyan]")
    console.print("=" * 60)

    # Find all ablation configs
    ablations_dir = Path(__file__).parent / "configs" / "ablations"
    config_files = sorted(ablations_dir.glob("*.yaml"))

    console.print(f"\nFound {len(config_files)} ablation configurations")

    # Run each ablation
    all_results = []
    for config_file in config_files:
        try:
            result = run_single_ablation(config_file)
            all_results.append(result)
        except Exception as e:
            console.print(f"[red]Error running {config_file.name}: {e}[/red]")

    # Display results summary
    console.print("\n[bold green]Results Summary[/bold green]")

    # Create results table
    table = Table(title="CJE Ablation Results")
    table.add_column("Experiment", style="cyan")
    table.add_column("Estimator", style="magenta")
    table.add_column("pi_clone", justify="right")
    table.add_column("pi_cot", justify="right")
    table.add_column("pi_bigger_model", justify="right")
    table.add_column("pi_bad", justify="right")

    for result in all_results:
        row = [
            result["experiment"].replace("arena_10k_", ""),
            result["estimator"],
        ]

        # Add policy estimates
        for policy in ["pi_clone", "pi_cot", "pi_bigger_model", "pi_bad"]:
            value = result["results"].get(policy, {}).get("mean", 0)
            row.append(f"{value:.3f}")

        table.add_row(*row)

    console.print(table)

    # If we have oracle results, show comparison
    if any(r["oracle"] for r in all_results):
        console.print("\n[bold]Oracle Comparison[/bold]")
        oracle_table = Table(title="Estimator vs Oracle")
        oracle_table.add_column("Policy", style="cyan")
        oracle_table.add_column("Oracle", justify="right", style="green")

        # Get first result with oracle data
        oracle_data = next(r["oracle"] for r in all_results if r["oracle"])

        for policy, oracle_value in sorted(oracle_data.items()):
            oracle_table.add_row(policy, f"{oracle_value:.3f}")

        console.print(oracle_table)

    console.print("\n[bold green]✅ All ablations completed![/bold green]")


if __name__ == "__main__":
    main()
