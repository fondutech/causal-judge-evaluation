#!/usr/bin/env python3
"""
Phase 2: Run complete CJE ablations with all estimators.

This script runs CJE ablations comparing:
1. Judge types: Deterministic vs Uncertainty
2. Estimators: IPS, SNIPS, Calibrated IPS, DRCPO, MRDR
"""

import subprocess
import json
import yaml  # type: ignore[import-untyped]
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import track
import sys
import time

console = Console()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def create_ablation_config(
    ablation_name: str,
    judge_type: str,  # "deterministic" or "uncertainty"
    estimator: str,  # "ips", "snips", "calibrated_ips", "drcpo", "mrdr"
    data_dir: Path = Path("../data"),
) -> Dict[str, Any]:
    """Create configuration for a specific ablation."""

    # Map estimator names to CJE classes
    estimator_map = {
        "ips": "cje.estimators.ips_only_estimators.MultiIPSEstimator",
        "snips": "cje.estimators.ips_only_estimators.MultiSNIPSEstimator",
        "calibrated_ips": "cje.estimators.ips_only_estimators.CalibratedIPSEstimator",
        "drcpo": "cje.estimators.doubly_robust_estimators.MultiDRCPOEstimator",
        "mrdr": "cje.estimators.doubly_robust_estimators.MultiMRDREstimator",
    }

    # Judge score files based on type
    judge_files = {
        "deterministic": {
            "p0": "p0_scored_deterministic.jsonl",
            "targets": "targets_scored_deterministic.jsonl",
        },
        "uncertainty": {
            "p0": "p0_scored_uncertainty.jsonl",
            "targets": "targets_scored_uncertainty.jsonl",
        },
    }

    config: Dict[str, Any] = {
        "experiment_name": f"arena_10k_{ablation_name}",
        "data": {
            "logging_policy_file": str(data_dir / judge_files[judge_type]["p0"]),
            "target_policy_file": str(data_dir / judge_files[judge_type]["targets"]),
            "output_dir": f"outputs/arena_10k_{ablation_name}",
        },
        "estimator": {
            "_target_": estimator_map[estimator],
        },
        "calibrator": {
            "_target_": "cje.calibration.isotonic.IsotonicCalibrator",
            "cv_folds": 5,
        },
        "oracle": {
            "enabled": True,
            "calibration_file": str(
                data_dir / "labeling/oracle_labels_calibration_detailed.jsonl"
            ),
            "validation_file": str(
                data_dir / "labeling/oracle_labels_validation_detailed.jsonl"
            ),
        },
        "judge": {
            "uncertainty_method": "structured" if judge_type == "uncertainty" else None,
        },
        "policies": {
            "logging": "pi_0",
            "target": ["pi_cot", "pi_bigger_model", "pi_bad"],
        },
        "logging": {
            "level": "INFO",
            "rich": True,
        },
    }

    # Add estimator-specific configurations
    if estimator == "calibrated_ips":
        config["estimator"]["clip_min"] = 0.01
        config["estimator"]["clip_max"] = 100.0
    elif estimator in ["drcpo", "mrdr"]:
        # These estimators need additional configuration
        config["estimator"]["n_folds"] = 5
        config["estimator"]["ridge_alpha"] = 0.1

    return config


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """Save configuration to YAML file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def run_cje_experiment(config_path: Path) -> Optional[Dict[str, Any]]:
    """Run CJE experiment with given config."""

    # Prepare command
    cmd = ["python", "-m", "cje.pipeline", "--config", str(config_path)]

    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent.parent),  # Run from repo root
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            console.print(f"[green]✅ Success[/green] ({elapsed_time:.1f}s)")

            # Try to load results
            output_dir = (
                Path(config_path).parent.parent
                / f"outputs/arena_10k_{config_path.stem}"
            )
            results_file = output_dir / "results.json"

            if results_file.exists():
                with open(results_file) as f:
                    results: Dict[str, Any] = json.load(f)
                    results["elapsed_time"] = elapsed_time
                    return results
            else:
                console.print("[yellow]⚠️  Results file not found[/yellow]")
                return {"status": "success", "elapsed_time": elapsed_time}
        else:
            console.print(f"[red]❌ Failed[/red] (exit code: {result.returncode})")
            if result.stderr:
                console.print(f"[red]Error:[/red] {result.stderr[:500]}...")
            return None

    except Exception as e:
        console.print(f"[red]❌ Exception: {e}[/red]")
        return None


def create_results_table(all_results: Dict[str, Optional[Dict[str, Any]]]) -> Table:
    """Create a comparison table of all results."""

    table = Table(title="CJE Ablation Results - Arena 10K Oracle Experiment")

    # Add columns
    table.add_column("Judge Type", style="cyan", width=12)
    table.add_column("Estimator", style="yellow", width=15)
    table.add_column("π_cot", justify="right", style="green")
    table.add_column("π_bigger", justify="right", style="green")
    table.add_column("π_bad", justify="right", style="red")
    table.add_column("Runtime", justify="right", style="dim")
    table.add_column("Status", justify="center")

    # Sort results for consistent display
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: (
            x[0].split("_")[0],  # Judge type first
            x[0].split("_")[1],  # Then estimator
        ),
    )

    for ablation_name, results in sorted_results:
        parts = ablation_name.split("_")
        judge_type = parts[0].capitalize()
        estimator = "_".join(parts[1:]).upper().replace("_", "-")

        if results and "policy_scores" in results:
            scores = results["policy_scores"]
            row = [
                judge_type,
                estimator,
                (
                    f"{scores.get('pi_cot', {}).get('mean', 'N/A'):.3f}"
                    if isinstance(scores.get("pi_cot", {}).get("mean"), (int, float))
                    else "N/A"
                ),
                (
                    f"{scores.get('pi_bigger_model', {}).get('mean', 'N/A'):.3f}"
                    if isinstance(
                        scores.get("pi_bigger_model", {}).get("mean"), (int, float)
                    )
                    else "N/A"
                ),
                (
                    f"{scores.get('pi_bad', {}).get('mean', 'N/A'):.3f}"
                    if isinstance(scores.get("pi_bad", {}).get("mean"), (int, float))
                    else "N/A"
                ),
                f"{results.get('elapsed_time', 0):.1f}s",
                "✅" if results else "❌",
            ]
        else:
            row = [judge_type, estimator, "N/A", "N/A", "N/A", "N/A", "❌"]

        table.add_row(*row)

    return table


def main() -> None:
    """Run all ablations."""

    console.print("\n[bold cyan]🔬 Arena 10K Oracle - CJE Ablations[/bold cyan]")
    console.print("=" * 60)

    # Check if Phase 1 is complete
    data_dir = Path(__file__).parent.parent / "data"
    required_files = [
        "p0_scored_deterministic.jsonl",
        "p0_scored_uncertainty.jsonl",
        "targets_scored_deterministic.jsonl",
        "targets_scored_uncertainty.jsonl",
        "labeling/oracle_labels_calibration_detailed.jsonl",
        "labeling/oracle_labels_validation_detailed.jsonl",
    ]

    missing_files = [f for f in required_files if not (data_dir / f).exists()]
    if missing_files:
        console.print("[red]❌ Missing required files from Phase 1:[/red]")
        for f in missing_files:
            console.print(f"   - {f}")
        console.print("\n[yellow]Please complete Phase 1 first![/yellow]")
        return

    # Define all ablations
    judge_types = ["deterministic", "uncertainty"]
    estimators = ["ips", "snips", "calibrated_ips", "drcpo", "mrdr"]

    total_ablations = len(judge_types) * len(estimators)
    console.print(f"\n📊 Running {total_ablations} ablations:")
    console.print(f"   • Judge types: {', '.join(judge_types)}")
    console.print(f"   • Estimators: {', '.join(estimators)}")

    # Confirm
    response = console.input("\n▶️  Continue? [Y/n]: ").strip().lower()
    if response == "n":
        console.print("Aborted.")
        return

    # Create configs directory
    configs_dir = Path(__file__).parent / "configs" / "ablations"
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    all_results = {}

    # Run each ablation
    console.print("\n[bold]Running ablations...[/bold]\n")

    for judge_type in judge_types:
        for estimator in estimators:
            ablation_name = f"{judge_type}_{estimator}"
            console.print(f"🔄 Running [cyan]{ablation_name}[/cyan]... ", end="")

            # Create config
            config = create_ablation_config(
                ablation_name, judge_type, estimator, data_dir
            )
            config_path = configs_dir / f"{ablation_name}.yaml"
            save_config(config, config_path)

            # Run experiment
            results = run_cje_experiment(config_path)
            all_results[ablation_name] = results

    # Display results table
    console.print("\n")
    table = create_results_table(all_results)
    console.print(table)

    # Save detailed results
    results_path = Path(__file__).parent / "results" / "ablation_results.json"
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, "w") as results_file:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ablations": all_results,
                "summary": {
                    "total_ablations": total_ablations,
                    "successful": sum(1 for r in all_results.values() if r is not None),
                    "failed": sum(1 for r in all_results.values() if r is None),
                },
            },
            results_file,
            indent=2,
        )

    console.print(f"\n💾 Results saved to: {results_path}")

    # Analysis suggestions
    console.print("\n[bold cyan]📊 Next steps for analysis:[/bold cyan]")
    console.print("1. Compare deterministic vs uncertainty judging impact")
    console.print("2. Analyze estimator performance across policies")
    console.print("3. Check calibration quality for each method")
    console.print("4. Validate against oracle labels")

    console.print("\n✅ [bold green]Ablations complete![/bold green]")


if __name__ == "__main__":
    main()
