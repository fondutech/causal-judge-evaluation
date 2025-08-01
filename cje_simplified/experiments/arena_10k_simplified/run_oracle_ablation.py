#!/usr/bin/env python3
"""
Orchestrate oracle coverage ablation study.

This script runs multiple experiments varying:
1. Oracle coverage (25%, 50%, 100%)
2. Estimator type (CalibratedIPS, RawIPS)
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


ABLATION_CONFIG: Dict[str, Any] = {
    "oracle_fractions": [0.25, 0.50, 1.00],
    "estimators": [
        {"name": "calibrated-ips", "display": "CalibratedIPS"},
        {"name": "raw-ips", "display": "RawIPS"},
    ],
    "seeds": [42],  # Can add more seeds for multiple runs
}


def run_single_experiment(
    data_file: Path, estimator: str, output_dir: Path, n_folds: int = 5
) -> Dict[str, Any]:
    """Run a single CJE analysis experiment."""

    # Create output filename
    data_name = data_file.stem
    output_file = output_dir / f"{estimator}_{data_name}.json"

    # Run analysis
    cmd = [
        "poetry",
        "run",
        "python",
        "run_cje_analysis.py",
        "--data",
        str(data_file),
        "--estimator",
        estimator,
        "--n-folds",
        str(n_folds),
        "--output",
        str(output_file),
    ]

    print(f"  Running: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ❌ Failed: {result.stderr}")
        return None

    # Load and return results
    with open(output_file) as f:
        data: Dict[str, Any] = json.load(f)
        return data


def extract_metrics(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Extract key metrics from a result."""
    if result is None:
        return None

    # Parse oracle fraction from path
    path_parts = result["dataset"]["path"].split("oracle_")
    if len(path_parts) > 1:
        frac_str = path_parts[1].split("_seed")[0].replace("_", ".")
    else:
        frac_str = "1.00"

    metrics = {
        "estimator": result["estimation"]["estimator"],
        "oracle_fraction": float(frac_str),
        "n_samples": result["dataset"]["n_samples"],
    }

    # Add policy estimates and SEs
    for policy, data in result["estimation"]["policies"].items():
        metrics[f"{policy}_estimate"] = data["estimate"]
        metrics[f"{policy}_se"] = data["standard_error"]
        metrics[f"{policy}_ci_width"] = data["ci_upper"] - data["ci_lower"]

    # Add weight diagnostics if available
    if "weight_diagnostics" in result:
        for policy, diag in result["weight_diagnostics"]["all_policies"].items():
            metrics[f"{policy}_ess"] = diag["ess_fraction"]
            metrics[f"{policy}_max_weight"] = diag["max_weight"]

    return metrics


def create_summary_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create summary table from results."""
    # Extract metrics
    rows = []
    for result in results:
        metrics = extract_metrics(result)
        if metrics:
            rows.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Sort by oracle fraction and estimator
    df["oracle_fraction"] = df["oracle_fraction"].astype(float)
    df = df.sort_values(["oracle_fraction", "estimator"])

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run oracle coverage ablation study")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("test_e2e_data/ablation_data"),
        help="Directory containing ablation datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_e2e_data/ablation_results"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Prepare ablation datasets before running experiments",
    )

    args = parser.parse_args()

    print("Oracle Coverage Ablation Study")
    print("=" * 50)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data if requested
    if args.prepare_data:
        print("\nPreparing ablation datasets...")
        for frac in ABLATION_CONFIG["oracle_fractions"]:
            for seed in ABLATION_CONFIG["seeds"]:
                frac_str = f"{frac:.2f}".replace(".", "_")
                output_file = args.data_dir / f"oracle_{frac_str}_seed{seed}.jsonl"
                if not output_file.exists():
                    cmd = [
                        "poetry",
                        "run",
                        "python",
                        "prepare_ablation_data.py",
                        "--oracle-fraction",
                        str(frac),
                        "--seed",
                        str(seed),
                        "--output",
                        str(output_file),
                    ]
                    print(f"  Creating {output_file.name}...")
                    subprocess.run(cmd, check=True)

    # Run experiments
    print(f"\nRunning experiments...")
    results = []

    for frac in ABLATION_CONFIG["oracle_fractions"]:
        for seed in ABLATION_CONFIG["seeds"]:
            # Format fraction with consistent decimal places
            frac_str = f"{frac:.2f}".replace(".", "_")
            data_file = args.data_dir / f"oracle_{frac_str}_seed{seed}.jsonl"

            if not data_file.exists():
                print(f"  ⚠️  Skipping {data_file} (not found)")
                continue

            print(f"\nOracle fraction: {frac:.0%}, Seed: {seed}")

            for estimator_config in ABLATION_CONFIG["estimators"]:
                estimator = estimator_config["name"]
                print(f"  Estimator: {estimator_config['display']}")

                result = run_single_experiment(
                    data_file, estimator, args.output_dir, args.n_folds
                )

                if result:
                    results.append(result)

    # Create summary
    print("\nCreating summary table...")
    df = create_summary_table(results)

    # Save as CSV
    csv_path = args.output_dir / "ablation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved to: {csv_path}")

    # Display key results
    print("\n" + "=" * 80)
    print("SUMMARY: Policy Estimates by Oracle Coverage")
    print("=" * 80)

    # Group by oracle fraction and estimator for cleaner display
    for policy in ["base", "clone", "unhelpful"]:
        print(f"\n{policy.upper()} Policy:")
        print("Oracle %  | Estimator      | Estimate ± SE      | 95% CI Width")
        print("-" * 65)

        for oracle_frac in sorted(df["oracle_fraction"].unique()):
            for estimator in ABLATION_CONFIG["estimators"]:
                est_name = estimator["name"]
                row = df[
                    (df["oracle_fraction"] == oracle_frac)
                    & (df["estimator"] == est_name)
                ]

                if not row.empty:
                    estimate = row[f"{policy}_estimate"].values[0]
                    se = row[f"{policy}_se"].values[0]
                    ci_width = row[f"{policy}_ci_width"].values[0]

                    print(
                        f"{oracle_frac:6.1%}   | {estimator['display']:14} | "
                        f"{estimate:.3f} ± {se:.3f}    | {ci_width:.3f}"
                    )

    # ESS comparison
    print("\n" + "=" * 80)
    print("EFFECTIVE SAMPLE SIZE (ESS) by Oracle Coverage")
    print("=" * 80)

    for policy in ["clone", "unhelpful"]:
        print(f"\n{policy.upper()} Policy:")
        print("Oracle %  | Estimator      | ESS %   | Max Weight")
        print("-" * 55)

        for oracle_frac in sorted(df["oracle_fraction"].unique()):
            for estimator in ABLATION_CONFIG["estimators"]:
                est_name = estimator["name"]
                row = df[
                    (df["oracle_fraction"] == oracle_frac)
                    & (df["estimator"] == est_name)
                ]

                if not row.empty:
                    ess = row[f"{policy}_ess"].values[0] * 100
                    max_w = row[f"{policy}_max_weight"].values[0]

                    print(
                        f"{oracle_frac:6.1%}   | {estimator['display']:14} | "
                        f"{ess:5.1f}%  | {max_w:8.1f}"
                    )

    print("\n✓ Ablation study complete!")


if __name__ == "__main__":
    main()
