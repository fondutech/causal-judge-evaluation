#!/usr/bin/env python3
"""
Run CJE ablation analysis directly using the estimators.
This bypasses the pipeline and works directly with the scored data.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from rich.console import Console
from rich.table import Table
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from cje.estimators import get_estimator
from cje.data.schema import CJESample
from cje.loggers.precomputed_sampler import PrecomputedMultiTargetSampler

console = Console()


def load_scored_data(
    judge_type: str = "deterministic",
) -> Dict[str, List[Dict[str, Any]]]:
    """Load scored data for analysis."""

    data_dir = Path("data")

    # Load logging policy data
    p0_file = data_dir / f"p0_scored_{judge_type}.jsonl"
    p0_data = []
    with open(p0_file) as f:
        for line in f:
            p0_data.append(json.loads(line))

    # Load target policy data
    targets_file = data_dir / f"targets_scored_{judge_type}.jsonl"
    targets_data = []
    with open(targets_file) as f:
        for line in f:
            targets_data.append(json.loads(line))

    # Group target data by policy
    target_by_policy: Dict[str, List[Dict[str, Any]]] = {
        "pi_cot": [],
        "pi_bigger_model": [],
        "pi_bad": [],
    }

    for item in targets_data:
        policy = item["model"]
        if policy in target_by_policy:
            target_by_policy[policy].append(item)

    return {"pi_0": p0_data, **target_by_policy}


def prepare_estimation_data(
    data: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Prepare data for estimation."""

    # Get unique prompt IDs
    prompt_ids = set()
    for policy_data in data.values():
        for item in policy_data:
            prompt_ids.add(item["prompt_id"])

    # Create estimation records
    records = []
    for prompt_id in sorted(prompt_ids):
        # Get logging policy data
        p0_item = next(
            (item for item in data["pi_0"] if item["prompt_id"] == prompt_id), None
        )
        if not p0_item:
            continue

        # Create base record
        record = {
            "uid": prompt_id,
            "context": p0_item["prompt"],
            "response": p0_item["response"],
            "reward": p0_item["judge_score"]["mean"],
            "judge_variance": p0_item["judge_score"]["variance"],
            "logp_pi0": p0_item.get("logp", -10.0),
            "target_logps": {},
            "target_rewards": {},
        }

        # Add target policy data
        for policy in ["pi_cot", "pi_bigger_model", "pi_bad"]:
            target_item = next(
                (item for item in data[policy] if item["prompt_id"] == prompt_id), None
            )
            if target_item:
                record["target_logps"][policy] = target_item.get("logp", -10.0)
                record["target_rewards"][policy] = target_item["judge_score"]["mean"]

        records.append(record)

    return records


def run_estimator_analysis(
    estimator_name: str, records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run analysis with a specific estimator."""

    results = {}

    # Build lookup for precomputed sampler
    # Format: (context, response) -> [logp_pi_cot, logp_pi_bigger, logp_pi_bad]
    logp_lookup: Dict[Tuple[str, str], List[float]] = {}
    policy_names = ["pi_cot", "pi_bigger_model", "pi_bad"]
    n_policies = len(policy_names)

    for record in records:
        key = (record["context"], record["response"])
        logps = []
        for policy in policy_names:
            if policy in record["target_logps"]:
                logps.append(record["target_logps"][policy])
            else:
                logps.append(-10.0)  # Default if missing
        logp_lookup[key] = logps

    # Create PrecomputedMultiTargetSampler
    sampler = PrecomputedMultiTargetSampler(
        logp_lookup=logp_lookup, n_policies=n_policies
    )

    # Create estimator with sampler
    try:
        estimator = get_estimator(estimator_name, sampler=sampler)
    except:
        # Fallback for estimators that don't use sampler
        estimator = get_estimator(estimator_name)

    # Prepare logged data in the format expected by estimators
    logs = []
    for record in records:
        log = {
            "context": record["context"],
            "response": record["response"],
            "reward": record["reward"],
            "logp": record["logp_pi0"],
        }
        logs.append(log)

    # Fit the estimator
    estimator.fit(logs)

    # Get estimates
    result = estimator.estimate()

    # Convert results to our format
    for i, policy in enumerate(policy_names):
        results[policy] = {
            "estimate": result.v_hat[i],
            "se": result.se[i],
            "ci_lower": result.v_hat[i] - 1.96 * result.se[i],
            "ci_upper": result.v_hat[i] + 1.96 * result.se[i],
            "n_samples": len(logs),
        }

    return results


def main() -> None:
    """Run complete ablation analysis."""

    console.print("[bold blue]🔬 Arena 10K Ablation Analysis[/bold blue]\n")

    # Create results table
    table = Table(title="CJE Ablation Results")
    table.add_column("Judge Type", style="cyan")
    table.add_column("Estimator", style="yellow")
    table.add_column("π_cot", justify="right", style="green")
    table.add_column("π_bigger", justify="right", style="green")
    table.add_column("π_bad", justify="right", style="red")
    table.add_column("Avg SE", justify="right")

    all_results = {}

    for judge_type in ["deterministic", "uncertainty"]:
        console.print(f"\n[bold]Processing {judge_type} judge scores...[/bold]")

        # Load data
        data = load_scored_data(judge_type)
        records = prepare_estimation_data(data)
        console.print(f"  Loaded {len(records)} samples")

        for estimator in ["IPS", "SNIPS", "CalibratedIPS", "DRCPO", "MRDR"]:
            console.print(f"  Running {estimator}...")

            results = run_estimator_analysis(estimator, records)

            # Store results
            key = f"{judge_type}_{estimator.lower()}"
            all_results[key] = results

            # Add to table
            avg_se = np.mean([r["se"] for r in results.values()])
            table.add_row(
                judge_type.capitalize(),
                estimator,
                f"{results['pi_cot']['estimate']:.3f} ± {results['pi_cot']['se']:.3f}",
                f"{results['pi_bigger_model']['estimate']:.3f} ± {results['pi_bigger_model']['se']:.3f}",
                f"{results['pi_bad']['estimate']:.3f} ± {results['pi_bad']['se']:.3f}",
                f"{avg_se:.3f}",
            )

    # Display results
    console.print("\n")
    console.print(table)

    # Save results
    output_dir = Path("phase2_cje_ablations/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "ablation_analysis.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    console.print(f"\n💾 Results saved to {results_file}")

    # Summary insights
    console.print("\n[bold]📊 Key Insights:[/bold]")

    # Compare judge types
    det_avg = np.mean(
        [
            all_results[f"deterministic_ips"]["pi_cot"]["estimate"],
            all_results[f"deterministic_ips"]["pi_bigger_model"]["estimate"],
        ]
    )
    unc_avg = np.mean(
        [
            all_results[f"uncertainty_ips"]["pi_cot"]["estimate"],
            all_results[f"uncertainty_ips"]["pi_bigger_model"]["estimate"],
        ]
    )

    console.print(f"  • Deterministic vs Uncertainty: {det_avg:.3f} vs {unc_avg:.3f}")

    # Best estimator
    best_se = float("inf")
    best_est = None
    for est in ["IPS", "SNIPS", "CalibratedIPS", "DRCPO", "MRDR"]:
        avg_se = np.mean(
            [
                all_results[f"deterministic_{est.lower()}"][p]["se"]
                for p in ["pi_cot", "pi_bigger_model", "pi_bad"]
            ]
        )
        if avg_se < best_se:
            best_se = avg_se
            best_est = est

    console.print(f"  • Lowest variance estimator: {best_est} (avg SE: {best_se:.3f})")

    # Policy ranking
    console.print("\n[bold]🏆 Policy Rankings (by IPS estimate):[/bold]")
    policies = ["pi_cot", "pi_bigger_model", "pi_bad"]
    scores = [(p, all_results["deterministic_ips"][p]["estimate"]) for p in policies]
    scores.sort(key=lambda x: x[1], reverse=True)

    for i, (policy, score) in enumerate(scores, 1):
        console.print(f"  {i}. {policy}: {score:.3f}")


if __name__ == "__main__":
    main()
