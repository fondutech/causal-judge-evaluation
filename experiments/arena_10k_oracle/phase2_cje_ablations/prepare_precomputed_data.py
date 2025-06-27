#!/usr/bin/env python3
"""
Prepare precomputed data for CJE ablations.

This script consolidates all our data into the format expected by CJE's
PrecomputedMultiTargetSampler.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


def load_teacher_forcing_data(file_path: str) -> List[Dict[str, Any]]:
    """Load P0 data with teacher forcing log probs."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_scored_data(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load scored data indexed by prompt_id."""
    data = {}
    if not Path(file_path).exists():
        console.print(f"[yellow]Warning: {file_path} not found[/yellow]")
        return data

    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            prompt_id = item.get("prompt_id")
            if prompt_id:
                data[prompt_id] = item
    return data


def prepare_cje_data(
    teacher_forcing_data: List[Dict[str, Any]],
    p0_scores: Dict[str, Dict[str, Any]],
    target_scores: Dict[str, Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Prepare data in CJE format.

    Returns:
        Dict with:
        - logp_lookup: Dict[Tuple[str, str], List[float]] mapping (context, response) to target logps
        - rows: List of data rows for CJE pipeline
        - policy_names: List of target policy names
    """
    logp_lookup = {}
    rows = []
    policy_names = None

    console.print(f"[blue]Processing {len(teacher_forcing_data)} items...[/blue]")

    for item in track(teacher_forcing_data, description="Preparing data"):
        prompt_id = item["prompt_id"]
        context = item["prompt"]
        response = item["response"]

        # Get P0 log prob and scores
        p0_logp = item["total_logprob"]

        # Get judge scores for P0 response
        p0_score_data = p0_scores.get(prompt_id, {})
        p0_judge_score = p0_score_data.get(
            "judge_score", {"mean": 0.5, "variance": 0.01}
        )

        # Get target policy log probs
        target_logps_dict = item.get("target_logps", {})

        if policy_names is None:
            policy_names = sorted(target_logps_dict.keys())

        # Convert to list format expected by PrecomputedMultiTargetSampler
        target_logps_list = [target_logps_dict[name] for name in policy_names]

        # Add to lookup
        logp_lookup[(context, response)] = target_logps_list

        # Create row for CJE pipeline
        row = {
            "prompt_id": prompt_id,
            "context": context,
            "response": response,
            "logp": p0_logp,  # Behavior policy log prob
            "judge_score": p0_judge_score,
            "reward": p0_judge_score["mean"],  # Use mean as reward
            "metadata": item.get("metadata", {}),
        }

        rows.append(row)

    # Add target response data if available
    if target_scores:
        console.print(f"[blue]Adding {len(target_scores)} target responses...[/blue]")
        for target_item in target_scores.values():
            context = target_item.get("prompt")
            response = target_item.get("response")
            if context and response and (context, response) not in logp_lookup:
                # For target-generated responses, we might not have teacher forcing data
                # Skip these for now - they're not needed for importance weighting
                pass

    return {
        "logp_lookup": logp_lookup,
        "rows": rows,
        "policy_names": policy_names,
        "n_policies": len(policy_names),
    }


def create_summary_stats(data: Dict[str, Any]) -> None:
    """Print summary statistics."""
    console.print("\n[bold]📊 Data Summary[/bold]")

    table = Table(title="Precomputed Data Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total rows", str(len(data["rows"])))
    table.add_row("Unique (context, response) pairs", str(len(data["logp_lookup"])))
    table.add_row("Number of target policies", str(data["n_policies"]))
    table.add_row("Policy names", ", ".join(data["policy_names"]))

    # Calculate importance weight statistics
    weights_stats = defaultdict(list)
    for row in data["rows"][:100]:  # Sample first 100
        context = row["context"]
        response = row["response"]
        p0_logp = row["logp"]

        target_logps = data["logp_lookup"].get((context, response), [])
        for i, (policy_name, target_logp) in enumerate(
            zip(data["policy_names"], target_logps)
        ):
            weight = np.exp(target_logp - p0_logp)
            weights_stats[policy_name].append(weight)

    # Add weight statistics
    table.add_row("", "")  # Separator
    table.add_row("[bold]Importance Weights (sample)[/bold]", "")

    for policy_name in data["policy_names"]:
        weights = weights_stats[policy_name]
        if weights:
            table.add_row(f"  {policy_name} mean", f"{np.mean(weights):.3f}")
            table.add_row(
                f"  {policy_name} range",
                f"[{np.min(weights):.3f}, {np.max(weights):.3f}]",
            )

    console.print(table)


def save_precomputed_data(data: Dict[str, Any], output_dir: Path) -> None:
    """Save data in format ready for CJE."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save rows as JSONL
    rows_file = output_dir / "precomputed_rows.jsonl"
    with open(rows_file, "w") as f:
        for row in data["rows"]:
            f.write(json.dumps(row) + "\n")

    # Save lookup table
    lookup_file = output_dir / "logp_lookup.json"
    # Convert tuple keys to strings for JSON serialization
    json_lookup = {
        f"{ctx}|||{resp}": logps for (ctx, resp), logps in data["logp_lookup"].items()
    }

    with open(lookup_file, "w") as f:
        json.dump(
            {
                "lookup": json_lookup,
                "policy_names": data["policy_names"],
                "n_policies": data["n_policies"],
            },
            f,
            indent=2,
        )

    console.print(f"\n[green]✅ Saved precomputed data to {output_dir}[/green]")
    console.print(f"  - Rows: {rows_file}")
    console.print(f"  - Lookup: {lookup_file}")


def main():
    """Prepare precomputed data for CJE ablations."""
    console.print(
        "[bold blue]🔧 Preparing Precomputed Data for CJE Ablations[/bold blue]\n"
    )

    # Define paths
    base_dir = Path("../data")

    # Check what data is available
    teacher_forcing_file = base_dir / "p0_with_target_logps.jsonl"
    checkpoint_file = base_dir / "p0_with_target_logps.checkpoint.jsonl"

    # Use checkpoint if main file doesn't exist
    if not teacher_forcing_file.exists() and checkpoint_file.exists():
        console.print(f"[yellow]Using checkpoint file: {checkpoint_file}[/yellow]")
        teacher_forcing_file = checkpoint_file

    if not teacher_forcing_file.exists():
        console.print(
            f"[red]Error: Teacher forcing data not found at {teacher_forcing_file}[/red]"
        )
        console.print(
            "[yellow]Wait for 02c_compute_target_logprobs.py to complete![/yellow]"
        )
        return

    # Load data
    console.print("📂 Loading data...")
    teacher_forcing_data = load_teacher_forcing_data(teacher_forcing_file)
    console.print(f"  ✓ Loaded {len(teacher_forcing_data)} items with teacher forcing")

    # Load P0 scores
    p0_det_scores = load_scored_data(base_dir / "p0_scored_deterministic.jsonl")
    p0_unc_scores = load_scored_data(base_dir / "p0_scored_uncertainty.jsonl")

    # Merge scores (prefer uncertainty scores when available)
    p0_scores = {**p0_det_scores, **p0_unc_scores}
    console.print(f"  ✓ Loaded scores for {len(p0_scores)} P0 responses")

    # Prepare CJE data
    console.print("\n🔄 Preparing CJE format...")
    cje_data = prepare_cje_data(teacher_forcing_data, p0_scores)

    # Show summary
    create_summary_stats(cje_data)

    # Save prepared data
    output_dir = Path("precomputed_data")
    save_precomputed_data(cje_data, output_dir)

    console.print("\n[bold green]✅ Data preparation complete![/bold green]")
    console.print("\nNext steps:")
    console.print("1. Wait for teacher forcing to complete if still running")
    console.print("2. Run CJE ablations with: python run_cje_with_precomputed.py")


if __name__ == "__main__":
    main()
