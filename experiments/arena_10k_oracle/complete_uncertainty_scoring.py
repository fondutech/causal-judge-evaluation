#!/usr/bin/env python3
"""
Complete uncertainty scoring with robust error handling and progress tracking.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Set
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from cje.judge.factory import JudgeFactory

console = Console()


def load_completed_ids(checkpoint_file: Path) -> Set[str]:
    """Load IDs that have already been processed."""
    completed = set()
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    uid = f"{data['prompt_id']}_{data['policy']}"
                    completed.add(uid)
                except:
                    pass
    return completed


def load_targets_to_score(target_file: Path, completed_ids: Set[str]) -> List[Dict]:
    """Load target responses that haven't been scored yet."""
    to_score = []
    with open(target_file) as f:
        for line in f:
            data = json.loads(line)
            uid = f"{data['prompt_id']}_{data['policy']}"
            if uid not in completed_ids:
                to_score.append(data)
    return to_score


def score_batch_with_retry(
    judge, batch: List[Dict], max_retries: int = 3
) -> List[Dict]:
    """Score a batch with retry logic."""
    results = []

    for item in batch:
        success = False
        for attempt in range(max_retries):
            try:
                score = judge.score(context=item["prompt"], response=item["response"])

                scored = item.copy()
                scored["judge_score"] = {"mean": score.mean, "variance": score.variance}
                results.append(scored)
                success = True
                break

            except Exception as e:
                if attempt == max_retries - 1:
                    console.print(
                        f"[red]Failed after {max_retries} attempts: {item['prompt_id']}[/red]"
                    )
                    console.print(f"[red]Error: {str(e)[:100]}[/red]")
                else:
                    time.sleep(2**attempt)  # Exponential backoff

    return results


def main():
    """Complete the uncertainty scoring."""
    console.print("[bold cyan]Completing Arena 10K Uncertainty Scoring[/bold cyan]")

    # Paths
    target_file = Path("data/target_responses.jsonl")
    checkpoint_file = Path("data/targets_scored_uncertainty.checkpoint.jsonl")
    output_file = Path("data/targets_scored_uncertainty.jsonl")

    # Load what's already done
    completed_ids = load_completed_ids(checkpoint_file)
    console.print(f"[green]Already completed: {len(completed_ids):,} entries[/green]")

    # Load what needs to be done
    to_score = load_targets_to_score(target_file, completed_ids)
    console.print(f"[yellow]Remaining to score: {len(to_score):,} entries[/yellow]")

    if not to_score:
        console.print("[green]✅ All entries already scored![/green]")
        return

    # Create judge
    console.print("[cyan]Creating uncertainty judge...[/cyan]")
    judge = JudgeFactory.create(
        model="accounts/fireworks/models/llama4-scout-instruct-basic",
        provider="fireworks",
        temperature=0.0,
        uncertainty_method="confidence_interval",
    )

    # Process in batches
    batch_size = 4
    total_batches = (len(to_score) + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("Scoring responses...", total=len(to_score))

        start_time = time.time()
        processed = 0

        for i in range(0, len(to_score), batch_size):
            batch = to_score[i : i + batch_size]

            # Score batch with retry
            results = score_batch_with_retry(judge, batch)

            # Append to checkpoint file
            if results:
                with open(checkpoint_file, "a") as f:
                    for result in results:
                        f.write(json.dumps(result) + "\n")

                processed += len(results)
                progress.update(task, advance=len(results))

                # Show rate
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (len(to_score) - processed) / rate if rate > 0 else 0

                progress.console.print(
                    f"[dim]Rate: {rate:.1f} items/sec | "
                    f"ETA: {remaining/60:.1f} minutes[/dim]"
                )

            # Small delay to avoid rate limits
            time.sleep(0.5)

    # Create final output file
    console.print("\n[cyan]Creating final output file...[/cyan]")

    all_scored = []
    with open(checkpoint_file) as f:
        for line in f:
            all_scored.append(json.loads(line))

    # Remove duplicates (keeping last occurrence)
    seen = {}
    for item in all_scored:
        uid = f"{item['prompt_id']}_{item['policy']}"
        seen[uid] = item

    # Write final file
    with open(output_file, "w") as f:
        for item in seen.values():
            f.write(json.dumps(item) + "\n")

    console.print(f"\n[green]✅ Complete! Scored {len(seen):,} unique entries[/green]")
    console.print(f"[green]Output: {output_file}[/green]")

    # Show summary by policy
    policy_counts = {}
    for item in seen.values():
        policy = item.get("policy", "unknown")
        policy_counts[policy] = policy_counts.get(policy, 0) + 1

    console.print("\n[cyan]Summary by policy:[/cyan]")
    for policy, count in sorted(policy_counts.items()):
        console.print(f"  {policy}: {count:,}")


if __name__ == "__main__":
    main()
