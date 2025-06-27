"""
Backfill log probabilities CLI - WITHOUT dangerous fallback values.

This tool adds log probabilities to existing datasets without corrupting data.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.progress import track

from ..types import LogProbResult
from ..loggers.base_policy import BasePolicy
from ..loggers.api_policy import create_api_policy

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="Backfill log probabilities for existing datasets")


@app.command()
def backfill(
    input_file: Path = typer.Argument(..., help="Input JSONL file"),
    output_file: Path = typer.Argument(..., help="Output JSONL file"),
    provider: str = typer.Option("openai", help="API provider"),
    model: str = typer.Option("gpt-3.5-turbo", help="Model name"),
    batch_size: int = typer.Option(10, help="Batch size for processing"),
    max_retries: int = typer.Option(3, help="Max retries per sample"),
    skip_existing: bool = typer.Option(True, help="Skip items that already have logp"),
    fail_on_error: bool = typer.Option(False, help="Stop on first error"),
) -> None:
    """
    Backfill log probabilities for a dataset.

    Key features:
    - No fallback values - failures are marked as null
    - Detailed error reporting
    - Progress tracking
    - Resume capability
    """
    console.print(f"[bold blue]Backfilling log probabilities[/bold blue]")
    console.print(f"Input: {input_file}")
    console.print(f"Output: {output_file}")
    console.print(f"Model: {provider}:{model}")

    # Initialize policy
    try:
        policy = create_api_policy(
            provider=provider,
            model_name=model,
            max_retries=max_retries,
        )
    except Exception as e:
        console.print(f"[red]Failed to initialize {provider} policy: {e}[/red]")
        raise typer.Exit(1)

    # Read input file
    try:
        with open(input_file, "r") as f:
            lines = [json.loads(line) for line in f]
        console.print(f"Loaded {len(lines)} items")
    except Exception as e:
        console.print(f"[red]Error reading input file: {e}[/red]")
        raise typer.Exit(1)

    # Process items
    success_count = 0
    skip_count = 0
    fail_count = 0
    results = []

    for item in track(lines, description="Processing items"):
        # Check if we should skip
        if skip_existing and "logp" in item:
            skip_count += 1
            results.append(item)
            continue

        # Get context and response
        context = item.get("context", "")
        response = item.get("response", "")

        if not context or not response:
            logger.warning(
                f"Skipping item {item.get('id', 'unknown')}: missing context or response"
            )
            item["logp"] = None
            item["logp_error"] = "missing_input"
            fail_count += 1
            results.append(item)
            continue

        # Compute log probability
        result = policy.compute_log_prob(context, response)

        if result.is_valid:
            # Success!
            item["logp"] = result.value
            item["logp_attempts"] = result.attempts
            success_count += 1
        else:
            # Failed - mark explicitly
            item["logp"] = None  # NOT -100.0 or any other fake value!
            item["logp_error"] = result.error
            item["logp_error_type"] = result.status.value
            item["logp_attempts"] = result.attempts
            fail_count += 1

            console.print(
                f"[yellow]Failed for item {item.get('id', 'unknown')}: "
                f"{result.status.value} - {result.error}[/yellow]"
            )

            if fail_on_error:
                console.print("[red]Stopping due to error (--fail-on-error set)[/red]")
                break

        results.append(item)

    # Write output
    try:
        with open(output_file, "w") as f:
            for item in results:
                f.write(json.dumps(item) + "\n")
        console.print(f"[green]Wrote {len(results)} items to {output_file}[/green]")
    except Exception as e:
        console.print(f"[red]Error writing output file: {e}[/red]")
        raise typer.Exit(1)

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Processed: {success_count}")
    console.print(f"  Skipped: {skip_count}")
    console.print(f"  Failed: {fail_count}")

    if fail_count > 0:
        console.print(
            f"\n[yellow]⚠️  {fail_count} items failed. "
            f"These have logp=null and error details.[/yellow]"
        )
        console.print("You can:")
        console.print("  1. Re-run with different model/settings")
        console.print("  2. Filter out failed items")
        console.print("  3. Use imputation methods")

    # Report statistics
    stats = policy.get_stats()
    console.print(f"\n[bold]Model Statistics:[/bold]")
    console.print(f"  Total calls: {stats['total_calls']}")
    console.print(f"  Success rate: {stats['success_rate']:.1%}")
    console.print(f"  Retry distribution: {stats['retry_distribution']}")
    console.print(f"  Error types: {stats['error_types']}")


@app.command()
def validate(
    input_file: Path = typer.Argument(..., help="JSONL file to validate"),
) -> None:
    """
    Validate a dataset for dangerous fallback values.

    Checks for:
    - Log probabilities of exactly -100.0
    - Log probabilities of exactly 0.0
    - Other suspicious values
    """
    console.print(f"[bold blue]Validating {input_file}[/bold blue]")

    suspicious_values = {
        -100.0: "FALLBACK_LOG_PROB",
        0.0: "Implies P=1.0 (perfect prediction)",
        -50.0: "Common arbitrary replacement",
    }

    found_issues = {v: 0 for v in suspicious_values}
    total_items = 0
    items_with_logp = 0

    try:
        with open(input_file, "r") as f:
            for i, line in enumerate(f):
                total_items += 1
                item = json.loads(line)

                if "logp" in item and item["logp"] is not None:
                    items_with_logp += 1
                    logp = item["logp"]

                    # Check for suspicious values
                    for value, desc in suspicious_values.items():
                        if abs(logp - value) < 1e-10:
                            found_issues[value] += 1
                            if found_issues[value] <= 5:  # Show first 5
                                console.print(
                                    f"[red]Line {i+1}: Found {desc} "
                                    f"(logp={logp})[/red]"
                                )
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        raise typer.Exit(1)

    # Report
    console.print(f"\n[bold]Validation Report:[/bold]")
    console.print(f"  Total items: {total_items}")
    console.print(f"  Items with logp: {items_with_logp}")

    has_issues = False
    for value, count in found_issues.items():
        if count > 0:
            has_issues = True
            console.print(
                f"  [red]Found {count} items with logp={value} "
                f"({suspicious_values[value]})[/red]"
            )

    if not has_issues:
        console.print("[green]✅ No suspicious log probability values found![/green]")
    else:
        console.print(
            "\n[red]⚠️  Dataset contains suspicious values that may corrupt results![/red]"
        )
        console.print("Consider re-computing these log probabilities.")

    # Don't return anything since function expects None


if __name__ == "__main__":
    app()
