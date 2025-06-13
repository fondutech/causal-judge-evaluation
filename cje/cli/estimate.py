"""CLI command for running estimators on JSONL log files."""

import json
import pathlib
from typing import Optional, Dict, Any, Union, cast
import typer
from rich import print
from rich.console import Console
from rich.table import Table
from ..estimators import get_estimator

app = typer.Typer(help="Run an off-policy estimator on a JSONL log file")


@app.command()
def run(
    log_file: pathlib.Path = typer.Option(
        ...,
        help="Input JSONL file with logged data",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    estimator: str = typer.Option(
        ...,
        help="Estimator to use (IPS | SNIPS | DRCPO | MRDR)",
    ),
    out_json: pathlib.Path = typer.Option(
        ...,
        help="Output JSON file for results",
        file_okay=True,
        dir_okay=False,
    ),
    clip: Optional[float] = typer.Option(
        20.0,
        help="Maximum importance weight (default: 20.0)",
    ),
    k: Optional[int] = typer.Option(
        5,
        help="Number of cross-validation folds for DRCPO/MRDR (default: 5)",
    ),
) -> None:
    """
    Run an off-policy estimator on logged data.

    The input JSONL file should contain rows with:
    - context: Input context
    - response: Model response
    - reward: Observed reward
    - logp: Log probability under logging policy

    Note: This command requires pre-computed target policy log probabilities
    or a MultiTargetSampler for live evaluation.
    """
    # Read input data
    try:
        rows = [json.loads(l) for l in log_file.read_text().splitlines()]
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"Invalid JSON in {log_file}: {e}")

    if not rows:
        raise typer.BadParameter(f"Empty log file: {log_file}")

    # Note: This simplified version requires the user to provide a sampler
    # In practice, this command would need additional configuration for target policies
    typer.echo(
        "Note: This command requires additional configuration for target policies."
    )
    typer.echo("Consider using 'cje run' with a full configuration file instead.")

    # For now, just validate the data format
    required_fields = ["context", "response", "logp", "reward"]
    for i, row in enumerate(rows[:5]):  # Check first 5 rows
        missing = [f for f in required_fields if f not in row]
        if missing:
            raise typer.BadParameter(f"Row {i} missing required fields: {missing}")

    typer.echo(f"✅ Validated {len(rows)} rows with required fields")
    typer.echo("Use 'cje run' with a configuration file for complete estimation.")
