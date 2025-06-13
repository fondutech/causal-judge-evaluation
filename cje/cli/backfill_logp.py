"""CLI command for backfilling log probabilities in existing datasets."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console

from ..loggers.policy import PolicyRunner
from ..utils.error_handling import FALLBACK_LOG_PROB

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(help="Backfill log probabilities for existing datasets")


@app.command()
def backfill_logp(
    input_file: str = typer.Argument(..., help="Input JSONL file"),
    output_file: str = typer.Argument(..., help="Output JSONL file"),
    model_name: str = typer.Option("gpt2", help="Model name for PolicyRunner"),
    batch_size: int = typer.Option(16, help="Batch size for processing"),
) -> None:
    """Backfill log probabilities for existing datasets."""

    if Path(input_file).resolve() == Path(output_file).resolve():
        console.print(
            "[red]Error: Output file cannot be the same as the input file.[/red]"
        )
        raise typer.Exit(1)

    # Initialize PolicyRunner
    try:
        runner = PolicyRunner(model_name=model_name)
        logger.info(f"Initialized PolicyRunner with model: {model_name}")
    except Exception as e:
        console.print(
            f"[red]Error initializing PolicyRunner for model '{model_name}': {e}[/red]"
        )
        raise typer.Exit(1)

    # Read input file
    try:
        with open(input_file, "r") as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]
        logger.info(f"Loaded {len(lines)} lines from {input_file}")
    except Exception as e:
        console.print(f"[red]Error reading input file '{input_file}': {e}[/red]")
        raise typer.Exit(1)

    # Process in batches
    processed_lines: List[Dict[str, Any]] = []

    with console.status("[bold blue]Processing batches..."):
        for i in range(0, len(lines), batch_size):
            batch = lines[i : i + batch_size]

            try:
                # Extract contexts and responses
                contexts = [item.get("context", "") for item in batch]
                responses = [item.get("response", "") for item in batch]

                # Compute log probabilities
                logps: List[float] = []
                for context, response in zip(contexts, responses):
                    if context and response:
                        logp_result = runner.log_prob(context, response)
                        # Handle the case where log_prob returns a tuple (logp, token_logps)
                        if isinstance(logp_result, tuple):
                            logp = logp_result[0]
                        else:
                            logp = logp_result
                        logps.append(float(logp))
                    else:
                        logger.warning(
                            f"Skipping item with missing context or response"
                        )
                        logps.append(FALLBACK_LOG_PROB)

                # Add logp to each item
                for j, (item, logp) in enumerate(zip(batch, logps)):
                    item["logp"] = logp
                    processed_lines.append(item)

                logger.info(
                    f"Processed batch {i//batch_size + 1}/{(len(lines) + batch_size - 1)//batch_size}"
                )

            except Exception as e:
                console.print(
                    f"[red]Error processing batch starting at line {i}: {e}[/red]"
                )
                raise typer.Exit(1)

    # Write output file
    try:
        with open(output_file, "w") as f:
            for item in processed_lines:
                f.write(json.dumps(item) + "\n")

        console.print(
            f"[green]✅ Processed {len(lines)} lines. Output written to {output_file}[/green]"
        )
        logger.info(f"Successfully wrote {len(processed_lines)} lines to {output_file}")

    except Exception as e:
        console.print(f"[red]Error writing output file '{output_file}': {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
