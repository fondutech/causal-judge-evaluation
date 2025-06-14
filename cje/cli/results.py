"""
Enhanced results display for CJE CLI.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import json

from ..results import EstimationResult
from ..results.visualization import create_summary_report

app = typer.Typer()
console = Console()


@app.command()
def show(
    results_path: Path = typer.Argument(..., help="Path to results JSON file"),
    visualize: bool = typer.Option(False, "--viz", help="Generate visualization plots"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Directory for plots"
    ),
):
    """Display CJE results in a formatted table with optional visualizations."""

    # Load results
    with open(results_path) as f:
        data = json.load(f)

    # Create results object
    results = EstimationResult.from_dict(data)

    # Display summary table
    table = Table(title="CJE Estimation Results", show_header=True)
    table.add_column("Policy", style="cyan")
    table.add_column("Estimate", justify="right", style="green")
    table.add_column("Std Error", justify="right")
    table.add_column("95% CI Lower", justify="right")
    table.add_column("95% CI Upper", justify="right")
    table.add_column("ESS %", justify="right")

    # Get policy names and ESS if available
    policy_names = data.get(
        "policy_names", [f"Policy {i+1}" for i in range(results.n_policies)]
    )
    ess_values = None
    if "metadata" in data and "ess_percentage" in data["metadata"]:
        ess_values = data["metadata"]["ess_percentage"]

    # Add rows
    for i in range(results.n_policies):
        ci_lower, ci_upper = results.confidence_interval(i)
        ess_str = f"{ess_values[i]:.1f}" if ess_values else "N/A"

        # Color ESS based on threshold
        if ess_values and ess_values[i] < 10:
            ess_str = f"[red]{ess_str}[/red]"
        elif ess_values and ess_values[i] < 30:
            ess_str = f"[yellow]{ess_str}[/yellow]"
        else:
            ess_str = f"[green]{ess_str}[/green]"

        table.add_row(
            policy_names[i],
            f"{results.v_hat[i]:.4f}",
            f"{results.se[i]:.4f}",
            f"{ci_lower:.4f}",
            f"{ci_upper:.4f}",
            ess_str,
        )

    console.print(table)

    # Show metadata summary
    if "metadata" in data:
        meta = data["metadata"]
        summary_lines = [
            f"Estimator: {meta.get('estimator_type', 'Unknown')}",
            f"Sample size: {results.n:,}",
            f"Cross-validation folds: {meta.get('k_folds', 'N/A')}",
            f"Weight clipping: {meta.get('clip_threshold', 'None')}",
        ]

        panel = Panel(
            "\n".join(summary_lines),
            title="Experiment Configuration",
            border_style="blue",
        )
        console.print(panel)

    # Generate visualizations if requested
    if visualize:
        console.print("\n[bold]Generating visualizations...[/bold]")
        output_dir = output_dir or Path("./cje_results_plots")
        output_dir.mkdir(exist_ok=True)

        figures = create_summary_report(results, policy_names, output_dir)

        console.print(f"[green]✓[/green] Saved {len(figures)} plots to {output_dir}")
        for name in figures:
            console.print(f"  - {name}.png")


@app.command()
def compare(
    baseline: Path = typer.Argument(..., help="Baseline results JSON"),
    treatment: Path = typer.Argument(..., help="Treatment results JSON"),
):
    """Compare two CJE results to see relative improvement."""

    # Load both results
    with open(baseline) as f:
        baseline_data = json.load(f)
    with open(treatment) as f:
        treatment_data = json.load(f)

    baseline_result = EstimationResult.from_dict(baseline_data)
    treatment_result = EstimationResult.from_dict(treatment_data)

    # Compute relative improvement
    table = Table(title="Treatment vs Baseline Comparison", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Treatment", justify="right")
    table.add_column("Difference", justify="right", style="green")
    table.add_column("Relative %", justify="right", style="green")

    for i in range(min(baseline_result.n_policies, treatment_result.n_policies)):
        base_val = baseline_result.v_hat[i]
        treat_val = treatment_result.v_hat[i]
        diff = treat_val - base_val
        rel_pct = 100 * diff / abs(base_val) if base_val != 0 else 0

        # Check if significant
        base_ci = baseline_result.confidence_interval(i)
        treat_ci = treatment_result.confidence_interval(i)

        is_significant = (treat_ci[0] > base_ci[1]) or (treat_ci[1] < base_ci[0])
        sig_marker = " *" if is_significant else ""

        table.add_row(
            f"Policy {i+1}",
            f"{base_val:.4f}",
            f"{treat_val:.4f}",
            f"{diff:+.4f}{sig_marker}",
            f"{rel_pct:+.1f}%",
        )

    console.print(table)
    console.print(
        "\n[dim]* indicates statistically significant difference at 95% level[/dim]"
    )


if __name__ == "__main__":
    app()
