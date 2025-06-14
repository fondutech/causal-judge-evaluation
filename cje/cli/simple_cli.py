"""
Simplified CLI for CJE - fewer commands, clearer interface.
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from ..core import run_cje, CJEResult, CJEConfig
from ..config.simple import create_example_config

app = typer.Typer(help="CJE: Causal Judge Evaluation for LLMs")
console = Console()


@app.command()
def run(
    config: Path = typer.Argument(..., help="Path to config YAML file"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run CJE evaluation from config file."""
    try:
        # Override output dir if specified
        if output_dir:
            import yaml

            with open(config) as f:
                config_data = yaml.safe_load(f)
            config_data["work_dir"] = str(output_dir)

            # Use temporary modified config
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config_data, f)
                temp_config = f.name

            result = run_cje(temp_config)
            Path(temp_config).unlink()
        else:
            result = run_cje(str(config))

        # Display results
        console.print("\n[bold green]✓ Evaluation Complete![/bold green]\n")
        console.print(result.summary())

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def show(
    results: Path = typer.Argument(..., help="Path to results JSON file"),
    plot: bool = typer.Option(False, "--plot", "-p", help="Generate plots"),
) -> None:
    """Display results from a previous run."""
    try:
        result = CJEResult.from_json(results)
        console.print(result.summary())

        if plot:
            from ..results.visualization import plot_policy_comparison
            import matplotlib.pyplot as plt

            fig = plot_policy_comparison(result, result.policy_names)
            plot_path = results.parent / "policy_comparison.png"
            fig.savefig(plot_path)
            console.print(f"\n[blue]Plot saved to {plot_path}[/blue]")

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def compare(
    before: Path = typer.Argument(..., help="Baseline results"),
    after: Path = typer.Argument(..., help="New results"),
) -> None:
    """Compare two CJE results."""
    try:
        result1 = CJEResult.from_json(before)
        result2 = CJEResult.from_json(after)

        # Create comparison table
        table = Table(title="Policy Comparison")
        table.add_column("Policy", style="cyan")
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")
        table.add_column("Change", justify="right", style="bold")

        for i in range(len(result1.policy_names)):
            if i < len(result2.policy_names):
                before_val = result1.estimates[i]
                after_val = result2.estimates[i]
                change = after_val - before_val

                change_str = f"{change:+.4f}"
                if change > 0:
                    change_str = f"[green]{change_str}[/green]"
                elif change < 0:
                    change_str = f"[red]{change_str}[/red]"

                table.add_row(
                    result1.policy_names[i],
                    f"{before_val:.4f}",
                    f"{after_val:.4f}",
                    change_str,
                )

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def init(
    output: Path = typer.Argument("config.yaml", help="Output config file path"),
) -> None:
    """Create an example config file to get started."""
    try:
        config = create_example_config()

        # Convert to YAML
        import yaml

        with open(output, "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

        console.print(f"[green]✓ Created example config at {output}[/green]")
        console.print("\nEdit this file to configure your experiment, then run:")
        console.print(f"  [cyan]cje run {output}[/cyan]")

    except Exception as e:
        console.print(f"[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Config file to validate"),
) -> None:
    """Validate a config file without running the experiment."""
    try:
        # Try to load config
        cfg = CJEConfig.from_yaml(config)

        # Check for common issues
        issues = []

        if not cfg.target_policies:
            issues.append("No target policies defined")

        if cfg.dataset.sample_limit and cfg.dataset.sample_limit < 10:
            issues.append("Sample limit very low (<10), results may be unreliable")

        if cfg.estimator.k_folds == 1:
            issues.append("k=1 cross-validation may lead to overfitting")

        if cfg.estimator.clip and cfg.estimator.clip < 5:
            issues.append("Clip value very low (<5), may introduce significant bias")

        # Display validation results
        if issues:
            console.print("[yellow]⚠ Configuration warnings:[/yellow]")
            for issue in issues:
                console.print(f"  - {issue}")
        else:
            console.print("[green]✓ Configuration looks good![/green]")

        # Show summary
        console.print(f"\n[bold]Configuration Summary:[/bold]")
        console.print(
            f"  Dataset: {cfg.dataset.name} (n={cfg.dataset.sample_limit or 'all'})"
        )
        console.print(f"  Logging policy: {cfg.logging_policy.name}")
        console.print(f"  Target policies: {len(cfg.target_policies)}")
        console.print(f"  Estimator: {cfg.estimator.name} (k={cfg.estimator.k_folds})")

    except Exception as e:
        console.print(f"[bold red]✗ Invalid config:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
