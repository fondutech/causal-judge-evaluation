"""CLI command to check optional dependencies."""

import click
from rich.console import Console
from rich.table import Table

from ..utils.imports import ImportChecker

console = Console()


@click.command()
def check_deps() -> None:
    """Check the status of all optional dependencies."""
    console.print("\n[bold]CJE Dependency Check[/bold]\n")

    # Check all dependencies
    ImportChecker.print_status()

    # Also show provider-specific information
    console.print("\n[bold]Provider Availability:[/bold]")

    # Import provider registry to see what's available
    from ..provider_registry import get_registry

    registry = get_registry()

    if registry._provider_info:
        table = Table(title="Available Providers")
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Features", style="yellow")

        for name, info in registry._provider_info.items():
            features = []
            if info.supports_logprobs:
                features.append("Output logprobs")
            if info.supports_full_sequence_logprobs:
                features.append("Full sequence logprobs")

            table.add_row(
                name.title(),
                "✓ Available",
                ", ".join(features) if features else "Basic generation only",
            )

        console.print(table)
    else:
        console.print(
            "[red]No providers could be loaded. Check your installation.[/red]"
        )

    console.print(
        "\n[dim]Run 'pip install cje[all]' to install all optional dependencies.[/dim]"
    )


if __name__ == "__main__":
    check_deps()
