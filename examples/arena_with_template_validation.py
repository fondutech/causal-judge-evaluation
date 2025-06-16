#!/usr/bin/env python3
"""
Example: Arena analysis with completions template validation.

This example demonstrates how to use the new completions template system
with proper validation for arena-style experiments.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from cje.loggers.api_policy import APIPolicyRunner
from cje.pipeline import run_pipeline
from rich import print as rprint
from rich.console import Console
from rich.table import Table

console = Console()


def validate_arena_models() -> None:
    """Validate that arena models work with their templates."""
    console.print("[bold blue]🔍 Validating Arena Model Templates[/bold blue]\n")

    # Define test configurations
    test_configs = [
        {
            "name": "Llama 4 Scout",
            "provider": "fireworks",
            "model": "accounts/fireworks/models/llama4-scout-instruct-basic",
            "template": "llama4",
        },
        {
            "name": "Llama 4 Maverick",
            "provider": "fireworks",
            "model": "accounts/fireworks/models/llama4-maverick-instruct-basic",
            "template": "llama4",
        },
        {
            "name": "Llama 3.1 8B",
            "provider": "fireworks",
            "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "template": "llama3",
        },
    ]

    # Create results table
    table = Table(title="Template Validation Results")
    table.add_column("Model", style="cyan")
    table.add_column("Template", style="magenta")
    table.add_column("Validation", style="green")
    table.add_column("Log Prob", style="yellow")

    for config in test_configs:
        try:
            # Initialize runner with explicit template
            runner = APIPolicyRunner(
                provider=config["provider"],
                model_name=config["model"],
                completions_template_format=config["template"],
                temperature=0.5,
            )

            # Validate teacher forcing
            validation_result = runner.validate_teacher_forcing()

            # If validation passes, it returns the log probability
            if validation_result is not None and validation_result > -10:
                status = "✅ Passed"
                logprob = f"{validation_result:.3f}"
            else:
                status = "⚠️  Warning"
                logprob = f"{validation_result:.3f}" if validation_result else "N/A"

        except Exception as e:
            status = f"❌ Failed: {str(e)[:30]}..."
            logprob = "N/A"

        table.add_row(config["name"], config["template"], status, logprob)

    console.print(table)
    console.print()


def run_arena_with_validation(config_name: str = "arena_test") -> Dict[str, Any]:
    """Run arena experiment with template validation."""
    console.print(
        f"[bold green]🏟️  Running Arena Experiment: {config_name}[/bold green]\n"
    )

    # First validate the models
    validate_arena_models()

    # Run the experiment
    console.print("[bold blue]🚀 Starting CJE Pipeline[/bold blue]")

    configs_dir = Path(__file__).parent.parent / "configs"
    results = run_pipeline(cfg_path=str(configs_dir), cfg_name=config_name)

    # Display summary
    console.print("\n[bold green]✅ Experiment Complete![/bold green]")

    if "estimate_result" in results:
        est = results["estimate_result"]
        console.print("\n[bold]Policy Rankings:[/bold]")

        # Create rankings table
        rankings_table = Table()
        rankings_table.add_column("Rank", style="cyan")
        rankings_table.add_column("Policy", style="magenta")
        rankings_table.add_column("Value", style="green")
        rankings_table.add_column("95% CI", style="yellow")

        # Sort policies by value
        policy_values = list(
            zip(est.policy_names, est.v_hat, est.v_ci_lower, est.v_ci_upper)
        )
        policy_values.sort(key=lambda x: x[1], reverse=True)

        for rank, (policy, value, ci_lower, ci_upper) in enumerate(policy_values, 1):
            rankings_table.add_row(
                str(rank), policy, f"{value:.3f}", f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            )

        console.print(rankings_table)

    return results


if __name__ == "__main__":
    # Example 1: Validate models only
    print("\n" + "=" * 60)
    print("Example 1: Validating Arena Models")
    print("=" * 60 + "\n")
    validate_arena_models()

    # Example 2: Run full experiment
    print("\n" + "=" * 60)
    print("Example 2: Running Full Arena Experiment")
    print("=" * 60 + "\n")

    try:
        results = run_arena_with_validation("arena_test")

        # Access specific results
        if "estimate_result" in results:
            est = results["estimate_result"]
            print(f"\nBest policy: {est.policy_names[est.v_hat.argmax()]}")
            print(f"Worst policy: {est.policy_names[est.v_hat.argmin()]}")

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        console.print(
            "\n[yellow]💡 Tip: Make sure you have set FIREWORKS_API_KEY[/yellow]"
        )
