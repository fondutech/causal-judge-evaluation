#!/usr/bin/env python3
"""Check which Fireworks models are available with your API key."""

import os
import requests  # type: ignore
from rich.console import Console
from rich.table import Table

console = Console()


def check_fireworks_access() -> None:
    """Check Fireworks API access and list available models."""

    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        console.print("[red]❌ FIREWORKS_API_KEY not set![/red]")
        return

    console.print(f"[green]✅ API key found:[/green] {api_key[:8]}...")

    # Test API access
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        # Get list of models
        response = requests.get(
            "https://api.fireworks.ai/inference/v1/models", headers=headers
        )

        if response.status_code == 200:
            models = response.json().get("data", [])
            console.print(f"\n[green]✅ API access confirmed![/green]")
            console.print(f"Found {len(models)} available models\n")

            # Create table of relevant models
            table = Table(title="Available Llama Models")
            table.add_column("Model ID", style="cyan")
            table.add_column("Type", style="green")

            # Filter for llama models
            llama_models = [m for m in models if "llama" in m["id"].lower()]

            for model in sorted(llama_models, key=lambda x: x["id"]):
                model_type = (
                    "chat" if "chat" in model.get("object", "") else "completion"
                )
                table.add_row(model["id"], model_type)

            console.print(table)

            # Check for specific models
            console.print("\n[bold]Checking experiment models:[/bold]")
            experiment_models = [
                "accounts/fireworks/models/llama4-scout-instruct-basic",
                "accounts/fireworks/models/llama4-maverick-instruct-basic",
                "accounts/fireworks/models/llama-v3-34b-instruct",
                "accounts/fireworks/models/llama-v3p1-70b-instruct",
            ]

            model_ids = [m["id"] for m in models]
            for exp_model in experiment_models:
                if exp_model in model_ids:
                    console.print(f"✅ {exp_model}")
                else:
                    console.print(f"❌ {exp_model} [red](not available)[/red]")

        else:
            console.print(f"[red]❌ API request failed:[/red] {response.status_code}")
            console.print(f"Response: {response.text}")

    except Exception as e:
        console.print(f"[red]❌ Error checking API:[/red] {e}")


if __name__ == "__main__":
    check_fireworks_access()
