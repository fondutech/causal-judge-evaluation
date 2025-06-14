#!/usr/bin/env python3
"""Arena CJE Experiment - Simplified Implementation.

Simple wrapper around the existing CJE pipeline for arena analysis.
Uses the standard CJE configuration and pipeline.
"""

from cje.pipeline import run_pipeline
from rich import print as rprint
from pathlib import Path
from typing import Dict, Any
import sys


def run_arena_experiment(config_name: str = "arena_test") -> Dict[str, Any]:
    """Run arena CJE experiment using the standard CJE pipeline.

    Args:
        config_name: Configuration file to use (default: arena_test)
    """
    rprint(f"🏟️ [bold blue]Running Arena CJE Experiment[/bold blue]")
    rprint(f"📋 Config: {config_name}")

    try:
        # Get configs directory
        configs_dir = Path("configs")
        if not configs_dir.exists():
            configs_dir = Path.cwd() / "configs"

        # Run the standard CJE pipeline
        results = run_pipeline(cfg_path=str(configs_dir), cfg_name=config_name)

        rprint("✅ [bold green]Arena experiment complete![/bold green]")
        return results

    except Exception as e:
        rprint(f"❌ [bold red]Arena experiment failed: {e}[/bold red]")
        raise


def main() -> Dict[str, Any]:
    """Main CLI entry point."""
    config_name = "arena_test"

    if len(sys.argv) > 1:
        config_name = sys.argv[1]

    return run_arena_experiment(config_name)


if __name__ == "__main__":
    main()
