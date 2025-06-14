#!/usr/bin/env python3
"""
Arena Research Experiment Runner

Simple interface to run the complete Arena CJE research experiment
using CJE infrastructure with research extensions.

Usage:
    # Run full research experiment
    python scripts/run_arena_research.py

    # Run with smaller sample size for testing
    python scripts/run_arena_research.py --samples 100

    # Run specific config
    python scripts/run_arena_research.py --config arena_research_experiment
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cje.research import ArenaResearchExperiment
from rich.console import Console

console = Console()


def main() -> None:
    """Main entry point for arena research experiment."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Arena CJE Research Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="arena_research_experiment",
        help="Configuration name to use",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs",
        help="Path to configuration directory",
    )
    parser.add_argument(
        "--samples", type=int, default=None, help="Override sample count for testing"
    )

    args = parser.parse_args()

    # Display configuration
    console.print("\n🔬 Arena CJE Research Experiment")
    console.print("─" * 50)
    console.print(f"Config: {args.config}")
    console.print(f"Config path: {args.config_path}")
    if args.samples:
        console.print(f"Sample override: {args.samples}")
    console.print("─" * 50)

    # TODO: Handle sample override by modifying config
    if args.samples:
        console.print(f"[yellow]Note: Sample override not yet implemented[/yellow]")
        console.print(
            f"[yellow]Edit {args.config}.yaml to change sample_limit[/yellow]"
        )

    try:
        # Create and run research experiment
        experiment = ArenaResearchExperiment(
            config_path=args.config_path, config_name=args.config
        )

        results = experiment.run_full_experiment()

        # Display final summary
        console.print("\n🎯 Research Experiment Summary")
        console.print("─" * 50)
        console.print(f"Total runtime: {results.total_runtime}")
        console.print(
            f"Health score: {results.diagnostics.get('health_score', 'N/A') if results.diagnostics else 'N/A'}"
        )
        console.print(
            f"All checks passed: {results.diagnostics.get('all_checks_passed', False) if results.diagnostics else False}"
        )
        console.print(f"Work directory: {results.base_results.get('work_dir', 'N/A')}")

        # Show phase timing breakdown
        if results.phase_times:
            console.print("\n⏱️ Phase Timing:")
            for phase, duration in results.phase_times.items():
                console.print(f"  {phase}: {duration}")

        console.print("\n✅ Research experiment completed successfully!")

    except Exception as e:
        console.print(f"\n[red]❌ Research experiment failed: {e}[/red]")
        import traceback

        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
