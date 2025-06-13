#!/usr/bin/env python3
"""Arena Analysis using Python Interface.

This script uses the existing CJE pipeline interface to run arena analysis
programmatically. Much simpler than CLI and gets all the same benefits.
"""

from cje.pipeline import run_pipeline
from rich import print as rprint
from rich.panel import Panel
from pathlib import Path
from typing import Dict, Any, Optional
import sys


def run_arena_analysis(
    config_name: str = "arena_analysis",
    sample_limit: int = 1000,
    estimator: str = "DRCPO",
) -> Dict[str, Any]:
    """Run arena analysis using the Python pipeline interface.

    Args:
        config_name: Name of the config file to use
        sample_limit: Number of samples to process
        estimator: Estimator to use (DRCPO, IPS, SNIPS, MRDR)
    """

    rprint(
        Panel.fit(
            f"🏟️ Arena Analysis (Python Interface)\n\n"
            f"• Config: {config_name}\n"
            f"• Samples: {sample_limit}\n"
            f"• Estimator: {estimator}\n\n"
            f"Using the enhanced APIPolicyRunner with:\n"
            f"• Safe error handling\n"
            f"• Automatic caching\n"
            f"• Progress tracking",
            title="Arena Analysis",
        )
    )

    try:
        # Get absolute path to configs directory
        # Assuming this script is in examples/ and configs/ is at root
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent  # Go up from examples/ to project root
        configs_dir = project_root / "configs"

        if not configs_dir.exists():
            # Try current working directory
            configs_dir = Path.cwd() / "configs"

        if not configs_dir.exists():
            raise FileNotFoundError(
                f"Configs directory not found. Looked in: {project_root / 'configs'} and {Path.cwd() / 'configs'}"
            )

        rprint(f"[blue]Using configs from: {configs_dir}[/blue]")

        # Run the pipeline with correct absolute config path
        # This uses the enhanced APIPolicyRunner automatically!
        results = run_pipeline(cfg_path=str(configs_dir), cfg_name=config_name)

        rprint("\n[bold green]✅ Arena analysis complete![/bold green]")

        # Print results summary
        analysis_type = results.get("analysis_type", "unknown")
        interpretation = results.get("interpretation", "No interpretation available")
        warning = results.get("warning")

        rprint(f"\n[bold blue]Analysis Type:[/bold blue] {analysis_type}")
        rprint(f"[bold blue]Interpretation:[/bold blue] {interpretation}")

        if warning:
            rprint(f"[yellow]⚠️  Warning:[/yellow] {warning}")

        # Show policy estimates
        if "v_hat" in results:
            v_hat = results["v_hat"]
            se = results.get("se", [])

            rprint(f"\n[bold green]Policy Estimates:[/bold green]")
            if isinstance(v_hat, list):
                for i, estimate in enumerate(v_hat):
                    stderr = se[i] if i < len(se) else 0.0
                    rprint(f"  Policy {i}: {estimate:.3f} ± {stderr:.3f}")
            else:
                stderr = se if isinstance(se, (int, float)) else 0.0
                rprint(f"  Estimate: {v_hat:.3f} ± {stderr:.3f}")

        # Add weight diagnostics
        try:
            from cje.utils.weight_diagnostics import (
                analyze_arena_weights,
                create_weight_summary_table,
            )

            # Load arena data for weight analysis
            metadata = results.get("metadata", {})
            work_dir = metadata.get("work_dir")

            if work_dir:
                import json
                from pathlib import Path

                # Look for arena data file
                arena_files = list(
                    Path(work_dir).glob(
                        "**/quick_judge_scores_cal_with_target_logp.jsonl"
                    )
                )
                if not arena_files:
                    arena_files = list(Path(work_dir).glob("**/*target_logp*.jsonl"))

                if arena_files:
                    arena_file = arena_files[0]
                    rprint(f"\n[bold cyan]Importance Weight Diagnostics:[/bold cyan]")

                    # Load and analyze
                    with open(arena_file, "r") as f:
                        arena_data = [json.loads(line) for line in f.readlines()]

                    diagnostics = analyze_arena_weights(arena_data)
                    if diagnostics:
                        # Show summary table
                        summary_table = create_weight_summary_table(diagnostics)
                        rprint(summary_table)

                        # Check for critical issues
                        critical_policies = [
                            name
                            for name, diag in diagnostics.items()
                            if diag.consistency_flag == "CRITICAL"
                        ]
                        warning_policies = [
                            name
                            for name, diag in diagnostics.items()
                            if diag.consistency_flag == "WARNING"
                        ]

                        if critical_policies:
                            rprint(
                                f"\n[red]❌ CRITICAL weight issues found in: {', '.join(critical_policies)}[/red]"
                            )
                            rprint(
                                f"[red]   This may indicate teacher forcing problems or policy inconsistencies[/red]"
                            )
                        elif warning_policies:
                            rprint(
                                f"\n[yellow]⚠️  Weight warnings for: {', '.join(warning_policies)}[/yellow]"
                            )
                            rprint(
                                f"[yellow]   Consider investigating low ESS or extreme weights[/yellow]"
                            )
                        else:
                            rprint(
                                f"\n[green]✅ All importance weights look healthy[/green]"
                            )

                        # Offer to create diagnostic plots
                        rprint(
                            f"\n[dim]💡 Tip: Create weight diagnostic plots with:[/dim]"
                        )
                        rprint(
                            f"[dim]   from cje.utils.weight_plots import create_weight_diagnostic_dashboard[/dim]"
                        )
                        rprint(
                            f"[dim]   create_weight_diagnostic_dashboard(arena_data)[/dim]"
                        )

        except Exception as e:
            rprint(f"\n[yellow]⚠️  Could not compute weight diagnostics: {e}[/yellow]")

        # Show where results are saved
        work_dir = results.get("metadata", {}).get("work_dir")
        if work_dir:
            rprint(f"\n[blue]Results saved to: {work_dir}[/blue]")

        return results

    except Exception as e:
        rprint(f"[red]❌ Arena analysis failed: {e}[/red]")
        raise


def run_quick_test() -> Optional[Dict[str, Any]]:
    """Run a quick test with arena_quick config."""

    rprint(
        Panel.fit(
            "🚀 Quick Arena Test\n\n"
            "This will run a fast test to verify everything works.\n"
            "Uses the arena_quick config with minimal samples.",
            title="Quick Test",
        )
    )

    try:
        results = run_arena_analysis(
            config_name="arena_quick", sample_limit=25, estimator="DRCPO"
        )

        rprint("\n[bold green]🎉 Quick test completed successfully![/bold green]")
        return results

    except Exception as e:
        rprint(f"[red]❌ Quick test failed: {e}[/red]")
        return None


def main() -> Optional[Dict[str, Any]]:
    """Main function with simple CLI."""

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        return run_quick_test()

    # Default arena analysis
    return run_arena_analysis()


if __name__ == "__main__":
    main()
