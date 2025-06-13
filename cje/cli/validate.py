"""CJE data validation CLI commands."""

import pathlib
import typer
from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from ..data.validation import validate_dataset, ValidationResult

app = typer.Typer(help="Validate CJE data files for quality and compliance")
console = Console()


@app.command("data")
def validate_data_cmd(
    file_path: pathlib.Path = typer.Argument(
        ..., help="Path to JSONL data file to validate"
    ),
    scenario: Optional[str] = typer.Option(
        None,
        help="Expected scenario type (1, 2, or 3). Auto-detected if not specified.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed validation information"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Only show errors, no warnings or info"
    ),
) -> None:
    """
    Validate a CJE data file for quality and compliance.

    This command checks your data file for:
    - Required fields and proper formatting
    - Data quality issues (log probabilities, ground truth labels)
    - Scenario compliance and recommendations
    - Statistical quality of labels and propensity scores

    Examples:
        cje validate data my_data.jsonl
        cje validate data my_data.jsonl --scenario 2 --verbose
        cje validate data my_data.jsonl --quiet  # Only show errors
    """

    if not file_path.exists():
        console.print(f"❌ File not found: {file_path}", style="red")
        raise typer.Exit(1)

    # Run validation
    with console.status("[bold blue]Validating data file..."):
        result = validate_dataset(file_path, scenario)

    # Display results based on verbosity
    if quiet:
        # Only show errors
        if result.errors:
            console.print(
                f"\n❌ {len(result.errors)} Error(s) Found:", style="red bold"
            )
            for error in result.errors:
                console.print(f"   • {error}", style="red")
            raise typer.Exit(1)
        else:
            console.print("✅ Data validation passed", style="green")

    elif verbose:
        # Show everything including detailed summary
        result.print_report()

        # Additional verbose information
        if result.summary:
            console.print("\n📋 Detailed Summary:", style="blue bold")
            summary_text = ""
            for key, value in result.summary.items():
                summary_text += f"{key}: {value}\n"

            console.print(
                Panel(summary_text.strip(), title="Data Analysis", border_style="blue")
            )

    else:
        # Standard output
        result.print_report()

    # Exit with appropriate code
    if result.errors:
        raise typer.Exit(1)
    elif result.warnings and not quiet:
        console.print("\n⚠️  Validation completed with warnings", style="yellow")
        raise typer.Exit(0)
    else:
        if not quiet:
            console.print("\n✅ Validation completed successfully", style="green")
        raise typer.Exit(0)


@app.command("quick")
def quick_check(
    file_path: pathlib.Path = typer.Argument(
        ..., help="Path to JSONL data file to check"
    )
) -> None:
    """
    Quick health check of a data file.

    Provides a fast overview of your data file structure and basic stats
    without detailed validation.
    """

    if not file_path.exists():
        console.print(f"❌ File not found: {file_path}", style="red")
        raise typer.Exit(1)

    from ..data.validation import analyze_data_summary

    with console.status("[bold blue]Analyzing data file..."):
        summary = analyze_data_summary(file_path)

    console.print(f"\n📊 Quick Analysis: {file_path.name}")
    console.print("=" * 50)

    # Basic stats
    console.print(f"📈 Total samples: {summary['total_samples']}")
    console.print(f"🎯 Scenario: {summary['scenario_type']}")

    # Data availability
    if summary["has_responses"] > 0:
        console.print(
            f"💬 Responses: {summary['has_responses']}/{summary['total_samples']}"
        )

    if summary["has_ground_truth"] > 0:
        percentage = (summary["has_ground_truth"] / summary["total_samples"]) * 100
        console.print(
            f"🏷️  Ground truth: {summary['has_ground_truth']}/{summary['total_samples']} ({percentage:.1f}%)"
        )

    if summary["has_log_probabilities"] > 0:
        console.print(
            f"📊 Log probabilities: {summary['has_log_probabilities']}/{summary['total_samples']}"
        )

    if summary["has_target_samples"] > 0:
        console.print(
            f"🎯 Target samples: {summary['has_target_samples']}/{summary['total_samples']}"
        )

    console.print("\n💡 Run 'cje validate data' for detailed validation")


@app.command("help-scenarios")
def help_scenarios() -> None:
    """
    Show help about CJE data scenarios and requirements.
    """

    scenarios_help = """
CJE Data Scenarios:

1️⃣  Scenario 1: Context Only
   • Fields: context
   • Use case: You have prompts, CJE generates responses
   • Example: {"context": "Write a summary..."}

2️⃣  Scenario 2: Complete Logs with Ground Truth  
   • Fields: context, response, logp, y_true
   • Use case: You have existing model logs + human labels
   • Example: {"context": "...", "response": "...", "logp": -2.5, "y_true": 0.8}

3️⃣  Scenario 3: Pre-computed Policy Data
   • Fields: context, target_samples
   • Use case: You pre-computed target policy responses
   • Example: {"context": "...", "target_samples": [{"response": "...", "logp": -1.2}]}

Required Fields:
✅ context: Always required (the input prompt)
✅ response: Required if you have y_true
✅ logp: Log probability (should be negative)
✅ y_true: Numeric ground truth labels

Quality Tips:
• Aim for 100+ ground truth examples for best calibration
• Ensure logp values are negative (log probabilities)
• Balance your ground truth labels when possible
• Check for duplicate responses with different logp values
"""

    console.print(
        Panel(
            scenarios_help.strip(),
            title="🔍 CJE Data Scenarios Guide",
            border_style="blue",
        )
    )


@app.command("config")
def validate_config_cmd(
    cfg_path: str = typer.Option(
        "cje.conf", "--cfg-path", help="Path to config directory"
    ),
    cfg_name: str = typer.Option(
        "experiment", "--cfg-name", help="Name of config file to validate"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed configuration information"
    ),
) -> None:
    """
    Validate a CJE configuration file.

    This command checks your configuration file for:
    - Required fields and proper structure
    - Valid parameter values and types
    - Consistency between configuration sections
    - Compatibility with available estimators and providers

    Examples:
        cje validate config --cfg-path configs --cfg-name experiment
        cje validate config --cfg-path . --cfg-name my_experiment --verbose
    """
    from pathlib import Path
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from ..config.unified import from_dict, validate_configuration
    from ..utils.error_handling import ConfigurationError

    # Convert to absolute path
    cfg_path_abs = Path(cfg_path).resolve()

    if not cfg_path_abs.exists():
        console.print(f"❌ Config directory not found: {cfg_path_abs}", style="red")
        raise typer.Exit(1)

    config_file = cfg_path_abs / f"{cfg_name}.yaml"
    if not config_file.exists():
        console.print(f"❌ Config file not found: {config_file}", style="red")
        raise typer.Exit(1)

    try:
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()

        # Initialize Hydra with the config directory
        with initialize_config_dir(config_dir=str(cfg_path_abs), version_base=None):
            # Compose the configuration
            cfg = compose(config_name=cfg_name)
            config_dict = OmegaConf.to_container(cfg, resolve=True)

        # Ensure config_dict is a dictionary
        if not isinstance(config_dict, dict):
            console.print(
                f"❌ Configuration must be a dictionary, got {type(config_dict)}",
                style="red",
            )
            raise typer.Exit(1)

        # Cast to proper type for mypy
        config_dict_typed: Dict[str, Any] = {str(k): v for k, v in config_dict.items()}

        # Validate using unified configuration system
        with console.status("[bold blue]Validating configuration..."):
            validation_errors = validate_configuration(config_dict_typed)

        if validation_errors:
            console.print(
                f"\n❌ {len(validation_errors)} Configuration Error(s) Found:",
                style="red bold",
            )
            for error in validation_errors:
                console.print(f"   • {error}", style="red")
            raise typer.Exit(1)
        else:
            console.print("✅ Configuration validation passed", style="green")

        if verbose:
            # Show detailed configuration structure
            console.print("\n📋 Configuration Structure:", style="blue bold")

            # Parse into CJE config for structured display
            try:
                cje_config = from_dict(config_dict_typed)

                console.print(f"   • Dataset: {cje_config.dataset.name}")
                console.print(f"   • Estimator: {cje_config.estimator.name}")
                console.print(
                    f"   • Target Policies: {len(cje_config.target_policies)}"
                )
                console.print(f"   • Judge Provider: {cje_config.judge.provider}")

                if hasattr(cje_config, "logging_policy") and cje_config.logging_policy:
                    console.print(
                        f"   • Logging Policy: {cje_config.logging_policy.model_name}"
                    )

                console.print(f"   • Work Directory: {cje_config.paths.work_dir}")

            except Exception as e:
                console.print(f"   • Raw config keys: {list(config_dict_typed.keys())}")

    except Exception as e:
        console.print(f"❌ Configuration validation failed: {e}", style="red")
        raise typer.Exit(1)
    finally:
        # Clean up Hydra
        GlobalHydra.instance().clear()


if __name__ == "__main__":
    app()
