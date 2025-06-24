"""Run CJE experiments via CLI using the modular pipeline."""

from pathlib import Path
import typer
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from rich.console import Console

from ..pipeline import CJEPipeline
from ..pipeline.config import PipelineConfig

console = Console()


def run(
    cfg_path: str = typer.Argument(..., help="Path to configuration directory"),
    cfg_name: str = typer.Argument(
        ..., help="Name of configuration file (without .yaml extension)"
    ),
    overrides: list[str] = typer.Option(
        None,
        "--override",
        "-o",
        help="Hydra-style overrides (e.g., -o dataset.sample_limit=10)",
    ),
) -> None:
    """Run CJE experiment using configuration file."""

    # Initialize Hydra with the config directory
    cfg_dir = Path(cfg_path).resolve()

    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        # Compose configuration with overrides
        cfg = compose(config_name=cfg_name, overrides=overrides or [])

        # Convert Hydra config to pipeline config
        pipeline_config = PipelineConfig.from_hydra_config(cfg)

        # Show configuration
        console.print("[bold blue]Configuration:[/bold blue]")
        console.print(OmegaConf.to_yaml(cfg))

        # Create and run pipeline
        pipeline = CJEPipeline(pipeline_config)
        results = pipeline.run()

        # Show summary
        console.print(
            "\n[bold green]✅ Experiment completed successfully![/bold green]"
        )
        console.print(f"Results saved to: {pipeline_config.work_dir / 'results.json'}")
