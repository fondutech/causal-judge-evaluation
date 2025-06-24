"""
Modular CJE experiment runner using the pipeline architecture.

This is a refactored version of run_experiment.py that uses the new modular pipeline.
"""

from __future__ import annotations
import pathlib
import time
import logging
import typer
from typing import Dict, Any
from hydra import initialize, compose, initialize_config_dir
from omegaconf import OmegaConf
from rich import print
from rich.console import Console
from pathlib import Path
import tempfile

from ..pipeline import CJEPipeline, PipelineConfig
from ..config.unified import from_dict, to_dict
from ..utils.error_handling import ConfigurationError

# Temporary file to store the work directory path
_WORK_DIR_FILE = Path(tempfile.gettempdir()) / "cje_last_work_dir.txt"


def setup_logging(work_dir: Path) -> logging.Logger:
    """Set up logging configuration."""
    log_dir = work_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "experiment.log"),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


def run(
    cfg_path: str = typer.Option("configs", "--cfg-path"),
    cfg_name: str = typer.Option("experiment", "--cfg-name"),
) -> None:
    """
    Execute CJE experiment using the modular pipeline.

    This function loads configuration and runs the full CJE pipeline:
    dataset → logging policy → judge → calibration → target policy → estimators
    """
    start_time = time.time()
    console = Console()

    # Load configuration
    config_path_abs = pathlib.Path(cfg_path).resolve()

    if config_path_abs.is_absolute() and "/" in cfg_path:
        with initialize_config_dir(
            version_base=None, config_dir=str(config_path_abs), job_name="cje_run"
        ):
            cfg = compose(config_name=cfg_name)
    else:
        with initialize(version_base=None, config_path=cfg_path):
            cfg = compose(config_name=cfg_name)

    print("[bold green]Configuration[/bold green]")
    print(OmegaConf.to_yaml(cfg))

    # Convert to unified configuration system
    try:
        cfg_container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_container, dict):
            raise ConfigurationError("Configuration must be a dictionary")

        # Ensure all keys are strings for proper typing
        cfg_dict: Dict[str, Any] = {str(k): v for k, v in cfg_container.items()}
        unified_config = from_dict(cfg_dict)
        print("[bold blue]✅ Configuration Valid[/bold blue]")

        # Convert back to OmegaConf for compatibility
        cfg = OmegaConf.create(to_dict(unified_config))

    except ConfigurationError as e:
        print(f"[bold red]❌ Configuration Error:[/bold red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[bold red]❌ Unexpected Error:[/bold red] {e}")
        raise typer.Exit(1)

    # Prepare work directory
    work_dir = pathlib.Path(cfg.paths.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(work_dir)
    logger.info(f"Starting CJE experiment: {cfg_name}")
    logger.info(f"Work directory: {work_dir}")
    logger.info(f"Config path: {cfg_path}")

    # Create pipeline configuration
    pipeline_config = PipelineConfig.from_hydra_config(cfg)

    # Create and run pipeline
    pipeline = CJEPipeline(pipeline_config, console=console)

    try:
        # Run the pipeline
        result = pipeline.run()

        # Update temporary file with work directory
        with open(_WORK_DIR_FILE, "w") as f:
            f.write(str(work_dir))

        # Print final summary
        total_time = time.time() - start_time
        console.print(
            f"\n[bold green]✅ Experiment completed successfully in "
            f"{total_time:.1f} seconds[/bold green]"
        )

        logger.info(f"Experiment completed in {total_time:.1f} seconds")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        console.print(f"\n[bold red]❌ Experiment failed: {e}[/bold red]")
        raise typer.Exit(1)


def get_last_work_dir() -> Path:
    """Get the work directory from the last run of the experiment."""
    if not _WORK_DIR_FILE.exists():
        raise RuntimeError("No experiment has been run yet")
    with open(_WORK_DIR_FILE, "r") as f:
        return Path(f.read())


# This module can be imported and the run function used directly
