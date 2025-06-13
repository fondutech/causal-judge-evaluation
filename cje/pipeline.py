from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, cast
import json
from hydra import initialize, compose, initialize_config_dir
from .cli.run_experiment import run as run_cli, get_last_work_dir


def run_pipeline(
    cfg_path: str = "cje.conf", cfg_name: str = "experiment"
) -> Dict[str, Any]:
    """Run the full CJE pipeline via Hydra config and return the result.

    This is a thin wrapper around the CLI implementation so that the
    experiment can be launched from Python code. It mirrors the
    ``cje run run`` command.

    Parameters
    ----------
    cfg_path: str, optional
        Path to the Hydra config directory. Defaults to ``"cje.conf"``.
    cfg_name: str, optional
        Name of the config file to load. Defaults to ``"experiment"``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with the estimator results.
    """
    # Execute the same logic as the CLI command
    run_cli(cfg_path=cfg_path, cfg_name=cfg_name)

    # Get the actual work directory from the CLI run (avoids timestamp mismatch)
    work_dir = get_last_work_dir()
    result_file = work_dir / "result.json"

    if not result_file.exists():
        raise FileNotFoundError(f"{result_file} does not exist")

    return cast(Dict[str, Any], json.loads(result_file.read_text()))
