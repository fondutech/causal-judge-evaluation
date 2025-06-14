"""Interactive Arena CJE Analysis.

Simple Python interface for running arena CJE experiments.
"""

from typing import Dict, Any
from pathlib import Path
from cje.pipeline import run_pipeline
from rich import print as rprint


class ArenaAnalyzer:
    """Simple arena CJE experiment runner."""

    def __init__(self, work_dir: str = "outputs/arena_test"):
        """Initialize analyzer.

        Args:
            work_dir: Working directory for results
        """
        self.work_dir = Path(work_dir)
        self.results: Dict[str, Any] = {}

    def run_experiment(self, config_name: str = "arena_test") -> Dict[str, Any]:
        """Run arena experiment.

        Args:
            config_name: Configuration file to use

        Returns:
            Experiment results
        """
        rprint(f"🏟️ Running arena experiment with config: {config_name}")

        # Get configs directory
        configs_dir = Path("configs")
        if not configs_dir.exists():
            configs_dir = Path.cwd() / "configs"

        # Run pipeline
        self.results = run_pipeline(cfg_path=str(configs_dir), cfg_name=config_name)

        rprint("✅ Experiment complete!")
        return self.results

    def get_results(self) -> Dict[str, Any]:
        """Get the last experiment results."""
        return self.results or {}


def run_arena_analysis(
    config_name: str = "arena_test", **kwargs: Any
) -> Dict[str, Any]:
    """Run arena analysis with specified config."""
    analyzer = ArenaAnalyzer()
    return analyzer.run_experiment(config_name=config_name)


if __name__ == "__main__":
    analyzer = ArenaAnalyzer()
    results = analyzer.run_experiment()
    print("Results:", results)
