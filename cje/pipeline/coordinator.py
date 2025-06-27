"""
CJE Pipeline Coordinator - Orchestrates the execution of all pipeline stages.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .config import PipelineConfig
from .stages import (
    DatasetStage,
    LoggingPolicyStage,
    JudgeStage,
    OracleStage,
    CalibrationStage,
    TargetPolicyStage,
)
from ..estimators import get_estimator
from ..utils.progress import ProgressMonitor, print_summary_table
from ..validation import validate_pipeline_data, assign_rewards_with_priority
from ..utils.error_handling import ConfigurationError

logger = logging.getLogger(__name__)


class CJEPipeline:
    """
    Main pipeline coordinator for CJE experiments.

    This class orchestrates the execution of all pipeline stages in the correct order,
    handling caching, validation, and error recovery.
    """

    def __init__(self, config: PipelineConfig, console: Optional[Console] = None):
        self.config = config
        self.console = console or Console()
        self.work_dir = config.work_dir

        # Initialize stages
        self.dataset_stage = DatasetStage(self.work_dir, self.console)
        self.logging_policy_stage = LoggingPolicyStage(self.work_dir, self.console)
        self.judge_stage = JudgeStage(self.work_dir, self.console)
        self.oracle_stage = (
            OracleStage(self.work_dir, self.console) if config.oracle_config else None
        )
        self.calibration_stage = CalibrationStage(self.work_dir, self.console)
        self.target_policy_stage = TargetPolicyStage(self.work_dir, self.console)

        # Track execution time
        self.start_time: Optional[float] = None
        self.stage_times: Dict[str, float] = {}

    def run(self) -> Dict[str, Any]:
        """
        Execute the full CJE pipeline.

        Returns:
            Dictionary containing experiment results and metadata
        """
        self.start_time = time.time()
        logger.info("Starting CJE pipeline execution")

        # Print pipeline configuration
        self._print_configuration()

        try:
            # 1. Load dataset
            rows = self._run_stage("dataset", self._load_dataset)

            # 2. Generate logging policy responses
            rows = self._run_stage(
                "logging_policy", self._generate_logging_responses, rows
            )

            # 3. Score with judge
            contexts_hash = self._get_contexts_hash(rows)
            rows = self._run_stage("judge", self._score_with_judge, rows, contexts_hash)

            # 4. Generate oracle labels (if enabled)
            if self.config.oracle_config and self.config.oracle_config.get("enabled"):
                judge_hash = self._get_judge_hash(contexts_hash)
                rows = self._run_stage(
                    "oracle", self._generate_oracle_labels, rows, judge_hash
                )

            # 5. Calibrate judge scores
            judge_hash = self._get_judge_hash(contexts_hash)
            oracle_enabled = bool(
                self.config.oracle_config and self.config.oracle_config.get("enabled")
            )
            rows = self._run_stage(
                "calibration", self._calibrate_scores, rows, judge_hash, oracle_enabled
            )

            # 6. Compute target policy log probabilities
            rows = self._run_stage(
                "target_policy", self._compute_target_logprobs, rows, contexts_hash
            )

            # 7. Run estimators
            results = self._run_stage("estimation", self._run_estimators, rows)

            # Print summary
            self._print_summary(results)

            # Save final results
            self._save_results(results, rows)

            return {
                "results": results,
                "rows": rows,
                "execution_time": time.time() - self.start_time,
                "stage_times": self.stage_times,
            }

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.console.print(f"[red]❌ Pipeline execution failed: {e}[/red]")
            raise

    def _run_stage(
        self, stage_name: str, stage_func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Run a pipeline stage with timing and error handling."""
        stage_start = time.time()

        self.console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        self.console.print(
            f"[bold cyan]Stage: {stage_name.replace('_', ' ').title()}[/bold cyan]"
        )
        self.console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

        try:
            result = stage_func(*args, **kwargs)
            stage_time = time.time() - stage_start
            self.stage_times[stage_name] = stage_time

            self.console.print(
                f"\n[green]✅ {stage_name.replace('_', ' ').title()} completed "
                f"in {stage_time:.1f}s[/green]\n"
            )

            return result

        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            self.console.print(f"[red]❌ Stage {stage_name} failed: {e}[/red]")
            raise

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset stage."""
        return self.dataset_stage.run(self.config.dataset_config)

    def _generate_logging_responses(
        self, rows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate logging policy responses."""
        contexts_hash = self._get_contexts_hash(rows)
        return self.logging_policy_stage.run(
            rows, self.config.logging_policy_config, contexts_hash
        )

    def _score_with_judge(
        self, rows: List[Dict[str, Any]], contexts_hash: str
    ) -> List[Dict[str, Any]]:
        """Score responses with judge."""
        return self.judge_stage.run(rows, self.config.judge_config, contexts_hash)

    def _generate_oracle_labels(
        self, rows: List[Dict[str, Any]], judge_hash: str
    ) -> List[Dict[str, Any]]:
        """Generate oracle labels."""
        return self.oracle_stage.run(rows, self.config.oracle_config, judge_hash)

    def _calibrate_scores(
        self, rows: List[Dict[str, Any]], judge_hash: str, oracle_enabled: bool
    ) -> List[Dict[str, Any]]:
        """Calibrate judge scores."""
        return self.calibration_stage.run(
            rows, self.config.calibration_config, judge_hash, oracle_enabled
        )

    def _compute_target_logprobs(
        self, rows: List[Dict[str, Any]], contexts_hash: str
    ) -> List[Dict[str, Any]]:
        """Compute target policy log probabilities."""
        return self.target_policy_stage.run(
            rows, self.config.target_policies_config, contexts_hash
        )

    def _run_estimators(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run causal estimators."""
        # Prepare data for estimation
        rows = self._prepare_for_estimation(rows)

        # Validate data
        validate_pipeline_data(rows)

        results = {}

        for estimator_config in self.config.estimator_configs:
            estimator_name = estimator_config["name"]
            estimator_params = estimator_config.get("params", {})

            logger.info(f"Running estimator: {estimator_name}")
            self.console.print(f"[blue]Running estimator: {estimator_name}[/blue]")

            try:
                # Create estimator
                estimator = get_estimator(estimator_name, **estimator_params)

                # Run estimation (cleaner API)
                estimate = estimator.estimate_from_logs(rows)
                results[estimator_name] = estimate

                # Print result
                self.console.print(
                    f"[green]✓ {estimator_name}: "
                    f"{estimate.estimates} ± {np.sqrt(estimate.standard_errors)}[/green]"  # type: ignore
                )

            except Exception as e:
                logger.error(f"Estimator {estimator_name} failed: {e}")
                self.console.print(f"[red]✗ {estimator_name} failed: {e}[/red]")
                results[estimator_name] = {"error": str(e)}  # type: ignore

        return results

    def _prepare_for_estimation(
        self, rows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare data for causal estimation."""
        # Ensure rewards are assigned
        if not all("reward" in row for row in rows):
            oracle_enabled = bool(
                self.config.oracle_config and self.config.oracle_config.get("enabled")
            )

            if oracle_enabled:
                # Oracle mode: use calibrated scores
                for row in rows:
                    if "score_cal" in row and row["score_cal"] is not None:
                        row["reward"] = row["score_cal"]
                    elif "score" in row and row["score"] is not None:
                        row["reward"] = row["score"]["mean"]
                    else:
                        row["reward"] = 0.0
            else:
                # Standard mode: assign from available sources
                assign_rewards_with_priority(rows)

        return rows

    def _get_contexts_hash(self, rows: List[Dict[str, Any]]) -> str:
        """Get contexts hash from dataset stage."""
        from ..cache import compute_contexts_hash

        return compute_contexts_hash(
            self.config.dataset_config, self.config.logging_policy_config
        )

    def _get_judge_hash(self, contexts_hash: str) -> str:
        """Get judge hash for dependency tracking."""
        from ..cache import compute_judge_hash

        return compute_judge_hash(contexts_hash, self.config.judge_config)

    def _print_configuration(self) -> None:
        """Print pipeline configuration summary."""
        self.console.print(
            Panel.fit(
                f"[bold green]CJE Pipeline Configuration[/bold green]\n\n"
                f"Work Directory: {self.config.work_dir}\n"
                f"Dataset: {self.config.dataset_config['name']}\n"
                f"Logging Policy: {self.config.logging_policy_config['model_name']}\n"
                f"Judge: {self.config.judge_config.get('model_name', self.config.judge_config.get('model', 'unknown'))}\n"
                f"Target Policies: {len(self.config.target_policies_config)}\n"
                f"Estimators: {len(self.config.estimator_configs)}\n"
                f"Oracle: {'Enabled' if self.config.oracle_config else 'Disabled'}",
                title="Configuration",
                border_style="green",
            )
        )

    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print results summary."""
        # Create results table
        table = Table(title="Estimation Results")
        table.add_column("Estimator", style="cyan")
        table.add_column("Policy", style="green")
        table.add_column("Estimate", style="yellow")
        table.add_column("Std Error", style="yellow")

        for estimator_name, estimate in results.items():
            if isinstance(estimate, dict) and "error" in estimate:
                table.add_row(estimator_name, "—", "ERROR", estimate["error"])
            else:
                for i, (value, variance) in enumerate(
                    zip(estimate.estimates, estimate.standard_errors)  # type: ignore
                ):
                    table.add_row(
                        estimator_name if i == 0 else "",
                        f"π_{i}",
                        f"{value:.4f}",
                        f"{np.sqrt(variance):.4f}",
                    )

        self.console.print(table)

        # Print timing summary
        total_time = time.time() - self.start_time
        timing_table = Table(title="Execution Times")
        timing_table.add_column("Stage", style="cyan")
        timing_table.add_column("Time (s)", style="yellow")
        timing_table.add_column("Percentage", style="green")

        for stage, stage_time in self.stage_times.items():
            percentage = (stage_time / total_time) * 100
            timing_table.add_row(
                stage.replace("_", " ").title(),
                f"{stage_time:.1f}",
                f"{percentage:.1f}%",
            )

        timing_table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{total_time:.1f}[/bold]",
            "[bold]100.0%[/bold]",
        )

        self.console.print(timing_table)

    def _save_results(
        self, results: Dict[str, Any], rows: List[Dict[str, Any]]
    ) -> None:
        """Save experiment results."""
        import json

        # Save results
        results_path = self.work_dir / "results.json"
        with open(results_path, "w") as f:
            # Convert results to serializable format
            serializable_results = {}
            for name, estimate in results.items():
                if hasattr(estimate, "to_dict"):
                    serializable_results[name] = estimate.to_dict()
                else:
                    serializable_results[name] = estimate

            json.dump(
                {
                    "results": serializable_results,
                    "config": self.config.to_dict(),
                    "execution_time": time.time() - self.start_time,
                    "stage_times": self.stage_times,
                    "num_samples": len(rows),
                },
                f,
                indent=2,
            )

        logger.info(f"Results saved to {results_path}")
        self.console.print(f"[green]💾 Results saved to {results_path}[/green]")
