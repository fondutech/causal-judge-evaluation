"""
Oracle labeling stage - Handles oracle label generation with caching.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from rich.console import Console

from ...cache import compute_oracle_hash, chunk_exists, load_chunk, save_chunk
from ...oracle_labeling import (
    add_oracle_labels,
    add_full_oracle_labels_with_holdout,
)

logger = logging.getLogger(__name__)


class OracleStage:
    """Handles oracle label generation and caching."""

    def __init__(self, work_dir: Path, console: Optional[Console] = None):
        self.work_dir = work_dir
        self.console = console or Console()

    def run(
        self,
        rows: List[Dict[str, Any]],
        oracle_config: Dict[str, Any],
        judge_hash: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate oracle labels for calibration.

        Args:
            rows: Input rows with judge scores
            oracle_config: Oracle configuration
            judge_hash: Hash of judge configuration for cache dependency

        Returns:
            Rows with oracle labels added
        """
        if not oracle_config.get("enabled", False):
            logger.info("Oracle analysis not enabled")
            return rows

        logger.info("Oracle analysis enabled - generating oracle labels")
        self.console.print("[bold blue]🔮 Oracle Analysis Enabled[/bold blue]")

        oracle_fraction = oracle_config["logging_policy_oracle_fraction"]

        # Compute hash for oracle configuration
        oracle_hash = compute_oracle_hash(
            judge_hash=judge_hash, oracle_config=oracle_config
        )

        # Check cache
        if chunk_exists(self.work_dir, "oracle_labels", oracle_hash):
            logger.info(f"Using cached oracle labels: {oracle_hash}")
            self.console.print(
                f"[green]📦 Using cached oracle labels: {oracle_hash}[/green]"
            )
            return load_chunk(self.work_dir, "oracle_labels", oracle_hash)

        # Generate oracle labels
        try:
            with self.console.status(
                f"[bold blue]Generating oracle labels | Hash: {oracle_hash}..."
            ):
                rows_with_oracle = self._generate_oracle_labels(
                    rows, oracle_config, oracle_fraction
                )

            # Save to cache
            self._save_to_cache(
                rows_with_oracle, oracle_hash, oracle_config, oracle_fraction
            )

            # Log statistics
            self._log_oracle_statistics(rows_with_oracle, oracle_fraction)

            return rows_with_oracle

        except Exception as e:
            logger.error(f"Oracle labeling failed: {e}")
            self.console.print(f"[red]❌ Oracle labeling failed: {e}[/red]")
            self.console.print("[yellow]Continuing without oracle analysis...[/yellow]")
            return rows

    def _generate_oracle_labels(
        self,
        rows: List[Dict[str, Any]],
        oracle_config: Dict[str, Any],
        oracle_fraction: float,
    ) -> List[Dict[str, Any]]:
        """Generate oracle labels based on configuration."""
        if oracle_fraction < 1.0:
            # Cost-efficient mode
            self.console.print(
                f"[blue]💰 Cost-efficient oracle mode: generating labels for "
                f"{oracle_fraction:.1%} of samples[/blue]"
            )
            return add_oracle_labels(
                rows,
                provider=oracle_config["provider"],
                model_name=oracle_config["model_name"],
                fraction=oracle_fraction,
                seed=oracle_config["seed"],
                template=oracle_config.get("template", "quick_judge"),
                temperature=oracle_config.get("temperature", 0.0),
                max_tokens=oracle_config.get("max_tokens", 50),
                score_field="y_true",  # Use y_true for calibration
            )
        else:
            # Full experimental design
            self.console.print(
                "[blue]🧪 Full experimental design: generating labels for all "
                "samples with holdout[/blue]"
            )
            return add_full_oracle_labels_with_holdout(
                rows,
                provider=oracle_config["provider"],
                model_name=oracle_config["model_name"],
                logging_policy_oracle_fraction=oracle_fraction,
                seed=oracle_config["seed"],
                template=oracle_config.get("template", "quick_judge"),
                temperature=oracle_config.get("temperature", 0.0),
                max_tokens=oracle_config.get("max_tokens", 50),
            )

    def _save_to_cache(
        self,
        rows: List[Dict[str, Any]],
        oracle_hash: str,
        oracle_config: Dict[str, Any],
        oracle_fraction: float,
    ) -> None:
        """Save oracle results to cache."""
        metadata = {
            "provider": oracle_config["provider"],
            "model_name": oracle_config["model_name"],
            "oracle_fraction": oracle_fraction,
            "sample_count": len(rows),
            "oracle_mode": (
                "cost_efficient" if oracle_fraction < 1.0 else "full_experimental"
            ),
        }

        save_chunk(
            self.work_dir,
            "oracle_labels",
            oracle_hash,
            rows,
            metadata=metadata,
        )

        logger.info(f"Oracle labels generated and cached: {oracle_hash}")
        self.console.print(
            f"[green]✅ Oracle labels generated and cached: {oracle_hash}[/green]"
        )

    def _log_oracle_statistics(
        self,
        rows: List[Dict[str, Any]],
        oracle_fraction: float,
    ) -> None:
        """Log statistics about oracle labeling."""
        oracle_count = sum(1 for row in rows if row.get("y_true") is not None)
        total_samples = len(rows)

        logger.info(
            f"Oracle labels available: {oracle_count}/{total_samples} samples "
            f"({oracle_count/total_samples:.1%})"
        )
        self.console.print(
            f"[green]✅ Oracle labels available: {oracle_count}/{total_samples} "
            f"samples ({oracle_count/total_samples:.1%})[/green]"
        )

        if oracle_fraction < 1.0:
            self.console.print(
                f"[blue]💰 Cost savings: {total_samples - oracle_count} fewer "
                f"oracle API calls[/blue]"
            )
        else:
            holdout_count = sum(
                1 for row in rows if row.get("oracle_holdout_mask", False)
            )
            available_count = sum(
                1 for row in rows if row.get("oracle_available_to_logging", False)
            )
            self.console.print(
                f"[blue]📊 Oracle breakdown: {available_count} for calibration, "
                f"{holdout_count} held out for evaluation[/blue]"
            )

    def prepare_for_calibration(
        self, rows: List[Dict[str, Any]], oracle_fraction: float
    ) -> None:
        """
        Prepare oracle data for calibration stage.

        Updates rows in-place to set y_true appropriately for calibration.
        """
        for row in rows:
            # Ensure score_raw field exists for calibration
            if "score" in row and "score_raw" not in row:
                row["score_raw"] = row["score"]

            if oracle_fraction < 1.0:
                # Cost-efficient mode: y_true already set by add_oracle_labels
                pass
            else:
                # Full experimental design: set y_true only for available samples
                if row.get("oracle_available_to_logging", False):
                    row["y_true"] = row["oracle_full"]
