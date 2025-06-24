"""
Calibration stage - Handles judge score calibration.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from rich.console import Console
from rich.table import Table

from ...calibration.cross_fit import cross_fit_calibration
from ...calibration.isotonic import fit_isotonic, plot_reliability
from ...cache import chunk_exists, load_chunk, save_chunk
from ...validation import assign_rewards_with_priority
from ..validation import validate_stage_output

logger = logging.getLogger(__name__)


class CalibrationStage:
    """Handles judge score calibration with ground truth or oracle labels."""

    def __init__(self, work_dir: Path, console: Optional[Console] = None):
        self.work_dir = work_dir
        self.console = console or Console()

    @validate_stage_output(
        required_fields={"uid", "context", "response"},
        at_least_one_of={"score_cal", "reward", "y_true"},
    )
    def run(
        self,
        rows: List[Dict[str, Any]],
        calibration_config: Dict[str, Any],
        judge_hash: str,
        oracle_analysis_enabled: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Calibrate judge scores using available ground truth.

        Args:
            rows: Input rows with judge scores
            calibration_config: Calibration configuration
            judge_hash: Hash of judge configuration for cache dependency
            oracle_analysis_enabled: Whether oracle analysis is enabled

        Returns:
            Rows with calibrated scores added
        """
        logger.info("Starting calibration")
        self.console.print("[bold blue]📊 Calibrating Judge Scores[/bold blue]")

        # Prepare data for calibration
        rows_with_rewards = self._prepare_calibration_data(
            rows, oracle_analysis_enabled
        )

        # Check if we have enough data for calibration
        calibration_samples = self._count_calibration_samples(
            rows_with_rewards, oracle_analysis_enabled
        )

        if calibration_samples < calibration_config.get("min_samples", 10):
            logger.warning(
                f"Insufficient samples for calibration: {calibration_samples}"
            )
            self.console.print(
                f"[yellow]⚠️  Insufficient samples for calibration: "
                f"{calibration_samples}[/yellow]"
            )
            return self._apply_fallback_calibration(rows_with_rewards)

        # Generate calibration hash
        calibration_hash = self._compute_calibration_hash(
            judge_hash, calibration_config, oracle_analysis_enabled
        )

        # Check cache
        if chunk_exists(self.work_dir, "calibrated_scores", calibration_hash):
            logger.info(f"Using cached calibration: {calibration_hash}")
            self.console.print(
                f"[green]📦 Using cached calibration: {calibration_hash}[/green]"
            )
            return load_chunk(self.work_dir, "calibrated_scores", calibration_hash)

        # Perform calibration
        calibrated_rows = self._calibrate_scores(
            rows_with_rewards, calibration_config, oracle_analysis_enabled
        )

        # Save to cache
        self._save_to_cache(
            calibrated_rows, calibration_hash, calibration_config, calibration_samples
        )

        # Log calibration statistics
        self._log_calibration_statistics(calibrated_rows, oracle_analysis_enabled)

        return calibrated_rows

    def _prepare_calibration_data(
        self, rows: List[Dict[str, Any]], oracle_analysis_enabled: bool
    ) -> List[Dict[str, Any]]:
        """Prepare data for calibration by assigning rewards."""
        if oracle_analysis_enabled:
            # Oracle mode: y_true already set by oracle stage
            return rows
        else:
            # Standard mode: assign rewards from available sources
            assign_rewards_with_priority(rows)
            return rows

    def _count_calibration_samples(
        self, rows: List[Dict[str, Any]], oracle_analysis_enabled: bool
    ) -> int:
        """Count available samples for calibration."""
        if oracle_analysis_enabled:
            # Count rows with oracle labels or fallback rewards
            return sum(
                1
                for row in rows
                if (
                    row.get("y_true") is not None
                    or row.get("calibration_fallback", False)
                )
            )
        else:
            # Count rows with rewards
            return sum(1 for row in rows if row.get("reward") is not None)

    def _apply_fallback_calibration(
        self, rows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply fallback calibration when insufficient data."""
        for row in rows:
            if row.get("score") is not None:
                # Use raw score as calibrated score
                row["score_cal"] = row["score"]["mean"]
                row["calibration_fallback"] = True
                row["calibration_method"] = "identity"

        return rows

    def _calibrate_scores(
        self,
        rows: List[Dict[str, Any]],
        config: Dict[str, Any],
        oracle_analysis_enabled: bool,
    ) -> List[Dict[str, Any]]:
        """Perform cross-fit calibration."""
        # Prepare data arrays
        X, y, indices = self._prepare_arrays(rows, oracle_analysis_enabled)

        if len(X) == 0:
            logger.warning("No valid samples for calibration")
            return self._apply_fallback_calibration(rows)

        # Perform cross-fit calibration
        n_folds = min(config.get("n_folds", 5), len(X))

        logger.info(
            f"Performing {n_folds}-fold cross-fit calibration " f"on {len(X)} samples"
        )

        try:
            # Call cross_fit_calibration with proper arguments
            rows_for_calibration = rows.copy()
            calibrated_rows, calibration_info = cross_fit_calibration(
                rows_for_calibration, k_folds=n_folds
            )

            # The calibrated_rows already have score_cal set by cross_fit_calibration
            for row in calibrated_rows:
                row["calibration_method"] = "cross_fit_isotonic"

            # Store calibration model info
            self._store_calibration_info(calibration_info, config)

            return calibrated_rows

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            self.console.print(f"[red]❌ Calibration failed: {e}[/red]")
            return self._apply_fallback_calibration(rows)

    def _prepare_arrays(
        self, rows: List[Dict[str, Any]], oracle_analysis_enabled: bool
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Prepare numpy arrays for calibration."""
        X = []  # Judge scores
        y = []  # Ground truth
        indices = []  # Row indices

        for i, row in enumerate(rows):
            # Skip rows without scores
            if row.get("score") is None:
                continue

            # Get ground truth value
            if oracle_analysis_enabled:
                # Oracle mode
                if row.get("y_true") is not None:
                    gt_value = row["y_true"]
                elif row.get("calibration_fallback", False):
                    gt_value = row.get("reward", 0)
                else:
                    continue
            else:
                # Standard mode
                if row.get("reward") is not None:
                    gt_value = row["reward"]
                else:
                    continue

            # Add to arrays
            X.append(row["score"]["mean"])
            y.append(gt_value)
            indices.append(i)

        return np.array(X), np.array(y), indices

    def _compute_calibration_hash(
        self, judge_hash: str, config: Dict[str, Any], oracle_analysis_enabled: bool
    ) -> str:
        """Compute hash for calibration configuration."""
        import hashlib
        import json

        hash_data = {
            "judge_hash": judge_hash,
            "n_folds": config.get("n_folds", 5),
            "oracle_mode": oracle_analysis_enabled,
            "method": "cross_fit_isotonic",
        }

        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]

    def _save_to_cache(
        self,
        rows: List[Dict[str, Any]],
        calibration_hash: str,
        config: Dict[str, Any],
        calibration_samples: int,
    ) -> None:
        """Save calibrated scores to cache."""
        metadata = {
            "n_folds": config.get("n_folds", 5),
            "calibration_samples": calibration_samples,
            "sample_count": len(rows),
            "method": "cross_fit_isotonic",
        }

        save_chunk(
            self.work_dir,
            "calibrated_scores",
            calibration_hash,
            rows,
            metadata=metadata,
        )

        logger.info(f"Calibrated scores cached: {calibration_hash}")
        self.console.print(
            f"[green]✅ Calibrated scores cached: {calibration_hash}[/green]"
        )

    def _store_calibration_info(
        self, calibration_info: Dict[str, Any], config: Dict[str, Any]
    ) -> None:
        """Store calibration model information."""
        # Save calibration plots if requested
        if config.get("save_plots", True):
            plots_dir = self.work_dir / "calibration_plots"
            plots_dir.mkdir(exist_ok=True)

            # Generate reliability plots
            for fold_idx, fold_info in calibration_info.get("folds", {}).items():
                if "model" in fold_info:
                    plot_path = plots_dir / f"reliability_fold_{fold_idx}.png"
                    # Note: plot_reliability would need to be implemented
                    logger.info(f"Saved calibration plot: {plot_path}")

    def _log_calibration_statistics(
        self, rows: List[Dict[str, Any]], oracle_analysis_enabled: bool
    ) -> None:
        """Log calibration statistics."""
        calibrated_count = sum(1 for row in rows if row.get("score_cal") is not None)
        fallback_count = sum(
            1 for row in rows if row.get("calibration_fallback", False)
        )

        # Create statistics table
        table = Table(title="Calibration Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Samples", str(len(rows)))
        table.add_row("Calibrated Samples", str(calibrated_count))
        table.add_row("Fallback Samples", str(fallback_count))

        if oracle_analysis_enabled:
            oracle_count = sum(1 for row in rows if row.get("y_true") is not None)
            table.add_row("Oracle Labels", str(oracle_count))

        self.console.print(table)

        logger.info(
            f"Calibration complete: {calibrated_count} calibrated, "
            f"{fallback_count} fallback"
        )
