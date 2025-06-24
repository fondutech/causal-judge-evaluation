"""
Judge scoring stage - Handles judge score generation with caching.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from rich.console import Console
from rich.progress import track

from ...judge.factory import JudgeFactory
from ...judge.judges import Judge
from ...judge.cached_judge import CachedJudge
from ...cache import compute_judge_hash, chunk_exists, load_chunk, save_chunk
from ...utils.error_handling import safe_call
from ..validation import validate_stage_output

logger = logging.getLogger(__name__)


class JudgeStage:
    """Handles judge score generation and caching."""

    def __init__(self, work_dir: Path, console: Optional[Console] = None):
        self.work_dir = work_dir
        self.console = console or Console()

    @validate_stage_output(
        required_fields={"uid", "context", "response", "score"},
        optional_fields={"score_raw", "judge_model"},
    )
    def run(
        self,
        rows: List[Dict[str, Any]],
        judge_config: Dict[str, Any],
        contexts_hash: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate judge scores for responses.

        Args:
            rows: Input rows with contexts and responses
            judge_config: Judge configuration
            contexts_hash: Hash of contexts for cache dependency

        Returns:
            Rows with judge scores added
        """
        logger.info("Starting judge scoring")
        self.console.print("[bold blue]⚖️  Generating Judge Scores[/bold blue]")

        # Compute judge hash
        judge_hash = compute_judge_hash(contexts_hash, judge_config)

        # Check cache
        if chunk_exists(self.work_dir, "judge_scores", judge_hash):
            logger.info(f"Using cached judge scores: {judge_hash}")
            self.console.print(
                f"[green]📦 Using cached judge scores: {judge_hash}[/green]"
            )
            return load_chunk(self.work_dir, "judge_scores", judge_hash)

        # Create judge with caching
        judge = self._create_cached_judge(judge_config, judge_hash)

        # Extract model name
        model_name = judge_config.get("model_name", "unknown")

        # Generate scores
        rows_with_scores = self._generate_scores(rows, judge, judge_config, model_name)

        # Save to cache
        self._save_to_cache(rows_with_scores, judge_hash, judge_config, model_name)

        # Log statistics
        self._log_score_statistics(rows_with_scores)

        return rows_with_scores

    def _create_cached_judge(
        self, config: Dict[str, Any], judge_hash: str
    ) -> CachedJudge:
        """Create judge with caching wrapper."""
        # Create base judge
        model_name = config.get("model_name")
        if not model_name:
            raise ValueError("Judge config must specify 'model_name'")

        base_judge = JudgeFactory.create(
            provider=config["provider"],
            model=model_name,
            template=config.get("template", "default_template"),
            uncertainty_method=config.get("uncertainty_method", "deterministic"),
            mc_samples=config.get("num_samples", 1),
            api_key=config.get("api_key"),
        )

        # Wrap with caching
        cache_dir = self.work_dir / "judge_cache" / judge_hash
        cache_dir.mkdir(parents=True, exist_ok=True)

        return CachedJudge(
            base_judge=base_judge, cache_size=config.get("cache_size", 1000)
        )

    def _generate_scores(
        self,
        rows: List[Dict[str, Any]],
        judge: CachedJudge,
        config: Dict[str, Any],
        model_name: str,
    ) -> List[Dict[str, Any]]:
        """Generate judge scores for all rows."""
        total_rows = len(rows)
        logger.info(f"Scoring {total_rows} samples")

        # Filter rows that need scoring
        rows_to_score = []
        score_indices = []

        for i, row in enumerate(rows):
            if "response" in row and row["response"] is not None:
                rows_to_score.append(row)
                score_indices.append(i)

        if not rows_to_score:
            logger.warning("No rows with responses to score")
            return rows

        # Score in batches
        batch_size = config.get("batch_size", 10)
        scored_rows = rows.copy()

        with self.console.status(
            f"[bold blue]Scoring {len(rows_to_score)} samples..."
        ) as status:
            for batch_start in range(0, len(rows_to_score), batch_size):
                batch_end = min(batch_start + batch_size, len(rows_to_score))
                batch_rows = rows_to_score[batch_start:batch_end]
                batch_indices = score_indices[batch_start:batch_end]

                # Update status
                status.update(
                    f"[bold blue]Scoring samples {batch_start+1}-{batch_end} "
                    f"of {len(rows_to_score)}..."
                )

                # Score batch
                for row, idx in zip(batch_rows, batch_indices):
                    try:
                        score = judge.score(
                            context=row["context"], response=row["response"]
                        )

                        # Add score to row
                        scored_rows[idx]["score"] = {
                            "mean": score.mean,
                            "variance": score.variance,
                        }
                        scored_rows[idx]["score_raw"] = scored_rows[idx]["score"]
                        scored_rows[idx]["judge_model"] = model_name

                    except Exception as e:
                        logger.error(f"Failed to score row {idx}: {e}")
                        # Add null score
                        scored_rows[idx]["score"] = None
                        scored_rows[idx]["score_raw"] = None
                        scored_rows[idx]["judge_error"] = str(e)

                # Log progress
                progress = batch_end / len(rows_to_score) * 100
                logger.info(f"Scoring progress: {progress:.1f}%")

        return scored_rows

    def _save_to_cache(
        self,
        rows: List[Dict[str, Any]],
        judge_hash: str,
        config: Dict[str, Any],
        model_name: str,
    ) -> None:
        """Save judge scores to cache."""
        metadata = {
            "provider": config["provider"],
            "model_name": model_name,
            "template": config.get("template", "default_template"),
            "uncertainty_method": config.get("uncertainty_method", "deterministic"),
            "sample_count": len(rows),
            "scored_count": sum(1 for r in rows if r.get("score") is not None),
        }

        save_chunk(self.work_dir, "judge_scores", judge_hash, rows, metadata=metadata)

        logger.info(f"Judge scores cached: {judge_hash}")
        self.console.print(
            f"[green]✅ Judge scores generated and cached: {judge_hash}[/green]"
        )

    def _log_score_statistics(self, rows: List[Dict[str, Any]]) -> None:
        """Log statistics about judge scores."""
        scores = []
        null_count = 0

        for row in rows:
            if row.get("score") is not None:
                scores.append(row["score"]["mean"])
            else:
                null_count += 1

        if scores:
            import numpy as np

            mean_score = np.mean(scores)
            std_score = np.std(scores)

            logger.info(
                f"Judge score statistics: mean={mean_score:.3f}, "
                f"std={std_score:.3f}, null={null_count}"
            )
            self.console.print(
                f"[blue]📊 Score statistics: μ={mean_score:.3f}, "
                f"σ={std_score:.3f}, null={null_count}[/blue]"
            )
        else:
            logger.warning("No valid judge scores generated")
            self.console.print("[yellow]⚠️  No valid judge scores generated[/yellow]")
