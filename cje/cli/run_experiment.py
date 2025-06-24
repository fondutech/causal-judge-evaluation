from __future__ import annotations
import json
import pathlib
import math
import typer
import shutil
import time
import logging
import numpy as np
from itertools import islice
from hydra import initialize, compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from rich import print
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    track,
)
from rich.logging import RichHandler
from cje.utils.progress import ProgressMonitor, console, print_summary_table
from typing import cast, Dict, Any, List, Optional, Union, Tuple
from enum import Enum
import warnings
from pathlib import Path
import tempfile
import inspect

from ..cache import (
    compute_contexts_hash,
    compute_judge_hash,
    compute_oracle_hash,
    compute_target_logprobs_hash,
    chunk_exists,
    load_chunk,
    save_chunk,
    list_cached_stages,
)
from ..data import load_dataset
from ..loggers.policy import PolicyRunner
from ..loggers.api_policy import APIPolicyRunner
from ..estimators import get_estimator
from ..estimators.base import Estimator
from ..loggers.multi_target_sampler import make_multi_sampler
from ..validation import validate_target_policy_computation, validate_pipeline_data
from ..config.unified import from_dict, to_dict, CJEConfig
from ..utils.error_handling import ConfigurationError, ValidationError
from ..testing import testing_mode
from ..utils.progress import (
    ProgressMonitor,
    console as progress_console,
    print_summary_table,
    track as progress_track,
)

# Temporary file to store the work directory path
_WORK_DIR_FILE = Path(tempfile.gettempdir()) / "cje_last_work_dir.txt"


def _determine_analysis_type(
    rows: List[Dict[str, Any]], oracle_analysis_enabled: bool
) -> Tuple[str, int]:
    """
    Determine analysis type based on available ground truth data.

    Args:
        rows: List of data rows
        oracle_analysis_enabled: Whether oracle analysis is active

    Returns:
        Tuple of (analysis_type, ground_truth_count)

    Analysis types:
        - "causal_inference": Sufficient ground truth for reliable causal inference
        - "causal_inference_sparse": Limited ground truth, causal inference with warnings
        - "llm_comparison": No ground truth, only LLM judge comparison
    """
    if oracle_analysis_enabled:
        # Oracle analysis: count calibrated scores and fallback rewards
        ground_truth_count = sum(
            1
            for row in rows
            if (
                row.get("score_cal") is not None
                or row.get("calibration_fallback", False)
            )
        )
    else:
        # Standard analysis: count explicit ground truth labels with responses
        ground_truth_count = sum(
            1
            for row in rows
            if (row.get("reward") is not None and row.get("reward") != 0.0)
            or (row.get("y_true") is not None and row.get("response") is not None)
        )

    has_ground_truth = ground_truth_count > 0
    min_threshold = max(10, len(rows) * 0.02)  # At least 10 samples or 2% of data

    if not has_ground_truth:
        return "llm_comparison", ground_truth_count
    elif ground_truth_count < min_threshold:
        return "causal_inference_sparse", ground_truth_count
    else:
        return "causal_inference", ground_truth_count


def setup_logging(work_dir: pathlib.Path) -> logging.Logger:
    """Set up structured logging with file and console output."""

    # Create logs directory
    log_dir = work_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    log_file = log_dir / "experiment.log"

    # Create logger
    logger = logging.getLogger("cje.experiment")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Rich console handler
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger


def run(
    cfg_path: str = typer.Option("cje.conf", "--cfg-path"),
    cfg_name: str = typer.Option("experiment", "--cfg-name"),
) -> None:
    """`cje run` executes dataset → logging policy → target policy → estimator."""

    start_time: float = time.time()
    console = Console()

    # Load configuration
    config_path_abs = pathlib.Path(cfg_path).resolve()

    if config_path_abs.is_absolute() and "/" in cfg_path:
        with initialize_config_dir(
            version_base=None, config_dir=str(config_path_abs), job_name="cje_test_run"
        ):
            cfg = compose(config_name=cfg_name)
    else:
        with initialize(version_base=None, config_path=cfg_path):
            cfg = compose(config_name=cfg_name)

    print("[bold green]Config[/bold green]")
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

        # Convert back to OmegaConf for compatibility with existing code
        cfg = OmegaConf.create(to_dict(unified_config))

    except ConfigurationError as e:
        print(f"[bold red]❌ Configuration Error:[/bold red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"[bold red]❌ Unexpected Error:[/bold red] {e}")
        raise typer.Exit(1)

    # 1. prepare work dir
    work = pathlib.Path(cfg.paths.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(work)
    logger.info(f"Starting CJE experiment: {cfg_name}")
    logger.info(f"Work directory: {work}")
    logger.info(f"Config path: {cfg_path}")

    # ------------------------------------------------------------
    # helper to build a policy runner from a Hydra config section
    # (works for both logging policy and target policy configurations)
    # ------------------------------------------------------------
    def _make_runner(pol_cfg: Any) -> Union[APIPolicyRunner, PolicyRunner]:
        provider = getattr(pol_cfg, "provider", None)
        if provider not in ["hf"]:
            # Build kwargs with only the parameters present in config
            kwargs: Dict[str, Any] = {
                "provider": provider,
                "model_name": pol_cfg.model_name,
            }
            # Add optional parameters only if present in config
            for param in ["max_new_tokens", "temperature", "top_p", "batch_size"]:
                if hasattr(pol_cfg, param):
                    kwargs[param] = getattr(pol_cfg, param)
            return APIPolicyRunner(**kwargs)
        # default: local HF model
        return PolicyRunner(pol_cfg.model_name)

    # 2. dataset
    with console.status("[bold blue]Loading dataset...") as status:
        logger.info(f"Loading dataset: {cfg.dataset.name}, split: {cfg.dataset.split}")

        ds = load_dataset(cfg.dataset.name, split=cfg.dataset.split)

        # Get dataset size if possible
        try:
            if hasattr(ds, "__len__"):
                dataset_size = len(ds)
                logger.info(f"Dataset loaded: {dataset_size} samples")
            else:
                logger.info("Dataset loaded (size unknown)")
        except:
            logger.info("Dataset loaded (size unknown)")

    # Optional: limit samples for quick testing
    sample_limit = getattr(cfg.dataset, "sample_limit", None)
    if sample_limit is not None:
        logger.info(f"Limiting to first {sample_limit} samples for testing")
        print(f"[yellow]Limiting to first {sample_limit} samples for testing[/yellow]")

        with console.status("[bold blue]Sampling limited dataset..."):
            # Cache the dataset samples to avoid multiple iterations
            # First, check if we can disable progress bar for cleaner output
            disable_progress_supported = False
            if hasattr(ds, "itersamples"):
                try:
                    disable_progress_supported = (
                        "disable_progress" in ds.itersamples.__code__.co_varnames  # type: ignore
                    )
                except AttributeError:
                    disable_progress_supported = False

            # Single iteration to collect samples
            if disable_progress_supported:
                try:
                    limited_samples = list(
                        islice(ds.itersamples(disable_progress=True), sample_limit)  # type: ignore
                    )
                except TypeError:
                    # Fallback if disable_progress fails
                    limited_samples = list(islice(ds.itersamples(), sample_limit))
            else:
                # Standard iteration without disable_progress
                if hasattr(ds, "itersamples"):
                    limited_samples = list(islice(ds.itersamples(), sample_limit))
                else:
                    # Fallback for datasets without itersamples
                    limited_samples = list(islice(iter(ds), sample_limit))  # type: ignore

            # Create a mock dataset object that returns the limited samples
            class LimitedDataset:
                def __init__(self, samples: List[Any]) -> None:
                    self.samples = samples

                def itersamples(self) -> Any:
                    return iter(self.samples)

            ds = cast(Any, LimitedDataset(limited_samples))
            logger.info(f"Limited dataset created with {len(limited_samples)} samples")

    # 3. logging policy π₀ - compute log probabilities for existing/generated responses
    log_json = work / "logs.jsonl"
    if not log_json.exists():
        logger.info("Starting logging policy π₀ computation phase")

        # Initialize logging policy runner
        with console.status("[bold blue]Initializing logging policy runner..."):
            provider = getattr(cfg.logging_policy, "provider", "hf")
            model_name = cfg.logging_policy.model_name
            logger.info(
                f"Creating {provider} logging policy runner with model: {model_name}"
            )
            logging_policy_runner = _make_runner(cfg.logging_policy)

        # Get total samples for progress bar
        try:
            if hasattr(ds, "samples"):  # LimitedDataset
                total_samples = len(ds.samples)
            elif hasattr(ds, "__len__"):
                total_samples = len(ds)
            else:
                total_samples = None
        except:
            total_samples = None

        logger.info(
            f"Computing log probabilities for {total_samples or 'unknown'} samples"
        )

        # Process samples with progress bar
        samples_processed = 0
        samples_with_existing_response = 0
        samples_generated = 0

        with log_json.open("w") as fh:
            # Create progress bar for sample processing
            samples_iter = ds.itersamples()
            if total_samples:
                samples_iter = progress_track(
                    samples_iter,
                    total=total_samples,
                    description="Computing log probabilities",
                )
            else:
                samples_iter = progress_track(
                    samples_iter,
                    description="Computing log probabilities",
                )

            for samp in samples_iter:
                ctx = samp.context

                # --------------------------------------------------
                # Use dataset's existing response if available;
                # otherwise generate with logging policy π₀ (fallback).
                # --------------------------------------------------
                text = samp.response
                if text:
                    # Scenario 2: Complete logs with ground truth
                    # Compute log-prob (and token-level log-probs) for the GIVEN response
                    # APIPolicyRunner doesn't support return_token_logprobs in log_prob
                    if isinstance(logging_policy_runner, APIPolicyRunner):
                        logp_tokens = logging_policy_runner.log_prob(
                            ctx,
                            text,
                            max_new_tokens=getattr(
                                cfg.logging_policy, "max_new_tokens", None
                            ),
                            temperature=getattr(
                                cfg.logging_policy, "temperature", None
                            ),
                            top_p=getattr(cfg.logging_policy, "top_p", None),
                        )
                        token_logps: List[float] = []
                    else:
                        # PolicyRunner supports return_token_logprobs
                        lp_result = logging_policy_runner.log_prob(
                            ctx,
                            text,
                            max_new_tokens=getattr(
                                cfg.logging_policy, "max_new_tokens", None
                            ),
                            temperature=getattr(
                                cfg.logging_policy, "temperature", None
                            ),
                            top_p=getattr(cfg.logging_policy, "top_p", None),
                            return_token_logprobs=True,
                        )

                        if isinstance(lp_result, tuple):
                            # lp_result is Tuple[float, List[float]] when return_token_logprobs=True
                            logp_tokens, token_logps = lp_result
                        else:
                            # lp_result is float when return_token_logprobs=False (shouldn't happen here)
                            logp_tokens = lp_result  # type: ignore[assignment]
                            token_logps = []
                    samples_with_existing_response += 1

                    # Use ground truth since it corresponds to this specific response
                    reward = samp.y_true
                else:
                    # Scenario 1: Context only
                    # Dataset has no response → generate one with logging policy π₀

                    # Use consistent generation method for API policies to ensure π₀ is computed
                    # the same way as πₖ (both use teacher forcing)
                    if isinstance(logging_policy_runner, APIPolicyRunner):
                        # Two-pass approach: generation + teacher forcing for consistent π₀
                        result = logging_policy_runner.generate_with_consistent_logp(
                            [ctx],
                            max_new_tokens=getattr(
                                cfg.logging_policy, "max_new_tokens", None
                            ),
                            temperature=getattr(
                                cfg.logging_policy, "temperature", None
                            ),
                            top_p=getattr(cfg.logging_policy, "top_p", None),
                        )[0]
                        text, logp_tokens = result
                        token_logps = []  # APIPolicyRunner doesn't return token logps
                    else:
                        # Local models: generation and teacher forcing are already consistent
                        generation_result = logging_policy_runner.generate_with_logp(
                            [ctx],
                            max_new_tokens=getattr(
                                cfg.logging_policy, "max_new_tokens", None
                            ),
                            temperature=getattr(
                                cfg.logging_policy, "temperature", None
                            ),
                            top_p=getattr(cfg.logging_policy, "top_p", None),
                            return_token_logprobs=True,
                        )[0]
                        # Handle variable-length tuple result safely
                        if (
                            isinstance(generation_result, tuple)
                            and len(generation_result) >= 2
                        ):
                            text = generation_result[0]
                            logp_tokens = generation_result[1]
                            token_logps = (
                                generation_result[2]
                                if len(generation_result) > 2
                                else []
                            )
                        else:
                            # Fallback for unexpected format
                            text = (
                                str(generation_result)
                                if not isinstance(generation_result, tuple)
                                else str(generation_result[0])
                            )
                            logp_tokens = 0.0
                            token_logps = []

                    samples_generated += 1

                    # No ground truth available since this is a generated response
                    # Will rely on judge evaluation instead
                    reward = None

                row = {
                    "uid": samp.uid,
                    "context": samp.context,
                    "response": text,
                    "logp": float(logp_tokens),
                    "token_logps": token_logps,
                    "action": str(cfg.logging_policy.model_name),
                    "reward": reward,
                    "max_new_tokens": getattr(
                        cfg.logging_policy, "max_new_tokens", 1024
                    ),
                    "temperature": getattr(cfg.logging_policy, "temperature", 0.1),
                    "top_p": getattr(cfg.logging_policy, "top_p", 1.0),
                }
                fh.write(json.dumps(row) + "\n")
                samples_processed += 1

        logger.info(
            f"Logging policy π₀ computation complete: {samples_processed} samples processed"
        )
        logger.info(
            f"  - Samples with existing responses: {samples_with_existing_response}"
        )
        logger.info(f"  - Samples with generated responses: {samples_generated}")
        print(f"[green]Wrote {log_json}[/green]")
    else:
        logger.info(f"Using existing logging policy logs: {log_json}")

    # 3b. attach judge score if not present
    # ------------------------------------------------------------
    # Fast path: allow users to skip judge + calibration entirely
    # ------------------------------------------------------------
    if getattr(cfg.judge, "skip", False):
        logger.info("Judge + calibration skipped (cfg.judge.skip = true)")
        print("[yellow]Judge + calibration skipped (cfg.judge.skip = true).[/yellow]")
        source_json = log_json
    else:
        logger.info("Starting judge scoring phase")

        # Load base contexts data for cache computation
        contexts_rows = [json.loads(l) for l in log_json.read_text().splitlines()]

        # Compute hash for judge configuration - include actual sample count
        # to ensure cache invalidation when sample_limit changes
        dataset_config_with_count = {
            **cfg.dataset,
            "actual_sample_count": len(contexts_rows),
        }
        contexts_hash = compute_contexts_hash(
            dataset_config=dataset_config_with_count,
            logging_policy_config=cfg.logging_policy,
        )
        judge_hash = compute_judge_hash(
            contexts_hash=contexts_hash, judge_config=cfg.judge
        )

        # Check for cached judge scores
        if chunk_exists(work, "judge_scores", judge_hash):
            logger.info(f"Using cached judge scores: {judge_hash}")
            print(f"[green]📦 Using cached judge scores: {judge_hash}[/green]")
            judge_rows = load_chunk(work, "judge_scores", judge_hash)
        else:
            with console.status(
                f"[bold blue]Running judge ({cfg.judge.provider}) with template {cfg.judge.template} | Hash: {judge_hash}..."
            ):
                logger.info(
                    f"Running judge: {cfg.judge.provider} with template: {cfg.judge.template} | Hash: {judge_hash}"
                )

                # Call judge logic directly instead of CLI function
                from ..judge import JudgeFactory  # Now uses unified factory
                from ..utils.score_storage import update_row_with_score

                # Create judge using modern factory system with uncertainty
                judge_kwargs = {
                    "template": cfg.judge.template,
                    "uncertainty_method": getattr(
                        cfg.judge, "uncertainty_method", "structured"
                    ),
                }
                if hasattr(cfg.judge, "temperature"):
                    judge_kwargs["temperature"] = cfg.judge.temperature
                if hasattr(cfg.judge, "max_tokens"):
                    judge_kwargs["max_tokens"] = cfg.judge.max_tokens
                if hasattr(cfg.judge, "mc_samples"):
                    judge_kwargs["mc_samples"] = cfg.judge.mc_samples

                # Map provider to explicit configuration
                provider = cfg.judge.provider
                model = cfg.judge.model_name

                # Create unified judge with uncertainty support
                judge_instance = JudgeFactory.create(
                    provider=provider, model=model, **judge_kwargs
                )

                # Store judge instance for later use in estimators
                judge_runner = judge_instance

                # Prepare samples for batch scoring
                samples = [
                    {"context": row["context"], "response": row["response"]}
                    for row in contexts_rows
                ]

                # Score all samples - now returns List[JudgeScore]
                scores = judge_instance.score_batch(samples)

                # Add scores to rows using unified storage format
                judge_rows = []
                for row, score in zip(contexts_rows, scores):
                    # Update row with structured score
                    new_row = update_row_with_score(row, score, "score_raw")
                    judge_rows.append(new_row)

                # Save to modular cache with uncertainty metadata
                save_chunk(
                    work,
                    "judge_scores",
                    judge_hash,
                    judge_rows,
                    metadata={
                        "provider": provider,
                        "model_name": model,
                        "template": cfg.judge.template,
                        "uncertainty_method": judge_kwargs.get(
                            "uncertainty_method", "structured"
                        ),
                        "sample_count": len(judge_rows),
                        "has_variance": True,  # Flag for unified format
                    },
                )

                logger.info(f"Judge scoring complete and cached: {judge_hash}")
                print(
                    f"[green]✅ Judge scoring complete and cached: {judge_hash}[/green]"
                )

                # Log variance statistics if available
                if any(r.get("score_raw_variance", 0) > 0 for r in judge_rows):
                    import numpy as np

                    variances = [r.get("score_raw_variance", 0.0) for r in judge_rows]
                    logger.info(
                        f"Judge uncertainty stats - mean variance: {np.mean(variances):.4f}, "
                        f"std: {np.std(variances):.4f}, max: {np.max(variances):.4f}"
                    )

        # Save judge scores for pipeline
        judge_json = work / f"{cfg.judge.template}_scores.jsonl"
        judge_json.write_text("\n".join(json.dumps(r) for r in judge_rows))

        logger.info(f"Judge scores available: {len(judge_rows)} samples")

        # 3b.5. Oracle Analysis Integration (before calibration)
        # Oracle Analysis Workflow:
        # 1. Generate oracle labels for ALL (context, response) pairs
        # 2. Hold out oracle fraction for evaluation, make remainder available for calibration
        # 3. Train calibration model: cheap judge scores → oracle scores (using available subset)
        # 4. Apply calibration to ALL samples → predicted rewards
        # 5. Run CJE estimation using predicted rewards

        if hasattr(cfg, "oracle") and getattr(cfg.oracle, "enabled", False):
            logger.info("Oracle analysis enabled - generating oracle labels")
            print("[bold blue]🔮 Oracle Analysis Enabled[/bold blue]")

            # Import oracle functions
            from ..oracle_labeling import (
                add_oracle_labels,
                add_full_oracle_labels_with_holdout,
                evaluate_against_oracle_holdout,
            )

            # Generate oracle labels
            oracle_config = cfg.oracle
            oracle_fraction = oracle_config.logging_policy_oracle_fraction

            # Compute hash for oracle configuration
            oracle_hash = compute_oracle_hash(
                judge_hash=judge_hash, oracle_config=cfg.oracle
            )

            # Check for cached oracle labels
            if chunk_exists(work, "oracle_labels", oracle_hash):
                logger.info(f"Using cached oracle labels: {oracle_hash}")
                print(f"[green]📦 Using cached oracle labels: {oracle_hash}[/green]")
                rows = load_chunk(work, "oracle_labels", oracle_hash)

                # Oracle data already loaded from cache
            else:
                # Use cached judge scores for oracle labeling
                rows = judge_rows.copy()

                try:
                    with console.status(
                        f"[bold blue]Generating oracle labels | Hash: {oracle_hash}..."
                    ):
                        if oracle_fraction < 1.0:
                            # Cost-efficient mode: Only generate oracle labels for the specified fraction
                            print(
                                f"[blue]💰 Cost-efficient oracle mode: generating labels for {oracle_fraction:.1%} of samples[/blue]"
                            )
                            rows_with_oracle = add_oracle_labels(
                                rows,
                                provider=oracle_config.provider,
                                model_name=oracle_config.model_name,
                                fraction=oracle_fraction,
                                seed=oracle_config.seed,
                                template=getattr(
                                    oracle_config, "template", "quick_judge"
                                ),
                                temperature=getattr(oracle_config, "temperature", 0.0),
                                max_tokens=getattr(oracle_config, "max_tokens", 50),
                                score_field="y_true",  # Use y_true for calibration
                            )
                        else:
                            # Full experimental design: Generate oracle labels for all samples with holdout
                            print(
                                f"[blue]🧪 Full experimental design: generating labels for all samples with holdout[/blue]"
                            )
                            rows_with_oracle = add_full_oracle_labels_with_holdout(
                                rows,
                                provider=oracle_config.provider,
                                model_name=oracle_config.model_name,
                                logging_policy_oracle_fraction=oracle_fraction,
                                seed=oracle_config.seed,
                                template=getattr(
                                    oracle_config, "template", "quick_judge"
                                ),
                                temperature=getattr(oracle_config, "temperature", 0.0),
                                max_tokens=getattr(oracle_config, "max_tokens", 50),
                            )

                        # Replace rows with oracle-enabled version
                        rows = rows_with_oracle

                        # Save to modular cache
                        save_chunk(
                            work,
                            "oracle_labels",
                            oracle_hash,
                            rows,
                            metadata={
                                "provider": oracle_config.provider,
                                "model_name": oracle_config.model_name,
                                "oracle_fraction": oracle_fraction,
                                "sample_count": len(rows),
                                "oracle_mode": (
                                    "cost_efficient"
                                    if oracle_fraction < 1.0
                                    else "full_experimental"
                                ),
                            },
                        )

                        # Store oracle data in a variable to preserve it through calibration
                        oracle_data_backup = rows.copy()

                        logger.info(
                            f"Oracle labels generated and cached: {oracle_hash}"
                        )
                        print(
                            f"[green]✅ Oracle labels generated and cached: {oracle_hash}[/green]"
                        )

                except Exception as e:
                    logger.error(f"Oracle labeling failed: {e}")
                    print(f"[red]❌ Oracle labeling failed: {e}[/red]")
                    print("[yellow]Continuing without oracle analysis...[/yellow]")
                    # Use judge rows without oracle enhancement
                    rows = judge_rows.copy()

            # Count oracle labels (whether from cache or fresh generation)
            oracle_count = sum(1 for row in rows if row.get("y_true") is not None)
            total_samples = len(rows)
            oracle_fraction = oracle_config.logging_policy_oracle_fraction

            logger.info(
                f"Oracle labels available: {oracle_count}/{total_samples} samples ({oracle_count/total_samples:.1%})"
            )
            print(
                f"[green]✅ Oracle labels available: {oracle_count}/{total_samples} samples ({oracle_count/total_samples:.1%})[/green]"
            )

            if oracle_fraction < 1.0:
                print(
                    f"[blue]💰 Cost savings: {total_samples - oracle_count} fewer oracle API calls[/blue]"
                )
            else:
                holdout_count = sum(
                    1 for row in rows if row.get("oracle_holdout_mask", False)
                )
                available_count = sum(
                    1 for row in rows if row.get("oracle_available_to_logging", False)
                )
                print(
                    f"[blue]📊 Oracle breakdown: {available_count} for calibration, {holdout_count} held out for evaluation[/blue]"
                )

            # Prepare data for calibration:
            # - copy 'score' to 'score_raw' for calibration input
            # - handle different oracle data formats
            for row in rows:
                # Ensure score_raw field exists for calibration
                if "score" in row and "score_raw" not in row:
                    row["score_raw"] = row["score"]

                if oracle_fraction < 1.0:
                    # Cost-efficient mode: add_oracle_labels() puts oracle scores directly in y_true
                    # No additional processing needed - y_true is already set for oracle samples
                    pass
                else:
                    # Full experimental design: add_full_oracle_labels_with_holdout() uses holdout structure
                    # Set y_true ONLY for samples available to logging policy (for calibration training)
                    # Don't set reward yet - that will be done after calibration using the calibration model
                    if row.get("oracle_available_to_logging", False):
                        row["y_true"] = row["oracle_full"]

            # Save oracle-enhanced data back to judge file so calibration can use it
            judge_json.write_text("\n".join(json.dumps(r) for r in rows))
            logger.info(f"Oracle-enhanced judge scores saved for calibration")

        # 3c. calibrate judge scores if not present
        cal_json = work / f"{cfg.judge.template}_scores_cal.jsonl"
        cal_png = work / f"{cfg.judge.template}_reliability.png"
        if not cal_json.exists():
            with console.status("[bold blue]Calibrating judge scores..."):
                logger.info("Running judge calibration")

                # Call calibration logic directly instead of CLI function
                from ..calibration.isotonic import fit_isotonic, plot_reliability

                # Load judge scores
                rows = [json.loads(l) for l in judge_json.read_text().splitlines()]

                # Filter rows to only include those with both score_raw and y_true (for oracle holdout compatibility)
                calibration_rows = []
                cal_variances = []
                for r in rows:
                    y_true = r.get("y_true")
                    if "score_raw" in r and y_true is not None:
                        score_raw = r["score_raw"]
                        if isinstance(score_raw, dict):
                            mean = float(score_raw.get("mean", 0.0))
                            variance = float(score_raw.get("variance", 0.0))
                        else:
                            # Should not happen with unified system
                            mean = float(score_raw)
                            variance = 0.0
                        calibration_rows.append((mean, float(y_true)))
                        cal_variances.append(variance)

                if len(calibration_rows) == 0:
                    raise ValueError(
                        "No samples available for calibration - all y_true values are missing"
                    )

                cal_scores: "np.ndarray[Any, Any]" = np.array([score for score, _ in calibration_rows], dtype=float)  # type: ignore[type-arg]
                cal_y_true: "np.ndarray[Any, Any]" = np.array([y_true for _, y_true in calibration_rows], dtype=float)  # type: ignore[type-arg]

                logger.info(
                    f"Using {len(calibration_rows)}/{len(rows)} samples for calibration"
                )

                # Log if we have variance information
                if any(v > 0 for v in cal_variances):
                    logger.info(
                        f"Calibrating with uncertainty: {sum(v > 0 for v in cal_variances)} "
                        f"samples have non-zero variance"
                    )

                # Fit isotonic calibration
                iso = fit_isotonic(cal_scores, cal_y_true)

                # Apply calibration model to ALL samples to get predicted rewards
                for r in rows:
                    if "score_raw" in r:
                        # Get raw score
                        score_raw = r["score_raw"]
                        if isinstance(score_raw, dict):
                            raw_mean = float(score_raw.get("mean", 0.0))
                            raw_variance = float(score_raw.get("variance", 0.0))
                        else:
                            # Should not happen with unified system
                            raw_mean = float(score_raw)
                            raw_variance = 0.0

                        # Apply calibration to mean
                        cal_mean = float(iso.predict([raw_mean])[0])

                        # Create calibrated score preserving variance
                        from ..judge.schemas import JudgeScore

                        cal_score = JudgeScore(mean=cal_mean, variance=raw_variance)

                        # Update row with calibrated score
                        r["score_cal"] = {
                            "mean": cal_score.mean,
                            "variance": cal_score.variance,
                        }

                # Clear y_true used for calibration training - it should not be used for CJE estimation
                # Keep oracle_full for evaluation, but clear y_true to avoid confusion
                for r in rows:
                    if r.get("oracle_available_to_logging", False):
                        r["y_true"] = None  # Clear calibration training label

                # Validate calibration: check for collapse to constant value
                calibrated_scores = [r["score_cal"] for r in rows]
                cal_min = min(calibrated_scores)
                cal_max = max(calibrated_scores)
                cal_range = cal_max - cal_min

                # Threshold for detecting calibration collapse (adjust as needed)
                CALIBRATION_COLLAPSE_THRESHOLD = 0.05

                if cal_range < CALIBRATION_COLLAPSE_THRESHOLD:
                    logger.warning(
                        f"Calibration collapse detected! Range of calibrated scores: {cal_range:.6f} "
                        f"(min={cal_min:.6f}, max={cal_max:.6f}). This indicates insufficient calibration data "
                        f"or poor judge-oracle correlation. Falling back to raw scores + oracle labels."
                    )
                    print(
                        f"[yellow]⚠️  Calibration Collapse Detected![/yellow]\n"
                        f"[yellow]   Range of calibrated scores: {cal_range:.6f} (threshold: {CALIBRATION_COLLAPSE_THRESHOLD})[/yellow]\n"
                        f"[yellow]   This means calibration mapped all scores to nearly the same value.[/yellow]\n"
                        f"[yellow]   Root cause: Insufficient calibration data ({len(calibration_rows)} samples) or poor judge-oracle correlation.[/yellow]\n"
                        f"[yellow]   Falling back to: raw judge scores + oracle labels (no calibration).[/yellow]"
                    )

                    # Fallback: use raw scores + oracle labels instead of calibrated scores
                    for r in rows:
                        if (
                            r.get("oracle_available_to_logging", False)
                            and r.get("oracle_full") is not None
                        ):
                            # For oracle-labeled samples available to logging policy, use the oracle label as reward
                            r["reward"] = r["oracle_full"]
                        else:
                            # For non-oracle samples, use raw judge score as reward proxy
                            r["reward"] = r.get("score_raw", 0.0)

                        # Mark as fallback for transparency
                        r["calibration_fallback"] = True

                    logger.info(
                        "Applied calibration fallback: using raw scores + oracle labels"
                    )
                    print(
                        "[blue]🔧 Applied calibration fallback: using raw scores + oracle labels as rewards[/blue]"
                    )

                else:
                    logger.info(
                        f"Calibration successful: score range {cal_range:.4f} (min={cal_min:.4f}, max={cal_max:.4f})"
                    )
                    print(
                        f"[green]✅ Calibration successful: score range {cal_range:.4f}[/green]"
                    )

                # Oracle fields are already preserved in rows from oracle labeling phase

                # Save calibrated results (or fallback results) with oracle metadata preserved
                cal_json.write_text("\n".join(json.dumps(r) for r in rows))

                # Generate reliability plot
                plot_reliability(cal_scores, cal_y_true, iso, cal_png)

                logger.info(f"Calibration complete: {cal_json}")
        else:
            logger.info(f"Using existing calibrated scores: {cal_json}")

        # 4. estimator
        source_json = (
            cal_json
            if cal_json.exists()
            else (judge_json if judge_json.exists() else log_json)
        )
        logger.info(f"Using data source: {source_json}")

    with console.status("[bold blue]Loading experiment data..."):
        rows = [json.loads(l) for l in source_json.read_text().splitlines()]
        logger.info(f"Loaded {len(rows)} samples from {source_json}")

        # Apply centralized reward assignment logic
        from ..validation import assign_rewards_with_priority

        oracle_analysis_enabled = hasattr(cfg, "oracle") and getattr(
            cfg.oracle, "enabled", False
        )

        assign_rewards_with_priority(
            rows,
            source_description=str(source_json),
            oracle_analysis_enabled=oracle_analysis_enabled,
        )

        # Critical: Verify all rows have numeric rewards after assignment
        rows_without_reward = sum(1 for r in rows if r.get("reward") is None)
        if rows_without_reward > 0:
            raise ValueError(
                f"{rows_without_reward} rows still have reward=None after assign_rewards_with_priority. "
                f"This should never happen - the function should raise an error instead."
            )

    # Determine analysis type based on available ground truth
    analysis_type, ground_truth_count = _determine_analysis_type(
        rows, oracle_analysis_enabled
    )

    logger.info(
        f"Analysis type: {analysis_type} ({ground_truth_count}/{len(rows)} samples with ground truth)"
    )

    if analysis_type == "llm_comparison":
        logger.warning(
            "No ground truth labels detected. This will be LLM judge comparison, "
            "not causal inference. Results may not reflect real user value."
        )
        print(
            "[yellow]⚠️  Warning: No ground truth labels detected.[/yellow]\n"
            "[yellow]This will compare LLM judge scores, not real business outcomes.[/yellow]\n"
            "[yellow]For causal inference, provide data with y_true labels corresponding to specific responses.[/yellow]"
        )
    elif analysis_type == "causal_inference_sparse":
        logger.info(
            f"Limited ground truth: {ground_truth_count}/{len(rows)} samples have labels. "
            f"Causal inference will proceed but may be less precise."
        )
        print(
            f"[blue]ℹ️  Note: {ground_truth_count}/{len(rows)} samples have ground truth labels.[/blue]\n"
            "[blue]Causal inference will proceed. More labels would improve precision.[/blue]"
        )
    else:  # causal_inference
        logger.info(
            f"Ground truth labels detected in {ground_truth_count}/{len(rows)} samples - performing causal inference"
        )
        print(
            f"[green]✅ Ground truth labels detected in {ground_truth_count}/{len(rows)} samples - performing causal inference[/green]"
        )

    try:
        # Early validation before reward assignment
        from ..validation import (
            validate_core_fields,
            validate_response_reward_correspondence,
        )

        validate_core_fields(rows, str(source_json))
        validate_response_reward_correspondence(rows, str(source_json))

        # Full validation after reward assignment
        validate_pipeline_data(rows, str(source_json), stage="pre_estimation")
    except ValidationError as e:
        logger.error(f"Data validation failed: {e}")
        raise ValueError(str(e))

    logger.info("Data validation complete")

    # --- compute logp_target under target policy π′ ---
    logger.info("Starting target policy π′ computation")

    # NEW: Use target_policies (plural) instead of target_policy (singular)
    target_policies_cfg = getattr(cfg, "target_policies", None)

    # Validate that target_policies is provided and not empty
    if not target_policies_cfg:
        raise ValueError(
            "target_policies cannot be empty. At least one target policy must be specified. "
            "Please update your configuration to use 'target_policies' instead of 'target_policy'."
        )

    K = len(target_policies_cfg)
    logger.info(f"Detected {K} target policies")

    # NEW: Detect pre-computed target log-probabilities --------------------
    policy_names: List[str] = []
    for i, p in enumerate(target_policies_cfg):
        if hasattr(p, "name"):
            policy_names.append(str(getattr(p, "name")))
        elif isinstance(p, dict) and "name" in p:
            policy_names.append(str(p["name"]))
        else:
            policy_names.append(f"policy_{i}")

    def _has_mapping(r: Dict[str, Any]) -> bool:
        # Check if we have pre-computed logp_target_all
        lt_all = r.get("logp_target_all")
        # Support both list (legacy) and dict (preferred) formats
        if isinstance(lt_all, dict):
            # Dict format: must have all policy names as keys
            return all(name in lt_all for name in policy_names)
        elif isinstance(lt_all, list):
            # List format: must have correct length (less safe, ordering dependent)
            return len(lt_all) == len(policy_names)
        return False

    precomputed_available = all(_has_mapping(r) for r in rows)

    precomputed_lookup: Dict[Tuple[str, str], List[float]] = {}
    sample_lookup: Dict[str, List[List[str]]] = {}
    sampler: Union[PrecomputedMultiTargetSampler, Any]  # Will be assigned either type

    if precomputed_available:
        logger.info(
            "Detected pre-computed logp_target_all – skipping API log-prob computation"
        )
        from ..loggers.precomputed_sampler import PrecomputedMultiTargetSampler

        for row in rows:
            ctx = str(row["context"])
            resp = str(row["response"])

            # Extract pre-computed log probabilities
            lt_all = row["logp_target_all"]
            if isinstance(lt_all, dict):
                # Dict format (preferred): explicit policy name mapping
                logps_vec = [float(lt_all[name]) for name in policy_names]
            else:
                # List format (legacy): assumes same order as policy_names
                lps = cast(List[Any], lt_all)
                logps_vec = [float(lp) for lp in lps]
                logger.warning(
                    f"Using list format for logp_target_all - consider switching to dict format "
                    f"with explicit policy names for better reliability"
                )
            precomputed_lookup[(ctx, resp)] = logps_vec

            # ---------- target samples (optional) ----------
            ts_map = row.get("target_samples", {})
            per_policy_resps: List[List[str]] = []
            for name in policy_names:
                if isinstance(ts_map, dict):
                    lst = ts_map.get(name, [])
                else:
                    lst = []
                if isinstance(lst, list):
                    per_policy_resps.append([str(x) for x in lst])
                else:
                    per_policy_resps.append([str(lst)])
            sample_lookup[ctx] = per_policy_resps

        sampler = PrecomputedMultiTargetSampler(
            precomputed_lookup, K, sample_lookup=sample_lookup
        )

        # Validate target log probabilities directly
        logger.info("Validating pre-computed target log probabilities ...")
        try:
            validate_target_policy_computation(rows)
        except ValidationError as e:
            logger.error(f"Pre-computed target policy validation failed: {e}")
            raise ValueError(str(e))
    else:
        # Fallback to live computation with MultiTargetSampler
        # Always use multi-policy infrastructure (single policy is just K=1)
        logger.info("Setting up multi-policy evaluation infrastructure")

        # Compute hash for target log probabilities configuration
        target_logprobs_hash = compute_target_logprobs_hash(
            contexts_hash=contexts_hash, target_policies_config=target_policies_cfg
        )

        # Check for cached target log probabilities
        if chunk_exists(work, "target_logprobs", target_logprobs_hash):
            logger.info(
                f"Using cached target log probabilities: {target_logprobs_hash}"
            )
            print(
                f"[green]📦 Using cached target log probabilities: {target_logprobs_hash}[/green]"
            )

            rows_with_target_logprobs = load_chunk(
                work, "target_logprobs", target_logprobs_hash
            )

            # Validate that we have the right number of samples and policy names
            if len(rows_with_target_logprobs) != len(rows):
                logger.warning(
                    f"Cache mismatch: expected {len(rows)} samples, got {len(rows_with_target_logprobs)}. Recomputing..."
                )
                cached_target_logprobs = False
            else:
                # Update rows with cached target log probabilities
                for i, cached_row in enumerate(rows_with_target_logprobs):
                    rows[i]["logp_target_all"] = cached_row["logp_target_all"]
                    if "target_samples" in cached_row:
                        rows[i]["target_samples"] = cached_row["target_samples"]
                cached_target_logprobs = True
        else:
            cached_target_logprobs = False

        if not cached_target_logprobs:
            # Create MultiTargetSampler for all target policies (works for K=1 too)
            with console.status(
                f"[bold blue]Creating multi-target sampler | Hash: {target_logprobs_hash}..."
            ):
                # P2: Extract diagnostics config to pass to sampler
                diagnostics_config: Optional[Dict[str, Any]] = None
                if hasattr(cfg, "diagnostics"):
                    diagnostics_container = OmegaConf.to_container(
                        cfg.diagnostics, resolve=True
                    )
                    if isinstance(diagnostics_container, dict):
                        diagnostics_config = {
                            str(k): v for k, v in diagnostics_container.items()
                        }

                multi_sampler = make_multi_sampler(
                    target_policies_cfg, diagnostics_config
                )
                logger.info(
                    f"Created MultiTargetSampler for {multi_sampler.K} policies | Hash: {target_logprobs_hash}"
                )

            # Compute log probabilities for all target policies
            logger.info(
                f"Computing log probabilities for {K} target policies on {len(rows)} samples | Hash: {target_logprobs_hash}"
            )

            contexts = [str(row["context"]) for row in rows]
            responses = [str(row["response"]) for row in rows]

            # Get log probability matrix (n, K)
            with console.status(
                f"[bold blue]Computing multi-policy log probabilities | Hash: {target_logprobs_hash}..."
            ):
                logp_matrix = multi_sampler.logp_matrix(contexts, responses)
                logger.info(f"Computed log probability matrix: {logp_matrix.shape}")

            # Store log probabilities for each policy in the rows
            for i, row in enumerate(rows):
                # Store as dict for explicit policy name mapping
                row["logp_target_all"] = {
                    policy_names[j]: float(logp_matrix[i, j]) for j in range(K)
                }

            # Save to modular cache
            save_chunk(
                work,
                "target_logprobs",
                target_logprobs_hash,
                rows,
                metadata={
                    "policy_names": policy_names,
                    "num_policies": K,
                    "sample_count": len(rows),
                    "target_policies_config": target_policies_cfg,
                },
            )

            logger.info(
                f"Target log probabilities computed and cached: {target_logprobs_hash}"
            )
            print(
                f"[green]✅ Target log probabilities computed and cached: {target_logprobs_hash}[/green]"
            )

        # Create the sampler (either from cached data or fresh computation)
        if cached_target_logprobs:
            # Build precomputed sampler from cached data
            from ..loggers.precomputed_sampler import PrecomputedMultiTargetSampler

            precomputed_lookup = {}
            sample_lookup = {}
            for row in rows:
                ctx = str(row["context"])
                resp = str(row["response"])

                lt_all = row["logp_target_all"]
                logps_vec = [float(lt_all[name]) for name in policy_names]
                precomputed_lookup[(ctx, resp)] = logps_vec

                # Handle target samples if available
                ts_map = row.get("target_samples", {})
                per_policy_resps = []
                for name in policy_names:
                    if isinstance(ts_map, dict):
                        lst = ts_map.get(name, [])
                    else:
                        lst = []
                    if isinstance(lst, list):
                        per_policy_resps.append([str(x) for x in lst])
                    else:
                        per_policy_resps.append([str(lst)])
                sample_lookup[ctx] = per_policy_resps

            sampler = PrecomputedMultiTargetSampler(
                precomputed_lookup, K, sample_lookup=sample_lookup
            )
        else:
            sampler = multi_sampler

        # Validate target log probabilities
        logger.info("Validating target log probabilities...")
        try:
            validate_target_policy_computation(rows)
        except ValidationError as e:
            logger.error(f"Target policy validation failed: {e}")
            raise ValueError(str(e))
        logger.info("Target policy validation complete")

    # Get estimator configuration
    estimator_cfg = cfg.estimator
    est_kwargs: Dict[str, Any] = {}
    if estimator_cfg:
        container = OmegaConf.to_container(estimator_cfg, resolve=True)
        if isinstance(container, dict):
            # Convert to Dict[str, Any] type with explicit casting
            est_kwargs = {str(k): v for k, v in container.items()}

    # Add judge runner and work directory to estimator kwargs if available (for DR-CPO/MRDR)
    if "judge_runner" in locals() and est_kwargs.get("name", "DRCPO") in [
        "DRCPO",
        "MRDR",
    ]:
        est_kwargs["judge_runner"] = judge_runner
        est_kwargs["work_dir"] = str(work)

    base_est_name = est_kwargs.pop("name", "DRCPO") if est_kwargs else "DRCPO"

    # Always use unified multi-policy infrastructure (single policy is just K=1)
    logger.info(
        f"Using unified multi-policy infrastructure for {base_est_name.upper()}"
    )

    # Use simplified estimator names directly
    est_name = base_est_name.upper()

    # Automatic fallback: MRDR requires samples_per_policy > 0 to work correctly
    # If samples_per_policy = 0, MRDR degenerates to difference estimates instead of absolute values
    # In this case, automatically fall back to DRCPO which handles samples_per_policy = 0 correctly
    if est_name == "MRDR" and est_kwargs.get("samples_per_policy", 2) == 0:
        logger.warning(
            "⚠️  MRDR with samples_per_policy=0 produces difference estimates instead of absolute values. "
            "Automatically switching to DRCPO which handles samples_per_policy=0 correctly."
        )
        est_name = "DRCPO"
        print(
            f"[yellow]⚠️  Auto-switched from MRDR to DRCPO due to samples_per_policy=0[/yellow]"
        )

    # Validate estimator name
    valid_estimators = ["DRCPO", "IPS", "SNIPS", "MRDR"]
    if est_name not in valid_estimators:
        raise ValueError(
            f"Estimator '{base_est_name}' is not supported. "
            f"Available estimators: {', '.join(valid_estimators)}"
        )

    # Pass the appropriate sampler to the estimator
    est_kwargs["sampler"] = sampler

    # Use introspection to filter parameters automatically
    estimator_cls = get_estimator.__globals__["_ESTIMATORS"][est_name]
    sig = inspect.signature(estimator_cls.__init__)

    # Filter to only include parameters that the estimator accepts
    filtered_kwargs = {}
    for param_name, param_value in est_kwargs.items():
        if param_name in sig.parameters:
            filtered_kwargs[param_name] = param_value
        else:
            logger.warning(
                f"Ignoring unsupported parameter '{param_name}' for {est_name} estimator"
            )

    est_kwargs = filtered_kwargs

    logger.info(f"Using estimator: {est_name}")
    logger.info(f"Estimator config: {est_kwargs}")

    # Sanity checklist before estimation (from code review)
    logger.info("Running pre-estimation sanity checks...")

    # Check 1: All rows must have numeric rewards
    assert all(
        "reward" in r and isinstance(r["reward"], (int, float)) for r in rows
    ), "All rows must have numeric rewards for estimation"

    # Check 2: All rows must have target policy log probabilities
    assert all(
        "logp_target_all" in r and len(r["logp_target_all"]) == K for r in rows
    ), f"All rows must have logp_target_all with {K} policies"

    # Check 3: For DR estimators, verify we have target samples if configured
    if est_name in {"DRCPO", "MRDR"} and est_kwargs.get("samples_per_policy", 1) > 0:
        # This is a soft check - we don't require target samples in all cases
        # (e.g., when using precomputed sampler or API policies that don't support sampling)
        target_sample_count = sum(1 for r in rows if r.get("target_samples", {}))
        if target_sample_count == 0:
            logger.warning(
                f"{est_name} configured with samples_per_policy={est_kwargs.get('samples_per_policy', 1)} "
                f"but no target samples found. Will fall back to context-only predictions."
            )
        else:
            logger.info(
                f"Found target samples in {target_sample_count}/{len(rows)} rows"
            )

    logger.info("Pre-estimation sanity checks passed")

    with console.status(f"[bold blue]Initializing {est_name} estimator..."):
        est = get_estimator(est_name, **est_kwargs)

    # Fit the estimator
    with console.status(f"[bold blue]Fitting estimator..."):
        logger.info("Fitting estimator to data")
        est.fit(rows, sampler=sampler)

    # Extract and save target samples from DR estimators
    if est_name in ["DRCPO", "MRDR"] and hasattr(est, "_all_target_samples"):
        logger.info("Extracting target samples from DR estimator")

        # Group target samples by context
        target_samples_by_context: Dict[str, Dict[str, List[str]]] = {}
        for sample in est._all_target_samples:
            ctx = sample["context"]
            policy_idx = sample["policy_idx"]
            response = sample["response"]

            if ctx not in target_samples_by_context:
                target_samples_by_context[ctx] = {name: [] for name in policy_names}

            # Map policy index to policy name
            if policy_idx < len(policy_names):
                policy_name = policy_names[policy_idx]
                target_samples_by_context[ctx][policy_name].append(response)

        # Update rows with target samples
        samples_added = 0
        for row in rows:
            ctx = row["context"]
            if ctx in target_samples_by_context:
                if "target_samples" not in row or not isinstance(
                    row["target_samples"], dict
                ):
                    row["target_samples"] = {}
                # Merge with existing samples if any
                for policy_name, samples in target_samples_by_context[ctx].items():  # type: ignore[assignment]
                    if policy_name not in row["target_samples"]:
                        row["target_samples"][policy_name] = []  # type: ignore[assignment]
                    # Ensure it's a list before extending
                    if not isinstance(row["target_samples"][policy_name], list):
                        row["target_samples"][policy_name] = []  # type: ignore[assignment]
                    row["target_samples"][policy_name].extend(samples)  # type: ignore[arg-type]
                    samples_added += len(samples)

        if samples_added > 0:
            logger.info(f"Added {samples_added} target samples to rows")

            # Re-save to cache with updated target samples
            if "target_logprobs_hash" in locals() and chunk_exists(
                work, "target_logprobs", target_logprobs_hash
            ):
                logger.info("Updating cache with target samples")
                save_chunk(
                    work,
                    "target_logprobs",
                    target_logprobs_hash,
                    rows,
                    metadata={
                        "policy_names": policy_names,
                        "num_policies": K,
                        "sample_count": len(rows),
                        "target_policies_config": target_policies_cfg,
                        "has_target_samples": True,
                    },
                )

    with console.status(f"[bold blue]Computing estimate..."):
        logger.info("Computing final estimate")
        res = est.estimate()

    # Save results with JSON serialization for numpy arrays
    result_file = work / "result.json"

    # Compute logging policy value using appropriate estimator
    if oracle_analysis_enabled:
        # Use doubly-robust estimator for logging policy value with partial oracle coverage
        logger.info("Computing logging policy value using doubly-robust estimator")

        # Get oracle inclusion probability
        oracle_fraction = getattr(cfg.oracle, "logging_policy_oracle_fraction", 0.25)

        # Count actual oracle coverage to verify
        oracle_available_count = sum(
            1 for row in rows if row.get("oracle_available_to_logging", False)
        )
        actual_oracle_fraction = (
            oracle_available_count / len(rows) if len(rows) > 0 else 0.0
        )

        logger.info(
            f"Oracle inclusion: configured={oracle_fraction:.3f}, actual={actual_oracle_fraction:.3f} ({oracle_available_count}/{len(rows)})"
        )

        # Validate oracle inclusion probability
        if actual_oracle_fraction == 0:
            logger.error("No oracle samples found - cannot compute DR estimate")
            raise ValueError(
                "Doubly-robust estimation requires at least some oracle samples"
            )

        # Use actual fraction if significantly different (handles random sampling variation)
        if abs(oracle_fraction - actual_oracle_fraction) > 0.1:
            logger.warning(
                f"Large discrepancy between configured ({oracle_fraction:.3f}) and actual ({actual_oracle_fraction:.3f}) oracle fraction"
            )
            logger.info("Using actual oracle fraction for DR estimation")
            pi_i = actual_oracle_fraction
        else:
            pi_i = oracle_fraction

        # Compute doubly-robust terms: m_i + (Z_i/π_i)(Y_i - m_i)
        # This gives an unbiased estimate of V(π_logging) even with partial oracle coverage
        # - m_i: calibrated cheap judge (outcome model prediction)
        # - (Z_i/π_i)(Y_i - m_i): inverse-probability weighted bias correction
        logger.info(f"Computing DR terms with inclusion probability π = {pi_i:.3f}")

        dr_terms: List[float] = []
        n_oracle_used = 0
        bias_corrections: List[float] = []

        for row in rows:
            m_i_raw = row.get("score_cal")
            m_i: float = (
                float(cast(Union[int, float], m_i_raw)) if m_i_raw is not None else 0.0
            )  # Calibrated cheap judge score
            Z_i = row.get(
                "oracle_available_to_logging", False
            )  # Oracle availability indicator

            if Z_i and pi_i > 0:
                # Oracle available: add bias correction term
                Y_i_raw = row.get("oracle_full")
                Y_i: float = (
                    float(cast(Union[int, float], Y_i_raw))
                    if Y_i_raw is not None
                    else 0.0
                )  # True oracle score
                bias_correction: float = (Y_i - m_i) / pi_i
                dr_term = m_i + bias_correction
                bias_corrections.append(bias_correction)
                n_oracle_used += 1
                # Note: We use m_i (calibrated judge) as the outcome model prediction
                # and Y_i (oracle) for bias correction. This is correct DR estimation.
                # No double-counting occurs because the DR formula properly combines them.
            else:
                # No oracle: just use calibrated score (bias correction = 0)
                dr_term = m_i

            dr_terms.append(dr_term)

        # Log bias correction statistics
        if bias_corrections:
            mean_bias_correction = sum(bias_corrections) / len(bias_corrections)
            logger.info(
                f"Mean bias correction: {mean_bias_correction:.4f} (applied to {len(bias_corrections)} samples)"
            )

        # Compute DR estimate and standard error
        logging_policy_value = sum(dr_terms) / len(dr_terms) if dr_terms else 0.0

        # Compute standard error using influence function
        if len(dr_terms) > 1:
            mean_dr = logging_policy_value
            variance = sum((term - mean_dr) ** 2 for term in dr_terms) / (
                len(dr_terms) - 1
            )
            logging_policy_se = (variance / len(dr_terms)) ** 0.5
        else:
            logging_policy_se = 0.0

        logger.info(
            f"Logging policy DR estimate: {logging_policy_value:.4f} ± {logging_policy_se:.4f}"
        )
        logger.info(
            f"Oracle samples used: {n_oracle_used}/{len(rows)}, inclusion probability: {pi_i:.3f}"
        )

    else:
        # For standard analysis: use all available rewards
        reward_values: List[float] = [
            float(row.get("reward", 0.0))
            for row in rows
            if row.get("reward") is not None
        ]
        logging_policy_value = (
            sum(reward_values) / len(reward_values) if reward_values else 0.0
        )

        # Compute standard error for empirical mean
        if len(reward_values) > 1:
            variance = sum((r - logging_policy_value) ** 2 for r in reward_values) / (
                len(reward_values) - 1
            )
            logging_policy_se = (variance / len(reward_values)) ** 0.5
        else:
            logging_policy_se = 0.0

        logger.info(
            f"Logging policy empirical estimate: {logging_policy_value:.4f} ± {logging_policy_se:.4f} from {len(reward_values)} samples"
        )

    # Calculate total runtime
    end_time: float = time.time()
    total_runtime: float = end_time - start_time

    # Create user-friendly results format
    from ..results.user_friendly import create_user_friendly_result

    # Prepare metadata for user-friendly formatting
    experiment_metadata = {
        "runtime_seconds": total_runtime,
        **to_jsonable(res).get("metadata", {}),
    }

    # Create user-friendly result
    user_friendly_result = create_user_friendly_result(
        estimation_result=res,
        policy_names=policy_names,
        logging_policy_value=logging_policy_value,
        logging_policy_se=logging_policy_se,
        analysis_type=analysis_type,
        **experiment_metadata,
    )

    # ==========================================
    # SAVE RESULTS: 2 Files for Different Audiences
    # ==========================================

    # 1. DECISION-ORIENTED RESULTS (result.json)
    # → Business stakeholders, product managers, decision makers
    # → Executive summary, recommendations, key findings
    # → Bootstrap CIs automatically enabled for small samples (n < 100)
    logger.info("Saving decision-oriented results for business stakeholders")
    with open(result_file, "w") as f:
        json.dump(to_jsonable(user_friendly_result), f, indent=2)

    # 2. TECHNICAL/DEBUGGING RESULTS (result_technical.json)
    # → Researchers, data scientists, algorithm developers
    # → Raw estimates, covariance matrices, detailed diagnostics
    logger.info("Saving technical results for researchers and debugging")
    res_dict = to_jsonable(res)
    # Remove oracle evaluation data if present (no longer needed)
    if "oracle_evaluation" in res_dict:
        del res_dict["oracle_evaluation"]

    technical_result = {
        **res_dict,
        "v_hat_logging": logging_policy_value,
        "se_logging": logging_policy_se,
        "policy_names": ["logging"] + policy_names,
        "analysis_type": analysis_type,
        "runtime_seconds": total_runtime,
        "interpretation": (
            "Expected improvement in real business outcomes"
            if analysis_type == "causal_inference"
            else (
                "Expected improvement in real business outcomes (limited calibration data)"
                if analysis_type == "causal_inference_sparse"
                else "Difference in LLM judge scores (not validated against real outcomes)"
            )
        ),
        "warning": (
            None
            if analysis_type == "causal_inference"
            else (
                "Results based on limited ground truth data - consider collecting more labels for higher precision"
                if analysis_type == "causal_inference_sparse"
                else "This is LLM comparison, not causal inference. Results may not reflect real user value."
            )
        ),
    }

    technical_result_file = work / "result_technical.json"
    with open(technical_result_file, "w") as f:
        json.dump(technical_result, f, indent=2)

    logger.info(f"Experiment complete! Total runtime: {total_runtime:.2f} seconds")
    logger.info(f"Decision-oriented results: {result_file}")
    logger.info(f"Technical results: {technical_result_file}")
    logger.info(f"Final estimate: {res}")

    print(f"[green]✅ Experiment completed in {total_runtime:.2f} seconds[/green]")

    if analysis_type == "causal_inference":
        print(f"[bold green]Causal Inference Result[/bold green]: {to_jsonable(res)}")
        print(
            "[blue]✅ Results represent expected improvement in real business outcomes[/blue]"
        )
    elif analysis_type == "causal_inference_sparse":
        print(
            f"[bold blue]Causal Inference Result (Limited Data)[/bold blue]: {to_jsonable(res)}"
        )
        print(
            "[blue]ℹ️  Results based on limited ground truth - more labels would improve precision[/blue]"
        )
    else:
        print(f"[bold yellow]LLM Comparison Result[/bold yellow]: {to_jsonable(res)}")
        print(
            "[yellow]⚠️  Results show LLM judge score differences, not real user value[/yellow]"
        )

    print(f"[blue]📊 Decision-oriented results: {result_file}[/blue]")
    print(f"[blue]🔧 Technical results: {technical_result_file}[/blue]")

    # Show bootstrap info for small samples
    if res.n < 100 and res.eif_components is not None:
        print(
            f"[green]🎯 Bootstrap CIs automatically enabled for small sample (n={res.n})[/green]"
        )

    # Check for reliability warnings
    from ..estimators.reliability import (
        create_reliability_warning_message,
        assess_ci_reliability,
        EstimatorMetadata,
    )

    try:
        # Create structured metadata from result
        structured_metadata = None
        if res.metadata:
            try:
                structured_metadata = EstimatorMetadata(**res.metadata)
            except (TypeError, ValueError):
                structured_metadata = EstimatorMetadata(
                    estimator_type=res.estimator_type,
                    bootstrap_available=res.eif_components is not None,
                )

        # Get bootstrap results from user-friendly result if available
        bootstrap_results = user_friendly_result.get("robust_inference", {}).get(
            "bootstrap_confidence_intervals"
        )

        # Assess reliability
        reliability_assessment = assess_ci_reliability(
            result=res,
            bootstrap_results=bootstrap_results,
            metadata=structured_metadata,
        )

        # Show warning if needed
        if reliability_assessment.rating in ["low", "unreliable"]:
            warning_message = create_reliability_warning_message(reliability_assessment)
            if warning_message:
                from rich.panel import Panel

                console.print("")  # Add spacing
                console.print(
                    Panel(
                        warning_message,
                        style=(
                            "red"
                            if reliability_assessment.rating == "unreliable"
                            else "yellow"
                        ),
                        title="⚠️  CI Reliability Warning",
                    )
                )

    except Exception as e:
        # Don't fail the entire experiment if reliability assessment fails
        logger.warning(f"Reliability assessment failed: {e}")

    # Update temporary file
    with open(_WORK_DIR_FILE, "w") as f:
        f.write(str(work))


def get_last_work_dir() -> Path:
    """Get the work directory from the last run of the experiment."""
    if not _WORK_DIR_FILE.exists():
        raise RuntimeError("No experiment has been run yet")
    with open(_WORK_DIR_FILE, "r") as f:
        return Path(f.read())


def to_jsonable(obj: Any) -> Any:
    """Convert numpy arrays and other non-JSON types to JSON-serializable format."""
    # Check if it's an EstimationResult object
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable(item) for item in obj]
    else:
        return obj


# No need for a typer app since this function is imported directly
