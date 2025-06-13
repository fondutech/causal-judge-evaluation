from __future__ import annotations

from typing import Sequence, List, Dict, Any, Callable, Optional, Union
import logging
import time
import concurrent.futures
from dataclasses import dataclass, field

from ..estimators.drcpo import MultiDRCPOEstimator
from ..loggers.multi_target_sampler import MultiTargetSampler
from ..judge.base import JudgeProtocol
from ..judge.factory import JudgeFactory

logger = logging.getLogger(__name__)


@dataclass
class PromptTuningConfig:
    """Configuration for prompt tuning."""

    max_workers: int = 4  # Parallel evaluation of prompts
    timeout: int = 300  # Timeout per prompt evaluation in seconds
    min_improvement: float = (
        0.001  # Minimum variance improvement to consider significant
    )
    early_stopping: bool = True  # Stop if no improvement for several iterations
    patience: int = 3  # Number of iterations without improvement before stopping
    save_results: bool = True  # Save detailed results
    estimator_config: Dict[str, Any] = field(default_factory=lambda: {"k": 3})


@dataclass
class PromptResult:
    """Result of evaluating a single prompt."""

    prompt: str
    variance: float
    v_hat: float
    ci_width: float
    n_samples: int
    evaluation_time: float
    error: Optional[str] = None


@dataclass
class TuningResult:
    """Complete result of prompt tuning."""

    best_prompt: str
    best_variance: float
    results: List[PromptResult]
    total_time: float
    n_prompts_evaluated: int
    early_stopped: bool = False


def tune_judge_prompt(
    prompt_grid: Sequence[str],
    logs: List[Dict[str, Any]],
    sampler: MultiTargetSampler,
    judge_type: str,
    judge_config: Dict[str, Any],
    config: Optional[PromptTuningConfig] = None,
) -> Optional[str]:
    """Return the prompt that minimises DR-CPO variance.

    Parameters
    ----------
    prompt_grid : list of prompt templates
        Candidate judge prompts.
    logs : list of log dictionaries
        Logged interactions with keys ``context`` and ``response``.
    sampler : MultiTargetSampler
        Sampler for the target policy used by ``DRCPOEstimator``.
    judge_type : str
        Type of judge ("openai", "anthropic", "prometheus", etc.)
    judge_config : dict
        Configuration for the judge.
    config : PromptTuningConfig, optional
        Configuration for the tuning process.
    """
    if not prompt_grid:
        return None

    config = config or PromptTuningConfig()
    result = tune_judge_prompt_detailed(
        prompt_grid, logs, sampler, judge_type, judge_config, config
    )
    return result.best_prompt if result else None


def tune_judge_prompt_detailed(
    prompt_grid: Sequence[str],
    logs: List[Dict[str, Any]],
    sampler: MultiTargetSampler,
    judge_type: str,
    judge_config: Dict[str, Any],
    config: PromptTuningConfig,
) -> Optional[TuningResult]:
    """Detailed prompt tuning with full results and configuration options."""
    start_time = time.time()
    results: List[PromptResult] = []

    best_prompt = prompt_grid[0]
    best_variance = float("inf")
    no_improvement_count = 0

    logger.info(f"Starting prompt tuning with {len(prompt_grid)} candidates")

    def evaluate_prompt(prompt: str) -> PromptResult:
        """Evaluate a single prompt."""
        prompt_start = time.time()
        try:
            # Create judge with custom template
            judge_config_with_prompt = judge_config.copy()

            # Extract provider and model from judge_type or config
            if judge_type in ["openai", "anthropic", "google"]:
                provider = judge_type
                model = judge_config_with_prompt.get("model", None)
                if not model:
                    # Set default models for each provider
                    default_models = {
                        "openai": "gpt-4o-mini",
                        "anthropic": "claude-3-haiku-20240307",
                        "google": "gemini-1.5-flash",
                    }
                    model = default_models.get(provider, "gpt-4o-mini")
            else:
                # Extract from config if not specified directly
                provider = judge_config_with_prompt.get("provider", "openai")
                model = judge_config_with_prompt.get("model", "gpt-4o-mini")

            # Remove provider/model from config to avoid duplication
            judge_config_with_prompt.pop("provider", None)
            judge_config_with_prompt.pop("model", None)

            # Set the custom template
            judge_config_with_prompt["template"] = prompt

            judge = JudgeFactory.create(
                provider=provider,
                model=model,
                use_cache=False,
                **judge_config_with_prompt,
            )

            scored = []
            for row in logs:
                r = judge.score(row["context"], row["response"])
                new = dict(row)
                new["reward"] = r
                scored.append(new)

            est = MultiDRCPOEstimator(sampler=sampler, **config.estimator_config)
            est.fit(scored)
            result = est.estimate()

            # Extract the first policy's statistics using new API
            v_hat_val = float(result.v_hat[0])
            variance_val = float(result.se[0] ** 2)  # variance is se squared
            ci_lower, ci_upper = result.confidence_interval(0.95)
            ci_width_val = float(ci_upper[0] - ci_lower[0])

            return PromptResult(
                prompt=prompt,
                variance=variance_val,
                v_hat=v_hat_val,
                ci_width=ci_width_val,
                n_samples=result.n,
                evaluation_time=time.time() - prompt_start,
            )

        except Exception as e:
            logger.warning(f"Error evaluating prompt '{prompt[:50]}...': {e}")
            return PromptResult(
                prompt=prompt,
                variance=float("inf"),
                v_hat=0.0,
                ci_width=float("inf"),
                n_samples=0,
                evaluation_time=time.time() - prompt_start,
                error=str(e),
            )

    # Evaluate prompts in parallel
    if config.max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_workers
        ) as executor:
            future_to_prompt = {
                executor.submit(evaluate_prompt, prompt): prompt
                for prompt in prompt_grid
            }

            for future in concurrent.futures.as_completed(
                future_to_prompt, timeout=config.timeout
            ):
                result = future.result()
                results.append(result)

                # Check for improvement
                if result.variance < best_variance - config.min_improvement:
                    best_variance = result.variance
                    best_prompt = result.prompt
                    no_improvement_count = 0
                    logger.info(
                        f"New best prompt (var={best_variance:.6f}): {result.prompt[:50]}..."
                    )
                else:
                    no_improvement_count += 1

                # Early stopping
                if config.early_stopping and no_improvement_count >= config.patience:
                    logger.info(
                        f"Early stopping after {no_improvement_count} iterations without improvement"
                    )
                    # Cancel remaining futures
                    for remaining_future in future_to_prompt:
                        if not remaining_future.done():
                            remaining_future.cancel()
                    break
    else:
        # Sequential evaluation
        for prompt in prompt_grid:
            result = evaluate_prompt(prompt)
            results.append(result)

            if result.variance < best_variance - config.min_improvement:
                best_variance = result.variance
                best_prompt = result.prompt
                no_improvement_count = 0
                logger.info(
                    f"New best prompt (var={best_variance:.6f}): {result.prompt[:50]}..."
                )
            else:
                no_improvement_count += 1

            if config.early_stopping and no_improvement_count >= config.patience:
                logger.info(
                    f"Early stopping after {no_improvement_count} iterations without improvement"
                )
                break

    total_time = time.time() - start_time
    early_stopped = (
        no_improvement_count >= config.patience if config.early_stopping else False
    )

    tuning_result = TuningResult(
        best_prompt=best_prompt,
        best_variance=best_variance,
        results=results,
        total_time=total_time,
        n_prompts_evaluated=len(results),
        early_stopped=early_stopped,
    )

    logger.info(
        f"Prompt tuning completed in {total_time:.2f}s. Best variance: {best_variance:.6f}"
    )

    return tuning_result


def generate_prompt_variants(
    base_template: str, variations: Optional[Dict[str, List[str]]] = None
) -> List[str]:
    """Generate prompt variants by substituting different components.

    Parameters
    ----------
    base_template : str
        Base template with placeholders like {instruction_style}, {output_format}
    variations : dict, optional
        Dict mapping placeholder names to lists of alternatives

    Returns
    -------
    list of str
        Generated prompt variants
    """
    if variations is None:
        variations = {
            "instruction_style": [
                "You are an expert evaluator.",
                "You are a helpful assistant tasked with evaluation.",
                "You are an experienced judge.",
                "Please carefully evaluate the following.",
            ],
            "output_format": [
                'Respond with JSON: {"score": <float>}.',
                "Provide only a number 1-10:",
                "Rate on a scale of 1-10 and provide only the number:",
                "Output format: SCORE=<number>",
            ],
            "evaluation_criteria": [
                "Rate the quality of the RESPONSE to the CONTEXT",
                "Evaluate how helpful and accurate the response is",
                "Assess the relevance and usefulness of the response",
                "Judge the overall quality and appropriateness",
            ],
        }

    import itertools

    # Extract placeholder names from template
    import re

    placeholders = re.findall(r"\{(\w+)\}", base_template)

    # Generate all combinations
    variant_values = []
    for placeholder in placeholders:
        if placeholder in variations:
            variant_values.append(variations[placeholder])
        else:
            variant_values.append(
                [f"{{{placeholder}}}"]
            )  # Keep original if no variations

    prompts = []
    for combination in itertools.product(*variant_values):
        prompt = base_template
        for placeholder, value in zip(placeholders, combination):
            prompt = prompt.replace(f"{{{placeholder}}}", value)
        prompts.append(prompt)

    return prompts


def auto_tune_judge(
    logs: List[Dict[str, Any]],
    sampler: MultiTargetSampler,
    judge_type: str = "openai",
    judge_config: Optional[Dict[str, Any]] = None,
    base_template: Optional[str] = None,
    config: Optional[PromptTuningConfig] = None,
) -> TuningResult:
    """Automatically tune a judge by generating and testing prompt variants.

    This is the highest-level interface that handles everything automatically.
    """
    judge_config = judge_config or {}
    config = config or PromptTuningConfig()

    if base_template is None:
        base_template = """
        {instruction_style}
        {evaluation_criteria} on a scale 1–10.
        {output_format}

        CONTEXT:
        {{context}}

        RESPONSE:
        {{response}}
        """

    # Generate prompt variants
    prompt_grid = generate_prompt_variants(base_template)
    logger.info(f"Generated {len(prompt_grid)} prompt variants")

    # Tune prompts
    result = tune_judge_prompt_detailed(
        prompt_grid, logs, sampler, judge_type, judge_config, config
    )

    if result is None:
        # Return a default result if tuning failed
        return TuningResult(
            best_prompt=prompt_grid[0] if prompt_grid else "",
            best_variance=float("inf"),
            results=[],
            total_time=0.0,
            n_prompts_evaluated=0,
            early_stopped=False,
        )

    logger.info(
        f"Auto-tuning completed. Best prompt achieved variance {result.best_variance:.6f}"
    )

    return result
