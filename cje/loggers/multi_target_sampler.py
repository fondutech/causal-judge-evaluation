"""
Multi-target sampler with NO fallback values.

Clean implementation that makes data corruption impossible.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, cast
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

from ..types import LogProbResult, LogProbStatus, SampleResult, BatchResult
from .base_policy import BasePolicy

logger = logging.getLogger(__name__)


class MultiTargetSampler:
    """
    Sample from multiple target policies and compute importance weights.

    Key improvements:
    - NO fallback values ever - explicit None for failures
    - Parallel execution for efficiency
    - Comprehensive failure tracking
    - Returns structured results that force explicit handling
    """

    def __init__(
        self,
        policies: Union[List[BasePolicy], Dict[str, BasePolicy]],
        base_policy_name: str,
        max_workers: int = 10,
        checkpoint_dir: Optional[Path] = None,
        enable_caching: bool = True,
        validate_identical: bool = True,
    ):
        """
        Initialize multi-target sampler.

        Args:
            policies: List of policy instances
            base_policy_name: Name of the base policy for importance weights
            max_workers: Max parallel workers
            checkpoint_dir: Directory for saving checkpoints
            enable_caching: Whether to cache results
            validate_identical: Whether to validate identical policies have same outputs
        """
        if not policies:
            raise ValueError("At least one policy required")

        # Handle both list and dict inputs
        if isinstance(policies, list):
            self.policies = {p.name: p for p in policies}
            self.policy_names = [p.name for p in policies]
        else:
            self.policies = policies
            self.policy_names = list(policies.keys())

        self.base_policy_name = base_policy_name

        if base_policy_name not in self.policies:
            raise ValueError(
                f"Base policy '{base_policy_name}' not found. "
                f"Available: {list(self.policies.keys())}"
            )

        self.max_workers = max_workers
        self.checkpoint_dir = checkpoint_dir
        self.enable_caching = enable_caching
        self.validate_identical = validate_identical

        # Tracking
        self.cache: Dict[str, SampleResult] = {}
        self.failure_tracker: Dict[str, List[Dict]] = defaultdict(list)
        self.timing_stats: Dict[str, List[float]] = defaultdict(list)

        if checkpoint_dir:
            checkpoint_dir.mkdir(exist_ok=True)

        logger.info(
            f"Initialized MultiTargetSampler with {len(policies)} policies, "
            f"base policy: {base_policy_name}"
        )

    @property
    def K(self) -> int:
        """Number of target policies."""
        return len(self.policies)

    def compute_log_probs(
        self,
        context: str,
        response: str,
        sample_id: Optional[str] = None,
    ) -> Dict[str, LogProbResult]:
        """
        Compute log probabilities for all policies.

        Args:
            context: Input context
            response: Response to evaluate
            sample_id: Optional sample identifier

        Returns:
            Dictionary mapping policy name to LogProbResult
            NEVER returns raw floats or uses fallback values!
        """
        results = {}

        for name, policy in self.policies.items():
            start = time.time()

            # Compute log prob - always returns LogProbResult
            result = policy.compute_log_prob(context, response)
            results[name] = result

            # Track timing
            elapsed = time.time() - start
            self.timing_stats[name].append(elapsed)

            # Track failures
            if not result.is_valid:
                self.failure_tracker[name].append(
                    {
                        "sample_id": sample_id or f"unknown_{time.time()}",
                        "status": result.status.value,
                        "error": result.error,
                        "attempts": result.attempts,
                        "timestamp": time.time(),
                    }
                )

                logger.warning(
                    f"Policy {name} failed for sample {sample_id}: "
                    f"{result.status.value} - {result.error}"
                )

        # Validate identical policies if requested
        if self.validate_identical:
            self._validate_identical_policies(results)

        return results

    def compute_importance_weights(
        self,
        log_prob_results: Dict[str, LogProbResult],
    ) -> Dict[str, Optional[float]]:
        """
        Compute importance weights relative to base policy.

        Args:
            log_prob_results: Dictionary of LogProbResults

        Returns:
            Dictionary mapping policy name to importance weight
            Returns None for any failed policies - no fake values!
        """
        weights = {}

        # Get base policy result
        base_result = log_prob_results.get(self.base_policy_name)
        if not base_result or not base_result.is_valid:
            # Base policy failed - all weights are None
            logger.error(
                f"Base policy {self.base_policy_name} failed - "
                f"cannot compute importance weights"
            )
            return {name: None for name in log_prob_results}

        base_logp = base_result.value

        # Compute weights for each policy
        for name, result in log_prob_results.items():
            if name == self.base_policy_name:
                weights[name] = 1.0  # Base always has weight 1
            elif result.is_valid:
                # Compute weight with clipping for numerical stability
                log_ratio = result.value - base_logp

                # Clip to prevent overflow
                if log_ratio > 20:
                    logger.warning(
                        f"Large log ratio {log_ratio:.2f} for {name}, clipping to 20"
                    )
                    log_ratio = 20.0
                elif log_ratio < -20:
                    logger.warning(
                        f"Small log ratio {log_ratio:.2f} for {name}, clipping to -20"
                    )
                    log_ratio = -20.0

                weights[name] = np.exp(log_ratio)
            else:
                # Failed policy has no weight - explicit None!
                weights[name] = None

        return weights

    def process_sample(
        self,
        sample_id: str,
        context: str,
        response: str,
    ) -> SampleResult:
        """
        Process a single sample through all policies.

        Returns:
            SampleResult with all policy results and weights
            No exceptions raised - all failures captured in results
        """
        # Check cache
        cache_key = self._get_cache_key(sample_id, context, response)
        if self.enable_caching and cache_key in self.cache:
            logger.debug(f"Cache hit for {sample_id}")
            return self.cache[cache_key]

        # Compute log probs for all policies
        policy_results = self.compute_log_probs(context, response, sample_id)

        # Compute importance weights
        importance_weights = self.compute_importance_weights(policy_results)

        # Create result
        result = SampleResult(
            sample_id=sample_id,
            context=context,
            response=response,
            policy_results=policy_results,
            importance_weights=importance_weights,
            metadata={
                "context_len": len(context),
                "response_len": len(response),
                "base_policy": self.base_policy_name,
                "timestamp": time.time(),
            },
        )

        # Cache if enabled
        if self.enable_caching:
            self.cache[cache_key] = result

        return result

    def process_batch(
        self,
        samples: List[Tuple[str, str, str]],  # (id, context, response)
        show_progress: bool = True,
        save_checkpoint_every: int = 100,
    ) -> BatchResult:
        """
        Process a batch of samples in parallel.

        Args:
            samples: List of (sample_id, context, response) tuples
            show_progress: Whether to show progress
            save_checkpoint_every: Save checkpoint every N samples

        Returns:
            BatchResult with all sample results
        """
        start_time = time.time()
        results = []

        logger.info(f"Processing batch of {len(samples)} samples")

        # Use thread pool for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(self.process_sample, sample_id, context, response): (
                    sample_id,
                    context,
                    response,
                )
                for sample_id, context, response in samples
            }

            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_sample):
                sample_info = future_to_sample[future]
                sample_id = sample_info[0]

                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Even catastrophic failures don't stop the batch
                    logger.error(f"Catastrophic failure for {sample_id}: {e}")

                    # Create a completely failed result
                    failed_result = SampleResult(
                        sample_id=sample_id,
                        context=sample_info[1],
                        response=sample_info[2],
                        policy_results={
                            name: LogProbResult(
                                status=LogProbStatus.API_ERROR,
                                error=f"Catastrophic: {e}",
                                attempts=0,
                            )
                            for name in self.policies
                        },
                        importance_weights={name: None for name in self.policies},
                        metadata={"catastrophic_error": str(e)},
                    )
                    results.append(failed_result)

                completed += 1

                # Progress update
                if show_progress and completed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(samples) - completed) / rate
                    logger.info(
                        f"Progress: {completed}/{len(samples)} "
                        f"({completed/len(samples)*100:.1f}%) "
                        f"Rate: {rate:.1f}/s, ETA: {eta:.0f}s"
                    )

                # Checkpoint
                if (
                    self.checkpoint_dir
                    and save_checkpoint_every > 0
                    and completed % save_checkpoint_every == 0
                ):
                    self._save_checkpoint(results, partial=True)

        # Calculate efficiency
        total_time = time.time() - start_time

        # Estimate ideal time (sum of all individual times)
        total_individual_time = sum(sum(times) for times in self.timing_stats.values())
        parallel_efficiency = (
            total_individual_time / (total_time * self.max_workers)
            if total_time > 0
            else 0
        )

        # Final checkpoint
        if self.checkpoint_dir:
            self._save_checkpoint(results, partial=False)

        # Create batch result
        batch_result = BatchResult(
            results=results,
            total_time_seconds=total_time,
            parallel_efficiency=parallel_efficiency,
            metadata={
                "num_policies": len(self.policies),
                "base_policy": self.base_policy_name,
                "max_workers": self.max_workers,
            },
        )

        logger.info(f"Batch processing complete: {batch_result.get_summary()}")

        return batch_result

    def _validate_identical_policies(self, results: Dict[str, LogProbResult]) -> None:
        """Validate that identical policies give identical results."""
        # Group policies by configuration signature
        policy_groups = defaultdict(list)

        for name, policy in self.policies.items():
            sig = self._get_policy_signature(policy)
            policy_groups[sig].append(name)

        # Check each group
        for sig, names in policy_groups.items():
            if len(names) > 1:
                # Get results for this group
                group_results = {
                    name: results[name] for name in names if name in results
                }

                # All should be either valid or invalid
                valid_results = [
                    (name, r.value) for name, r in group_results.items() if r.is_valid
                ]

                if len(valid_results) > 1:
                    # Check if all values are close
                    values = [v for _, v in valid_results]
                    if not np.allclose(values, values[0], rtol=1e-5):
                        logger.warning(
                            f"Identical policies have different log probs: "
                            f"{valid_results}"
                        )

    def _get_policy_signature(self, policy: BasePolicy) -> str:
        """Get signature for policy configuration."""
        # This is a simplified version - enhance based on your policy attributes
        return f"{policy.model_id}"

    def _get_cache_key(self, sample_id: str, context: str, response: str) -> str:
        """Generate cache key."""
        # Use hashes to keep keys manageable
        return f"{sample_id}:{hash(context)}:{hash(response)}"

    def _save_checkpoint(
        self, results: List[SampleResult], partial: bool = False
    ) -> None:
        """Save results checkpoint."""
        timestamp = int(time.time())
        suffix = "partial" if partial else "final"
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{timestamp}_{suffix}.jsonl"

        with open(checkpoint_path, "w") as f:
            for result in results:
                # Convert to JSON-serializable format
                data = {
                    "sample_id": result.sample_id,
                    "context": result.context[:500],  # Truncate for space
                    "response": result.response[:500],
                    "policy_results": {
                        name: {
                            "status": r.status.value,
                            "value": r.value,
                            "error": r.error,
                            "attempts": r.attempts,
                        }
                        for name, r in result.policy_results.items()
                    },
                    "importance_weights": result.importance_weights,
                    "metadata": result.metadata,
                }
                f.write(json.dumps(data) + "\n")

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics."""
        analytics: Dict[str, Any] = {
            "policies": {},
            "failures": {},
            "timing": {},
            "cache": {
                "enabled": self.enable_caching,
                "size": len(self.cache),
            },
        }

        # Policy statistics
        for name, policy in self.policies.items():
            analytics["policies"][name] = policy.get_stats()

        # Failure analysis
        for name, failures in self.failure_tracker.items():
            if failures:
                by_status: Dict[str, int] = defaultdict(int)
                for f in failures:
                    by_status[f["status"]] += 1
                analytics["failures"][name] = dict(by_status)

        # Timing statistics
        for name, times in self.timing_stats.items():
            if times:
                analytics["timing"][name] = {
                    "mean_ms": np.mean(times) * 1000,
                    "std_ms": np.std(times) * 1000,
                    "min_ms": np.min(times) * 1000,
                    "max_ms": np.max(times) * 1000,
                }

        return analytics

    def importance_weights_matrix(
        self,
        contexts: List[str],
        responses: List[str],
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute importance weights for all samples.

        Args:
            contexts: List of contexts
            responses: List of responses
            show_progress: Whether to show progress bar

        Returns:
            Tuple of:
            - Weight matrix of shape (n_samples, n_policies)
            - Dictionary of weight statistics
        """
        n_samples = len(contexts)
        n_policies = len(self.policies)
        weights_matrix = np.zeros((n_samples, n_policies))

        # Process all samples
        samples = [
            (f"sample_{i}", ctx, resp)
            for i, (ctx, resp) in enumerate(zip(contexts, responses))
        ]
        batch_result = self.process_batch(samples, show_progress=show_progress)

        # Extract weights into matrix
        for i, result in enumerate(batch_result.results):
            for j, policy_name in enumerate(self.policy_names):
                weight = result.importance_weights.get(policy_name)
                if weight is not None:
                    weights_matrix[i, j] = weight
                else:
                    weights_matrix[i, j] = np.nan

        # Compute statistics
        weight_stats = self._compute_weight_statistics(weights_matrix)

        return weights_matrix, weight_stats

    def _compute_weight_statistics(self, weights_matrix: np.ndarray) -> Dict[str, Any]:
        """Compute statistics for weight matrix."""
        stats: Dict[str, Any] = {
            "n_samples": weights_matrix.shape[0],
            "n_policies": weights_matrix.shape[1],
            "policy_stats": {},
        }

        for j, policy_name in enumerate(self.policy_names):
            weights = weights_matrix[:, j]
            valid_weights = weights[~np.isnan(weights)]

            if len(valid_weights) > 0:
                policy_stats: Dict[str, Any] = {
                    "mean": float(np.mean(valid_weights)),
                    "std": float(np.std(valid_weights)),
                    "min": float(np.min(valid_weights)),
                    "max": float(np.max(valid_weights)),
                    "n_valid": len(valid_weights),
                    "n_missing": len(weights) - len(valid_weights),
                }
                stats["policy_stats"][policy_name] = policy_stats
            else:
                stats["policy_stats"][policy_name] = {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "n_valid": 0,
                    "n_missing": len(weights),
                }

        return stats


def make_multi_sampler(
    policies: Union[Dict[str, BasePolicy], List[BasePolicy]],
    base_policy_name: Optional[str] = None,
    **kwargs: Any,
) -> MultiTargetSampler:
    """
    Factory function to create a MultiTargetSampler.

    Args:
        policies: Either a dict of name->policy or list of policies
        base_policy_name: Name of base policy for importance weighting
        **kwargs: Additional arguments for MultiTargetSampler

    Returns:
        Configured MultiTargetSampler instance
    """
    # Convert list to dict if needed
    if isinstance(policies, list):
        policy_dict = {
            getattr(p, "name", f"policy_{i}"): p for i, p in enumerate(policies)
        }
    else:
        policy_dict = policies

    # Determine base policy if not specified
    if base_policy_name is None:
        if len(policy_dict) > 0:
            base_policy_name = list(policy_dict.keys())[0]
            logger.info(f"No base policy specified, using: {base_policy_name}")

    return MultiTargetSampler(
        policies=policy_dict, base_policy_name=base_policy_name, **kwargs
    )
