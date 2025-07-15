"""
Precomputed sampler for working with teacher-forced data.

This sampler allows all CJE estimators to work with precomputed log probabilities
from teacher forcing, without needing to call actual language models.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging

from ..utils.importance_weights import (
    compute_importance_weight,
    compute_weight_statistics,
)

logger = logging.getLogger(__name__)


class PrecomputedSampler:
    """
    A sampler that uses precomputed log probabilities.

    This implements the same interface as MultiTargetSampler but uses
    precomputed values instead of calling policy APIs. Perfect for:
    - Working with teacher-forced datasets
    - Reproducing results without API access
    - Fast experimentation with different estimators

    Args:
        data: List of dicts with precomputed log probabilities
        target_policies: List of target policy names
        base_policy_field: Field name for base policy log prob (default: "total_logprob")
        target_logps_field: Field name for target log probs dict (default: "target_logps")
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        target_policies: List[str],
        base_policy_field: str = "total_logprob",
        target_logps_field: str = "target_logps",
        prompt_field: str = "prompt",
        response_field: str = "response",
        max_importance_weight: int = 50,
    ):
        self.data = data
        self.target_policies = target_policies
        self.K = len(target_policies)
        self.policy_names = target_policies
        self.max_importance_weight = max_importance_weight

        # Field names
        self.base_policy_field = base_policy_field
        self.target_logps_field = target_logps_field
        self.prompt_field = prompt_field
        self.response_field = response_field

        # Build lookup index for fast matching
        self._build_index()

        # For compatibility with MultiTargetSampler
        self.runners = [None] * self.K  # Dummy runners

    def _build_index(self) -> None:
        """Build index for fast context/response lookup."""
        self.index = {}
        for item in self.data:
            key = (item[self.prompt_field], item[self.response_field])
            self.index[key] = item

    def importance_weights_matrix(
        self,
        contexts: List[str],
        responses: List[str],
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute importance weights from precomputed log probabilities.

        Returns:
            Tuple of (weights_matrix, statistics_dict)
            weights_matrix: Shape (n, K) array of importance weights
            statistics_dict: Dictionary with weight statistics
        """
        n = len(contexts)
        weights = np.zeros((n, self.K))

        # Track missing data
        missing_count = 0

        for i, (ctx, resp) in enumerate(zip(contexts, responses)):
            key = (ctx, resp)

            if key in self.index:
                item = self.index[key]
                p0_logp = item[self.base_policy_field]

                # Check for valid base log prob
                if p0_logp is None or not np.isfinite(p0_logp):
                    logger.warning(f"Invalid base log prob for sample {i}")
                    weights[i, :] = np.nan
                    continue

                # Compute weights for each target policy
                target_logps = item[self.target_logps_field]
                for j, policy in enumerate(self.target_policies):
                    if policy in target_logps:
                        target_logp = target_logps[policy]

                        if target_logp is None or not np.isfinite(target_logp):
                            weights[i, j] = np.nan
                        else:
                            # Use shared utility for consistent weight calculation
                            weights[i, j] = compute_importance_weight(
                                target_logp=target_logp,
                                base_logp=p0_logp,
                                max_weight=self.max_importance_weight,
                            )
                    else:
                        logger.warning(f"Policy {policy} not found in target_logps")
                        weights[i, j] = np.nan
            else:
                missing_count += 1
                weights[i, :] = np.nan

        if missing_count > 0:
            logger.warning(
                f"Could not find {missing_count}/{n} samples in precomputed data"
            )

        # Compute statistics using shared utility
        stats = compute_weight_statistics(weights, self.target_policies)

        if show_progress:
            logger.info(
                f"Computed weights for {n} samples, ESS: {stats['ess_percentage']:.1f}%"
            )

        return weights, stats

    def logp_matrix(self, contexts: List[str], responses: List[str]) -> np.ndarray:
        """
        Get log probability matrix (for compatibility).

        Returns:
            Matrix of shape (n, K) with log probabilities
        """
        n = len(contexts)
        logp_matrix = np.zeros((n, self.K))

        for i, (ctx, resp) in enumerate(zip(contexts, responses)):
            key = (ctx, resp)

            if key in self.index:
                item = self.index[key]
                target_logps = item[self.target_logps_field]

                for j, policy in enumerate(self.target_policies):
                    if policy in target_logps:
                        logp_matrix[i, j] = target_logps[policy]
                    else:
                        logp_matrix[i, j] = np.nan
            else:
                logp_matrix[i, :] = np.nan

        return logp_matrix

    def sample_many(self, state: Any, n: int = 1) -> List[List[str]]:
        """
        Dummy implementation for compatibility.
        PrecomputedSampler doesn't support generation.
        """
        return [[] for _ in range(self.K)]

    @classmethod
    def from_jsonl(
        cls, jsonl_path: str, target_policies: Optional[List[str]] = None, **kwargs: Any
    ) -> "PrecomputedSampler":
        """
        Create a PrecomputedSampler from a JSONL file.

        Args:
            jsonl_path: Path to JSONL file with precomputed data
            target_policies: List of target policies (auto-detected if None)
            **kwargs: Additional arguments for PrecomputedSampler

        Returns:
            PrecomputedSampler instance
        """
        import json

        data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                data.append(json.loads(line))

        # Auto-detect target policies if not provided
        if target_policies is None and data:
            target_logps_field = kwargs.get("target_logps_field", "target_logps")
            if target_logps_field in data[0]:
                target_policies = list(data[0][target_logps_field].keys())
                logger.info(f"Auto-detected target policies: {target_policies}")
            else:
                raise ValueError(
                    f"Could not find {target_logps_field} field for auto-detection"
                )

        return cls(data, target_policies, **kwargs)
