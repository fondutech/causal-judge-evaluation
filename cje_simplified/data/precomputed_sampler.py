"""Precomputed sampler for CJE estimation."""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import logging

from .models import Dataset, Sample
from .factory import DatasetFactory

logger = logging.getLogger(__name__)


class PrecomputedSampler:
    """Wrapper around Dataset that provides CJE-specific operations.

    This class takes a Dataset and adds methods needed for importance sampling
    estimation like weight computation, filtering, and diagnostic checks.
    """

    def __init__(
        self,
        data_or_dataset: Union[Dataset, List[Dict[str, Any]]],
        target_policies: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Initialize with either a Dataset or raw data.

        Args:
            data_or_dataset: Either a Dataset instance or raw data list
            target_policies: Target policy names (only used if data_or_dataset is a list)
            **kwargs: Additional arguments passed to DatasetFactory

        Raises:
            ValueError: If any samples are missing rewards
        """
        if isinstance(data_or_dataset, Dataset):
            self.dataset = data_or_dataset
        else:
            # Create Dataset from raw data using factory
            factory = DatasetFactory()
            self.dataset = factory.create_from_data(
                data_or_dataset, target_policies=target_policies
            )

        # Validate that all samples have rewards
        samples_without_rewards = [
            i for i, sample in enumerate(self.dataset.samples) if sample.reward is None
        ]
        if samples_without_rewards:
            raise ValueError(
                f"PrecomputedSampler requires all samples to have rewards. "
                f"Found {len(samples_without_rewards)} samples without rewards. "
                f"Please calibrate your dataset first using calibrate_dataset()."
            )

        self.target_policies = self.dataset.target_policies

        # Prepare formatted data
        self.formatted_data = self._format_for_estimators()

    @classmethod
    def from_jsonl(
        cls, file_path: str, target_policies: Optional[List[str]] = None, **kwargs: Any
    ) -> "PrecomputedSampler":
        """Create sampler from JSONL file.

        Args:
            file_path: Path to JSONL file
            target_policies: Optional list of target policy names
            **kwargs: Additional arguments passed to DatasetFactory

        Returns:
            PrecomputedSampler instance
        """
        factory = DatasetFactory()
        dataset = factory.create_from_jsonl(file_path, target_policies)
        return cls(dataset)

    def _format_for_estimators(self) -> List[Dict[str, Any]]:
        """Format data for CJE estimators.

        Returns list of dicts with:
        - context: prompt text
        - response: generated text
        - base_policy_logprob: base policy log prob
        - reward: calibrated reward
        - target_policy_logprobs: dict of target log probs
        """
        formatted = []
        n_missing_base = 0
        n_missing_target = {policy: 0 for policy in self.target_policies}

        for sample in self.dataset.samples:
            # Skip samples without valid base log prob
            if sample.base_policy_logprob is None:
                n_missing_base += 1
                continue

            # Check all required target policies have valid log probs
            valid_targets = {}
            skip_record = False
            for policy in self.target_policies:
                logp = sample.target_policy_logprobs.get(policy)
                if logp is None:
                    n_missing_target[policy] += 1
                    skip_record = True
                    break
                valid_targets[policy] = logp

            if skip_record:
                continue

            formatted.append(
                {
                    "context": sample.prompt,
                    "response": sample.response,
                    "base_policy_logprob": sample.base_policy_logprob,
                    "reward": sample.reward,
                    "target_policy_logprobs": valid_targets,
                }
            )

        # Report filtering statistics
        n_total = len(self.dataset.samples)
        n_valid = len(formatted)
        n_filtered = n_total - n_valid

        if n_filtered > 0:
            filter_pct = (n_filtered / n_total) * 100
            logger.warning(
                f"Filtered {n_filtered}/{n_total} samples ({filter_pct:.1f}%) due to missing log probabilities:\n"
                f"  - Missing base_policy_logprob: {n_missing_base}\n"
                f"  - Missing target policy logprobs: {n_missing_target}"
            )

            if filter_pct > 50:
                logger.error(
                    f"WARNING: More than 50% of samples filtered! Only {n_valid}/{n_total} samples remain. "
                    f"This may significantly impact estimation quality."
                )

        if not formatted:
            raise ValueError(
                f"No valid records after filtering! All {n_total} samples had invalid log probabilities.\n"
                f"  - Missing base_policy_logprob: {n_missing_base}\n"
                f"  - Missing target policy logprobs: {n_missing_target}"
            )

        if n_valid < 10:
            logger.warning(
                f"Only {n_valid} valid samples available for estimation. "
                f"Results may be unreliable with such small sample size."
            )

        return formatted

    def get_data_for_policy(self, target_policy: str) -> Optional[List[Dict[str, Any]]]:
        """Get formatted data for a specific target policy.

        Returns data in format expected by estimators:
        - reward: float
        - base_policy_logprob: base policy log prob
        - policy_logprob: target policy log prob

        Note: Returns data from formatted_data to ensure consistency with weights.
        """
        if target_policy not in self.target_policies:
            return None

        # Use the same formatted_data that was used for weights to ensure consistency
        policy_data = []
        for record in self.formatted_data:
            # Check if this record has the target policy logprob
            if target_policy in record["target_policy_logprobs"]:
                policy_data.append(
                    {
                        "reward": record["reward"],
                        "base_policy_logprob": record["base_policy_logprob"],
                        "policy_logprob": record["target_policy_logprobs"][
                            target_policy
                        ],
                        "prompt": record["context"],
                        "response": record["response"],
                    }
                )

        return policy_data if policy_data else None

    def compute_raw_weights(self, target_policy: str) -> np.ndarray:
        """Compute raw importance weights WITHOUT any clipping.

        Returns truly raw weights: exp(log_p_target - log_p_base)
        Only handles numerical overflow protection, no artificial clipping.

        Args:
            target_policy: Name of target policy

        Returns:
            Array of raw importance weights
        """
        if target_policy not in self.target_policies:
            raise ValueError(f"Unknown target policy: {target_policy}")

        weights = []
        for record in self.formatted_data:
            base_logp = record["base_policy_logprob"]
            target_logp = record["target_policy_logprobs"][target_policy]

            # Compute weight with overflow protection
            log_ratio = target_logp - base_logp

            # Practical thresholds for importance weights:
            # - log_ratio > 100 means target is 10^43 times more likely (unrealistic)
            # - log_ratio < -100 means target is 10^-43 times as likely (effectively 0)
            # These are still very extreme but prevent numerical overflow
            if log_ratio > 100:
                weight = np.exp(100)  # ~2.7e43, extreme but manageable
            elif log_ratio < -100:
                weight = 0.0  # Effectively zero weight
            else:
                weight = np.exp(log_ratio)

            weights.append(weight)

        return np.array(weights)

    def compute_importance_weights(
        self, target_policy: str, clip_weight: Optional[float] = None
    ) -> np.ndarray:
        """Compute importance weights for a target policy with optional clipping.

        Args:
            target_policy: Name of target policy
            clip_weight: Maximum weight value for variance control.
                        If None (default), no clipping is applied.
                        Set to a finite value (e.g., 100.0) to clip weights.

        Returns:
            Array of importance weights
        """
        # Get raw weights
        raw_weights = self.compute_raw_weights(target_policy)

        # Apply clipping if requested (and if it would actually clip anything)
        if (
            clip_weight is not None and clip_weight < 1e9
        ):  # Only clip if reasonable threshold
            weights_array = np.clip(raw_weights, 0, clip_weight)

            # Log clipping statistics
            n_clipped = np.sum(raw_weights > clip_weight)
            if n_clipped > 0:
                max_raw = np.max(raw_weights)
                logger.info(
                    f"Weight clipping for policy '{target_policy}': "
                    f"{n_clipped}/{len(raw_weights)} weights clipped to {clip_weight} "
                    f"(max raw weight: {max_raw:.2f})"
                )
        else:
            weights_array = raw_weights

        # Log statistics
        logger.debug(
            f"Weight statistics for '{target_policy}': "
            f"mean={weights_array.mean():.3f}, std={weights_array.std():.3f}, "
            f"min={weights_array.min():.3f}, max={weights_array.max():.3f}"
        )

        return weights_array

    def get_rewards(self) -> np.ndarray:
        """Get array of calibrated rewards."""
        return np.array([s.reward for s in self.dataset.samples])

    def get_contexts(self) -> List[str]:
        """Get list of contexts/prompts."""
        return [s.prompt for s in self.dataset.samples]

    def get_responses(self) -> List[str]:
        """Get list of responses."""
        return [s.response for s in self.dataset.samples]

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset.

        Note: This returns the total number of samples in the dataset.
        For the number of samples with valid log probabilities that will
        be used for estimation, use n_valid_samples.
        """
        return self.dataset.n_samples

    @property
    def n_valid_samples(self) -> int:
        """Number of valid samples with all required log probabilities.

        This is the number of samples that will actually be used for estimation,
        after filtering out samples with missing log probabilities.
        """
        return len(self.formatted_data)

    @property
    def n_samples_total(self) -> int:
        """Total number of samples in the dataset (before filtering).

        Deprecated: Use n_samples instead.
        """
        return self.dataset.n_samples

    @property
    def n_samples_valid(self) -> int:
        """Number of valid samples for estimation.

        Deprecated: Use n_valid_samples instead.
        """
        return self.n_valid_samples

    @property
    def n_policies(self) -> int:
        """Number of target policies."""
        return len(self.target_policies)

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        dataset_summary = self.dataset.summary()
        filter_rate = (
            1.0 - (self.n_valid_samples / self.n_samples) if self.n_samples > 0 else 0.0
        )

        return {
            "n_samples": self.n_samples,  # Keep for backwards compatibility
            "n_samples_valid": self.n_valid_samples,
            "n_samples_total": self.n_samples,  # Same as n_samples
            "n_samples_filtered": self.n_samples - self.n_valid_samples,
            "filter_rate": filter_rate,
            "n_policies": self.n_policies,
            "target_policies": self.target_policies,
            "reward_mean": dataset_summary["reward_mean"],
            "reward_std": dataset_summary["reward_std"],
            "valid_samples_per_policy": dataset_summary["valid_samples_per_policy"],
        }
