"""Precomputed sampler for CJE estimation."""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import logging

from .models import Dataset
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
        self._formatted_to_dataset_idx = []  # Track mapping for O(1) lookup
        n_missing_base = 0
        n_missing_target = {policy: 0 for policy in self.target_policies}

        for i, sample in enumerate(self.dataset.samples):
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
            self._formatted_to_dataset_idx.append(i)

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
        for i, record in enumerate(self.formatted_data):
            # Check if this record has the target policy logprob
            if target_policy in record["target_policy_logprobs"]:
                # Get the corresponding sample for metadata
                sample = self.dataset.samples[self._get_sample_index(i)]
                policy_data.append(
                    {
                        "reward": record["reward"],
                        "base_policy_logprob": record["base_policy_logprob"],
                        "policy_logprob": record["target_policy_logprobs"][
                            target_policy
                        ],
                        "prompt": record["context"],
                        "response": record["response"],
                        "prompt_id": sample.prompt_id,
                        "judge_score": sample.metadata.get("judge_score"),
                        "cv_fold": sample.metadata.get(
                            "cv_fold"
                        ),  # Include fold info for DR
                    }
                )

        return policy_data if policy_data else None

    def _get_valid_indices(self, target_policy: str) -> np.ndarray:
        """Get indices of valid samples for a target policy.

        Returns indices into the original dataset.samples array.
        """
        valid_indices = []
        for i, sample in enumerate(self.dataset.samples):
            # Check if sample has valid data for this policy
            if (
                sample.base_policy_logprob is not None
                and sample.target_policy_logprobs.get(target_policy) is not None
            ):
                valid_indices.append(i)
        return np.array(valid_indices)

    def _get_sample_index(self, formatted_index: int) -> int:
        """Map from formatted_data index back to dataset.samples index.

        This is needed because formatted_data filters out invalid samples.
        """
        if formatted_index >= len(self._formatted_to_dataset_idx):
            raise IndexError(f"Formatted index {formatted_index} out of range")
        return self._formatted_to_dataset_idx[formatted_index]

    def compute_log_ratios(self, target_policy: str) -> np.ndarray:
        """Compute log importance ratios (log p_target - log p_base).

        Args:
            target_policy: Name of target policy

        Returns:
            Array of log ratios (may contain -inf for zero weights)
        """
        if target_policy not in self.target_policies:
            raise ValueError(f"Unknown target policy: {target_policy}")

        log_ratios = np.array(
            [
                record["target_policy_logprobs"][target_policy]
                - record["base_policy_logprob"]
                for record in self.formatted_data
            ],
            dtype=np.float64,
        )

        # NaN -> -inf (zero weight)
        log_ratios[np.isnan(log_ratios)] = -np.inf

        return log_ratios

    def compute_raw_weights(self, target_policy: str) -> np.ndarray:
        """Compute raw importance weights WITHOUT scaling.

        Returns truly raw weights: exp(log_p_target - log_p_base)
        Only guards against overflow to inf, no scaling applied.

        Args:
            target_policy: Name of target policy

        Returns:
            Array of raw importance weights
        """
        log_ratios = self.compute_log_ratios(target_policy)

        # Clamp only to avoid overflow to inf, keep underflow (->0) natural
        max_log = np.log(np.finfo(np.float64).max)  # ~709.78
        clamped = np.minimum(log_ratios, max_log)

        # Report extreme values
        n_clamped = np.sum(log_ratios > max_log)
        if n_clamped > 0:
            logger.debug(
                f"Clamped {n_clamped} extreme log-ratios for {target_policy} "
                f"(max was {np.max(log_ratios[np.isfinite(log_ratios)]):.1f}) to prevent overflow"
            )

        weights = np.exp(clamped)

        # Clean up non-finite values (from -inf log ratios)
        weights[~np.isfinite(weights)] = 0.0

        return np.asarray(weights)

    def compute_hajek_weights(self, target_policy: str) -> np.ndarray:
        """Compute mean-one (SNIPS/Hájek) weights using stable log-sum-exp.

        These weights have mean exactly 1.0 and are computed in a numerically
        stable way using the log-sum-exp trick.

        Args:
            target_policy: Name of target policy

        Returns:
            Array of Hájek weights with mean=1.0
        """
        log_ratios = self.compute_log_ratios(target_policy)
        n = len(log_ratios)

        # Handle all -inf case (all weights would be zero)
        finite_mask = np.isfinite(log_ratios)
        if not finite_mask.any():
            logger.warning(
                f"All log-ratios are -inf for {target_policy}; returning zeros"
            )
            return np.zeros_like(log_ratios)

        # Log-sum-exp trick: subtract max for numerical stability
        max_log = np.max(log_ratios[finite_mask])

        # Compute stable exponentials
        stable_exp = np.zeros_like(log_ratios)
        stable_exp[finite_mask] = np.exp(log_ratios[finite_mask] - max_log)

        # Sum of weights
        sum_weights = stable_exp.sum()
        if sum_weights == 0.0:
            logger.warning(
                f"Sum of weights is zero for {target_policy}; returning zeros"
            )
            return np.zeros_like(log_ratios)

        # Normalize to mean=1: w_i = n * exp(lr_i) / sum(exp(lr))
        hajek_weights = (n * stable_exp) / sum_weights

        # Verify mean is 1.0 (within floating point precision)
        actual_mean = hajek_weights.mean()
        if abs(actual_mean - 1.0) > 1e-10:
            logger.debug(
                f"Hájek weights for {target_policy} have mean {actual_mean:.12f} (expected 1.0)"
            )

        return np.asarray(hajek_weights)

    def compute_importance_weights(
        self,
        target_policy: str,
        clip_weight: Optional[float] = None,
        mode: str = "hajek",
    ) -> np.ndarray:
        """Compute importance weights for a target policy with optional clipping.

        Args:
            target_policy: Name of target policy
            clip_weight: Maximum weight value for variance control.
                        If None (default), no clipping is applied.
                        Set to a finite value (e.g., 100.0) to clip weights.
            mode: "hajek" for mean-one weights (default), "raw" for unnormalized

        Returns:
            Array of importance weights
        """
        # Get weights based on mode
        if mode == "hajek":
            weights_array = self.compute_hajek_weights(target_policy)
        elif mode == "raw":
            weights_array = self.compute_raw_weights(target_policy)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'hajek' or 'raw'")

        # Apply clipping if requested (after Hájek normalization for interpretability)
        if clip_weight is not None and np.isfinite(clip_weight):
            n_clipped = np.sum(weights_array > clip_weight)
            if n_clipped > 0:
                max_weight = np.max(weights_array)
                weights_array = np.minimum(weights_array, clip_weight)
                logger.info(
                    f"Clipped {n_clipped}/{len(weights_array)} weights for {target_policy} "
                    f"to {clip_weight} (max was {max_weight:.2f})"
                )

        # Log statistics
        logger.debug(
            f"Weight statistics for '{target_policy}': "
            f"mean={weights_array.mean():.3f}, std={weights_array.std():.3f}, "
            f"min={weights_array.min():.3f}, max={weights_array.max():.3f}"
        )

        return np.asarray(weights_array)

    def get_rewards(self) -> np.ndarray:
        """Get array of calibrated rewards."""
        return np.array([s.reward for s in self.dataset.samples])

    def get_judge_scores(self) -> Optional[np.ndarray]:
        """Get array of judge scores from metadata.

        Returns:
            Array of judge scores if available in all valid samples, None otherwise.
        """
        # Only get judge scores for valid (formatted) samples
        judge_scores = []
        for idx in self._formatted_to_dataset_idx:
            sample = self.dataset.samples[idx]
            score = sample.metadata.get("judge_score")
            if score is None:
                return None  # Not all samples have judge scores
            judge_scores.append(score)
        return np.array(judge_scores)

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
