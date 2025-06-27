"""
Mock multi-target sampler for testing without external dependencies.

This module provides a mock implementation of MultiTargetSampler that coordinates
multiple mock policy runners for multi-policy evaluation testing.
"""

import numpy as np
from typing import List, Sequence, Union, Tuple, Any, Dict, Optional

from .policy_runners import MockPolicyRunner, MockAPIPolicyRunner
from ...loggers.multi_target_sampler import MultiTargetSampler


class MockMultiTargetSampler(MultiTargetSampler):
    """
    Mock implementation of MultiTargetSampler for testing.

    Coordinates multiple mock policy runners to simulate multi-policy evaluation
    without loading actual models or making API calls.
    """

    def __init__(
        self,
        runners: Sequence[Union[MockPolicyRunner, MockAPIPolicyRunner]],
        consistent_ordering: bool = True,
    ):
        """
        Initialize with a list of mock policy runners.

        Args:
            runners: Sequence of mock policy runners, one for each target policy πᵏ
            consistent_ordering: If True, ensures deterministic ordering for testing
        """
        if not runners:
            raise ValueError("At least one policy runner must be provided")

        # Initialize parent class with mock policies
        # Create mock base policies from runners
        mock_policies = []
        for i, runner in enumerate(runners):
            from ...loggers.base_policy import BasePolicy

            class MockPolicy(BasePolicy):
                def __init__(self, runner: Any, idx: int) -> None:
                    super().__init__(
                        name=f"mock_policy_{idx}",
                        model_id=getattr(runner, "model_name", f"mock_{idx}"),
                    )
                    self.runner = runner

                def _compute_log_prob_impl(self, context: str, response: str) -> float:
                    result = self.runner.log_prob(context, response)
                    if isinstance(result, tuple):
                        return float(result[0])
                    return float(result)

            mock_policies.append(MockPolicy(runner, i))

        # Initialize with base_policy_name as first policy
        policy_list: List[BasePolicy] = mock_policies  # type: ignore[assignment]
        super().__init__(policies=policy_list, base_policy_name="mock_policy_0")

        self.runners = list(runners)  # Keep original runners for compatibility
        self.consistent_ordering = consistent_ordering

        # For testing consistency - track which runner corresponds to which policy
        self.runner_names = [
            getattr(runner, "model_name", f"mock_policy_{i}")
            for i, runner in enumerate(self.runners)
        ]

    def logp_many(self, context: str, response: str) -> List[float]:
        """
        Return log πᵏ(response | context) for every mock runner.

        Args:
            context: Input context string
            response: Response sequence to evaluate

        Returns:
            List of log probabilities, length K (one per target policy)
        """
        logps = []
        for runner in self.runners:
            try:
                logp_result = runner.log_prob(context, response)
                # Handle the case where log_prob returns a tuple (logp, token_logps)
                if isinstance(logp_result, tuple):
                    logp = logp_result[0]
                else:
                    logp = logp_result
                logps.append(float(logp))
            except Exception as e:
                # Fallback for robustness in testing
                print(f"Warning: Mock runner failed, using fallback logp: {e}")
                logps.append(-20.0)  # Reasonable fallback

        return logps

    def sample_many(
        self,
        context: str,
        n: int = 1,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> List[List[str]]:
        """
        Sample n responses from each of the K target policies.

        Args:
            context: Input context string
            n: Number of samples per policy
            temperature: Optional temperature override
            **kwargs: Additional generation parameters

        Returns:
            List of K lists, each containing n response strings
        """
        all_samples = []

        for runner in self.runners:
            try:
                # Generate samples using the mock runner
                gen_kwargs = kwargs.copy()
                if temperature is not None:
                    gen_kwargs["temperature"] = temperature

                results = runner.generate_with_logp(prompts=[context] * n, **gen_kwargs)

                # Extract just the response strings
                samples = [
                    result[0] for result in results
                ]  # result is (response, logp, ...)
                all_samples.append(samples)

            except Exception as e:
                # Fallback for robustness in testing
                print(
                    f"Warning: Mock runner failed during sampling, using fallback: {e}"
                )
                fallback_samples = [f"Mock response {i}" for i in range(n)]
                all_samples.append(fallback_samples)

        return all_samples

    def logp_matrix(self, contexts: List[str], responses: List[str]) -> np.ndarray:
        """
        Compute log probabilities for multiple context-response pairs.

        Args:
            contexts: List of input contexts
            responses: List of responses (same length as contexts)

        Returns:
            Matrix of shape (len(contexts), K) where entry (i,k) is
            log πᵏ(responses[i] | contexts[i])
        """
        if len(contexts) != len(responses):
            raise ValueError("contexts and responses must have the same length")

        N = len(contexts)
        logp_matrix: np.ndarray = np.zeros((N, self.K), dtype=np.float64)

        for i, (context, response) in enumerate(zip(contexts, responses)):
            logps = self.logp_many(context, response)
            logp_matrix[i, :] = logps

        return np.asarray(logp_matrix)

    def importance_weights_matrix(
        self,
        contexts: List[str],
        responses: List[str],
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute importance weights for multiple context-response pairs.

        This is a simplified mock implementation that matches the parent signature.

        Args:
            contexts: List of input contexts
            responses: List of responses
            show_progress: Whether to show progress (ignored in mock)

        Returns:
            Tuple of (weights_matrix, statistics_dict)
        """
        if len(contexts) != len(responses):
            raise ValueError("contexts and responses must have the same length")

        # Get target policy log probabilities
        logp_target_matrix: np.ndarray = self.logp_matrix(contexts, responses)

        # For mock, use first policy as behavior policy
        logp_behavior = logp_target_matrix[:, 0]

        # Convert to numpy arrays for vectorized operations
        logp_behavior_array: np.ndarray = np.array(
            logp_behavior, dtype=np.float64
        ).reshape(
            -1, 1
        )  # Shape: (N, 1)

        # Compute importance weights in log space, then exponentiate
        # w = exp(log π^k - log π₀)
        log_weights: np.ndarray = logp_target_matrix - logp_behavior_array
        weights: np.ndarray = np.exp(log_weights)

        # Apply reasonable bounds for testing
        weights = np.clip(weights, 0.0, 1000.0)

        weights = np.nan_to_num(weights, nan=0.0, posinf=1000.0, neginf=0.0)

        weights_matrix = np.asarray(weights, dtype=np.float64)

        # Compute basic statistics for the mock
        n_samples = len(contexts)
        n_policies = weights_matrix.shape[1]

        # Compute ESS per policy
        ess_values = []
        for k in range(n_policies):
            w_k = weights_matrix[:, k]
            if w_k.sum() > 0:
                ess_k = (w_k.sum()) ** 2 / (w_k**2).sum()
            else:
                ess_k = 0.0
            ess_values.append(ess_k)

        # Mock statistics
        statistics = {
            "ess_values": ess_values,
            "ess_percentage": np.mean(ess_values) / n_samples * 100,
            "n_clipped": 0,  # Mock doesn't track actual clipping
            "clip_fraction": 0.0,
            "weight_range": (
                float(np.min(weights_matrix)),
                float(np.max(weights_matrix)),
            ),
            "stabilization_applied": False,
            "n_samples": n_samples,
            "n_policies": n_policies,
        }
        return weights_matrix, statistics

    def get_runner_info(self) -> List[Dict[str, Any]]:
        """
        Get information about each mock runner for debugging/testing.

        Returns:
            List of dictionaries with runner information
        """
        info = []
        for i, runner in enumerate(self.runners):
            runner_info = {
                "index": i,
                "name": self.runner_names[i],
                "type": type(runner).__name__,
                "temperature": getattr(runner, "temperature", None),
                "max_new_tokens": getattr(runner, "max_new_tokens", None),
            }

            # Add provider info for API runners
            if hasattr(runner, "provider"):
                runner_info["provider"] = runner.provider

            # Add model config info if available
            if hasattr(runner, "config"):
                runner_info["quality"] = runner.config.quality
                runner_info["creativity"] = runner.config.creativity
                runner_info["verbosity"] = runner.config.verbosity

            info.append(runner_info)

        return info

    def test_consistency(
        self, context: str, response: str, num_trials: int = 3
    ) -> Dict[str, Any]:
        """
        Test consistency of mock runners for debugging.

        Args:
            context: Test context
            response: Test response
            num_trials: Number of trials to test consistency

        Returns:
            Dictionary with consistency test results
        """
        results: Dict[str, Any] = {
            "consistent": True,
            "logp_trials": [],
            "max_variance": 0.0,
            "runner_names": self.runner_names,
        }

        # Test multiple trials
        for trial in range(num_trials):
            trial_logps = self.logp_many(context, response)
            results["logp_trials"].append(trial_logps)

        # Check consistency across trials
        if num_trials > 1:
            logp_array = np.array(results["logp_trials"])  # Shape: (trials, K)
            variances = np.var(logp_array, axis=0)  # Variance per runner
            results["max_variance"] = float(np.max(variances))

            # Should be very consistent for mock runners (deterministic)
            if results["max_variance"] > 0.01:
                results["consistent"] = False

        return results

    def __repr__(self) -> str:
        """String representation for debugging."""
        runner_types = [type(r).__name__ for r in self.runners]
        return f"MockMultiTargetSampler(K={self.K}, runners={runner_types})"

    def __len__(self) -> int:
        """Number of target policies."""
        return self.K


# Factory function for easy creation
def create_mock_multi_sampler(
    policy_configs: List[Dict[str, Any]], consistent_ordering: bool = True
) -> MockMultiTargetSampler:
    """
    Factory function to create MockMultiTargetSampler from policy configurations.

    Args:
        policy_configs: List of policy configuration dictionaries
        consistent_ordering: If True, ensures deterministic ordering for testing

    Returns:
        MockMultiTargetSampler instance

    Example:
        configs = [
            {"model_name": "gpt-4", "provider": "openai", "temperature": 0.1},
            {"model_name": "gpt-3.5-turbo", "provider": "openai", "temperature": 0.7},
            {"model_name": "sshleifer/tiny-gpt2", "temperature": 1.0}  # Local model
        ]
        sampler = create_mock_multi_sampler(configs)
    """
    runners: List[Union[MockPolicyRunner, MockAPIPolicyRunner]] = []

    for config in policy_configs:
        config = config.copy()  # Don't modify original

        # Determine runner type based on provider
        provider = config.pop("provider", None)
        model_name = config.pop(
            "model_name", "mock-model"
        )  # Extract model_name separately

        runner: Union[MockPolicyRunner, MockAPIPolicyRunner]
        if provider in ["openai", "anthropic", "google"]:
            # API-based runner
            runner = MockAPIPolicyRunner(
                provider=provider, model_name=model_name, **config
            )
        else:
            # Local runner (default)
            runner = MockPolicyRunner(model_name=model_name, **config)

        runners.append(runner)

    return MockMultiTargetSampler(runners, consistent_ordering=consistent_ordering)


# Convenience function for creating simple temperature-based policies
def create_temperature_sweep_sampler(
    base_model: str = "gpt-4",
    provider: str = "openai",
    temperatures: List[float] = [0.1, 0.5, 1.0],
    **base_kwargs: Any,
) -> MockMultiTargetSampler:
    """
    Create a sampler with multiple temperature settings for the same base model.

    Args:
        base_model: Base model name
        provider: Provider name (for API models) or None for local
        temperatures: List of temperatures to test
        **base_kwargs: Additional parameters for all runners

    Returns:
        MockMultiTargetSampler with temperature sweep

    Example:
        # Test temperature effects with OpenAI models
        sampler = create_temperature_sweep_sampler(
            base_model="gpt-4",
            provider="openai",
            temperatures=[0.1, 0.7, 1.2]
        )

        # Test with local model
        sampler = create_temperature_sweep_sampler(
            base_model="sshleifer/tiny-gpt2",
            provider=None,
            temperatures=[0.0, 0.5, 1.0]
        )
    """
    configs = []

    for temp in temperatures:
        config = base_kwargs.copy()
        config.update({"model_name": base_model, "temperature": temp})

        if provider:
            config["provider"] = provider

        configs.append(config)

    return create_mock_multi_sampler(configs)
