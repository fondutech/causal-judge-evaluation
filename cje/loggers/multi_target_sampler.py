"""
Multi-target policy sampler for efficient multi-policy evaluation.

This module provides the MultiTargetSampler class that can evaluate
logged sequences under multiple target policies simultaneously,
returning vectorized probabilities and samples.
"""

from typing import (
    List,
    Sequence,
    Union,
    Tuple,
    Any,
    Dict,
    Protocol,
    Optional,
    TYPE_CHECKING,
    cast,
)
import numpy as np
import logging

if TYPE_CHECKING:
    from ..config.unified import TargetPolicyConfig

from .policy import PolicyRunner
from .api_policy import APIPolicyRunner
from ..utils.error_handling import (
    safe_call,
    ErrorSeverity,
    FALLBACK_LOG_PROB,
    FALLBACK_RESPONSE,
    PolicyError,
)
from ..provider_registry import get_registry

logger = logging.getLogger(__name__)


class PolicyRunnerProtocol(Protocol):
    """Protocol for policy runners to allow mock classes in tests."""

    def log_prob(
        self, context: str, response: str, **kwargs: Any
    ) -> Union[float, Tuple[float, List[float]]]:
        """Compute log probability of response given context."""
        ...

    def generate_with_logp(
        self, prompts: List[str], **kwargs: Any
    ) -> List[Tuple[str, float, Any]]:
        """Generate responses with log probabilities."""
        ...


class MultiTargetSampler:
    """
    Holds K PolicyRunner instances and gives vectorised access
    to log-probabilities and samples under each target policy.

    This class enables efficient evaluation of logged sequences
    under multiple target policies {π¹, π², ..., πᴷ} for
    multi-policy evaluation scenarios.
    """

    def __init__(
        self,
        runners: Sequence[Union[PolicyRunner, APIPolicyRunner, PolicyRunnerProtocol]],
        policy_names: Optional[Sequence[str]] = None,
        log_ratio_clip: float = 20.0,  # P2: Make configurable via YAML
    ):
        """
        Initialize with a list of policy runners.

        Args:
            runners: Sequence of PolicyRunner, APIPolicyRunner, or compatible instances,
                    one for each target policy πᵏ
            policy_names: Optional sequence of policy names corresponding to runners.
                         If not provided, defaults to "policy_0", "policy_1", etc.
            log_ratio_clip: Hard clipping threshold for log ratios (default: 20.0)
        """
        if not runners:
            raise ValueError("At least one policy runner must be provided")

        self.runners = list(runners)  # Convert to list for internal use
        self.K = len(self.runners)  # Number of target policies
        self.log_ratio_clip = log_ratio_clip  # P2: Store configurable parameter

        # Store policy names
        if policy_names is not None:
            if len(policy_names) != self.K:
                raise ValueError(
                    f"Number of policy names ({len(policy_names)}) must match number of runners ({self.K})"
                )
            self.policy_names = list(policy_names)
        else:
            self.policy_names = [f"policy_{i}" for i in range(self.K)]

        self._logprob_warning_shown = False  # Track if we've shown the warning

    def logp_many(self, context: str, response: str) -> List[float]:
        """
        Return log πᵏ(response | context) for every runner.

        Args:
            context: Input context string
            response: Response sequence to evaluate

        Returns:
            List of log probabilities, length K (one per target policy)
        """
        logps = []

        for i, runner in enumerate(self.runners):
            policy_name = (
                self.policy_names[i] if i < len(self.policy_names) else f"policy_{i}"
            )

            logp_result = safe_call(
                runner.log_prob,  # type: ignore[arg-type]
                context,
                response,
                error_context=f"Computing log probability for policy {i} ({runner})",
                fallback=FALLBACK_LOG_PROB,
            )
            # Handle the case where log_prob returns a tuple (logp, token_logps)
            if isinstance(logp_result, tuple):
                logp = logp_result[0]
            else:
                logp = logp_result

            final_logp = float(logp if logp is not None else FALLBACK_LOG_PROB)
            logps.append(final_logp)

        # Validate identical policies for consistency
        self._validate_identical_policies_consistency(logps, context, response)

        return logps

    def _validate_identical_policies_consistency(
        self, logps: List[float], context: str, response: str
    ) -> None:
        """
        Validate that identical policies return identical log probabilities.

        Raises a detailed error if identical policies produce different results.
        This serves as a critical system health check.

        Args:
            logps: List of computed log probabilities
            context: Input context (for error reporting)
            response: Response text (for error reporting)
        """
        identical_groups = self._find_identical_policy_groups()

        if not identical_groups:
            # No identical policies found, nothing to validate
            return

        tolerance = 1e-10  # Very strict tolerance for identical policies
        validation_errors = []

        for group_name, policy_indices in identical_groups.items():
            if len(policy_indices) < 2:
                continue  # Need at least 2 policies to compare

            # Get log probabilities for this group
            group_logps = [logps[i] for i in policy_indices]
            group_names = [self.policy_names[i] for i in policy_indices]

            # Check if all results in this group are identical
            reference_logp = group_logps[0]
            differences = [abs(logp - reference_logp) for logp in group_logps[1:]]
            max_difference = max(differences) if differences else 0.0

            if max_difference > tolerance:
                # Build detailed error information
                policy_details = []
                for j, idx in enumerate(policy_indices):
                    runner = self.runners[idx]
                    details = {
                        "index": idx,
                        "name": group_names[j],
                        "logp": group_logps[j],
                        "signature": self._get_policy_signature(runner),
                    }
                    policy_details.append(details)

                error_info = {
                    "group": group_name,
                    "max_difference": max_difference,
                    "tolerance": tolerance,
                    "policy_details": policy_details,
                    "context_preview": context[:200],
                    "response_preview": response[:200],
                }
                validation_errors.append(error_info)

        if validation_errors:
            self._raise_identical_policies_error(validation_errors)

    def _raise_identical_policies_error(self, validation_errors: List[Dict]) -> None:
        """
        Raise a detailed error about identical policies producing different results.

        Args:
            validation_errors: List of validation error details
        """
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        error_msg = "\n🚨 IDENTICAL POLICIES VALIDATION FAILED 🚨\n"
        error_msg += "Identical policy configurations are producing different log probabilities!\n"
        error_msg += (
            "This indicates a serious issue with API determinism or configuration.\n\n"
        )

        for error in validation_errors:
            error_msg += f"Group: {error['group']}\n"
            error_msg += f"Max difference: {error['max_difference']:.10f} (tolerance: {error['tolerance']:.10f})\n"
            error_msg += f"Context: {error['context_preview']}...\n"
            error_msg += f"Response: {error['response_preview']}...\n\n"

            error_msg += "Policy Details:\n"
            for policy in error["policy_details"]:
                error_msg += f"  - Policy {policy['index']} ({policy['name']}): logp={policy['logp']:.10f}\n"
                error_msg += f"    Signature: {policy['signature']}\n"
            error_msg += "\n"

        error_msg += "POSSIBLE CAUSES:\n"
        error_msg += "1. API non-determinism (temperature > 0 with same seed)\n"
        error_msg += "2. Different API endpoints or model versions\n"
        error_msg += "3. Caching issues or stale connections\n"
        error_msg += "4. Configuration drift between policy instances\n\n"

        error_msg += "RECOMMENDED ACTIONS:\n"
        error_msg += "1. Check API provider documentation for determinism guarantees\n"
        error_msg += "2. Verify all policy configurations are truly identical\n"
        error_msg += "3. Consider using temperature=0 for deterministic behavior\n"
        error_msg += "4. Report issue to API provider if determinism is expected\n"

        # Print to console for immediate visibility
        console.print(
            Panel(error_msg, title="🚨 SYSTEM VALIDATION FAILURE", border_style="red")
        )

        # Raise exception to halt execution
        raise ValueError(
            f"Identical policies produced different log probabilities! "
            f"Found {len(validation_errors)} validation failures. "
            f"See console output above for detailed diagnostics."
        )

    def _find_identical_policy_groups(self) -> Dict[str, List[int]]:
        """
        Find groups of identical policies.

        Returns:
            Dict mapping group names to lists of policy indices that are identical.
        """
        signature_to_indices: Dict[str, List[int]] = {}

        for i, runner in enumerate(self.runners):
            signature = self._get_policy_signature(runner)

            if signature not in signature_to_indices:
                signature_to_indices[signature] = []
            signature_to_indices[signature].append(i)

        # Filter to only groups with multiple policies
        identical_groups = {
            f"group_{j}": indices
            for j, (signature, indices) in enumerate(signature_to_indices.items())
            if len(indices) > 1
        }

        if identical_groups:
            logger.debug(
                f"Found {len(identical_groups)} groups of identical policies for validation"
            )

        return identical_groups

    def _get_policy_signature(self, runner: Any) -> str:
        """
        Create a unique signature for a policy runner based on its configuration.

        Args:
            runner: Policy runner instance

        Returns:
            String signature uniquely identifying the policy configuration
        """
        signature_parts = []

        # Add core attributes that define policy behavior
        if hasattr(runner, "model_name"):
            signature_parts.append(f"model:{runner.model_name}")
        if hasattr(runner, "provider"):
            signature_parts.append(f"provider:{runner.provider}")
        if hasattr(runner, "temperature"):
            signature_parts.append(f"temp:{runner.temperature}")
        if hasattr(runner, "max_new_tokens"):
            signature_parts.append(f"max_tokens:{runner.max_new_tokens}")
        if hasattr(runner, "top_p"):
            signature_parts.append(f"top_p:{runner.top_p}")
        if hasattr(runner, "system_prompt"):
            signature_parts.append(f"system:{hash(runner.system_prompt or '')}")
        if hasattr(runner, "user_message_template"):
            signature_parts.append(f"template:{hash(runner.user_message_template)}")

        return "|".join(signature_parts)

    def sample_many(self, context: str, n: int = 1) -> List[List[str]]:
        """
        Sample `n` responses from each policy.

        Args:
            context: Input context string
            n: Number of samples to draw from each policy

        Returns:
            List of length K, where each element is a list of n response strings
            from the corresponding target policy πᵏ
        """
        all_samples = []

        for i, runner in enumerate(self.runners):
            samples = self._sample_from_runner(runner, context, n, i)
            all_samples.append(samples)

        return all_samples

    def _sample_from_runner(
        self,
        runner: Union[PolicyRunner, APIPolicyRunner, PolicyRunnerProtocol],
        context: str,
        n: int,
        runner_index: int,
    ) -> List[str]:
        """Sample from a single runner with unified error handling."""

        def _try_generate_with_logp() -> List[str]:
            if not hasattr(runner, "generate_with_logp"):
                raise PolicyError(f"Runner {runner} doesn't support generate_with_logp")
            results = runner.generate_with_logp([context] * n)
            return [
                str(result[0]) for result in results
            ]  # Extract text only and ensure string type

        def _try_generate() -> List[str]:
            if not hasattr(runner, "generate"):
                raise PolicyError(f"Runner {runner} doesn't support generate")
            # Most runners don't have a generate method, only generate_with_logp
            # This is a fallback that likely won't be used
            result = getattr(runner, "generate")([context] * n)
            return (
                [str(r) for r in result] if isinstance(result, list) else [str(result)]
            )

        def _create_fallback_samples() -> List[str]:
            return [f"{FALLBACK_RESPONSE}_{i}" for i in range(n)]

        # Try generate_with_logp first
        try:
            samples = safe_call(
                _try_generate_with_logp,
                error_context=f"Sampling from policy {runner_index} using generate_with_logp",
                fallback=None,
            )
        except Exception:
            samples = None

        if samples is not None:
            return samples

        # Try generate as fallback (though most runners don't have this method)
        try:
            samples = safe_call(
                _try_generate,
                error_context=f"Sampling from policy {runner_index} using generate",
                fallback=None,
            )
        except Exception:
            samples = None

        if samples is not None:
            return samples

        # Final fallback to dummy samples
        return _create_fallback_samples()

    def logp_matrix(self, contexts: List[str], responses: List[str]) -> "np.ndarray[Any, Any]":  # type: ignore[type-arg]
        """
        Compute log probability matrix for multiple context-response pairs.

        Args:
            contexts: List of context strings
            responses: List of response strings

        Returns:
            Log probability matrix of shape (n, K) where n = len(contexts)
        """
        if len(contexts) != len(responses):
            raise ValueError(
                f"Contexts and responses must have same length: {len(contexts)} vs {len(responses)}"
            )

        n = len(contexts)
        logp_matrix: "np.ndarray[Any, Any]" = np.zeros((n, self.K))  # type: ignore[type-arg]

        for i, (context, response) in enumerate(zip(contexts, responses)):
            logps = self.logp_many(context, response)
            logp_matrix[i, :] = logps

        return logp_matrix

    def importance_weights_matrix(
        self,
        contexts: List[str],
        responses: List[str],
        logp_behavior: List[float],
        clip: Optional[float] = None,
        stabilize: bool = True,
        return_stats: bool = False,
    ) -> "np.ndarray[Any, Any]" | Tuple["np.ndarray[Any, Any]", Dict[str, Any]]:  # type: ignore[type-arg]
        """
        Compute importance weights matrix for multiple policies.

        Args:
            contexts: List of context strings
            responses: List of response strings
            logp_behavior: Log probabilities under behavior policy
            clip: Clipping value for importance weights (None for no clipping - research mode)
            stabilize: Whether to apply numerical stabilization for extreme log differences
            return_stats: Whether to return detailed statistics alongside weights

        Returns:
            If return_stats=False: Importance weights matrix of shape (n, K)
            If return_stats=True: Tuple of (weights_matrix, statistics_dict)
        """
        logp_matrix = self.logp_matrix(contexts, responses)

        # 🔧 P1 FIX: Cast to float64 BEFORE subtraction to prevent overflow
        # logp_matrix may be float32; subtracting large opposite-sign numbers can overflow to ±inf
        logp_matrix = logp_matrix.astype(np.float64)
        logp_behavior_array: "np.ndarray[Any, Any]" = np.array(logp_behavior, dtype=np.float64)  # type: ignore[type-arg]

        # Compute log importance weights: log π'(s|x) - log π₀(s|x)
        log_weights_matrix: "np.ndarray[Any, Any]" = (  # type: ignore[type-arg]
            logp_matrix - logp_behavior_array[:, np.newaxis]
        )

        # 🔧 INTERVENTION 1: Hard log-ratio clipping to prevent astronomical weights
        # Clip log ratios to prevent exp(±log_ratio_clip) overflow/underflow
        # See MultiTargetSampler.__init__() for configuration
        original_log_range = (np.min(log_weights_matrix), np.max(log_weights_matrix))
        if np.any(np.abs(log_weights_matrix) > self.log_ratio_clip):
            from cje.utils.progress import console

            console.print(
                f"[yellow]✂️  Hard clipping log ratios to ±{self.log_ratio_clip} (prevents exp overflow)[/yellow]"
            )
            console.print(
                f"   • Original range: [{original_log_range[0]:.1f}, {original_log_range[1]:.1f}]"
            )
            log_weights_matrix = np.clip(
                log_weights_matrix, -self.log_ratio_clip, self.log_ratio_clip
            )
            console.print(
                f"   • Clipped range: [{np.min(log_weights_matrix):.1f}, {np.max(log_weights_matrix):.1f}]"
            )

        # Declare weights_matrix once
        weights_matrix: "np.ndarray[Any, Any]"  # type: ignore[type-arg]

        # Add warning for no clipping in research mode
        if clip is None:
            from cje.utils.progress import console

            console.print(
                "[yellow]🔬 Research mode: No importance weight clipping enabled. "
                "This may produce extreme variance but preserves theoretical unbiasedness.[/yellow]"
            )

        if stabilize:
            # 🔧 INTERVENTION 2: Softer stabilization that preserves weight diversity
            # Check if we still need stabilization after hard clipping
            needs_stabilization = np.any(np.abs(log_weights_matrix) > 10)

            if needs_stabilization:
                from cje.utils.progress import console

                console.print(
                    "[yellow]🔧 Applying soft numerical stabilization (preserves weight diversity)[/yellow]"
                )

                # Softer approach: subtract 75th percentile per policy instead of global max
                # This prevents winner-take-all while treating each policy fairly
                percentile_75_per_policy = np.percentile(log_weights_matrix, 75, axis=0)
                stabilized_log_weights = log_weights_matrix - percentile_75_per_policy

                # Apply clipping in stabilized space (if clipping enabled)
                if clip is not None:
                    log_clip = np.log(clip)
                    # More generous clipping bounds to preserve diversity
                    max_stabilized = np.max(stabilized_log_weights)
                    stabilized_log_weights = np.clip(
                        stabilized_log_weights,
                        max_stabilized - log_clip,  # Lower bound preserves ratios
                        max_stabilized,  # Upper bound at current max
                    )

                # Exponentiate stabilized weights (cast to float64 to prevent overflow)
                weights_matrix = np.exp(stabilized_log_weights.astype(np.float64))

                # Report stabilization details
                console.print(
                    f"   • Original log weight range: [{np.min(log_weights_matrix):.1f}, {np.max(log_weights_matrix):.1f}]"
                )
                console.print(
                    f"   • Stabilized log weight range: [{np.min(stabilized_log_weights):.1f}, {np.max(stabilized_log_weights):.1f}]"
                )
                console.print(
                    f"   • Raw weight range: [{np.min(weights_matrix):.2e}, {np.max(weights_matrix):.2e}]"
                )

                # Compute ESS for diagnostics (no normalization applied)
                ess_values = []
                for k in range(weights_matrix.shape[1]):
                    w_k = weights_matrix[:, k]
                    ess_k = (w_k.sum()) ** 2 / (w_k**2).sum() if w_k.sum() > 0 else 0
                    ess_values.append(ess_k)

                n_samples = len(weights_matrix)
                ess_percentages = [100 * ess / n_samples for ess in ess_values]
                avg_ess_percentage = np.mean(ess_percentages)

                console.print(
                    f"   📊 ESS per policy: {[f'{ess:.1f}' for ess in ess_values]} / {n_samples}"
                )
                console.print(
                    f"   📊 ESS percentages: {[f'{pct:.1f}%' for pct in ess_percentages]} (avg: {avg_ess_percentage:.1f}%)"
                )
                console.print(
                    f"   ✅ Final weight range: [{np.min(weights_matrix):.2e}, {np.max(weights_matrix):.2e}]"
                )
                console.print(f"   📈 Weight means: {np.mean(weights_matrix, axis=0)}")
                console.print(
                    f"   🔧 Weights will be calibrated per-fold in estimators"
                )

                # 🔧 INTERVENTION 3: ESS guard-rail with per-policy warnings
                critical_policies = []
                warning_policies = []

                for k, ess_pct in enumerate(ess_percentages):
                    policy_name = (
                        f"Policy {k}"
                        if not hasattr(self, "policy_names")
                        else (
                            self.policy_names[k]
                            if k < len(self.policy_names)
                            else f"Policy {k}"
                        )
                    )

                    if ess_pct < 5.0:
                        critical_policies.append((policy_name, ess_pct))
                    elif ess_pct < 15.0:
                        warning_policies.append((policy_name, ess_pct))

                if critical_policies:
                    console.print(f"[red]🚨 CRITICAL ESS detected![/red]")
                    for name, ess_pct in critical_policies:
                        console.print(
                            f"[red]   • {name}: {ess_pct:.1f}% - estimates will be unreliable![/red]"
                        )
                    console.print(
                        "[red]   💡 Solutions: (1) Increase sample size, (2) Use DRCPO/MRDR instead of IPS[/red]"
                    )
                elif warning_policies:
                    console.print(f"[yellow]⚠️  LOW ESS warnings:[/yellow]")
                    for name, ess_pct in warning_policies:
                        console.print(
                            f"[yellow]   • {name}: {ess_pct:.1f}% - estimates may be noisy[/yellow]"
                        )
                    console.print(
                        "[yellow]   💡 Consider: More samples or different target policies[/yellow]"
                    )
                else:
                    console.print(
                        f"   ✅ All policies have healthy ESS (min: {min(ess_percentages):.1f}%)"
                    )

                if clip is None:
                    console.print(f"   📊 No clipping applied (research mode)")
                else:
                    console.print(f"   ✂️  Clipping threshold: {clip}")

                # Check if stabilization preserved differences
                unique_weights_per_sample = [
                    len(np.unique(weights_matrix[i, :]))
                    for i in range(weights_matrix.shape[0])
                ]
                if any(n > 1 for n in unique_weights_per_sample):
                    console.print(f"   ✅ Preserved weight differences across policies")
                else:
                    console.print(f"   ⚠️  All weights became identical (extreme case)")
            else:
                # Standard processing for normal cases
                if clip is not None:
                    log_clip = np.log(clip)
                    log_weights_matrix = np.clip(
                        log_weights_matrix, -log_clip, log_clip
                    )
                weights_matrix = np.exp(log_weights_matrix.astype(np.float64))
        else:
            # Original approach without stabilization
            if clip is not None:
                log_clip = np.log(clip)
                log_weights_matrix = np.clip(log_weights_matrix, -log_clip, log_clip)
            weights_matrix = np.exp(log_weights_matrix.astype(np.float64))

        # Final check only to prevent negative weights (should be unnecessary)
        weights_matrix = np.maximum(weights_matrix, 0)

        if return_stats:
            # Collect detailed statistics for reliability assessment
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

            # Clipping statistics
            n_clipped = 0
            if clip is not None:
                # Check if any weights hit the clipping threshold
                # For stabilized weights, the threshold may be adjusted
                if stabilize:
                    # After stabilization, check against log_clip threshold in stabilized space
                    log_clip = np.log(clip)
                    # Reconstruct which weights would have been clipped in original space
                    original_log_weights = log_weights_matrix
                    n_clipped = int(
                        np.sum(np.abs(original_log_weights) > 50)
                    )  # Stabilization threshold
                else:
                    # Direct clipping case
                    n_clipped = int(
                        np.sum(weights_matrix >= clip * 0.99)
                    )  # Near clipping threshold

            statistics = {
                "ess_values": ess_values,
                "ess_percentage": np.mean(ess_values) / n_samples * 100,
                "n_clipped": int(n_clipped),
                "clip_fraction": n_clipped / (n_samples * n_policies),
                "weight_range": (
                    float(np.min(weights_matrix)),
                    float(np.max(weights_matrix)),
                ),
                "stabilization_applied": stabilize
                and locals().get("needs_stabilization", False),
                "n_samples": n_samples,
                "n_policies": n_policies,
            }

            return weights_matrix, statistics
        else:
            return weights_matrix


def make_multi_sampler(
    target_policies_cfg: Sequence[Union[Dict[str, Any], Any, "TargetPolicyConfig"]],
    diagnostics_config: Optional[
        Dict[str, Any]
    ] = None,  # P2: Accept diagnostics config
) -> MultiTargetSampler:
    """
    Create a MultiTargetSampler from configuration.

    Args:
        target_policies_cfg: List of target policy configurations (dicts, config objects, or TargetPolicyConfig)
        diagnostics_config: Optional diagnostics configuration (e.g., log_ratio_clip)

    Returns:
        MultiTargetSampler instance

    Raises:
        ValueError: If any target policy uses a provider that doesn't support full sequence logprobs
    """
    from .api_policy import APIPolicyRunner
    from .policy import PolicyRunner
    from ..provider_registry import get_registry

    # Get registry to validate providers
    registry = get_registry()

    # Extract diagnostics parameters
    log_ratio_clip = 20.0  # Default value
    if diagnostics_config and "log_ratio_clip" in diagnostics_config:
        log_ratio_clip = float(diagnostics_config["log_ratio_clip"])

    runners: List[Union[PolicyRunner, APIPolicyRunner]] = []
    policy_names: List[str] = []
    for i, policy_cfg in enumerate(target_policies_cfg):
        # Convert any config object to dictionary
        policy_dict: Optional[Dict[str, Any]] = None

        if hasattr(policy_cfg, "_content"):
            # Handle OmegaConf objects
            try:
                from omegaconf import OmegaConf

                container = OmegaConf.to_container(policy_cfg, resolve=True)
                if isinstance(container, dict):
                    policy_dict = cast(Dict[str, Any], container)
                else:
                    raise ValueError(
                        f"Policy {i}: OmegaConf object did not convert to dict"
                    )
            except Exception as e:
                raise ValueError(f"Policy {i}: Failed to convert OmegaConf object: {e}")
        elif hasattr(policy_cfg, "__dict__"):
            # Handle dataclass objects
            try:
                from dataclasses import asdict, is_dataclass

                if is_dataclass(policy_cfg) and not isinstance(policy_cfg, type):
                    # Only call asdict on dataclass instances, not types
                    policy_dict = asdict(policy_cfg)
                else:
                    policy_dict = {
                        k: v
                        for k, v in policy_cfg.__dict__.items()
                        if not k.startswith("_")
                    }
            except ImportError:
                policy_dict = {
                    k: v
                    for k, v in policy_cfg.__dict__.items()
                    if not k.startswith("_")
                }
        elif isinstance(policy_cfg, dict):
            # It's already a dictionary
            policy_dict = policy_cfg
        elif hasattr(policy_cfg, "get") and hasattr(policy_cfg, "keys"):
            # It's a dictionary-like object
            policy_dict = dict(policy_cfg)  # type: ignore[arg-type]
        else:
            # Try to treat as dict directly
            try:
                policy_dict = dict(policy_cfg)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                # Last resort - try to convert to dict if it has items()
                if hasattr(policy_cfg, "items"):
                    policy_dict = dict(policy_cfg.items())  # type: ignore[arg-type]
                else:
                    raise ValueError(
                        f"Policy {i}: Unable to convert policy config to dictionary. Got: {type(policy_cfg)}"
                    )

        # Validate required fields - no inference allowed
        if not policy_dict:
            raise ValueError(f"Policy {i}: Policy configuration is empty or None")
        if "name" not in policy_dict:
            raise ValueError(
                f"Policy {i}: 'name' is required and must be explicitly specified. Available keys: {list(policy_dict.keys())}"
            )
        if "model_name" not in policy_dict:
            raise ValueError(
                f"Policy {i}: 'model_name' is required and must be explicitly specified. Available keys: {list(policy_dict.keys())}"
            )
        if "provider" not in policy_dict:
            raise ValueError(
                f"Policy {i}: 'provider' is required and must be explicitly specified. Available keys: {list(policy_dict.keys())}"
            )

        # Extract policy name
        policy_name = str(policy_dict["name"])
        policy_names.append(policy_name)

        provider = policy_dict["provider"]
        model_name = policy_dict["model_name"]

        # CRITICAL VALIDATION: Check if provider supports full sequence logprobs
        # Skip validation for mock providers since they support everything for testing
        if provider in ["openai", "anthropic", "google"] and provider != "mock":
            # These providers don't support full sequence logprobs for policy evaluation
            supported_providers = ["together", "fireworks", "hf", "mock"]
            raise ValueError(
                f"Policy {i} (provider='{provider}'): This provider does not support full sequence "
                f"log probabilities required for policy evaluation. "
                f"Supported providers: {supported_providers}"
            )

        # Create the appropriate runner based on provider
        runner: Union[PolicyRunner, APIPolicyRunner]

        # No inference - user must explicitly specify provider
        if provider in ["openai", "anthropic", "google", "together", "fireworks"]:
            # API-based model
            api_kwargs = {
                k: v
                for k, v in policy_dict.items()
                if k
                in [
                    "max_new_tokens",
                    "temperature",
                    "top_p",
                    "batch_size",
                    "system_prompt",
                    "user_message_template",
                ]
            }
            runner = APIPolicyRunner(  # type: ignore[arg-type]
                provider=provider,
                model_name=model_name,
                **api_kwargs,
            )
        elif provider == "mock":
            # Mock provider for testing
            try:
                from ..testing.mocks.policy_runners import MockAPIPolicyRunner

                api_kwargs = {
                    k: v
                    for k, v in policy_dict.items()
                    if k in ["max_new_tokens", "temperature", "top_p", "batch_size"]
                }
                runner = MockAPIPolicyRunner(
                    provider=provider,
                    model_name=model_name,
                    **api_kwargs,
                )  # type: ignore[assignment]
            except ImportError:
                raise ValueError(
                    f"Policy {i}: Mock provider requested but testing module not available"
                )
        elif provider == "hf":
            # Local HuggingFace model
            hf_kwargs = {
                k: v
                for k, v in policy_dict.items()
                if k
                in [
                    "device",
                    "max_new_tokens",
                    "temperature",
                    "top_p",
                    "system_prompt",
                    "user_message_template",
                    "text_format",
                ]
            }
            runner = PolicyRunner(  # type: ignore[arg-type]
                model_name=model_name,
                **hf_kwargs,
            )
        else:
            raise ValueError(
                f"Policy {i}: Unknown provider '{provider}'. Valid providers: hf, openai, anthropic, google, together, fireworks, mock"
            )

        runners.append(runner)

    return MultiTargetSampler(runners, policy_names, log_ratio_clip)
