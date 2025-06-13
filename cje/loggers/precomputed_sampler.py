"""Pre-computed multi-target sampler.

This sampler is used when the dataset already contains log-probabilities under each
 target policy for every (context,response) pair (Scenario 3).
It provides the same public interface expected by estimators but never calls the
underlying language-model APIs, making the pipeline completely offline.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Sequence, Optional
import numpy as np


class PrecomputedMultiTargetSampler:
    """Sampler that serves pre-computed log-probabilities.

    Parameters
    ----------
    logp_lookup: Dict[Tuple[str, str], List[float]]
        Mapping from (context, response) to a length-K list of log-probabilities
        under each of the K target policies.
    n_policies: int
        Number of target policies (K).
    sample_lookup: Optional[Dict[str, List[List[str]]]]
        Optional mapping context → List[len-K] of response lists.
        Each entry sample_lookup[ctx][k] is a list of responses pre-sampled
        from target policy πᵏ.  These will be returned by :py:meth:`sample_many`.
    """

    def __init__(
        self,
        logp_lookup: Dict[Tuple[str, str], List[float]],
        n_policies: int,
        sample_lookup: Optional[Dict[str, List[List[str]]]] = None,
    ):
        """Create sampler.

        Parameters
        ----------
        logp_lookup: Mapping (context, response) → logp list (len-K)
        n_policies: Number of target policies (K)
        sample_lookup: Optional mapping context → List[len-K] of response lists.
            Each entry sample_lookup[ctx][k] is a list of responses pre-sampled
            from target policy πᵏ.  These will be returned by :py:meth:`sample_many`.
        """

        if n_policies <= 0:
            raise ValueError("n_policies must be at least 1")
        self._logp_lookup = logp_lookup
        self._sample_lookup = sample_lookup or {}
        self.K = n_policies

    # ------------------------------------------------------------------
    # Log-probabilities
    # ------------------------------------------------------------------
    def logp_many(self, context: str, response: str) -> List[float]:
        """Return pre-computed log πᵏ(response|context) for all K policies."""
        key = (context, response)
        if key not in self._logp_lookup:
            raise KeyError(
                "Precomputed log-probability missing for given (context, response)."
            )
        probs = self._logp_lookup[key]
        if len(probs) != self.K:
            raise ValueError(
                f"Expected {self.K} log-probabilities, found {len(probs)} for key {key}"
            )
        return [float(p) for p in probs]

    def logp_matrix(self, contexts: List[str], responses: List[str]) -> "np.ndarray[Any, Any]":  # type: ignore
        if len(contexts) != len(responses):
            raise ValueError("contexts and responses length mismatch")
        n = len(contexts)
        mat = np.zeros((n, self.K), dtype=float)  # type: ignore
        for i, (c, r) in enumerate(zip(contexts, responses)):
            mat[i, :] = self.logp_many(c, r)
        return mat

    # ------------------------------------------------------------------
    # Importance weights
    # ------------------------------------------------------------------
    def importance_weights_matrix(
        self,
        contexts: List[str],
        responses: List[str],
        logp_behavior: List[float],
        clip: Optional[float] = None,
        stabilize: bool = True,
        return_stats: bool = False,
    ) -> "np.ndarray[Any, Any]" | Tuple["np.ndarray[Any, Any]", Dict[str, Any]]:  # type: ignore
        """
        Compute importance weights matrix for multiple policies.

        Args:
            contexts: List of context strings
            responses: List of response strings
            logp_behavior: Log probabilities under behavior policy
            clip: Clipping value for importance weights (applied to final weights, not log-weights)
            stabilize: Whether to apply numerical stabilization for extreme log differences

        Returns:
            Importance weights matrix of shape (n, K)
        """
        import numpy as np

        target_logp = self.logp_matrix(contexts, responses)
        logp_beh = np.array(logp_behavior, dtype=np.float64)  # type: ignore

        # Compute log importance weights: log π'(s|x) - log π₀(s|x)
        log_weights_matrix = target_logp - logp_beh[:, None]

        # 🔧 INTERVENTION 1: Hard log-ratio clipping to prevent astronomical weights
        # Clip log ratios to ±20 (exp(20) ≈ 485M max weight ratio)
        log_ratio_clip = 20.0
        original_log_range = (np.min(log_weights_matrix), np.max(log_weights_matrix))
        if np.any(np.abs(log_weights_matrix) > log_ratio_clip):
            try:
                from cje.utils.progress import console

                console.print(
                    f"[yellow]✂️  Hard clipping log ratios to ±{log_ratio_clip} (prevents exp overflow)[/yellow]"
                )
                console.print(
                    f"   • Original range: [{original_log_range[0]:.1f}, {original_log_range[1]:.1f}]"
                )
                log_weights_matrix = np.clip(
                    log_weights_matrix, -log_ratio_clip, log_ratio_clip
                )
                console.print(
                    f"   • Clipped range: [{np.min(log_weights_matrix):.1f}, {np.max(log_weights_matrix):.1f}]"
                )
            except ImportError:
                # Fallback if console not available
                log_weights_matrix = np.clip(
                    log_weights_matrix, -log_ratio_clip, log_ratio_clip
                )

        # Track whether stabilization was actually applied
        stabilization_actually_applied = False

        if stabilize:
            # 🔧 INTERVENTION 2: Softer stabilization that preserves weight diversity
            # Check if we still need stabilization after hard clipping
            needs_stabilization = np.any(np.abs(log_weights_matrix) > 10)

            if needs_stabilization:
                stabilization_actually_applied = True
                # Import console here to avoid dependency issues
                try:
                    from cje.utils.progress import console

                    console.print(
                        "[yellow]🔧 Applying soft numerical stabilization (preserves weight diversity)[/yellow]"
                    )

                    # Softer approach: subtract 75th percentile per policy instead of global max
                    # This prevents winner-take-all while treating each policy fairly
                    percentile_75_per_policy = np.percentile(
                        log_weights_matrix, 75, axis=0
                    )
                    stabilized_log_weights = (
                        log_weights_matrix - percentile_75_per_policy
                    )

                    # Apply clipping in stabilized space (if enabled)
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
                    w = np.exp(stabilized_log_weights.astype(np.float64))

                    # Report stabilization details
                    console.print(
                        f"   • Original log weight range: [{np.min(log_weights_matrix):.1f}, {np.max(log_weights_matrix):.1f}]"
                    )
                    console.print(
                        f"   • Stabilized log weight range: [{np.min(stabilized_log_weights):.1f}, {np.max(stabilized_log_weights):.1f}]"
                    )
                    console.print(
                        f"   • Final weight range: [{np.min(w):.2e}, {np.max(w):.2e}]"
                    )

                    # 🔧 INTERVENTION 3: ESS guard-rail with per-policy warnings
                    ess_values = []
                    for k in range(w.shape[1]):
                        w_k = w[:, k]
                        ess_k = (
                            (w_k.sum()) ** 2 / (w_k**2).sum() if w_k.sum() > 0 else 0
                        )
                        ess_values.append(ess_k)

                    n_samples = len(w)
                    ess_percentages = [100 * ess / n_samples for ess in ess_values]
                    avg_ess_percentage = np.mean(ess_percentages)

                    console.print(
                        f"   📊 ESS per policy: {[f'{ess:.1f}' for ess in ess_values]} / {n_samples}"
                    )
                    console.print(
                        f"   📊 ESS percentages: {[f'{pct:.1f}%' for pct in ess_percentages]} (avg: {avg_ess_percentage:.1f}%)"
                    )

                    # Per-policy ESS guard-rails
                    critical_policies = []
                    warning_policies = []

                    for k, ess_pct in enumerate(ess_percentages):
                        policy_name = f"Policy {k}"  # PrecomputedSampler doesn't have policy names

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

                    # Check if stabilization preserved differences
                    unique_weights_per_sample = [
                        len(np.unique(w[i, :])) for i in range(w.shape[0])
                    ]
                    if any(n > 1 for n in unique_weights_per_sample):
                        console.print(
                            f"   ✅ Preserved weight differences across policies"
                        )
                    else:
                        console.print(
                            f"   ⚠️  All weights became identical (extreme case)"
                        )

                except ImportError:
                    # Fallback if console is not available
                    percentile_75_per_policy = np.percentile(
                        log_weights_matrix, 75, axis=0
                    )
                    stabilized_log_weights = (
                        log_weights_matrix - percentile_75_per_policy
                    )
                    if clip is not None:
                        log_clip = np.log(clip)
                        max_stabilized = np.max(stabilized_log_weights)
                        stabilized_log_weights = np.clip(
                            stabilized_log_weights,
                            max_stabilized - log_clip,
                            max_stabilized,
                        )
                    w = np.exp(stabilized_log_weights.astype(np.float64))
            else:
                # Standard clipping for normal cases (if enabled)
                if clip is not None:
                    log_clip = np.log(clip)
                    log_weights_matrix = np.clip(
                        log_weights_matrix, -log_clip, log_clip
                    )
                w = np.exp(log_weights_matrix.astype(np.float64))
        else:
            # Original approach without stabilization
            if clip is not None:
                log_clip = np.log(clip)
                log_weights_matrix = np.clip(log_weights_matrix, -log_clip, log_clip)
            w = np.exp(log_weights_matrix.astype(np.float64))

        # Final clipping only to prevent negative weights (should be unnecessary)
        w = np.maximum(w, 0)

        if return_stats:
            # Collect detailed statistics for reliability assessment
            n_samples = len(contexts)
            n_policies = w.shape[1]

            # Compute ESS per policy
            ess_values = []
            for k in range(n_policies):
                w_k = w[:, k]
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
                    n_clipped = np.sum(
                        np.abs(log_weights_matrix) > 50
                    )  # Stabilization threshold
                else:
                    # Direct clipping case
                    n_clipped = np.sum(w >= clip * 0.99)  # Near clipping threshold

            statistics = {
                "ess_values": ess_values,
                "ess_percentage": np.mean(ess_values) / n_samples * 100,
                "n_clipped": int(n_clipped),
                "clip_fraction": n_clipped / (n_samples * n_policies),
                "weight_range": (float(np.min(w)), float(np.max(w))),
                "stabilization_applied": stabilization_actually_applied,
                "n_samples": n_samples,
                "n_policies": n_policies,
            }

            return w, statistics  # type: ignore
        else:
            return w  # type: ignore

    # ------------------------------------------------------------------
    # Sampling interface (unused in pre-computed setting)
    # ------------------------------------------------------------------
    def sample_many(self, context: str, n: int = 1) -> List[List[str]]:  # noqa: D401
        """Return *n* samples per policy for a given context.

        If pre-computed samples were provided, they are returned (trimmed or
        duplicated to match *n*).  Otherwise returns empty strings.
        """
        if context in self._sample_lookup:
            out: List[List[str]] = []
            for k in range(self.K):
                policy_samples = (
                    self._sample_lookup[context][k]
                    if k < len(self._sample_lookup[context])
                    else []
                )
                if len(policy_samples) >= n:
                    out.append(policy_samples[:n])
                else:
                    # pad by repeating last sample or empty string
                    pad_with = policy_samples[-1] if policy_samples else ""
                    out.append(policy_samples + [pad_with] * (n - len(policy_samples)))
            return out
        # Fallback: return empty strings
        return [[""] * n for _ in range(self.K)]
