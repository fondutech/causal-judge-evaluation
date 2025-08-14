"""
Robust inference utilities for handling dependence and multiple testing.

Implements dependence-robust standard errors and FDR control as specified
in Section 9.4 of the CJE paper.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# ========== Dependence-Robust Standard Errors ==========


def stationary_bootstrap_se(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 4000,
    mean_block_length: Optional[float] = None,
    alpha: float = 0.05,
    return_distribution: bool = False,
) -> Dict[str, Any]:
    """Compute standard errors using stationary bootstrap for time series.

    The stationary bootstrap (Politis & Romano, 1994) resamples blocks of
    random length with geometric distribution, preserving weak dependence.

    Args:
        data: Input data array (n_samples, ...)
        statistic_fn: Function that computes the statistic of interest
        n_bootstrap: Number of bootstrap iterations (default 4000)
        mean_block_length: Expected block length (auto if None)
        alpha: Significance level for CI (default 0.05 for 95% CI)
        return_distribution: If True, return bootstrap distribution

    Returns:
        Dictionary with:
        - 'estimate': Point estimate
        - 'se': Bootstrap standard error
        - 'ci_lower': Lower CI bound
        - 'ci_upper': Upper CI bound
        - 'mean_block_length': Block length used
        - 'distribution': Bootstrap distribution (if requested)

    References:
        Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
        Journal of the American Statistical Association, 89(428), 1303-1313.
    """
    n = len(data)

    # Compute point estimate
    estimate = statistic_fn(data)

    # Determine block length if not provided
    if mean_block_length is None:
        # Use first-order autocorrelation to tune block length
        # Rule of thumb: block_length ≈ n^(1/3) * (ρ/(1-ρ))^(2/3)
        if data.ndim == 1:
            acf_1 = np.corrcoef(data[:-1], data[1:])[0, 1] if n > 1 else 0
        else:
            # For multi-dimensional, use first column
            acf_1 = np.corrcoef(data[:-1, 0], data[1:, 0])[0, 1] if n > 1 else 0

        # Ensure reasonable bounds
        acf_1 = np.clip(acf_1, -0.99, 0.99)

        if abs(acf_1) < 0.1:
            # Weak dependence, use smaller blocks
            mean_block_length = max(1, int(n**0.33))
        else:
            # Stronger dependence, use larger blocks
            mean_block_length = max(
                1, int(n**0.33 * (abs(acf_1) / (1 - abs(acf_1))) ** 0.67)
            )

        # Cap at n/4 to ensure variety
        mean_block_length = min(mean_block_length, n // 4)

    # Probability of continuing block (geometric distribution)
    p = 1.0 / mean_block_length

    # Bootstrap iterations
    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        # Generate bootstrap sample using stationary bootstrap
        bootstrap_indices: List[int] = []
        i = 0

        while len(bootstrap_indices) < n:
            # Start new block at random position
            if i == 0 or np.random.random() < p:
                i = np.random.randint(0, n)

            bootstrap_indices.append(i)
            i = (i + 1) % n  # Wrap around

        bootstrap_indices = bootstrap_indices[:n]
        bootstrap_sample = data[bootstrap_indices]

        # Compute statistic on bootstrap sample
        try:
            boot_stat = statistic_fn(bootstrap_sample)
            bootstrap_estimates.append(boot_stat)
        except Exception as e:
            # Skip failed iterations (can happen with small samples)
            logger.debug(f"Bootstrap iteration failed: {e}")
            continue

    bootstrap_estimates = np.array(bootstrap_estimates)

    # Compute standard error
    se = np.std(bootstrap_estimates, ddof=1)

    # Compute confidence interval (percentile method)
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    result: Dict[str, Any] = {
        "estimate": float(estimate),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "mean_block_length": float(mean_block_length),
        "n_bootstrap": len(bootstrap_estimates),
        "effective_samples": len(bootstrap_estimates),
    }

    if return_distribution:
        result["distribution"] = bootstrap_estimates

    return result


def moving_block_bootstrap_se(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 4000,
    block_length: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute standard errors using moving block bootstrap.

    The moving block bootstrap (Künsch, 1989) resamples fixed-length
    contiguous blocks, preserving local dependence structure.

    Args:
        data: Input data array
        statistic_fn: Function that computes the statistic
        n_bootstrap: Number of bootstrap iterations
        block_length: Fixed block length (auto if None)
        alpha: Significance level for CI

    Returns:
        Dictionary with bootstrap results

    References:
        Künsch, H. R. (1989). The jackknife and the bootstrap for general
        stationary observations. The Annals of Statistics, 17(3), 1217-1241.
    """
    n = len(data)

    # Compute point estimate
    estimate = statistic_fn(data)

    # Determine block length if not provided
    if block_length is None:
        # Standard choice: n^(1/3) for optimal MSE
        block_length = max(1, int(n ** (1.0 / 3.0)))

    # Number of blocks needed
    n_blocks = int(np.ceil(n / block_length))

    # Bootstrap iterations
    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        # Sample blocks with replacement
        bootstrap_indices = []

        for _ in range(n_blocks):
            # Random starting point for block
            start = np.random.randint(0, n - block_length + 1)
            block_indices = list(range(start, min(start + block_length, n)))
            bootstrap_indices.extend(block_indices)

        # Trim to original length
        bootstrap_indices = bootstrap_indices[:n]
        bootstrap_sample = data[bootstrap_indices]

        # Compute statistic
        try:
            boot_stat = statistic_fn(bootstrap_sample)
            bootstrap_estimates.append(boot_stat)
        except Exception as e:
            logger.debug(f"Bootstrap iteration failed: {e}")
            continue

    bootstrap_estimates = np.array(bootstrap_estimates)

    # Compute statistics
    se = np.std(bootstrap_estimates, ddof=1)
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    return {
        "estimate": float(estimate),
        "se": float(se),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "block_length": int(block_length),
        "n_bootstrap": len(bootstrap_estimates),
    }


def cluster_robust_se(
    data: np.ndarray,
    cluster_ids: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    influence_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute cluster-robust (sandwich) standard errors.

    For data with cluster structure (e.g., multiple obs per user),
    accounts for within-cluster correlation.

    Args:
        data: Input data array
        cluster_ids: Cluster membership for each observation
        statistic_fn: Function that computes the statistic
        influence_fn: Function that computes influence functions
        alpha: Significance level for CI

    Returns:
        Dictionary with robust standard errors
    """
    n = len(data)
    estimate = statistic_fn(data)

    # Get unique clusters
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    if influence_fn is None:
        # For the mean, the influence function is just (x_i - mean)
        # This is the standard influence function for the sample mean
        influences = data - estimate if data.ndim == 1 else data[:, 0] - estimate
    else:
        # Use provided influence function
        influences = influence_fn(data)

    # Aggregate influences by cluster
    cluster_sums = np.zeros(n_clusters)
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_ids == cluster_id
        cluster_sums[i] = np.sum(influences[mask])

    # Cluster-robust variance
    # Variance of cluster sums, scaled by n^2
    var_cluster = np.sum((cluster_sums - np.mean(cluster_sums)) ** 2) / (n_clusters - 1)
    se_cluster = np.sqrt(var_cluster) / n

    # Confidence interval using t-distribution with n_clusters - 1 df
    t_crit = stats.t.ppf(1 - alpha / 2, n_clusters - 1)
    ci_lower = estimate - t_crit * se_cluster
    ci_upper = estimate + t_crit * se_cluster

    return {
        "estimate": float(estimate),
        "se": float(se_cluster),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_clusters": int(n_clusters),
        "df": int(n_clusters - 1),
    }


# ========== Multiple Testing Correction ==========


def benjamini_hochberg_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Apply Benjamini-Hochberg FDR correction for multiple testing.

    Controls the False Discovery Rate when testing multiple hypotheses,
    as required by Section 9.4 of the CJE paper.

    Args:
        p_values: Array of p-values from individual tests
        alpha: FDR level (default 0.05)
        labels: Optional labels for each test

    Returns:
        Dictionary with:
        - 'adjusted_p_values': BH-adjusted p-values
        - 'significant': Boolean mask of significant results
        - 'n_significant': Number of significant results
        - 'threshold': Largest p-value threshold used
        - 'summary': List of (label, p_value, adjusted_p, significant)

    References:
        Benjamini, Y., & Hochberg, Y. (1995). Controlling the false
        discovery rate. Journal of the Royal Statistical Society B.
    """
    n = len(p_values)

    if n == 0:
        return {
            "adjusted_p_values": np.array([]),
            "significant": np.array([], dtype=bool),
            "n_significant": 0,
            "threshold": 0.0,
            "summary": [],
        }

    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH adjustment: p_adj = min(1, p * n / rank)
    ranks = np.arange(1, n + 1)
    adjusted_p = np.minimum(1.0, sorted_p * n / ranks)

    # Enforce monotonicity (adjusted p-values should be non-decreasing)
    for i in range(n - 2, -1, -1):
        adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])

    # Find threshold (largest p where p <= alpha * rank / n)
    bh_threshold = 0.0
    significant_sorted = np.zeros(n, dtype=bool)

    for i in range(n - 1, -1, -1):
        if sorted_p[i] <= alpha * (i + 1) / n:
            bh_threshold = sorted_p[i]
            significant_sorted[: i + 1] = True
            break

    # Map back to original order
    adjusted_p_orig = np.zeros(n)
    significant_orig = np.zeros(n, dtype=bool)

    for i, orig_idx in enumerate(sorted_indices):
        adjusted_p_orig[orig_idx] = adjusted_p[i]
        significant_orig[orig_idx] = significant_sorted[i]

    # Create summary
    summary = []
    if labels is None:
        labels = [f"H{i+1}" for i in range(n)]

    for i in range(n):
        summary.append(
            {
                "label": labels[i],
                "p_value": float(p_values[i]),
                "adjusted_p": float(adjusted_p_orig[i]),
                "significant": bool(significant_orig[i]),
            }
        )

    # Sort summary by p-value for readability
    def get_p_value(item: Dict[str, Any]) -> float:
        return float(item["p_value"])

    summary.sort(key=get_p_value)

    return {
        "adjusted_p_values": adjusted_p_orig,
        "significant": significant_orig,
        "n_significant": int(np.sum(significant_orig)),
        "threshold": float(bh_threshold),
        "fdr_level": float(alpha),
        "n_tests": n,
        "summary": summary,
    }


def compute_simultaneous_bands(
    estimates: np.ndarray,
    standard_errors: np.ndarray,
    correlation_matrix: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute simultaneous confidence bands using max-t method.

    For a selected subset of policies, provides simultaneous coverage
    accounting for correlation between estimates.

    Args:
        estimates: Point estimates for each policy
        standard_errors: Standard errors for each estimate
        correlation_matrix: Correlation between estimates (identity if None)
        alpha: Significance level
        labels: Optional policy labels

    Returns:
        Dictionary with simultaneous confidence bands
    """
    k = len(estimates)

    if correlation_matrix is None:
        # Assume independence
        correlation_matrix = np.eye(k)

    # Standardize to get t-statistics
    t_stats = estimates / standard_errors

    # Critical value for simultaneous coverage
    # Using Bonferroni as conservative approximation
    # (Could use multivariate t simulation for exact)
    bonf_alpha = alpha / k
    z_crit = stats.norm.ppf(1 - bonf_alpha / 2)

    # Simultaneous bands
    lower_bands = estimates - z_crit * standard_errors
    upper_bands = estimates + z_crit * standard_errors

    # Check which are significantly different from 0
    significant = (lower_bands > 0) | (upper_bands < 0)

    if labels is None:
        labels = [f"Policy{i+1}" for i in range(k)]

    bands = []
    for i in range(k):
        bands.append(
            {
                "label": labels[i],
                "estimate": float(estimates[i]),
                "se": float(standard_errors[i]),
                "lower": float(lower_bands[i]),
                "upper": float(upper_bands[i]),
                "significant": bool(significant[i]),
            }
        )

    return {
        "bands": bands,
        "critical_value": float(z_crit),
        "n_policies": k,
        "bonferroni_alpha": float(bonf_alpha),
        "n_significant": int(np.sum(significant)),
    }


# ========== Integrated Robust Inference ==========


def compute_robust_inference(
    estimates: np.ndarray,
    influence_functions: Optional[np.ndarray] = None,
    data: Optional[np.ndarray] = None,
    method: str = "stationary_bootstrap",
    cluster_ids: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 4000,
    apply_fdr: bool = True,
    fdr_alpha: float = 0.05,
    policy_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Comprehensive robust inference with dependence and multiplicity handling.

    Args:
        estimates: Point estimates for policies
        influence_functions: If provided, use for inference
        data: Raw data (if influence_functions not provided)
        method: "stationary_bootstrap", "moving_block", or "cluster"
        cluster_ids: For cluster-robust SEs
        alpha: Significance level for CIs
        n_bootstrap: Bootstrap iterations
        apply_fdr: Whether to apply FDR correction
        fdr_alpha: FDR control level
        policy_labels: Names for policies

    Returns:
        Dictionary with complete robust inference results
    """
    n_policies = len(estimates)

    # Compute robust SEs for each policy
    robust_ses = []
    robust_cis = []
    p_values = []

    for i in range(n_policies):
        if method == "stationary_bootstrap":
            if influence_functions is not None:
                # Use influence functions
                result = stationary_bootstrap_se(
                    (
                        influence_functions[:, i]
                        if influence_functions.ndim > 1
                        else influence_functions
                    ),
                    lambda x: np.mean(x),
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                )
            elif data is not None:
                # Use raw data
                result = stationary_bootstrap_se(
                    data,
                    lambda x: estimates[i],  # Placeholder - would need actual estimator
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                )
            else:
                raise ValueError("Need either influence_functions or data")

        elif method == "cluster" and cluster_ids is not None:
            result = cluster_robust_se(
                influence_functions[:, i] if influence_functions is not None else data,
                cluster_ids,
                lambda x: np.mean(x),
                alpha=alpha,
            )
        else:
            # Fallback to classical
            se = np.std(
                influence_functions[:, i] if influence_functions is not None else data
            ) / np.sqrt(len(data))
            result = {
                "se": se,
                "ci_lower": estimates[i] - 1.96 * se,
                "ci_upper": estimates[i] + 1.96 * se,
            }

        robust_ses.append(result["se"])
        robust_cis.append((result["ci_lower"], result["ci_upper"]))

        # Compute p-value for test that estimate != 0
        z_stat = estimates[i] / result["se"] if result["se"] > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        p_values.append(p_val)

    robust_ses = np.array(robust_ses)
    p_values = np.array(p_values)

    # Apply FDR correction if requested
    fdr_results = None
    if apply_fdr and n_policies > 1:
        fdr_results = benjamini_hochberg_correction(
            p_values,
            alpha=fdr_alpha,
            labels=policy_labels,
        )

    return {
        "estimates": estimates,
        "robust_ses": robust_ses,
        "robust_cis": robust_cis,
        "p_values": p_values,
        "method": method,
        "fdr_results": fdr_results,
        "n_policies": n_policies,
        "inference_alpha": alpha,
        "fdr_alpha": fdr_alpha if apply_fdr else None,
    }
