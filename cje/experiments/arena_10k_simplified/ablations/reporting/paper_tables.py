"""
High information-density tables for paper.

This module generates the core tables (1-3) and diagnostic tables (A1-A6)
optimized for information density and clarity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from pathlib import Path
import json


def compute_debiased_rmse(
    results: List[Dict[str, Any]],
    policies: Optional[List[str]] = None,
    n_oracle_default: int = 4989,
) -> Dict[str, float]:
    """Compute oracle-noise-debiased RMSE (RMSE^d) per estimator config.

    RMSE^d is defined as sqrt(mean_p(max{0, (est_p - oracle_p)^2 - Var_oracle_p})).
    This adjusts for oracle sampling noise, providing a fairer accuracy metric when
    oracle truths are computed from finite samples.

    Args:
        results: List of experiment results
        policies: Policies to include (default: well-behaved only)
        n_oracle_default: Fallback oracle sample size if not inferable

    Returns:
        Dict mapping estimator config to RMSE^d
    """
    if policies is None:
        policies = ["clone", "parallel_universe_prompt", "premium"]

    rmse_by_config: Dict[str, List[float]] = {}

    for result in results:
        config_key = create_config_key(result)
        estimates = result.get("estimates", {}) or {}
        truths = result.get("oracle_truths", {}) or {}

        if not estimates or not truths:
            continue

        debiased_sq_errors: List[float] = []

        # Per-policy oracle counts if available; fall back to global default
        n_oracle_map = result.get("n_oracle_per_policy", {}) or {}
        n_oracle_global = int(
            result.get("n_oracle", n_oracle_default) or n_oracle_default
        )

        for pol in policies:
            if pol in estimates and pol in truths:
                est = float(estimates[pol])
                tru = float(truths[pol])
                err2 = (est - tru) ** 2
                n_pol = int(n_oracle_map.get(pol, n_oracle_global) or n_oracle_global)
                var_oracle = compute_oracle_variance(tru, n_pol)
                debiased = max(0.0, err2 - var_oracle)
                debiased_sq_errors.append(debiased)

        if debiased_sq_errors:
            rmse_d = float(np.sqrt(np.mean(debiased_sq_errors)))
            rmse_by_config.setdefault(config_key, []).append(rmse_d)

    # Average across seeds/experiments
    return {k: float(np.mean(v)) for k, v in rmse_by_config.items()}


def compute_interval_score_oa(
    results: List[Dict[str, Any]], alpha: float = 0.05
) -> Dict[str, float]:
    """Compute oracle-adjusted interval score (geometric mean).

    Interval score = width + 2/α * (coverage penalty)
    Lower is better - balances sharpness and calibration.

    Args:
        results: List of experiment results
        alpha: Significance level (default 0.05 for 95% CIs)

    Returns:
        Dict mapping estimator config to geometric mean interval score
    """
    scores_by_config: Dict[str, List[float]] = {}

    for result in results:
        config_key = create_config_key(result)

        # Get oracle-adjusted CIs and coverage (prefer robust, fall back to regular)
        robust_cis = result.get("robust_confidence_intervals") or result.get(
            "confidence_intervals", {}
        )
        oracle_truths = result.get("oracle_truths", {})

        if not robust_cis or not oracle_truths:
            continue

        scores = []
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy in robust_cis and policy in oracle_truths:
                ci = robust_cis[policy]
                truth = oracle_truths[policy]

                # Interval score formula
                width = ci[1] - ci[0]
                if truth < ci[0]:
                    penalty = 2 / alpha * (ci[0] - truth)
                elif truth > ci[1]:
                    penalty = 2 / alpha * (truth - ci[1])
                else:
                    penalty = 0

                score = width + penalty
                scores.append(score)

        if scores:
            # Use geometric mean to reduce impact of outliers
            geom_mean = np.exp(np.mean(np.log(scores)))
            if config_key not in scores_by_config:
                scores_by_config[config_key] = []
            scores_by_config[config_key].append(geom_mean)

    return {k: np.exp(np.mean(np.log(v))) for k, v in scores_by_config.items()}


def compute_oracle_variance(oracle_truth: float, n_oracle: int = 4989) -> float:
    """Compute conservative Bernoulli variance estimate for oracle.

    The oracle truth is always computed from the complete ~5k dataset,
    regardless of the experiment's sample size.

    Args:
        oracle_truth: Oracle point estimate (probability)
        n_oracle: Number of samples used to compute oracle truth (default 4989)

    Returns:
        Variance of oracle estimate
    """
    # Conservative Bernoulli variance (max at p=0.5)
    # Note: Oracle is always computed from full ~5k dataset
    return min(oracle_truth * (1 - oracle_truth), 0.25) / n_oracle


def compute_debiased_interval_score(
    results: List[Dict[str, Any]], alpha: float = 0.05, n_oracle: int = 4989
) -> Dict[str, float]:
    """Compute interval score accounting for oracle sampling uncertainty.

    Unlike the standard interval score which treats oracle as fixed truth,
    this accounts for the fact that oracle is a sample mean with variance.

    Args:
        results: List of experiment results
        alpha: Significance level (default 0.05 for 95% CIs)
        n_oracle: Number of samples used for oracle truth

    Returns:
        Dict mapping estimator config to debiased interval score
    """
    from scipy import stats as scipy_stats

    scores_by_config: Dict[str, List[float]] = {}

    for result in results:
        config_key = create_config_key(result)

        # Get robust CIs (which include OUA)
        robust_cis = result.get("robust_confidence_intervals") or result.get(
            "confidence_intervals", {}
        )
        oracle_truths = result.get("oracle_truths", {})

        if not robust_cis or not oracle_truths:
            continue

        scores = []
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy in robust_cis and policy in oracle_truths:
                ci_lower, ci_upper = robust_cis[policy]
                oracle_mean = oracle_truths[policy]

                # Compute oracle SE
                oracle_var = compute_oracle_variance(oracle_mean, n_oracle)
                oracle_se = np.sqrt(oracle_var)

                # Width component
                width = ci_upper - ci_lower

                if oracle_se > 1e-10:  # Oracle has uncertainty
                    # Compute expected penalty under oracle uncertainty
                    # Oracle ~ N(oracle_mean, oracle_se^2) approximately

                    # Standardize CI bounds relative to oracle distribution
                    z_lower = (ci_lower - oracle_mean) / oracle_se
                    z_upper = (ci_upper - oracle_mean) / oracle_se

                    # Expected penalty for undercoverage
                    # E[penalty | oracle < ci_lower] * P(oracle < ci_lower)
                    prob_below = scipy_stats.norm.cdf(z_lower)
                    if prob_below > 1e-10:
                        # Expected distance given oracle is below CI
                        # E[|X - ci_lower| | X < ci_lower] where X ~ N(0,1)
                        # This is E[ci_lower - X | X < z_lower] = -z_lower + pdf(z_lower)/cdf(z_lower)
                        exp_dist_below = oracle_se * abs(
                            scipy_stats.norm.pdf(z_lower) / prob_below - z_lower
                        )
                        penalty_below = (2 / alpha) * prob_below * exp_dist_below
                    else:
                        penalty_below = 0

                    # E[penalty | oracle > ci_upper] * P(oracle > ci_upper)
                    prob_above = 1 - scipy_stats.norm.cdf(z_upper)
                    if prob_above > 1e-10:
                        # Expected distance given oracle is above CI
                        # E[|X - ci_upper| | X > ci_upper] where X ~ N(0,1)
                        # This is E[X - ci_upper | X > z_upper] = pdf(z_upper)/(1-cdf(z_upper)) - z_upper
                        exp_dist_above = oracle_se * abs(
                            scipy_stats.norm.pdf(z_upper) / prob_above + z_upper
                        )
                        penalty_above = (2 / alpha) * prob_above * exp_dist_above
                    else:
                        penalty_above = 0

                    score = width + penalty_below + penalty_above
                else:
                    # Fall back to standard interval score if no oracle uncertainty
                    if oracle_mean < ci_lower:
                        penalty = 2 / alpha * (ci_lower - oracle_mean)
                    elif oracle_mean > ci_upper:
                        penalty = 2 / alpha * (oracle_mean - ci_upper)
                    else:
                        penalty = 0
                    score = width + penalty

                scores.append(score)

        if scores:
            # Geometric mean across policies
            geom_mean = np.exp(np.mean(np.log(scores)))
            if config_key not in scores_by_config:
                scores_by_config[config_key] = []
            scores_by_config[config_key].append(geom_mean)

    return {k: np.exp(np.mean(np.log(v))) for k, v in scores_by_config.items()}


def compute_debiased_calibration_score(
    results: List[Dict[str, Any]], target: float = 0.95, n_oracle: int = 4989
) -> Dict[str, float]:
    """Compute calibration score accounting for oracle sampling uncertainty.

    Instead of binary coverage, computes probabilistic coverage given
    that oracle is itself a random variable.

    Args:
        results: List of experiment results
        target: Target coverage (default 0.95)
        n_oracle: Number of samples used for oracle truth

    Returns:
        Dict mapping estimator config to debiased calibration score
    """
    from scipy import stats as scipy_stats

    calib_by_config: Dict[str, List[float]] = {}

    for result in results:
        config_key = create_config_key(result)

        # Get robust CIs (which include OUA)
        robust_cis = result.get("robust_confidence_intervals") or result.get(
            "confidence_intervals", {}
        )
        oracle_truths = result.get("oracle_truths", {})

        if not robust_cis or not oracle_truths:
            continue

        coverage_probs = []
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy in robust_cis and policy in oracle_truths:
                ci_lower, ci_upper = robust_cis[policy]
                oracle_mean = oracle_truths[policy]

                # Compute oracle SE
                oracle_var = compute_oracle_variance(oracle_mean, n_oracle)
                oracle_se = np.sqrt(oracle_var)

                if oracle_se > 1e-10:  # Oracle has uncertainty
                    # Probability that true value is in CI given oracle uncertainty
                    # True value ~ N(oracle_mean, oracle_se^2)
                    z_lower = (ci_lower - oracle_mean) / oracle_se
                    z_upper = (ci_upper - oracle_mean) / oracle_se

                    # P(true value in CI) = P(z_lower < Z < z_upper)
                    coverage_prob = scipy_stats.norm.cdf(
                        z_upper
                    ) - scipy_stats.norm.cdf(z_lower)
                else:
                    # No oracle uncertainty, use binary coverage
                    coverage_prob = 1.0 if ci_lower <= oracle_mean <= ci_upper else 0.0

                coverage_probs.append(coverage_prob)

        if coverage_probs:
            mean_coverage = np.mean(coverage_probs)
            calib_score = abs(mean_coverage - target)

            if config_key not in calib_by_config:
                calib_by_config[config_key] = []
            calib_by_config[config_key].append(calib_score)

    return {k: np.mean(v) for k, v in calib_by_config.items()}


def compute_calibration_score(
    results: List[Dict[str, Any]], target: float = 0.95
) -> Dict[str, float]:
    """Compute calibration score: mean |coverage - target|.

    Args:
        results: List of experiment results
        target: Target coverage (default 0.95)

    Returns:
        Dict mapping estimator config to calibration score (lower is better)
    """
    calib_by_config: Dict[str, List[float]] = {}

    for result in results:
        config_key = create_config_key(result)

        # Count coverage (prefer robust, fall back to regular)
        robust_cis = result.get("robust_confidence_intervals") or result.get(
            "confidence_intervals", {}
        )
        oracle_truths = result.get("oracle_truths", {})

        if not robust_cis or not oracle_truths:
            continue

        covered = []
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy in robust_cis and policy in oracle_truths:
                ci = robust_cis[policy]
                truth = oracle_truths[policy]
                covered.append(ci[0] <= truth <= ci[1])

        if covered:
            coverage = np.mean(covered)
            calib_score = abs(coverage - target)

            if config_key not in calib_by_config:
                calib_by_config[config_key] = []
            calib_by_config[config_key].append(calib_score)

    return {k: np.mean(v) for k, v in calib_by_config.items()}


def compute_se_geomean(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute geometric mean of robust SEs (sharpness proxy).

    Args:
        results: List of experiment results

    Returns:
        Dict mapping estimator config to geometric mean SE
    """
    se_by_config: Dict[str, List[float]] = {}

    for result in results:
        config_key = create_config_key(result)

        # Get robust SEs (prefer robust, fall back to base)
        ses = result.get("robust_standard_errors") or result.get("standard_errors", {})

        se_values = []
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy in ses and ses[policy] is not None and ses[policy] > 0:
                se_values.append(ses[policy])

        if se_values:
            geom_mean = np.exp(np.mean(np.log(se_values)))
            if config_key not in se_by_config:
                se_by_config[config_key] = []
            se_by_config[config_key].append(geom_mean)

    return {k: np.exp(np.mean(np.log(v))) for k, v in se_by_config.items()}


def compute_runtime_efficiency(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute runtime efficiency metrics.

    Args:
        results: List of experiment results

    Returns:
        Dict mapping estimator config to {'se_per_second': ..., 'gate_rate': ...}
    """
    efficiency_by_config: Dict[str, Dict[str, List[float]]] = {}

    for result in results:
        config_key = create_config_key(result)

        # Get runtime
        runtime = result.get("runtime_s")
        if runtime is None or runtime <= 0:
            continue

        # Get SEs for efficiency calculation
        ses = result.get("robust_standard_errors") or result.get("standard_errors", {})
        estimates = result.get("estimates", {})

        # Compute SE per second (lower is better - more efficient)
        se_values = []
        gate_count = 0
        total_policies = 0

        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            total_policies += 1
            if policy in ses and ses[policy] is not None and ses[policy] > 0:
                se_values.append(ses[policy])
            else:
                # Check if estimate is NaN/None (gated)
                if (
                    policy not in estimates
                    or estimates[policy] is None
                    or np.isnan(estimates[policy])
                ):
                    gate_count += 1

        if config_key not in efficiency_by_config:
            efficiency_by_config[config_key] = {"se_per_sec": [], "gate_rate": []}

        # SE per second (geometric mean SE / runtime)
        if se_values:
            geom_mean_se = np.exp(np.mean(np.log(se_values)))
            se_per_sec = geom_mean_se / runtime
            efficiency_by_config[config_key]["se_per_sec"].append(se_per_sec)

        # Gate rate (fraction of policies that failed)
        if total_policies > 0:
            gate_rate = gate_count / total_policies
            efficiency_by_config[config_key]["gate_rate"].append(gate_rate)

    # Average across experiments
    result_dict = {}
    for config, metrics in efficiency_by_config.items():
        result_dict[config] = {
            "se_per_second": (
                np.mean(metrics["se_per_sec"]) if metrics["se_per_sec"] else np.nan
            ),
            "gate_rate": (
                np.mean(metrics["gate_rate"]) * 100 if metrics["gate_rate"] else 0.0
            ),
        }

    return result_dict


def compute_ranking_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute ranking metrics (Kendall tau-b, Top-1 accuracy, Pairwise accuracy, Top-1 regret).

    Args:
        results: List of experiment results

    Returns:
        Dict mapping estimator config to {'kendall_tau': ..., 'top1_acc': ...,
                                           'pairwise_acc': ..., 'top1_regret': ...}
    """
    from scipy.stats import kendalltau
    from itertools import combinations

    ranking_by_config: Dict[str, Dict[str, List[float]]] = {}

    for result in results:
        config_key = create_config_key(result)

        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})

        # Get values for well-behaved policies
        est_values = []
        true_values = []
        policies = []
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy in estimates and policy in oracle_truths:
                est_values.append(estimates[policy])
                true_values.append(oracle_truths[policy])
                policies.append(policy)

        if len(est_values) >= 2:
            # Kendall tau-b (handles ties)
            tau, _ = kendalltau(est_values, true_values)

            # Top-1 accuracy
            if len(est_values) >= 3:
                best_est = np.argmax(est_values)
                best_true = np.argmax(true_values)
                top1_correct = best_est == best_true
            else:
                top1_correct = np.nan

            # Pairwise accuracy: fraction of correctly ordered pairs
            if len(est_values) >= 2:
                correct_pairs = 0
                total_pairs = 0
                for i, j in combinations(range(len(est_values)), 2):
                    # Check if ordering is preserved
                    est_ordering = est_values[i] > est_values[j]
                    true_ordering = true_values[i] > true_values[j]
                    if est_ordering == true_ordering:
                        correct_pairs += 1
                    total_pairs += 1
                pairwise_acc = (
                    correct_pairs / total_pairs if total_pairs > 0 else np.nan
                )
            else:
                pairwise_acc = np.nan

            # Top-1 regret: difference from true best value
            if len(est_values) >= 2:
                # What we selected as best
                selected_idx = np.argmax(est_values)
                selected_true_value = true_values[selected_idx]

                # True best value
                best_true_value = np.max(true_values)

                # Regret is the difference (higher is worse)
                top1_regret = best_true_value - selected_true_value
            else:
                top1_regret = np.nan

            if config_key not in ranking_by_config:
                ranking_by_config[config_key] = {
                    "tau": [],
                    "top1": [],
                    "pairwise": [],
                    "regret": [],
                }

            ranking_by_config[config_key]["tau"].append(tau)
            if not np.isnan(top1_correct):
                ranking_by_config[config_key]["top1"].append(top1_correct)
            if not np.isnan(pairwise_acc):
                ranking_by_config[config_key]["pairwise"].append(pairwise_acc)
            if not np.isnan(top1_regret):
                ranking_by_config[config_key]["regret"].append(top1_regret)

    # Average
    result_dict = {}
    for config, metrics in ranking_by_config.items():
        result_dict[config] = {
            "kendall_tau": np.mean(metrics["tau"]) if metrics["tau"] else np.nan,
            "top1_acc": np.mean(metrics["top1"]) * 100 if metrics["top1"] else np.nan,
            "pairwise_acc": (
                np.mean(metrics["pairwise"]) * 100 if metrics["pairwise"] else np.nan
            ),
            "top1_regret": np.mean(metrics["regret"]) if metrics["regret"] else np.nan,
        }

    return result_dict


def create_config_key(result: Dict[str, Any]) -> str:
    """Create a configuration key for grouping results."""
    spec = result.get("spec", {})
    estimator = spec.get("estimator", "unknown")

    # Check for flags in spec.extra first, then fall back to top-level
    extra = spec.get("extra", {})
    use_calib = extra.get(
        "use_weight_calibration", result.get("use_weight_calibration", False)
    )
    use_iic = extra.get("use_iic", result.get("use_iic", False))

    # Special cases that always have calibration
    if estimator in ["calibrated-ips", "orthogonalized-ips", "oc-dr-cpo", "stacked-dr"]:
        return f"{estimator} (iic={use_iic})"
    elif estimator in ["raw-ips", "tr-cpo", "tr-cpo-e"]:
        # Never calibrated
        return f"{estimator} (iic={use_iic})"
    else:
        return f"{estimator} (calib={use_calib}, iic={use_iic})"


def compute_robust_bounds(
    df: pd.DataFrame,
    percentile_low: float = 5,
    percentile_high: float = 95,
) -> Dict[str, Tuple[float, float]]:
    """Compute robust normalization bounds using percentiles.

    Args:
        df: DataFrame with metric columns
        percentile_low: Lower percentile for clipping (default 5)
        percentile_high: Upper percentile for clipping (default 95)

    Returns:
        Dict mapping metric names to (min, max) bounds
    """
    bounds = {}

    # Include both regular and debiased versions
    for col in [
        "RMSE_d",
        "IntervalScore_OA",
        "IntervalScore_d",
        "CalibScore",
        "CalibScore_d",
        "SE_GeoMean",
    ]:
        if col in df.columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                bounds[col] = (
                    float(np.percentile(valid_values, percentile_low)),
                    float(np.percentile(valid_values, percentile_high)),
                )
            else:
                bounds[col] = (0.0, 1.0)

    # Ranking metrics have natural bounds
    bounds["Kendall_tau"] = (-1.0, 1.0)
    bounds["Top1_Acc"] = (0.0, 100.0)
    bounds["Pairwise_Acc"] = (0.0, 100.0)

    # Top-1 regret: use percentiles since it's unbounded
    if "Top1_Regret" in df.columns:
        valid_values = df["Top1_Regret"].dropna()
        if len(valid_values) > 0:
            bounds["Top1_Regret"] = (
                float(np.percentile(valid_values, percentile_low)),
                float(np.percentile(valid_values, percentile_high)),
            )
        else:
            bounds["Top1_Regret"] = (0.0, 0.1)

    return bounds


def compute_outlier_robust_bounds(
    df: pd.DataFrame,
    outlier_method: str = "iqr",
    iqr_multiplier: float = 1.5,
) -> Dict[str, Tuple[float, float]]:
    """Compute normalization bounds with outlier handling.

    Uses IQR method to detect outliers, then applies min-max on capped values.
    This prevents extreme outliers from dominating the normalization while
    preserving the full range of typical values.

    Args:
        df: DataFrame with metric columns
        outlier_method: Method for outlier detection ('iqr' or 'none')
        iqr_multiplier: Multiplier for IQR fence (default 1.5)

    Returns:
        Dict mapping metric names to (min, max) bounds
    """
    bounds = {}

    # Metrics that need outlier handling
    metrics_with_outliers = [
        "RMSE_d",
        "IntervalScore_OA",
        "IntervalScore_d",
        "CalibScore",
        "CalibScore_d",
        "SE_GeoMean",
        "Top1_Regret",
    ]

    for col in metrics_with_outliers:
        if col not in df.columns:
            continue

        values = df[col].dropna().values
        if len(values) == 0:
            bounds[col] = (0.0, 1.0)
            continue

        if outlier_method == "iqr" and len(values) >= 4:
            # IQR-based outlier detection
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1

            # Calculate fences
            lower_fence = q1 - iqr_multiplier * iqr
            upper_fence = q3 + iqr_multiplier * iqr

            # Cap values at fences
            capped_values = np.clip(values, lower_fence, upper_fence)

            # Use min-max of capped values
            bounds[col] = (float(capped_values.min()), float(capped_values.max()))
        else:
            # Fall back to simple min-max
            bounds[col] = (float(values.min()), float(values.max()))

    # Ranking metrics with natural bounds (no outlier handling needed)
    bounds["Kendall_tau"] = (-1.0, 1.0)
    bounds["Top1_Acc"] = (0.0, 100.0)
    bounds["Pairwise_Acc"] = (0.0, 100.0)

    return bounds


def compute_aggregate_score(
    row: pd.Series,
    normalize_bounds: Dict[str, Tuple[float, float]],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute aggregate ranking score for an estimator.

    Components (weights adjusted per user request):
    - Ranking quality (30%): Kendall τ + Top-1 accuracy (higher is better)
    - Accuracy (25%): RMSE^d (lower is better)
    - Efficiency (25%): SE GeoMean (lower is better, proxy for sample efficiency)
    - Calibration (20%): CalibScore + IntervalScore^OA (lower is better)

    Args:
        row: Series with estimator metrics
        normalize_bounds: Dict with (min, max) bounds for each metric

    Returns:
        Aggregate score in [0, 100] where higher is better
    """
    score_components = []
    weight_list = []

    # Top-level weights across components (must sum to 1)
    # Defaults: 25% accuracy, 25% efficiency, 30% ranking, 20% calibration
    w = {
        "accuracy": 0.25,
        "efficiency": 0.25,
        "ranking": 0.30,
        "calibration": 0.20,
    }
    if isinstance(weights, dict):
        # Merge without trusting malformed input
        for k in list(w.keys()):
            if k in weights and isinstance(weights[k], (int, float)):
                w[k] = float(weights[k])
        # Renormalize to sum to 1
        total = sum(max(0.0, v) for v in w.values())
        if total > 0:
            for k in w:
                w[k] = max(0.0, w[k]) / total

    # Accuracy component (25%) - RMSE^d
    if not pd.isna(row.get("RMSE_d")):
        min_val, max_val = normalize_bounds["RMSE_d"]
        # Lower is better, so invert
        if max_val > min_val:
            normalized = 1 - (row["RMSE_d"] - min_val) / (max_val - min_val)
        else:
            normalized = 0.5
        score_components.append(normalized)
        weight_list.append(w["accuracy"])

    # Ranking component - Split between multiple metrics
    ranking_scores = []
    ranking_weights = []

    # Kendall tau (25% of ranking weight)
    if not pd.isna(row.get("Kendall_tau")):
        min_val, max_val = normalize_bounds.get("Kendall_tau", (-1, 1))
        # Higher is better, map from [-1, 1] to [0, 1]
        if max_val > min_val:
            normalized = (row["Kendall_tau"] - min_val) / (max_val - min_val)
        else:
            normalized = 0.5
        ranking_scores.append(normalized)
        ranking_weights.append(0.25)

    # Pairwise accuracy (30% of ranking weight - more robust than tau)
    if not pd.isna(row.get("Pairwise_Acc")):
        # Already in [0, 100], normalize to [0, 1]
        normalized = row["Pairwise_Acc"] / 100
        ranking_scores.append(normalized)
        ranking_weights.append(0.30)

    # Top-1 accuracy (25% of ranking weight)
    if not pd.isna(row.get("Top1_Acc")):
        # Already in [0, 100], normalize to [0, 1]
        normalized = row["Top1_Acc"] / 100
        ranking_scores.append(normalized)
        ranking_weights.append(0.25)

    # Top-1 regret (20% of ranking weight - penalty for wrong selection)
    if not pd.isna(row.get("Top1_Regret")):
        min_val, max_val = normalize_bounds.get("Top1_Regret", (0, 0.1))
        # Lower is better (it's a regret)
        if max_val > min_val:
            normalized = 1 - (row["Top1_Regret"] - min_val) / (max_val - min_val)
        else:
            normalized = 0.5
        ranking_scores.append(normalized)
        ranking_weights.append(0.20)

    if ranking_scores:
        ranking_score = np.average(ranking_scores, weights=ranking_weights)
        score_components.append(ranking_score)
        weight_list.append(w["ranking"])

    # Calibration component (20%) - Split between CalibScore and IntervalScore
    # Prefer debiased versions when available (they account for oracle uncertainty)
    calib_scores = []
    calib_weights = []

    # CalibScore - prefer debiased version
    calib_metric = (
        "CalibScore_d"
        if "CalibScore_d" in row and not pd.isna(row.get("CalibScore_d"))
        else "CalibScore"
    )
    if not pd.isna(row.get(calib_metric)):
        min_val, max_val = normalize_bounds.get(
            calib_metric, normalize_bounds.get("CalibScore", (0, 100))
        )
        # Lower is better (distance from 95% coverage)
        if max_val > min_val:
            normalized = 1 - (row[calib_metric] - min_val) / (max_val - min_val)
        else:
            normalized = 0.5
        calib_scores.append(normalized)
        calib_weights.append(0.5)  # 50% of calibration weight

    # IntervalScore - prefer debiased version
    interval_metric = (
        "IntervalScore_d"
        if "IntervalScore_d" in row and not pd.isna(row.get("IntervalScore_d"))
        else "IntervalScore_OA"
    )
    if not pd.isna(row.get(interval_metric)):
        min_val, max_val = normalize_bounds.get(
            interval_metric, normalize_bounds.get("IntervalScore_OA", (0, 1))
        )
        # Lower is better
        if max_val > min_val:
            normalized = 1 - (row[interval_metric] - min_val) / (max_val - min_val)
        else:
            normalized = 0.5
        calib_scores.append(normalized)
        calib_weights.append(0.5)  # 50% of calibration weight

    if calib_scores:
        # Weighted average of calibration components
        calib_score = np.average(calib_scores, weights=calib_weights)
        score_components.append(calib_score)
        weight_list.append(w["calibration"])

    # Efficiency component (25%) - SE GeoMean
    if not pd.isna(row.get("SE_GeoMean")):
        min_val, max_val = normalize_bounds["SE_GeoMean"]
        # Lower is better (more efficient)
        if max_val > min_val:
            normalized = 1 - (row["SE_GeoMean"] - min_val) / (max_val - min_val)
        else:
            normalized = 0.5
        score_components.append(normalized)
        weight_list.append(w["efficiency"])

    # Compute weighted average, handling missing components
    if score_components:
        # Renormalize weights to sum to 1
        weights_array = np.array(weight_list)
        weights_normalized = weights_array / weights_array.sum()
        aggregate = np.average(score_components, weights=weights_normalized)
        return float(aggregate * 100)  # Scale to 0-100
    else:
        return 0  # No valid metrics


def compute_mae(
    results: List[Dict[str, Any]], policies: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute Mean Absolute Error (MAE) per estimator configuration.

    Args:
        results: List of experiment results
        policies: Policies to include (default well-behaved)

    Returns:
        Dict mapping estimator config to MAE
    """
    if policies is None:
        policies = ["clone", "parallel_universe_prompt", "premium"]

    mae_by_config: Dict[str, List[float]] = {}

    for result in results:
        config_key = create_config_key(result)
        estimates = result.get("estimates", {}) or {}
        truths = result.get("oracle_truths", {}) or {}
        if not estimates or not truths:
            continue
        abs_errors = []
        for pol in policies:
            if pol in estimates and pol in truths:
                abs_errors.append(abs(float(estimates[pol]) - float(truths[pol])))
        if abs_errors:
            mae = float(np.mean(abs_errors))
            mae_by_config.setdefault(config_key, []).append(mae)

    return {k: float(np.mean(v)) for k, v in mae_by_config.items()}


def get_weight_preset(preset_name: str = "balanced") -> Dict[str, float]:
    """Get predefined weight presets for aggregate scoring.

    Args:
        preset_name: One of 'balanced', 'ranking', 'accuracy', 'inference'

    Returns:
        Dict with weights for accuracy, efficiency, ranking, calibration
    """
    presets = {
        "balanced": {
            "accuracy": 0.25,
            "efficiency": 0.25,
            "ranking": 0.30,
            "calibration": 0.20,
        },
        "ranking": {
            "accuracy": 0.20,
            "efficiency": 0.20,
            "ranking": 0.50,
            "calibration": 0.10,
        },
        "accuracy": {
            "accuracy": 0.40,
            "efficiency": 0.20,
            "ranking": 0.10,
            "calibration": 0.30,
        },
        "inference": {
            "accuracy": 0.20,
            "efficiency": 0.30,
            "ranking": 0.10,
            "calibration": 0.40,
        },
    }
    return presets.get(preset_name, presets["balanced"])


def generate_leaderboard(
    results: List[Dict[str, Any]],
    output_format: str = "dataframe",
    include_aggregate: bool = True,
    include_debiased: bool = True,
    weight_preset: str = "balanced",
    use_robust_bounds: bool = True,
) -> Any:
    """Generate Table 1: Estimator Leaderboard with optional aggregate ranking.

    Args:
        results: List of experiment results
        output_format: "dataframe", "latex", or "markdown"
        include_aggregate: Whether to compute and include aggregate score
        include_debiased: Whether to include debiased metrics
        weight_preset: Weight preset name ('balanced', 'ranking', 'accuracy', 'inference')
        use_robust_bounds: Whether to use robust percentile-based normalization

    Returns:
        Formatted table
    """
    # Compute all metrics
    rmse_d = compute_debiased_rmse(results)
    interval_scores = compute_interval_score_oa(results)
    calib_scores = compute_calibration_score(results)
    se_geomeans = compute_se_geomean(results)
    ranking_metrics = compute_ranking_metrics(results)

    # Compute debiased metrics if requested
    if include_debiased:
        interval_scores_d = compute_debiased_interval_score(results)
        calib_scores_d = compute_debiased_calibration_score(results)

    # Build rows
    rows = []
    all_configs = set(rmse_d.keys()) | set(interval_scores.keys())

    for config in sorted(all_configs):
        row = {
            "Estimator": config,
            "RMSE_d": rmse_d.get(config, np.nan),
            "IntervalScore_OA": interval_scores.get(config, np.nan),
            "CalibScore": calib_scores.get(config, np.nan)
            * 100,  # Convert to percentage
            "SE_GeoMean": se_geomeans.get(config, np.nan),
            "Kendall_tau": ranking_metrics.get(config, {}).get("kendall_tau", np.nan),
            "Top1_Acc": ranking_metrics.get(config, {}).get("top1_acc", np.nan),
            "Pairwise_Acc": ranking_metrics.get(config, {}).get("pairwise_acc", np.nan),
            "Top1_Regret": ranking_metrics.get(config, {}).get("top1_regret", np.nan),
        }

        # Add debiased metrics if computed
        if include_debiased:
            row["IntervalScore_d"] = interval_scores_d.get(config, np.nan)
            row["CalibScore_d"] = (
                calib_scores_d.get(config, np.nan) * 100
            )  # Convert to percentage

        rows.append(row)

    df = pd.DataFrame(rows)

    if include_aggregate and len(df) > 0:
        # Compute normalization bounds
        if use_robust_bounds:
            # Use outlier-robust min-max bounds (better for skewed distributions)
            normalize_bounds = compute_outlier_robust_bounds(df)
        else:
            # Use min/max bounds (original approach)
            normalize_bounds = {
                "RMSE_d": (df["RMSE_d"].min(), df["RMSE_d"].max()),
                "IntervalScore_OA": (
                    df["IntervalScore_OA"].min(),
                    df["IntervalScore_OA"].max(),
                ),
                "CalibScore": (df["CalibScore"].min(), df["CalibScore"].max()),
                "SE_GeoMean": (df["SE_GeoMean"].min(), df["SE_GeoMean"].max()),
                "Kendall_tau": (df["Kendall_tau"].min(), df["Kendall_tau"].max()),
                "Top1_Acc": (0, 100),  # Fixed bounds for percentage
                "Pairwise_Acc": (0, 100),
                "Top1_Regret": (df["Top1_Regret"].min(), df["Top1_Regret"].max()),
            }

        # Get weight preset
        weights = get_weight_preset(weight_preset)

        # Compute aggregate scores
        df["AggScore"] = df.apply(
            lambda row: compute_aggregate_score(row, normalize_bounds, weights), axis=1
        )

        # Sort by aggregate score (higher is better)
        df = df.sort_values("AggScore", ascending=False, na_position="last")

        # Add rank column
        df["Rank"] = range(1, len(df) + 1)
    else:
        # Sort by RMSE_d as primary metric
        df = df.sort_values("RMSE_d")

    if output_format == "dataframe":
        return df
    elif output_format == "latex":
        return format_leaderboard_latex(df, include_aggregate)
    elif output_format == "markdown":
        return format_leaderboard_markdown(df, include_aggregate)
    else:
        raise ValueError(f"Unknown format: {output_format}")


def format_leaderboard_latex(df: pd.DataFrame, include_aggregate: bool = True) -> str:
    """Format leaderboard as LaTeX table with best/second-best highlighting."""
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Estimator Leaderboard (Well-Behaved Policies Only)}")
    latex.append("\\label{tab:leaderboard}")

    if include_aggregate and "AggScore" in df.columns:
        latex.append("\\begin{tabular}{cl|c|cccccc}")
        latex.append("\\toprule")
        latex.append(
            "Rank & Estimator & Score & RMSE$^d$ $\\downarrow$ & IS$^{OA}$ $\\downarrow$ & CalibScore $\\downarrow$ & SE GM $\\downarrow$ & K-$\\tau$ $\\uparrow$ & Top-1 $\\uparrow$ \\\\"
        )
    else:
        latex.append("\\begin{tabular}{l|cccccc}")
        latex.append("\\toprule")
        latex.append(
            "Estimator & RMSE$^d$ $\\downarrow$ & IS$^{OA}$ $\\downarrow$ & CalibScore $\\downarrow$ & SE GM $\\downarrow$ & Kendall $\\tau$ $\\uparrow$ & Top-1 \\% $\\uparrow$ \\\\"
        )
    latex.append("\\midrule")

    # Find best and second-best for each metric
    best_idx = {}
    second_idx = {}

    for col in ["RMSE_d", "IntervalScore_OA", "CalibScore", "SE_GeoMean"]:
        # Filter out NaN values first, then sort
        valid_df = df[~df[col].isna()]
        if len(valid_df) >= 1:
            sorted_vals = valid_df[col].sort_values()
            best_idx[col] = sorted_vals.index[0]
        if len(valid_df) >= 2:
            sorted_vals = valid_df[col].sort_values()
            second_idx[col] = sorted_vals.index[1]

    for col in ["Kendall_tau", "Top1_Acc"]:
        # Filter out NaN values first, then sort descending
        valid_df = df[~df[col].isna()]
        if len(valid_df) >= 1:
            sorted_vals = valid_df[col].sort_values(ascending=False)
            best_idx[col] = sorted_vals.index[0]
        if len(valid_df) >= 2:
            sorted_vals = valid_df[col].sort_values(ascending=False)
            second_idx[col] = sorted_vals.index[1]

    # Format rows
    for idx, row in df.iterrows():
        cells = []

        if include_aggregate and "AggScore" in df.columns:
            cells.append(str(row.get("Rank", "")))
            cells.append(row["Estimator"])
            # Format aggregate score with bold
            score = row.get("AggScore", np.nan)
            if pd.isna(score):
                cells.append("--")
            else:
                cells.append(f"\\textbf{{{score:.1f}}}")
        else:
            cells.append(row["Estimator"])

        for col, fmt in [
            ("RMSE_d", ".4f"),
            ("IntervalScore_OA", ".4f"),
            ("CalibScore", ".1f"),
            ("SE_GeoMean", ".4f"),
            ("Kendall_tau", ".3f"),
            ("Top1_Acc", ".1f"),
        ]:
            val = row[col]
            if pd.isna(val):
                cells.append("--")
            else:
                formatted = f"{val:{fmt}}"
                if idx == best_idx.get(col):
                    formatted = f"\\textbf{{{formatted}}}"
                elif idx == second_idx.get(col):
                    formatted = f"\\underline{{{formatted}}}"
                cells.append(formatted)

        latex.append(" & ".join(cells) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")

    if include_aggregate and "AggScore" in df.columns:
        latex.append(
            "\\footnotesize{Aggregate Score: 30\\% ranking (K-$\\tau$+Top-1), 25\\% accuracy (RMSE$^d$), 25\\% efficiency (SE GM), 20\\% calibration (CalibScore+IS$^{OA}$)}"
        )

    latex.append("\\end{table}")

    return "\n".join(latex)


def format_leaderboard_markdown(
    df: pd.DataFrame, include_aggregate: bool = True
) -> str:
    """Format leaderboard as Markdown table."""
    # Round values for display
    df_display = df.copy()
    df_display["RMSE_d"] = df_display["RMSE_d"].round(4)
    df_display["IntervalScore_OA"] = df_display["IntervalScore_OA"].round(4)
    df_display["CalibScore"] = df_display["CalibScore"].round(1)
    df_display["SE_GeoMean"] = df_display["SE_GeoMean"].round(4)
    df_display["Kendall_tau"] = df_display["Kendall_tau"].round(3)
    df_display["Top1_Acc"] = df_display["Top1_Acc"].round(1)
    df_display["Pairwise_Acc"] = df_display["Pairwise_Acc"].round(1)
    df_display["Top1_Regret"] = df_display["Top1_Regret"].round(4)

    # Round debiased metrics if present
    if "IntervalScore_d" in df_display.columns:
        df_display["IntervalScore_d"] = df_display["IntervalScore_d"].round(4)
    if "CalibScore_d" in df_display.columns:
        df_display["CalibScore_d"] = df_display["CalibScore_d"].round(1)

    if include_aggregate and "AggScore" in df_display.columns:
        df_display["AggScore"] = df_display["AggScore"].round(1)
        # Reorder columns to put rank and score first, including debiased metrics
        cols = [
            "Rank",
            "Estimator",
            "AggScore",
            "RMSE_d",
            "IntervalScore_OA",
            "IntervalScore_d",
            "CalibScore",
            "CalibScore_d",
            "SE_GeoMean",
            "Kendall_tau",
            "Top1_Acc",
            "Pairwise_Acc",
            "Top1_Regret",
        ]
        df_display = df_display[[c for c in cols if c in df_display.columns]]

    return str(df_display.to_markdown(index=False))


def compute_paired_deltas(
    results: List[Dict[str, Any]],
    toggle: str = "use_weight_calibration",
    within: str = "estimator",
    hold_constant: Tuple[str, ...] = ("seed_base", "sample_size", "oracle_coverage"),
    focus_on_variance: bool = False,
) -> pd.DataFrame:
    """Compute paired differences for design choice effects.

    Args:
        results: List of experiment results
        toggle: Variable to toggle (e.g., 'use_weight_calibration', 'use_iic')
        within: Group comparisons within this variable
        hold_constant: Variables that must match for pairing
        focus_on_variance: If True, focus on SE/CI metrics instead of point estimates

    Returns:
        DataFrame with delta statistics and significance tests
    """
    # Build pairing index
    paired_data: Dict[Tuple[str, Tuple[Any, ...], bool], List[Dict[str, Any]]] = {}

    for result in results:
        spec = result.get("spec", {})

        # Extract key components
        within_val = spec.get(within, "unknown")
        toggle_val = result.get(toggle, spec.get("extra", {}).get(toggle, False))

        # Build matching key
        match_key = tuple(
            (
                spec.get(h)
                if h != "seed_base"
                else result.get("seed", spec.get("seed_base"))
            )
            for h in hold_constant
        )

        full_key = (within_val, match_key, toggle_val)

        if full_key not in paired_data:
            paired_data[full_key] = []
        paired_data[full_key].append(result)

    # Compute deltas for matched pairs
    delta_rows = []

    # Group by (within_val, match_key) and find on/off pairs
    for within_val in set(k[0] for k in paired_data.keys()):
        deltas_rmse = []
        deltas_interval: List[float] = []
        deltas_calib: List[float] = []
        deltas_se: List[float] = []
        deltas_tau: List[float] = []
        deltas_se_geomean = []

        for match_key in set(k[1] for k in paired_data.keys() if k[0] == within_val):
            # Find on and off versions
            key_on = (within_val, match_key, True)
            key_off = (within_val, match_key, False)

            if key_on in paired_data and key_off in paired_data:
                # Should be one result each for matched experiments
                if len(paired_data[key_on]) == 1 and len(paired_data[key_off]) == 1:
                    result_on = paired_data[key_on][0]
                    result_off = paired_data[key_off][0]

                    if not focus_on_variance:
                        # Compute deltas for point estimates
                        rmse_on = result_on.get("rmse_vs_oracle")
                        rmse_off = result_off.get("rmse_vs_oracle")
                        if rmse_on is not None and rmse_off is not None:
                            deltas_rmse.append(rmse_on - rmse_off)
                    else:
                        # Focus on SE changes for IIC
                        ses_on = result_on.get(
                            "robust_standard_errors"
                        ) or result_on.get("standard_errors", {})
                        ses_off = result_off.get(
                            "robust_standard_errors"
                        ) or result_off.get("standard_errors", {})

                        se_vals_on = []
                        se_vals_off = []
                        for policy in ["clone", "parallel_universe_prompt", "premium"]:
                            if policy in ses_on and policy in ses_off:
                                if ses_on[policy] > 0 and ses_off[policy] > 0:
                                    se_vals_on.append(ses_on[policy])
                                    se_vals_off.append(ses_off[policy])

                        if se_vals_on and se_vals_off:
                            # Geometric mean of SEs
                            geom_on = np.exp(np.mean(np.log(se_vals_on)))
                            geom_off = np.exp(np.mean(np.log(se_vals_off)))
                            # Percent change in SE
                            deltas_se_geomean.append(
                                (geom_on - geom_off) / geom_off * 100
                                if geom_off > 0
                                else 0
                            )

                    # Add more delta computations...
                    # (Similar for interval score, calibration, ranking)

        if deltas_rmse or deltas_se_geomean:
            # Compute statistics based on focus
            if not focus_on_variance and deltas_rmse:
                delta_row = {
                    within: within_val,
                    "n_pairs": len(deltas_rmse),
                    "ΔRMSE_d": np.mean(deltas_rmse),
                    "ΔRMSE_d_se": np.std(deltas_rmse) / np.sqrt(len(deltas_rmse)),
                    "ΔRMSE_d_p": (
                        stats.wilcoxon(deltas_rmse, alternative="two-sided").pvalue
                        if len(deltas_rmse) > 5 and len(set(deltas_rmse)) > 1
                        else np.nan
                    ),
                }

                # Add bootstrap CI
                if len(deltas_rmse) >= 20:
                    bootstrap_means = []
                    for _ in range(1000):
                        sample = np.random.choice(
                            deltas_rmse, size=len(deltas_rmse), replace=True
                        )
                        bootstrap_means.append(np.mean(sample))
                    delta_row["ΔRMSE_d_ci_low"] = np.percentile(bootstrap_means, 2.5)
                    delta_row["ΔRMSE_d_ci_high"] = np.percentile(bootstrap_means, 97.5)

                delta_rows.append(delta_row)
            elif focus_on_variance and deltas_se_geomean:
                delta_row = {
                    within: within_val,
                    "n_pairs": len(deltas_se_geomean),
                    "ΔSE_pct": np.mean(deltas_se_geomean),
                    "ΔSE_pct_se": np.std(deltas_se_geomean)
                    / np.sqrt(len(deltas_se_geomean)),
                    "ΔSE_pct_p": (
                        stats.wilcoxon(
                            deltas_se_geomean, alternative="two-sided"
                        ).pvalue
                        if len(deltas_se_geomean) > 5
                        and len(set(deltas_se_geomean)) > 1
                        else np.nan
                    ),
                }

                # Add bootstrap CI for SE change
                if len(deltas_se_geomean) >= 20:
                    bootstrap_means = []
                    for _ in range(1000):
                        sample = np.random.choice(
                            deltas_se_geomean, size=len(deltas_se_geomean), replace=True
                        )
                        bootstrap_means.append(np.mean(sample))
                    delta_row["ΔSE_pct_ci_low"] = np.percentile(bootstrap_means, 2.5)
                    delta_row["ΔSE_pct_ci_high"] = np.percentile(bootstrap_means, 97.5)

                delta_rows.append(delta_row)

    return pd.DataFrame(delta_rows)


def generate_delta_tables(
    results: List[Dict[str, Any]], output_format: str = "dataframe"
) -> Dict[str, Any]:
    """Generate Table 2: Effects of Design Choices (paired deltas).

    Args:
        results: List of experiment results
        output_format: "dataframe", "latex", or "markdown"

    Returns:
        Dict with panels A (calibration) and B (IIC)
    """
    # Panel A: Weight calibration effect (affects point estimates)
    panel_a = compute_paired_deltas(
        results, toggle="use_weight_calibration", focus_on_variance=False
    )

    # Panel B: IIC effect (affects SEs only, not point estimates)
    panel_b = compute_paired_deltas(results, toggle="use_iic", focus_on_variance=True)

    if output_format == "dataframe":
        return {"calibration": panel_a, "iic": panel_b}
    elif output_format == "latex":
        return {
            "calibration": format_delta_latex(
                panel_a, "Weight Calibration (SIMCal) Effect"
            ),
            "iic": format_delta_latex(panel_b, "IIC Effect"),
        }
    else:
        return {"calibration": panel_a.to_markdown(), "iic": panel_b.to_markdown()}


def format_delta_latex(df: pd.DataFrame, title: str) -> str:
    """Format delta table as LaTeX with significance markers."""
    latex = []
    latex.append(f"\\subsubsection{{{title}}}")

    # Check if this is an SE-focused table (for IIC)
    if "ΔSE_pct" in df.columns:
        latex.append("\\begin{tabular}{l|ccc}")
        latex.append("\\toprule")
        latex.append("Estimator & $\\Delta$SE (\\%) & CI & p-value \\\\")
        latex.append("\\midrule")

        for _, row in df.iterrows():
            cells = [row["estimator"]]

            # Format SE percentage change
            se_change = row.get("ΔSE_pct", np.nan)
            if not pd.isna(se_change):
                formatted = f"{se_change:.1f}\\%"

                # Add CI if available
                ci_low = row.get("ΔSE_pct_ci_low")
                ci_high = row.get("ΔSE_pct_ci_high")
                if ci_low is not None and ci_high is not None:
                    ci_str = f"[{ci_low:.1f}, {ci_high:.1f}]"
                else:
                    ci_str = "--"

                # Add p-value
                p_val = row.get("ΔSE_pct_p", np.nan)
                if not pd.isna(p_val):
                    if p_val < 0.001:
                        p_str = "< 0.001***"
                    elif p_val < 0.01:
                        p_str = f"{p_val:.3f}**"
                    elif p_val < 0.05:
                        p_str = f"{p_val:.3f}*"
                    else:
                        p_str = f"{p_val:.3f}"
                else:
                    p_str = "--"

                cells.extend([formatted, ci_str, p_str])
            else:
                cells.extend(["--", "--", "--"])

            latex.append(" & ".join(cells) + " \\\\")
    else:
        # Original format for RMSE-focused tables
        latex.append("\\begin{tabular}{l|ccccc}")
        latex.append("\\toprule")
        latex.append(
            "Estimator & $\\Delta$RMSE$^d$ & $\\Delta$IS$^{OA}$ & $\\Delta$CalibScore & $\\Delta$SE & $\\Delta\\tau$ \\\\"
        )
        latex.append("\\midrule")

        for _, row in df.iterrows():
            cells = [row["estimator"]]

            # Format each delta with CI and significance
            for metric in [
                "RMSE_d",
                "IntervalScore_OA",
                "CalibScore",
                "SE_GeoMean",
                "Kendall_tau",
            ]:
                delta_key = f"Δ{metric}"
                if delta_key in row:
                    val = row[delta_key]
                    ci_low = row.get(f"{delta_key}_ci_low")
                    ci_high = row.get(f"{delta_key}_ci_high")
                    p_val = row.get(f"{delta_key}_p")

                    # Format value
                    formatted = f"{val:.4f}"

                    # Add CI if available
                    if ci_low is not None and ci_high is not None:
                        formatted += f" [{ci_low:.4f}, {ci_high:.4f}]"

                    # Add significance marker
                    if p_val is not None:
                        if p_val < 0.001:
                            formatted += "***"
                        elif p_val < 0.01:
                            formatted += "**"
                        elif p_val < 0.05:
                            formatted += "*"

                    # Add direction arrow
                    if val > 0.001:
                        formatted += " ↑"
                    elif val < -0.001:
                        formatted += " ↓"

                    cells.append(formatted)
                else:
                    cells.append("--")

            latex.append(" & ".join(cells) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")

    return "\n".join(latex)


def compute_stacking_diagnostics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract and analyze stacking diagnostics.

    Since stacking-specific diagnostics aren't saved, we'll compare
    stacked-dr variants on available metrics.

    Args:
        results: List of experiment results

    Returns:
        DataFrame with stacking efficiency comparison
    """
    rows = []

    # Group by estimator and configuration
    estimator_data: Dict[str, Dict[str, List[float]]] = {}

    for result in results:
        spec = result.get("spec", {})
        estimator = spec.get("estimator", "")
        if not estimator.startswith("stacked"):
            continue

        config_key = create_config_key(result)

        if config_key not in estimator_data:
            estimator_data[config_key] = {
                "ses": [],
                "rmses": [],
                "coverages": [],
                "runtimes": [],
                "ess_values": [],
            }

        # Collect available metrics
        ses = result.get("standard_errors", {})
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy in ses and ses[policy] is not None:
                estimator_data[config_key]["ses"].append(ses[policy])

        rmse = result.get("rmse_vs_oracle")
        if rmse is not None:
            estimator_data[config_key]["rmses"].append(rmse)

        runtime = result.get("runtime_s")
        if runtime is not None:
            estimator_data[config_key]["runtimes"].append(runtime)

        # ESS relative - it's stored as a dict by policy
        ess_rel = result.get("ess_relative")
        if ess_rel and isinstance(ess_rel, dict):
            for policy in ["clone", "parallel_universe_prompt", "premium"]:
                if policy in ess_rel and ess_rel[policy] is not None:
                    estimator_data[config_key]["ess_values"].append(ess_rel[policy])

    # Build comparison rows
    for config_key, data in estimator_data.items():
        if data["ses"] and data["rmses"]:
            row = {
                "Estimator": config_key,
                "SE_GeoMean": (
                    np.exp(np.mean(np.log(data["ses"]))) if data["ses"] else np.nan
                ),
                "RMSE": np.mean(data["rmses"]) if data["rmses"] else np.nan,
                "Runtime_Median": (
                    np.median(data["runtimes"]) if data["runtimes"] else np.nan
                ),
                "ESS_Mean": (
                    np.mean(data["ess_values"]) * 100 if data["ess_values"] else np.nan
                ),
                "N_Experiments": len(data["rmses"]),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by estimator name for consistent ordering
    if not df.empty:
        df = df.sort_values("Estimator")

    return df


def generate_stacking_table(
    results: List[Dict[str, Any]], output_format: str = "dataframe"
) -> Any:
    """Generate Table 3: Stacked-DR Efficiency & Stability.

    Args:
        results: List of experiment results
        output_format: "dataframe", "latex", or "markdown"

    Returns:
        Formatted table
    """
    df = compute_stacking_diagnostics(results)

    if output_format == "dataframe":
        return df
    elif output_format == "latex":
        return format_stacking_latex(df)
    else:
        return df.to_markdown()


def format_stacking_latex(df: pd.DataFrame) -> str:
    """Format stacking performance as LaTeX table."""
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Stacked-DR Performance Summary}")
    latex.append("\\label{tab:stacking}")

    if df.empty:
        # Empty table
        latex.append("\\begin{tabular}{l}")
        latex.append("\\toprule")
        latex.append("No stacking data available \\\\")
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
    else:
        latex.append("\\begin{tabular}{l|cccc}")
        latex.append("\\toprule")
        latex.append("Metric & Value \\\\")
        latex.append("\\midrule")

        # Since we only have one estimator now, show as key-value pairs
        if len(df) == 1:
            row = df.iloc[0]

            # Format metrics as rows
            metrics = [
                ("RMSE", row.get("RMSE"), ".4f"),
                ("SE (Geometric Mean)", row.get("SE_GeoMean"), ".4f"),
                ("Runtime (median, s)", row.get("Runtime_Median"), ".1f"),
                ("N Experiments", row.get("N_Experiments"), ".0f"),
            ]

            for metric_name, value, fmt in metrics:
                if not pd.isna(value):
                    latex.append(f"{metric_name} & {value:{fmt}} \\\\")
                else:
                    latex.append(f"{metric_name} & -- \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")

    latex.append("\\end{table}")

    return "\n".join(latex)


def main() -> None:
    """Generate all paper tables."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper tables")
    parser.add_argument(
        "--results", type=Path, default=Path("results/all_experiments.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("tables/"))
    parser.add_argument(
        "--format", choices=["dataframe", "latex", "markdown"], default="latex"
    )
    args = parser.parse_args()

    # Load results
    results = []
    with open(args.results) as f:
        for line in f:
            results.append(json.loads(line))

    # Create output directory
    args.output.mkdir(exist_ok=True, parents=True)

    # Generate tables
    print("Generating Table 1: Leaderboard...")
    leaderboard = generate_leaderboard(results, args.format)
    if args.format == "latex":
        (args.output / "main" / "table1_leaderboard.tex").write_text(leaderboard)
    elif args.format == "markdown":
        (args.output / "main" / "table1_leaderboard.md").write_text(leaderboard)

    print("Generating Table 2: Design Choice Effects...")
    delta_tables = generate_delta_tables(results, args.format)
    if args.format == "latex":
        (args.output / "main" / "table2a_calibration.tex").write_text(
            delta_tables["calibration"]
        )
        (args.output / "main" / "table2b_iic.tex").write_text(delta_tables["iic"])
    elif args.format == "markdown":
        (args.output / "main" / "table2_deltas.md").write_text(
            f"## Calibration Effects\n\n{delta_tables['calibration']}\n\n"
            f"## IIC Effects\n\n{delta_tables['iic']}"
        )

    print("Generating Table 3: Stacking Diagnostics...")
    stacking = generate_stacking_table(results, args.format)
    if args.format == "latex":
        (args.output / "main" / "table3_stacking.tex").write_text(stacking)
    elif args.format == "markdown":
        (args.output / "main" / "table3_stacking.md").write_text(stacking)

    print(f"Tables written to {args.output}/")


if __name__ == "__main__":
    main()
