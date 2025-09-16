"""
Appendix tables focused on diagnostics and SIMCal gains.

This module now prioritizes robust, actionable diagnostics centered on
effective sample size (ESS) improvements from SIMCal and weight stability
changes. Legacy generators remain available but are not invoked by default.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json


def generate_quadrant_leaderboard(
    results: List[Dict[str, Any]], include_debiased: bool = True
) -> pd.DataFrame:
    """Generate Table A1: Quadrant-specific leaderboard.

    Shows RMSE_d and CalibScore broken down by data regime quadrant.
    Optionally includes debiased versions that account for oracle uncertainty.

    Args:
        results: List of experiment results
        include_debiased: Whether to include debiased calibration scores

    Returns:
        DataFrame with quadrant-specific metrics
    """
    from .paper_tables import create_config_key, compute_oracle_variance
    from scipy import stats as scipy_stats

    metrics_by_quadrant: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}

    for result in results:
        config_key = create_config_key(result)

        # Compute quadrant from spec
        spec = result.get("spec", {})
        size = spec.get("sample_size", 0)
        coverage = spec.get("oracle_coverage", 0)

        if size <= 1000:
            size_label = "Small"
        else:
            size_label = "Large"

        if coverage <= 0.1:
            cov_label = "LowOracle"
        else:
            cov_label = "HighOracle"

        quadrant = f"{size_label}-{cov_label}"

        if quadrant not in metrics_by_quadrant:
            metrics_by_quadrant[quadrant] = {}
        if config_key not in metrics_by_quadrant[quadrant]:
            metrics_by_quadrant[quadrant][config_key] = {
                "rmse": [],
                "coverage": [],
                "coverage_debiased": [],  # For debiased calibration
            }

        # Collect RMSE
        rmse = result.get("rmse_vs_oracle")
        if rmse is not None:
            metrics_by_quadrant[quadrant][config_key]["rmse"].append(rmse)

        # Collect coverage for calibration score
        robust_cis = result.get("robust_confidence_intervals") or result.get(
            "confidence_intervals", {}
        )
        oracle_truths = result.get("oracle_truths", {})

        covered = []
        coverage_probs = []  # For debiased version

        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy in robust_cis and policy in oracle_truths:
                ci_lower, ci_upper = robust_cis[policy]
                oracle_mean = oracle_truths[policy]

                # Standard binary coverage
                covered.append(ci_lower <= oracle_mean <= ci_upper)

                # Debiased probabilistic coverage
                if include_debiased:
                    oracle_var = compute_oracle_variance(oracle_mean, n_oracle=4989)
                    oracle_se = np.sqrt(oracle_var)

                    if oracle_se > 1e-10:
                        # Probability that true value is in CI given oracle uncertainty
                        z_lower = (ci_lower - oracle_mean) / oracle_se
                        z_upper = (ci_upper - oracle_mean) / oracle_se
                        coverage_prob = scipy_stats.norm.cdf(
                            z_upper
                        ) - scipy_stats.norm.cdf(z_lower)
                    else:
                        coverage_prob = (
                            1.0 if ci_lower <= oracle_mean <= ci_upper else 0.0
                        )

                    coverage_probs.append(coverage_prob)

        if covered:
            coverage = np.mean(covered)
            metrics_by_quadrant[quadrant][config_key]["coverage"].append(coverage)

        if coverage_probs:
            coverage_debiased = np.mean(coverage_probs)
            metrics_by_quadrant[quadrant][config_key]["coverage_debiased"].append(
                coverage_debiased
            )

    # Build table with quadrants as columns
    rows = []
    all_configs: set[str] = set()
    for q_metrics in metrics_by_quadrant.values():
        all_configs.update(q_metrics.keys())

    for config in sorted(all_configs):
        row = {"Estimator": config}

        for quadrant in [
            "Small-LowOracle",
            "Small-HighOracle",
            "Large-LowOracle",
            "Large-HighOracle",
        ]:
            abbrev = {
                "Small-LowOracle": "SL",
                "Small-HighOracle": "SH",
                "Large-LowOracle": "LL",
                "Large-HighOracle": "LH",
            }[quadrant]

            if (
                quadrant in metrics_by_quadrant
                and config in metrics_by_quadrant[quadrant]
            ):
                rmse_vals = metrics_by_quadrant[quadrant][config]["rmse"]
                cov_vals = metrics_by_quadrant[quadrant][config]["coverage"]
                cov_debiased_vals = metrics_by_quadrant[quadrant][config][
                    "coverage_debiased"
                ]

                if rmse_vals:
                    row[f"{abbrev}_RMSE"] = np.mean(rmse_vals)
                else:
                    row[f"{abbrev}_RMSE"] = np.nan

                if cov_vals:
                    row[f"{abbrev}_Calib"] = abs(np.mean(cov_vals) - 0.95) * 100
                else:
                    row[f"{abbrev}_Calib"] = np.nan

                # Add debiased calibration if requested
                if include_debiased and cov_debiased_vals:
                    row[f"{abbrev}_Calib_d"] = (
                        abs(np.mean(cov_debiased_vals) - 0.95) * 100
                    )
                elif include_debiased:
                    row[f"{abbrev}_Calib_d"] = np.nan
            else:
                row[f"{abbrev}_RMSE"] = np.nan
                row[f"{abbrev}_Calib"] = np.nan
                if include_debiased:
                    row[f"{abbrev}_Calib_d"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def generate_bias_patterns_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate Table A2: Bias Patterns.

    Shows mean bias, mean |bias|, and per-policy biases with significance.

    Args:
        results: List of experiment results

    Returns:
        DataFrame with bias analysis
    """
    from .paper_tables import create_config_key

    bias_by_config: Dict[str, Dict[str, List[float]]] = {}

    for result in results:
        config_key = create_config_key(result)

        if config_key not in bias_by_config:
            bias_by_config[config_key] = {
                "all_errors": [],
                "clone_errors": [],
                "parallel_errors": [],
                "premium_errors": [],
                "unhelpful_errors": [],
            }

        estimates = result.get("estimates", {})
        oracle_truths = result.get("oracle_truths", {})

        # Collect errors by policy
        for policy in ["clone", "parallel_universe_prompt", "premium", "unhelpful"]:
            if policy in estimates and policy in oracle_truths:
                error = estimates[policy] - oracle_truths[policy]

                if policy == "parallel_universe_prompt":
                    bias_by_config[config_key]["parallel_errors"].append(error)
                else:
                    bias_by_config[config_key][f"{policy}_errors"].append(error)

                # Add to overall (well-behaved only)
                if policy != "unhelpful":
                    bias_by_config[config_key]["all_errors"].append(error)

    # Compute statistics
    rows = []
    for config, errors_dict in bias_by_config.items():
        row: Dict[str, Any] = {"Estimator": config}

        # Overall bias (well-behaved only)
        if errors_dict["all_errors"]:
            all_errors = errors_dict["all_errors"]
            row["Mean_Bias"] = np.mean(all_errors)
            row["Mean_Abs_Bias"] = np.mean(np.abs(all_errors))
            row["Bias_SE"] = np.std(all_errors) / np.sqrt(len(all_errors))

            # Classify pattern
            mean_bias = float(row["Mean_Bias"])
            mean_abs_bias = float(row["Mean_Abs_Bias"])
            if mean_bias < -0.005:
                row["Pattern"] = "Negative"
            elif mean_bias > 0.005:
                row["Pattern"] = "Positive"
            elif mean_abs_bias < 0.01:
                row["Pattern"] = "Unbiased"
            else:
                row["Pattern"] = "Mixed"

        # Per-policy bias with t-stats
        for policy, key in [
            ("clone", "clone_errors"),
            ("parallel", "parallel_errors"),
            ("premium", "premium_errors"),
        ]:
            if errors_dict[key]:
                errors = errors_dict[key]
                mean_bias = np.mean(errors)
                se_bias = np.std(errors) / np.sqrt(len(errors))
                t_stat = abs(mean_bias / se_bias) if se_bias > 0 else 0

                row[f"{policy}_bias"] = mean_bias
                row[f"{policy}_t"] = t_stat
                row[f"{policy}_sig"] = "*" if t_stat > 2 else ""

        rows.append(row)

    return pd.DataFrame(rows).sort_values("Mean_Abs_Bias")


def generate_mae_summary_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate Table A7: MAE Summary (well-behaved policies).

    Provides a robust accuracy view complementary to RMSE^d.

    Args:
        results: List of experiment results

    Returns:
        DataFrame with Estimator and MAE
    """
    from .paper_tables import create_config_key
    from .paper_tables import compute_mae

    mae_map = compute_mae(results)
    rows = []
    for cfg in sorted(mae_map.keys()):
        rows.append({"Estimator": cfg, "MAE": mae_map[cfg]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("MAE")
    return df


def generate_overlap_diagnostics_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate Table A3: Overlap & Tail Diagnostics.

    Buckets ESS%, Tail index, and Hellinger affinity into Good/OK/Poor.

    Args:
        results: List of experiment results

    Returns:
        DataFrame with bucketed diagnostics
    """
    from .paper_tables import create_config_key

    diagnostics_by_config: Dict[str, Dict[str, List[float]]] = {}

    for result in results:
        config_key = create_config_key(result)

        if config_key not in diagnostics_by_config:
            diagnostics_by_config[config_key] = {
                "ess_pct": [],
                "tail_index": [],
                "hellinger": [],
            }

        # Extract diagnostics - ESS is at top level as dict by policy
        ess_rel = result.get("ess_relative")
        if ess_rel is not None and isinstance(ess_rel, dict):
            # Average ESS across well-behaved policies and convert to percentage
            ess_values = []
            for policy in ["clone", "parallel_universe_prompt", "premium"]:
                if policy in ess_rel:
                    ess_values.append(ess_rel[policy])
            if ess_values:
                # ess_relative is already a percentage in results
                avg_ess_pct = np.mean(ess_values)
                diagnostics_by_config[config_key]["ess_pct"].append(avg_ess_pct)

        # Use top-level per-policy diagnostics captured by the ablation harness
        tail_alpha = result.get("tail_alpha") or {}
        hellinger_aff = result.get("hellinger_affinity") or {}
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy in tail_alpha and tail_alpha[policy] is not None:
                diagnostics_by_config[config_key]["tail_index"].append(
                    float(tail_alpha[policy])
                )
            if policy in hellinger_aff and hellinger_aff[policy] is not None:
                diagnostics_by_config[config_key]["hellinger"].append(
                    float(hellinger_aff[policy])
                )

    # Bucket and aggregate
    rows = []
    for config, diag_dict in diagnostics_by_config.items():
        row = {"Estimator": config}

        # ESS% bucketing (>20 good, 10-20 OK, <10 poor)
        if diag_dict["ess_pct"]:
            ess_vals = diag_dict["ess_pct"]
            row["ESS_Good"] = np.mean([e > 20 for e in ess_vals]) * 100
            row["ESS_OK"] = np.mean([10 <= e <= 20 for e in ess_vals]) * 100
            row["ESS_Poor"] = np.mean([e < 10 for e in ess_vals]) * 100
            row["ESS_Median"] = np.median(ess_vals)

        # Tail index bucketing (>2 finite variance, 1.5-2 OK, <1.5 poor)
        if diag_dict["tail_index"]:
            tail_vals = diag_dict["tail_index"]
            row["Tail_Good"] = np.mean([t > 2 for t in tail_vals]) * 100
            row["Tail_OK"] = np.mean([1.5 <= t <= 2 for t in tail_vals]) * 100
            row["Tail_Poor"] = np.mean([t < 1.5 for t in tail_vals]) * 100
            row["Tail_Median"] = np.median(tail_vals)

        # Hellinger bucketing (>0.5 good, 0.3-0.5 OK, <0.3 poor)
        if diag_dict["hellinger"]:
            hell_vals = diag_dict["hellinger"]
            row["Hell_Good"] = np.mean([h > 0.5 for h in hell_vals]) * 100
            row["Hell_OK"] = np.mean([0.3 <= h <= 0.5 for h in hell_vals]) * 100
            row["Hell_Poor"] = np.mean([h < 0.3 for h in hell_vals]) * 100
            row["Hell_Median"] = np.median(hell_vals)

        rows.append(row)

    return pd.DataFrame(rows)


def generate_oracle_adjustment_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate Table A4: Oracle Adjustment Share.

    Shows the proportion of uncertainty from oracle slice calibration.

    Args:
        results: List of experiment results

    Returns:
        DataFrame with OA analysis
    """
    from .paper_tables import create_config_key

    oa_by_config: Dict[str, Dict[str, List[float]]] = {}

    for result in results:
        config_key = create_config_key(result)

        if config_key not in oa_by_config:
            oa_by_config[config_key] = {
                "oa_shares": [],
                "coverage_base": [],
                "coverage_oa": [],
            }

        # Check if we have OA information
        base_ses = result.get("standard_errors", {})
        robust_ses = result.get("robust_standard_errors", {})

        # Calculate OA share if we have both
        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy in base_ses and policy in robust_ses:
                base_se = base_ses[policy]
                robust_se = robust_ses[policy]

                if base_se > 0 and robust_se > 0:
                    # OA share = (robust^2 - base^2) / robust^2
                    oa_share = max(0, (robust_se**2 - base_se**2) / robust_se**2)
                    oa_by_config[config_key]["oa_shares"].append(oa_share)

        # Coverage with and without OA
        robust_cis = result.get("robust_confidence_intervals") or result.get(
            "confidence_intervals", {}
        )
        base_cis = result.get("confidence_intervals", {})
        oracle_truths = result.get("oracle_truths", {})

        for policy in ["clone", "parallel_universe_prompt", "premium"]:
            if policy in oracle_truths:
                truth = oracle_truths[policy]

                if policy in base_cis:
                    ci = base_cis[policy]
                    oa_by_config[config_key]["coverage_base"].append(
                        ci[0] <= truth <= ci[1]
                    )

                if policy in robust_cis:
                    ci = robust_cis[policy]
                    oa_by_config[config_key]["coverage_oa"].append(
                        ci[0] <= truth <= ci[1]
                    )

    # Aggregate
    rows = []
    for config, oa_dict in oa_by_config.items():
        row: Dict[str, Any] = {"Estimator": config}

        if oa_dict["oa_shares"]:
            row["OA_Share_Mean"] = np.mean(oa_dict["oa_shares"]) * 100
            row["OA_Share_Median"] = np.median(oa_dict["oa_shares"]) * 100
            row["OA_Share_Max"] = np.max(oa_dict["oa_shares"]) * 100

        if oa_dict["coverage_base"]:
            row["Coverage_Base"] = np.mean(oa_dict["coverage_base"]) * 100

        if oa_dict["coverage_oa"]:
            row["Coverage_OA"] = np.mean(oa_dict["coverage_oa"]) * 100

        if "Coverage_Base" in row and "Coverage_OA" in row:
            row["Coverage_Diff"] = row["Coverage_OA"] - row["Coverage_Base"]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("OA_Share_Mean", ascending=False)


def generate_boundary_outlier_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate Table A5: Calibration Boundary Analysis with RCS-Lite.

    Shows distance to boundaries and outlier detection rates.

    Args:
        results: List of experiment results

    Returns:
        DataFrame with boundary analysis
    """
    from .paper_tables import create_config_key

    boundary_by_config: Dict[str, Dict[str, List[float]]] = {}

    for result in results:
        config_key = create_config_key(result)

        if config_key not in boundary_by_config:
            boundary_by_config[config_key] = {
                "min_distances": [],
                "outlier_flags": [],
                "unhelpful_distances": [],
                "cal_ranges": [],
                "cal_mins": [],
                "cal_maxs": [],
            }

        # Get calibrated reward range
        cal_min = result.get("calibrated_reward_min")
        cal_max = result.get("calibrated_reward_max")

        if cal_min is not None and cal_max is not None:
            estimates = result.get("estimates", {})

            # Store calibration range info for RCS check
            cal_range = cal_max - cal_min
            boundary_by_config[config_key]["cal_ranges"].append(cal_range)
            boundary_by_config[config_key]["cal_mins"].append(cal_min)
            boundary_by_config[config_key]["cal_maxs"].append(cal_max)

            for policy in ["clone", "parallel_universe_prompt", "premium"]:
                if policy in estimates:
                    est = estimates[policy]
                    # Distance to nearest boundary
                    dist = min(est - cal_min, cal_max - est)
                    boundary_by_config[config_key]["min_distances"].append(dist)

            # Special handling for unhelpful
            if "unhelpful" in estimates:
                est = estimates["unhelpful"]
                dist = min(est - cal_min, cal_max - est)
                boundary_by_config[config_key]["unhelpful_distances"].append(dist)

                # Check if outlier (using adaptive threshold)
                cal_range = cal_max - cal_min
                threshold = min(0.2 * cal_range, 0.15)
                is_outlier = dist < threshold
                boundary_by_config[config_key]["outlier_flags"].append(is_outlier)

    # Aggregate
    rows = []
    for config, boundary_dict in boundary_by_config.items():
        row = {"Estimator": config}

        if boundary_dict["min_distances"]:
            row["Mean_Dist_Boundary"] = np.mean(boundary_dict["min_distances"])
            row["Min_Dist_Boundary"] = np.min(boundary_dict["min_distances"])
            row["Pct_Near_Boundary"] = (
                np.mean([d < 0.1 for d in boundary_dict["min_distances"]]) * 100
            )

        if boundary_dict["unhelpful_distances"]:
            row["Unhelpful_Mean_Dist"] = np.mean(boundary_dict["unhelpful_distances"])
            row["Unhelpful_Min_Dist"] = np.min(boundary_dict["unhelpful_distances"])

        if boundary_dict["outlier_flags"]:
            row["Outlier_Rate"] = np.mean(boundary_dict["outlier_flags"]) * 100

        # Simple RCS-Lite check based on calibration range
        if boundary_dict["cal_ranges"]:
            avg_range = np.mean(boundary_dict["cal_ranges"])
            avg_min = np.mean(boundary_dict["cal_mins"])
            avg_max = np.mean(boundary_dict["cal_maxs"])

            # Good support: range > 0.4 AND covers [0.1, 0.9]
            if avg_range > 0.4 and avg_min <= 0.1 and avg_max >= 0.9:
                row["Support"] = "Good"
            # OK support: range > 0.3 AND covers [0.2, 0.8]
            elif avg_range > 0.3 and avg_min <= 0.2 and avg_max >= 0.8:
                row["Support"] = "OK"
            # Weak support: everything else
            else:
                row["Support"] = "Weak"

        rows.append(row)

    return pd.DataFrame(rows).sort_values("Pct_Near_Boundary", ascending=False)


def generate_runtime_complexity_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Generate Table A6: Runtime & Complexity.

    Shows runtime, fold counts, and computational complexity.

    Args:
        results: List of experiment results

    Returns:
        DataFrame with runtime analysis
    """
    from .paper_tables import create_config_key

    runtime_by_config = {}

    for result in results:
        config_key = create_config_key(result)
        spec = result.get("spec", {})

        if config_key not in runtime_by_config:
            runtime_by_config[config_key] = {
                "runtimes": [],
                "sample_sizes": [],
                "estimator": spec.get("estimator", "unknown"),
            }

        runtime = result.get("runtime_s")
        if runtime is not None:
            runtime_by_config[config_key]["runtimes"].append(runtime)
            runtime_by_config[config_key]["sample_sizes"].append(
                spec.get("sample_size", 0)
            )

    # Aggregate and compute complexity
    rows = []
    for config, runtime_dict in runtime_by_config.items():
        row = {"Estimator": config}

        if runtime_dict["runtimes"]:
            row["Runtime_Median"] = np.median(runtime_dict["runtimes"])
            row["Runtime_P90"] = np.percentile(runtime_dict["runtimes"], 90)

            # Estimate computational complexity
            estimator = runtime_dict["estimator"]
            if estimator in ["raw-ips", "calibrated-ips", "orthogonalized-ips"]:
                row["Complexity"] = "O(n)"
                row["N_Folds"] = "0"
            elif estimator.startswith("stacked"):
                row["Complexity"] = "O(M*K*n)"
                row["N_Folds"] = "20"
                row["M_Components"] = "5" if estimator == "stacked-dr" else "4"
            else:
                row["Complexity"] = "O(K*n)"
                row["N_Folds"] = "20"

            # Runtime per sample (normalized)
            if runtime_dict["sample_sizes"]:
                avg_n = np.mean(runtime_dict["sample_sizes"])
                row["Runtime_per_1k"] = row["Runtime_Median"] / (avg_n / 1000)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("Runtime_Median")


# ==========================
# New SIMCal-focused tables
# ==========================


def _config_key_for_pairing(result: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """Build a pairing key for within-config comparisons.

    Key dimensions: (estimator family, sample_size, oracle_coverage, use_iic, seed_base)
    """
    spec = result.get("spec", {})
    est = spec.get("estimator")
    size = spec.get("sample_size")
    cov = spec.get("oracle_coverage")
    seed = spec.get("seed_base", result.get("seed"))
    extra = spec.get("extra", {})
    use_iic = bool(extra.get("use_iic", result.get("use_iic", False)))

    if not est or size is None or cov is None:
        return None

    # Map estimator to a family key for IPS pairing
    family = est
    if est in ("raw-ips", "calibrated-ips", "orthogonalized-ips"):
        family = "ips"

    return (family, size, cov, use_iic, seed)


def _index_results_by_spec(
    results: List[Dict[str, Any]],
) -> Dict[Tuple, Dict[str, Dict[str, Any]]]:
    """Index results for pairing SIMCal on/off within the same config.

    Returns a dict keyed by pairing key. Each value is a dict with possible entries:
      - 'raw_ips': corresponding raw-ips result
      - 'calibrated_ips': calibrated-ips result
      - For DR families ('dr-cpo', 'mrdr', 'tmle'):
          'dr_off': use_weight_calibration=False result
          'dr_on':  use_weight_calibration=True result
    """
    index: Dict[Tuple, Dict[str, Dict[str, Any]]] = {}
    for r in results:
        if not r.get("success"):
            continue
        key = _config_key_for_pairing(r)
        if key is None:
            continue
        spec = r.get("spec", {})
        est = spec.get("estimator")
        extra = spec.get("extra", {})
        use_cal = bool(
            extra.get("use_weight_calibration", r.get("use_weight_calibration", False))
        )

        bucket = index.setdefault(key, {})
        if est == "raw-ips":
            bucket["raw_ips"] = r
        elif est == "calibrated-ips":
            bucket["calibrated_ips"] = r
        elif est == "orthogonalized-ips":
            bucket["ortho_ips"] = r
        elif est in ("dr-cpo", "mrdr", "tmle"):
            bucket["dr_on" if use_cal else "dr_off"] = r
        # Skip families that never/always calibrate (tr-cpo*, oc-dr-cpo, stacked-dr*)

    return index


def _get_policy_metric(
    result: Dict[str, Any], key: str, policy: str
) -> Optional[float]:
    d = result.get(key) or {}
    if not isinstance(d, dict):
        return None
    val = d.get(policy)
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def generate_simcal_ess_gain_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """SIMCal ESS gains by estimator family and policy.

    Produces within-config paired comparisons:
      - IPS: raw-ips vs calibrated-ips
      - DR:  dr-cpo/mrdr/tmle with/without use_weight_calibration

    Columns:
      Estimator, Pairs, and per-policy: ESS%_raw, ESS%_simcal, xGain
    """
    idx = _index_results_by_spec(results)

    rows: List[Dict[str, Any]] = []
    for key, bucket in idx.items():
        family = key[0]

        # Restrict to IPS family for ESS (weights-only) focus
        if family != "ips":
            continue

        # Build paired tuples (raw vs calibrated)
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        if "raw_ips" in bucket and "calibrated_ips" in bucket:
            pairs.append((bucket["raw_ips"], bucket["calibrated_ips"]))

        if not pairs:
            continue

        # Aggregate per-policy ESS% and gains
        policies = ["clone", "parallel_universe_prompt", "premium"]
        agg = {p: {"raw": [], "sim": [], "gain": []} for p in policies}

        for raw, sim in pairs:
            for p in policies:
                ess_raw = _get_policy_metric(raw, "ess_relative", p)
                ess_sim = _get_policy_metric(sim, "ess_relative", p)
                if ess_raw is None or ess_sim is None or ess_raw <= 0:
                    continue
                agg[p]["raw"].append(ess_raw)
                agg[p]["sim"].append(ess_sim)
                agg[p]["gain"].append(ess_sim / ess_raw)

        # Only keep if we have some data
        if not any(agg[p]["gain"] for p in policies):
            continue

        row: Dict[str, Any] = {
            "Estimator": "ips",
            "Pairs": len(pairs),
        }
        for p in policies:
            raw_mean = np.mean(agg[p]["raw"]) if agg[p]["raw"] else np.nan
            sim_mean = np.mean(agg[p]["sim"]) if agg[p]["sim"] else np.nan
            gain_mean = np.mean(agg[p]["gain"]) if agg[p]["gain"] else np.nan
            abbrev = {
                "clone": "Clone",
                "parallel_universe_prompt": "ParaU",
                "premium": "Premium",
            }[p]
            row[f"{abbrev}_ESS%_raw"] = raw_mean
            row[f"{abbrev}_ESS%_simcal"] = sim_mean
            row[f"{abbrev}_xGain"] = gain_mean

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        # Sort by average xGain across available policies (descending)
        gain_cols = [c for c in df.columns if c.endswith("_xGain")]
        df["Avg_xGain"] = df[gain_cols].mean(axis=1, skipna=True)
        df = df.sort_values("Avg_xGain", ascending=False).drop(columns=["Avg_xGain"])
    return df


def generate_weights_ess_raw_vs_simcal_table(
    results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Raw weights vs SIMCal weights ESS% (weights-only view, estimator-agnostic).

    Pairs across both IPS and DR families where calibration toggles exist within the same
    (sample_size, oracle_coverage, use_iic, seed) key. Reports per-policy means:
      - Raw_ESS%, SimCal_ESS%, xGain (SimCal/Raw)
      - Pairs: number of matched configs contributing
    """
    idx = _index_results_by_spec(results)
    policies = ["clone", "parallel_universe_prompt", "premium"]

    # Collect pairs from IPS (raw vs calibrated-ips) and DR (use_cal=False vs True)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for key, bucket in idx.items():
        # IPS pairs
        if "raw_ips" in bucket and "calibrated_ips" in bucket:
            pairs.append((bucket["raw_ips"], bucket["calibrated_ips"]))
        # DR pairs (dr-cpo/mrdr/tmle)
        if "dr_off" in bucket and "dr_on" in bucket:
            pairs.append((bucket["dr_off"], bucket["dr_on"]))

    # Aggregate
    agg = {p: {"raw": [], "sim": [], "gain": []} for p in policies}
    for raw, sim in pairs:
        for p in policies:
            ess_raw = _get_policy_metric(raw, "ess_relative", p)
            ess_sim = _get_policy_metric(sim, "ess_relative", p)
            if ess_raw is None or ess_sim is None or ess_raw <= 0:
                continue
            agg[p]["raw"].append(ess_raw)
            agg[p]["sim"].append(ess_sim)
            agg[p]["gain"].append(ess_sim / ess_raw)

    row: Dict[str, Any] = {"View": "weights_raw_vs_simcal", "Pairs": len(pairs)}
    for p in policies:
        abbrev = {
            "clone": "Clone",
            "parallel_universe_prompt": "ParaU",
            "premium": "Premium",
        }[p]
        row[f"{abbrev}_Raw_ESS%"] = np.mean(agg[p]["raw"]) if agg[p]["raw"] else np.nan
        row[f"{abbrev}_SimCal_ESS%"] = (
            np.mean(agg[p]["sim"]) if agg[p]["sim"] else np.nan
        )
        row[f"{abbrev}_xGain"] = np.mean(agg[p]["gain"]) if agg[p]["gain"] else np.nan

    return pd.DataFrame([row])


def generate_simcal_weight_diag_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """SIMCal weight diagnostics deltas (simcal - raw) by estimator family and policy.

    Columns per policy:
      ΔTailα (higher better), ΔCV (lower better), ΔMaxW (lower better), ΔZeroMass (lower better)
    """
    idx = _index_results_by_spec(results)
    rows: List[Dict[str, Any]] = []

    for key, bucket in idx.items():
        family = key[0]
        # Restrict to IPS family for weight diagnostics focus
        if family != "ips":
            continue
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        if "raw_ips" in bucket and "calibrated_ips" in bucket:
            pairs.append((bucket["raw_ips"], bucket["calibrated_ips"]))

        if not pairs:
            continue

        policies = ["clone", "parallel_universe_prompt", "premium"]
        agg = {p: {"dtail": [], "dcv": [], "dmaxw": [], "dzm": []} for p in policies}

        for raw, sim in pairs:
            for p in policies:
                tail_raw = _get_policy_metric(raw, "tail_alpha", p)
                tail_sim = _get_policy_metric(sim, "tail_alpha", p)
                cv_raw = _get_policy_metric(raw, "weight_cv", p)
                cv_sim = _get_policy_metric(sim, "weight_cv", p)
                mw_raw = _get_policy_metric(raw, "max_weight", p)
                mw_sim = _get_policy_metric(sim, "max_weight", p)
                zm_raw = _get_policy_metric(raw, "mass_concentration", p)
                zm_sim = _get_policy_metric(sim, "mass_concentration", p)

                if (
                    tail_raw is None
                    or tail_sim is None
                    or cv_raw is None
                    or cv_sim is None
                    or mw_raw is None
                    or mw_sim is None
                    or zm_raw is None
                    or zm_sim is None
                ):
                    continue

                agg[p]["dtail"].append(tail_sim - tail_raw)
                agg[p]["dcv"].append(cv_sim - cv_raw)
                agg[p]["dmaxw"].append(mw_sim - mw_raw)
                agg[p]["dzm"].append(zm_sim - zm_raw)

        if not any(agg[p]["dtail"] for p in policies):
            continue

        row: Dict[str, Any] = {"Estimator": family, "Pairs": len(pairs)}
        for p in policies:
            abbrev = {
                "clone": "Clone",
                "parallel_universe_prompt": "ParaU",
                "premium": "Premium",
            }[p]
            row[f"{abbrev}_ΔTailα"] = (
                np.mean(agg[p]["dtail"]) if agg[p]["dtail"] else np.nan
            )
            row[f"{abbrev}_ΔCV"] = np.mean(agg[p]["dcv"]) if agg[p]["dcv"] else np.nan
            row[f"{abbrev}_ΔMaxW"] = (
                np.mean(agg[p]["dmaxw"]) if agg[p]["dmaxw"] else np.nan
            )
            row[f"{abbrev}_ΔZeroMass"] = (
                np.mean(agg[p]["dzm"]) if agg[p]["dzm"] else np.nan
            )

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        # Sort by improvement in tails (primary) then CV decrease
        sort_cols = [c for c in df.columns if c.endswith("ΔTailα")] + [
            c for c in df.columns if c.endswith("ΔCV")
        ]
        df["Score"] = df[sort_cols].apply(
            lambda s: np.nanmean(s.values.astype(float)), axis=1
        )
        df = df.sort_values("Score", ascending=False).drop(columns=["Score"])
    return df


def format_appendix_latex(df: pd.DataFrame, table_num: str, caption: str) -> str:
    """Format appendix table as LaTeX."""
    latex = []
    latex.append(f"\\begin{{table}}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{tab:{table_num}}}")

    # Generate column specification based on DataFrame
    n_cols = len(df.columns)
    col_spec = "l|" + "c" * (n_cols - 1)
    latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex.append("\\toprule")

    # Header
    headers = [col.replace("_", " ") for col in df.columns]
    latex.append(" & ".join(headers) + " \\\\")
    latex.append("\\midrule")

    # Data rows
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                cells.append("--")
            elif isinstance(val, (int, np.integer)):
                cells.append(str(val))
            elif isinstance(val, (float, np.floating)):
                if val < 0.01:
                    cells.append(f"{val:.4f}")
                elif val < 1:
                    cells.append(f"{val:.3f}")
                elif val < 100:
                    cells.append(f"{val:.1f}")
                else:
                    cells.append(f"{val:.0f}")
            else:
                cells.append(str(val))
        latex.append(" & ".join(cells) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def main() -> None:
    """Generate SIMCal-focused appendix tables."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate appendix tables")
    parser.add_argument(
        "--results", type=Path, default=Path("results/all_experiments.jsonl")
    )
    parser.add_argument("--output", type=Path, default=Path("tables/appendix/"))
    parser.add_argument(
        "--format", choices=["dataframe", "latex", "markdown"], default="latex"
    )
    args = parser.parse_args()

    # Load results
    results = []
    with open(args.results) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Filter to successful and dedupe by (estimator, size, coverage, seed, use_calib, use_iic)
    results = [r for r in results if r.get("success")]
    seen = set()
    deduped = []
    for r in results:
        spec = r.get("spec", {})
        est = spec.get("estimator")
        size = spec.get("sample_size")
        cov = spec.get("oracle_coverage")
        seed = spec.get("seed_base", r.get("seed"))
        extra = spec.get("extra", {})
        use_cal = extra.get(
            "use_weight_calibration", r.get("use_weight_calibration", False)
        )
        use_iic = extra.get("use_iic", r.get("use_iic", False))
        key = (est, size, cov, seed, bool(use_cal), bool(use_iic))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    results = deduped

    # Create output directory
    args.output.mkdir(exist_ok=True, parents=True)

    # New default: SIMCal-centric diagnostics (IPS-focused)
    tables: List[Tuple[str, str, Any]] = [
        (
            "S1",
            "Raw vs SIMCal ESS% (weights-only)",
            generate_weights_ess_raw_vs_simcal_table,
        ),
        ("S2", "SIMCal Weight Diagnostics (Deltas)", generate_simcal_weight_diag_table),
        # Legacy (disabled by default): uncomment to generate
        # ("A1", "Quadrant Leaderboard", generate_quadrant_leaderboard),
        # ("A2", "Bias Patterns", generate_bias_patterns_table),
        # ("A3", "Overlap & Tail Diagnostics", generate_overlap_diagnostics_table),
        # ("A4", "Oracle Adjustment Share", generate_oracle_adjustment_table),
        # ("A5", "Calibration Boundary Analysis", generate_boundary_outlier_table),
        # ("A6", "Runtime & Complexity", generate_runtime_complexity_table),
    ]

    for table_num, caption, generator in tables:
        print(f"Generating Table {table_num}: {caption}...")
        df = generator(results)

        if args.format == "latex":
            latex = format_appendix_latex(df, table_num, caption)
            (args.output / f"table{table_num}.tex").write_text(latex)
        elif args.format == "markdown":
            (args.output / f"table{table_num}.md").write_text(
                df.to_markdown(index=False)
            )
        else:
            df.to_csv(args.output / f"table{table_num}.csv", index=False)

    print(f"Appendix tables written to {args.output}/")


if __name__ == "__main__":
    main()
