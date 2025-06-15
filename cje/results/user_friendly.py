"""
User-friendly result formatting for CJE experiments.

This module transforms technical estimation results into intuitive, business-oriented
formats that are optimized for end-user consumption and decision-making.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from ..estimators.results import EstimationResult
from ..estimators.reliability import (
    assess_ci_reliability,
    EstimatorMetadata,
    ReliabilityAssessment,
)


def create_user_friendly_result(
    estimation_result: EstimationResult,
    policy_names: List[str],
    logging_policy_value: float,
    logging_policy_se: float,
    analysis_type: str,
    **metadata: Any,
) -> Dict[str, Any]:
    """
    Transform technical estimation results into a user-friendly format.

    Args:
        estimation_result: Technical estimation result object
        policy_names: Names of the target policies
        logging_policy_value: Baseline logging policy value
        logging_policy_se: Baseline logging policy standard error
        analysis_type: Type of analysis performed
        **metadata: Additional metadata from the experiment

    Returns:
        User-friendly result dictionary optimized for business consumption
    """

    # Import reliability assessment
    from ..estimators.reliability import assess_ci_reliability, EstimatorMetadata

    # Create structured metadata for reliability assessment
    structured_metadata = None
    if estimation_result.metadata:
        # Create EstimatorMetadata from the stored metadata
        structured_metadata = EstimatorMetadata(**estimation_result.metadata)

    # Find best policy
    best_policy_idx = int(np.argmax(estimation_result.v_hat))
    best_policy_name = (
        policy_names[best_policy_idx]
        if best_policy_idx < len(policy_names)
        else f"policy_{best_policy_idx}"
    )

    # Get confidence intervals
    ci_lower, ci_upper = estimation_result.confidence_interval()

    # Detect and correct difference estimates (samples_per_policy=0) to absolute values
    potential_config_issue = (
        abs(np.mean(estimation_result.v_hat)) < 0.1
        and abs(logging_policy_value) > 0.3
        and np.all(np.abs(estimation_result.v_hat) < 0.1)
    )

    if potential_config_issue:
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            "🔧 Detected difference estimates (samples_per_policy=0). "
            "Converting to absolute values by adding baseline."
        )

        # Convert difference estimates to absolute values
        # v_hat_absolute = v_hat_difference + baseline_value
        # The baseline is the outcome model's expectation for the logging policy
        baseline_value = logging_policy_value

        # Convert to absolute values
        v_hat_absolute = [v_diff + baseline_value for v_diff in estimation_result.v_hat]

        # Update the estimation result with absolute values
        estimation_result.v_hat = np.array(v_hat_absolute)

        # Also correct the confidence intervals by adding the baseline
        ci_lower_corrected = ci_lower + baseline_value
        ci_upper_corrected = ci_upper + baseline_value

        # Update the cached confidence intervals
        ci_lower = ci_lower_corrected
        ci_upper = ci_upper_corrected

        logger.info(
            f"Converted to absolute values: {v_hat_absolute} "
            f"(added baseline: {baseline_value:.4f})"
        )

    # Compute policy vs baseline differences
    policy_comparisons = []
    for i, policy_name in enumerate(policy_names):
        # Now that we've corrected difference estimates to absolute values,
        # we can always compute differences normally
        diff = estimation_result.v_hat[i] - logging_policy_value

        # Compute significance test vs baseline
        if estimation_result.covariance_matrix is not None:
            # Use proper variance for difference
            diff_var = estimation_result.covariance_matrix[i, i] + logging_policy_se**2
            diff_se = np.sqrt(diff_var)
        else:
            # Conservative estimate
            diff_se = np.sqrt(estimation_result.se[i] ** 2 + logging_policy_se**2)

        z_score = diff / diff_se if diff_se > 0 else 0
        from scipy import stats

        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        policy_comparisons.append(
            {
                "policy_name": policy_name,
                "estimate": round(float(estimation_result.v_hat[i]), 4),
                "confidence_interval": [
                    round(float(ci_lower[i]), 4),
                    round(float(ci_upper[i]), 4),
                ],
                "standard_error": round(float(estimation_result.se[i]), 4),
                "vs_baseline": {
                    "difference": round(float(diff), 4),
                    "difference_se": round(float(diff_se), 4),
                    "significant": p_value < 0.05,
                    "p_value": round(float(p_value), 3),
                    "confidence_level": "95%",
                },
            }
        )

    # Automatically enable bootstrap results for small samples (n < 100) or if explicitly requested
    bootstrap_results = None
    should_compute_bootstrap = estimation_result.eif_components is not None and (
        estimation_result.n < 100 or metadata.get("force_bootstrap", False)
    )

    if should_compute_bootstrap:
        try:
            bootstrap_ci = estimation_result.bootstrap_confidence_intervals(
                n_bootstrap=1000, seed=42, confidence_level=0.95
            )
            bootstrap_results = {
                "available": True,
                "auto_enabled": estimation_result.n < 100,
                "confidence_intervals": {
                    policy_name: [
                        round(float(bootstrap_ci["ci_lower"][i]), 4),
                        round(float(bootstrap_ci["ci_upper"][i]), 4),
                    ]
                    for i, policy_name in enumerate(policy_names)
                },
                "standard_errors": {
                    policy_name: round(float(bootstrap_ci["bootstrap_se"][i]), 4)
                    for i, policy_name in enumerate(policy_names)
                },
                "note": "Non-parametric confidence intervals robust to small samples and non-normal distributions"
                + (" (auto-enabled for n < 100)" if estimation_result.n < 100 else ""),
            }
        except Exception as e:
            bootstrap_results = {
                "available": False,
                "note": f"Bootstrap computation failed: {str(e)}",
            }

    # Assess CI reliability using the new system
    reliability_assessment = assess_ci_reliability(
        result=estimation_result,
        bootstrap_results=bootstrap_results,
        metadata=structured_metadata,
    )

    # Generate recommendations
    recommendations = _generate_recommendations(
        estimation_result,
        policy_comparisons,
        reliability_assessment,
        analysis_type,
        metadata,
    )

    # Create user-friendly result
    user_result = {
        "summary": {
            "experiment_type": _get_experiment_description(analysis_type),
            "best_policy": best_policy_name,
            "confidence_level": reliability_assessment.rating,
            "sample_size": estimation_result.n,
            "estimator": _get_estimator_description(estimation_result.estimator_type),
            "interpretation": _get_interpretation(analysis_type),
            "key_finding": _generate_key_finding(policy_comparisons, best_policy_name),
        },
        "policies": {
            policy_comp["policy_name"]: {
                "estimate": policy_comp["estimate"],
                "confidence_interval": policy_comp["confidence_interval"],
                "standard_error": policy_comp["standard_error"],
                "vs_baseline": policy_comp["vs_baseline"],
            }
            for policy_comp in policy_comparisons
        },
        "baseline": {
            "logging_policy": {
                "estimate": round(float(logging_policy_value), 4),
                "confidence_interval": [
                    round(float(logging_policy_value - 1.96 * logging_policy_se), 4),
                    round(float(logging_policy_value + 1.96 * logging_policy_se), 4),
                ],
                "standard_error": round(float(logging_policy_se), 4),
                "description": "Current production policy baseline",
            }
        },
        "diagnostics": {
            "sample_quality": _assess_sample_quality_structured(
                estimation_result, structured_metadata
            ),
            "ci_reliability": reliability_assessment.to_dict(),
            "recommendations": recommendations,
        },
    }

    # Add bootstrap results if available
    if bootstrap_results and bootstrap_results["available"]:
        user_result["robust_inference"] = {
            "bootstrap_confidence_intervals": bootstrap_results,
            "recommendation": "Consider bootstrap intervals for additional robustness, especially with small samples",
        }

    # Add minimal technical details for reference (full details in result_technical.json)
    user_result["technical_summary"] = {
        "estimator": estimation_result.estimator_type,
        "sample_size": estimation_result.n,
        "policies_evaluated": estimation_result.n_policies,
        "analysis_type": analysis_type,
        "runtime_seconds": metadata.get("runtime_seconds"),
        "note": "Full technical details available in result_technical.json",
    }

    return user_result


def _assess_sample_quality_structured(
    result: EstimationResult, metadata: Optional["EstimatorMetadata"]
) -> Dict[str, Any]:
    """Assess sample quality using structured metadata."""
    quality = {
        "sample_size": result.n,
        "sample_size_assessment": (
            "adequate" if result.n >= 100 else "moderate" if result.n >= 50 else "small"
        ),
    }

    # Use structured metadata if available
    if metadata:
        if metadata.ess_percentage is not None:
            quality["effective_sample_size"] = f"{metadata.ess_percentage:.1f}%"
            quality["overlap_quality"] = (
                "good"
                if metadata.ess_percentage >= 50
                else "moderate" if metadata.ess_percentage >= 20 else "poor"
            )

        if metadata.clip_fraction > 0:
            quality["clipping_rate"] = (
                f"{metadata.clip_fraction*100:.1f}% of weights clipped"
            )

        if metadata.weight_range:
            min_w, max_w = metadata.weight_range
            quality["weight_range"] = f"[{min_w:.2e}, {max_w:.2e}]"

    return quality


def _generate_recommendations(
    result: EstimationResult,
    comparisons: List[Dict[str, Any]],
    reliability_assessment: "ReliabilityAssessment",
    analysis_type: str,
    metadata: Dict[str, Any],
) -> List[str]:
    """Generate actionable recommendations based on reliability assessment."""

    recommendations = []

    # Start with reliability-based recommendations
    recommendations.extend(reliability_assessment.recommendations)

    # Check for significant differences
    significant_diffs = [
        comp for comp in comparisons if comp["vs_baseline"]["significant"]
    ]

    if not significant_diffs:
        recommendations.append(
            "No statistically significant differences detected between policies and baseline"
        )
        recommendations.append(
            "Current production policy appears competitive with tested alternatives"
        )
    else:
        best_policy = max(comparisons, key=lambda x: x["estimate"])
        recommendations.append(
            f"Consider adopting {best_policy['policy_name']} - shows significant improvement"
        )

        if reliability_assessment.rating in ["low", "unreliable"]:
            recommendations.append(
                "Validate results with additional data before implementation"
            )

    # Analysis type specific recommendations
    if analysis_type == "llm_comparison":
        recommendations.append(
            "⚠️  Results based on LLM judge scores - validate with real user feedback"
        )
    elif analysis_type == "causal_inference_sparse":
        recommendations.append(
            "Limited ground truth data - consider collecting more oracle labels"
        )

    return recommendations


def _generate_key_finding(comparisons: List[Dict[str, Any]], best_policy: str) -> str:
    """Generate a key finding summary."""

    significant_improvements = [
        comp
        for comp in comparisons
        if comp["vs_baseline"]["significant"] and comp["vs_baseline"]["difference"] > 0
    ]

    if not significant_improvements:
        return "No policy shows statistically significant improvement over baseline"
    elif len(significant_improvements) == 1:
        policy = significant_improvements[0]
        return f"{policy['policy_name']} shows significant improvement (+{policy['vs_baseline']['difference']:.3f})"
    else:
        return f"{len(significant_improvements)} policies show significant improvements, {best_policy} performs best"


def _get_estimator_description(estimator_type: str) -> str:
    """Get human-readable estimator description."""

    descriptions = {
        "IPS": "Inverse Propensity Scoring",
        "SNIPS": "Self-Normalized Inverse Propensity Scoring",
        "DRCPO": "Doubly-Robust Cross-Policy Optimization",
        "MRDR": "More-Robust Doubly-Robust",
        "SWDR": "Self-Normalized Doubly-Robust (robust to poor overlap)",
    }

    return descriptions.get(estimator_type, estimator_type)


def _get_experiment_description(analysis_type: str) -> str:
    """Get human-readable experiment description."""

    descriptions = {
        "causal_inference": "Causal inference with ground truth validation",
        "causal_inference_sparse": "Causal inference with limited ground truth",
        "llm_comparison": "LLM judge comparison (not validated)",
    }

    return descriptions.get(analysis_type, analysis_type)


def _get_interpretation(analysis_type: str) -> str:
    """Get interpretation of what the results mean."""

    interpretations = {
        "causal_inference": "Results represent expected changes in real business outcomes",
        "causal_inference_sparse": "Results represent expected changes in real business outcomes (with limited validation)",
        "llm_comparison": "Results show LLM judge score differences (real-world impact uncertain)",
    }

    return interpretations.get(
        analysis_type, "Results interpretation depends on analysis type"
    )
