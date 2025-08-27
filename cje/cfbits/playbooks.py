"""Practical usage playbooks for CF-bits.

This module provides ready-to-use functions for common CF-bits workflows,
handling the two main scenarios: (A) fresh draws with DR, and (B) logging-only IPS.
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
import numpy as np
import logging

from .sampling import compute_ifr_aess, compute_sampling_width, compute_estimator_eif
from .overlap import estimate_overlap_floors
from .core import compute_cfbits, apply_gates
from .identification import compute_identification_width

if TYPE_CHECKING:
    from ..estimators.base_estimator import BaseCJEEstimator

logger = logging.getLogger(__name__)


def cfbits_report_fresh_draws(
    estimator: "BaseCJEEstimator",
    policy: str,
    n_boot: int = 800,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
    compute_tail_index: bool = False,
) -> Dict[str, Any]:
    """CF-bits report for fresh draws scenario (DR/TMLE).

    Use when you have:
    - Fresh draws from target policy with judge scores
    - DR/MRDR/TMLE estimator with cross-fitted nuisance models
    - Oracle labels for calibration

    Args:
        estimator: Fitted DR/TMLE estimator
        policy: Target policy name
        n_boot: Bootstrap samples for overlap CIs
        alpha: Significance level (0.05 for 95% CI)
        random_state: Random seed for reproducibility
        compute_tail_index: Whether to compute Hill tail index

    Returns:
        Complete CF-bits report dictionary
    """
    report: Dict[str, Any] = {"policy": policy, "scenario": "fresh_draws"}

    # 1. Sampling width with OUA
    logger.info(f"Computing sampling width for {policy}")
    wvar, var_components = compute_sampling_width(
        estimator,
        policy,
        alpha=alpha,
        use_iic=True,  # Use IIC for variance reduction
        compute_oua=True,
    )
    report["sampling_width"] = {
        "wvar": float(wvar),
        "var_main": float(var_components.var_main),
        "var_oracle": float(var_components.var_oracle),
        "var_total": float(var_components.var_total),
    }

    # 2. Efficient IF and IFR
    logger.info("Computing efficiency metrics")
    phi = estimator.get_influence_functions(policy)
    eif = compute_estimator_eif(estimator, policy)

    if phi is not None:
        efficiency = compute_ifr_aess(
            phi, eif=eif, n=len(phi), var_oracle=var_components.var_oracle
        )
        report["efficiency"] = {
            "ifr_main": float(efficiency.ifr_main),
            "ifr_oua": float(efficiency.ifr_oua),
            "aess_main": float(efficiency.aess_main),
            "aess_oua": float(efficiency.aess_oua),
            "var_phi": float(efficiency.var_phi),
            "var_eif": float(efficiency.var_eif),
        }
    else:
        logger.warning("No influence functions available")
        efficiency = None
        report["efficiency"] = None

    # 3. Structural floors on logging data
    logger.info("Computing structural overlap floors")
    try:
        # Get logging pool judge scores and RAW importance weights
        # Important: Use raw weights, not calibrated, to measure structural overlap
        S_log = estimator.sampler.get_judge_scores()
        W_log = estimator.sampler.compute_importance_weights(policy, mode="hajek")

        if S_log is not None and W_log is not None:
            floors = estimate_overlap_floors(
                S_log,
                W_log,
                method="conservative",
                n_boot=n_boot,
                alpha=alpha,
                random_state=random_state,
            )
            report["overlap"] = {
                "aessf": float(floors.aessf),
                "aessf_lcb": float(floors.ci_aessf[0]),
                "aessf_ucb": float(floors.ci_aessf[1]),
                "bc": float(floors.bc),
                "chi2_s": float(floors.chi2_s),
            }
        else:
            logger.warning("Cannot compute overlap: missing judge scores or weights")
            floors = None
            report["overlap"] = None
    except Exception as e:
        logger.warning(f"Failed to compute overlap: {e}")
        floors = None
        report["overlap"] = None

    # 4. Identification width (placeholder for now)
    wid, wid_diag = compute_identification_width(estimator, policy, alpha=alpha)
    report["identification"] = {
        "wid": float(wid),
        "diagnostics": wid_diag,
    }

    # 5. CF-bits computation
    logger.info("Computing CF-bits")
    cfbits = compute_cfbits(
        w0=1.0,
        wid=wid,
        wvar=wvar,
        ifr_main=efficiency.ifr_main if efficiency else None,
        ifr_oua=efficiency.ifr_oua if efficiency else None,
    )
    report["cfbits"] = {
        "bits_tot": float(cfbits.bits_tot),
        "bits_id": float(cfbits.bits_id) if cfbits.bits_id else None,
        "bits_var": float(cfbits.bits_var) if cfbits.bits_var else None,
        "w_tot": float(cfbits.w_tot),
        "w_max": float(cfbits.w_max),
        "dominant": "identification" if cfbits.w_id > cfbits.w_var else "sampling",
    }

    # 6. Reliability gates
    logger.info("Applying reliability gates")

    # Compute variance ratio
    var_oracle_ratio = None
    if phi is not None and len(phi) > 0:
        var_oracle_ratio = var_components.var_oracle / max(
            var_components.var_main / len(phi), 1e-12
        )

    # Compute tail index if requested
    tail_index = None
    if compute_tail_index and W_log is not None:
        try:
            from ..diagnostics.tail_diagnostics import estimate_tail_index

            tail_index = estimate_tail_index(W_log)
        except:
            logger.debug("Could not compute tail index")

    decision = apply_gates(
        aessf=floors.aessf if floors else None,
        aessf_lcb=floors.ci_aessf[0] if floors else None,
        ifr=efficiency.ifr_oua if efficiency else None,
        tail_index=tail_index,
        var_oracle_ratio=var_oracle_ratio,
    )
    report["gates"] = {
        "state": decision.state,
        "reasons": decision.reasons,
        "suggestions": decision.suggestions,
    }

    # 7. Summary interpretation
    report["summary"] = _generate_summary(report)

    return report


def cfbits_report_logging_only(
    estimator: "BaseCJEEstimator",
    policy: str,
    n_boot: int = 800,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
    compute_tail_index: bool = True,
) -> Dict[str, Any]:
    """CF-bits report for logging-only scenario (IPS/Cal-IPS).

    Use when you have:
    - Only logged data (no fresh draws)
    - RawIPS or CalibratedIPS estimator
    - Limited ability to estimate EIF

    Args:
        estimator: Fitted IPS/Cal-IPS estimator
        policy: Target policy name
        n_boot: Bootstrap samples for overlap CIs
        alpha: Significance level (0.05 for 95% CI)
        random_state: Random seed for reproducibility
        compute_tail_index: Whether to compute Hill tail index

    Returns:
        Complete CF-bits report dictionary
    """
    report: Dict[str, Any] = {"policy": policy, "scenario": "logging_only"}

    # 1. Sampling width (with IIC if available)
    logger.info(f"Computing sampling width for {policy}")
    wvar, var_components = compute_sampling_width(
        estimator, policy, alpha=alpha, use_iic=True, compute_oua=True
    )
    report["sampling_width"] = {
        "wvar": float(wvar),
        "var_main": float(var_components.var_main),
        "var_oracle": float(var_components.var_oracle),
        "var_total": float(var_components.var_total),
    }

    # 2. IFR (limited for Cal-IPS)
    logger.info("Computing efficiency metrics")
    phi = estimator.get_influence_functions(policy)
    estimator_type = estimator.__class__.__name__

    if phi is not None:
        if estimator_type == "RawIPS":
            # For RawIPS, IF = EIF
            efficiency = compute_ifr_aess(
                phi, eif=phi, n=len(phi), var_oracle=var_components.var_oracle
            )
            report["efficiency"] = {
                "ifr_main": 1.0,  # IF = EIF
                "ifr_oua": float(efficiency.ifr_oua),
                "aess_main": float(len(phi)),
                "aess_oua": float(efficiency.aess_oua),
                "var_phi": float(efficiency.var_phi),
            }
        else:
            # For Cal-IPS, we don't have true EIF
            var_phi = np.var(phi, ddof=1)
            n = len(phi)

            # Can only compute OUA share
            oua_share = (n * var_components.var_oracle) / (
                var_phi + n * var_components.var_oracle
            )
            report["efficiency"] = {
                "ifr_main": None,  # Unknown without EIF
                "ifr_oua": None,
                "oua_share": float(oua_share),
                "var_phi": float(var_phi),
                "note": "EIF unavailable for CalibratedIPS",
            }
            efficiency = None
    else:
        report["efficiency"] = None
        efficiency = None

    # 3. Structural floors (critical for IPS)
    logger.info("Computing structural overlap floors")
    try:
        # Use RAW weights for structural overlap, not calibrated weights
        S_log = estimator.sampler.get_judge_scores()
        W_log = estimator.sampler.compute_importance_weights(policy, mode="hajek")

        if S_log is not None and W_log is not None:
            floors = estimate_overlap_floors(
                S_log,
                W_log,
                method="conservative",
                n_boot=n_boot,
                alpha=alpha,
                random_state=random_state,
            )
            report["overlap"] = {
                "aessf": float(floors.aessf),
                "aessf_lcb": float(floors.ci_aessf[0]),
                "aessf_ucb": float(floors.ci_aessf[1]),
                "bc": float(floors.bc),
                "chi2_s": float(floors.chi2_s),
            }
        else:
            floors = None
            report["overlap"] = None
    except Exception as e:
        logger.warning(f"Failed to compute overlap: {e}")
        floors = None
        report["overlap"] = None

    # 4. Identification width (placeholder)
    wid = 0.1  # Conservative placeholder for logging-only
    report["identification"] = {
        "wid": float(wid),
        "note": "Conservative placeholder for logging-only scenario",
    }

    # 5. CF-bits (limited without EIF)
    logger.info("Computing CF-bits")
    cfbits = compute_cfbits(
        w0=1.0,
        wid=wid,
        wvar=wvar,
        ifr_main=efficiency.ifr_main if efficiency else None,
        ifr_oua=efficiency.ifr_oua if efficiency else None,
    )
    report["cfbits"] = {
        "bits_tot": float(cfbits.bits_tot),
        "bits_var": None,  # Often unavailable without EIF
        "w_tot": float(cfbits.w_tot),
        "w_max": float(cfbits.w_max),
        "dominant": "sampling",  # Usually sampling-limited
    }

    # 6. Reliability gates (stricter for IPS)
    logger.info("Applying reliability gates")

    # Compute tail index (important for IPS)
    tail_index = None
    if compute_tail_index and W_log is not None:
        try:
            # Simple Hill estimator
            W_sorted = np.sort(W_log)[::-1]
            k = max(10, int(np.sqrt(len(W_sorted))))
            if k < len(W_sorted):
                tail_index = k / np.sum(np.log(W_sorted[:k] / W_sorted[k]))
        except:
            logger.debug("Could not compute tail index")

    # Variance ratio
    var_oracle_ratio = None
    if phi is not None and len(phi) > 0:
        var_oracle_ratio = var_components.var_oracle / max(
            var_components.var_main / len(phi), 1e-12
        )

    decision = apply_gates(
        aessf=floors.aessf if floors else None,
        aessf_lcb=floors.ci_aessf[0] if floors else None,
        ifr=efficiency.ifr_oua if efficiency and efficiency.ifr_oua else None,
        tail_index=tail_index,
        var_oracle_ratio=var_oracle_ratio,
    )
    report["gates"] = {
        "state": decision.state,
        "reasons": decision.reasons,
        "suggestions": decision.suggestions,
    }

    # 7. IPS-specific warnings
    if floors and floors.ci_aessf[0] < 0.05:
        report["warning"] = "CRITICAL: A-ESSF < 5% - consider refusing this evaluation"
    elif floors and floors.ci_aessf[0] < 0.20:
        report["warning"] = "Poor overlap - strongly recommend DR with fresh draws"

    report["summary"] = _generate_summary(report)

    return report


def _generate_summary(report: Dict[str, Any]) -> str:
    """Generate human-readable summary from CF-bits report."""
    parts = []

    # Scenario
    scenario = report.get("scenario", "unknown")
    parts.append(f"Scenario: {scenario}")

    # Gate state
    gate_state = report.get("gates", {}).get("state", "UNKNOWN")
    parts.append(f"Reliability: {gate_state}")

    # Key metrics
    if report.get("overlap"):
        aessf_lcb = report["overlap"]["aessf_lcb"]
        parts.append(f"A-ESSF LCB: {aessf_lcb:.1%}")

    if report.get("efficiency"):
        eff = report["efficiency"]
        if eff.get("ifr_oua") is not None:
            parts.append(f"IFR (OUA): {eff['ifr_oua']:.1%}")
        elif eff.get("oua_share") is not None:
            parts.append(f"OUA share: {eff['oua_share']:.1%}")

    # CF-bits
    if report.get("cfbits"):
        cf = report["cfbits"]
        parts.append(f"Total bits: {cf['bits_tot']:.2f}")
        parts.append(f"Width: {cf['w_tot']:.3f}")

    # Main recommendation
    if gate_state == "REFUSE":
        parts.append("→ Do not use this estimate")
    elif gate_state == "CRITICAL":
        parts.append("→ Use with extreme caution")
    elif gate_state == "WARNING":
        parts.append("→ Consider improvements")
    else:
        parts.append("→ Estimate appears reliable")

    return " | ".join(parts)
