"""Sampling width and efficiency metrics for CF-bits.

This module computes:
- IFR (Information Fraction): ratio of efficient IF variance to actual IF variance
- aESS (adjusted Effective Sample Size): n × IFR
- Sampling width (Wvar): statistical uncertainty from finite samples
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
import numpy as np
import logging
from scipy import stats

if TYPE_CHECKING:
    from ..estimators.base_estimator import BaseCJEEstimator

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyStats:
    """Efficiency metrics for an estimator.

    Attributes:
        ifr_main: Information Fraction (no OUA) = Var(EIF)/Var(IF)
        ifr_oua: Information Fraction incl. oracle = Var(EIF)/(Var(IF)+n*Var_oracle)
        aess_main: n × ifr_main
        aess_oua: n × ifr_oua
        var_phi: Variance of actual influence function
        var_eif: Variance of efficient influence function
        var_oracle: Oracle jackknife variance (per-sample scale)
    """

    ifr_main: float
    ifr_oua: float
    aess_main: float
    aess_oua: float
    var_phi: float
    var_eif: float
    var_oracle: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "ifr_main": self.ifr_main,
            "ifr_oua": self.ifr_oua,
            "aess_main": self.aess_main,
            "aess_oua": self.aess_oua,
            "var_phi": self.var_phi,
            "var_eif": self.var_eif,
            "var_oracle": self.var_oracle,
        }


@dataclass
class SamplingVariance:
    """Variance components for sampling width.

    Attributes:
        var_main: Main influence function variance
        var_oracle: Oracle uncertainty augmentation variance
        var_total: Total variance (var_main/n + var_oracle)
    """

    var_main: float
    var_oracle: float
    var_total: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "var_main": self.var_main,
            "var_oracle": self.var_oracle,
            "var_total": self.var_total,
        }


def compute_ifr_aess(
    phi: np.ndarray,
    eif: Optional[np.ndarray] = None,
    n: Optional[int] = None,
    var_oracle: float = 0.0,
) -> EfficiencyStats:
    """Compute Information Fraction Ratio and adjusted ESS.

    IFR measures how close the estimator is to the efficiency bound:
    - IFR_main = Var(EIF) / Var(IF) ∈ (0, 1]
    - IFR_OUA = Var(EIF) / (Var(IF) + n*Var_oracle) ∈ (0, 1]

    aESS is the equivalent sample size if we had an efficient estimator:
    - aESS_main = n × IFR_main
    - aESS_OUA = n × IFR_OUA

    Args:
        phi: Actual influence function values (per-sample contributions)
        eif: Efficient influence function values (if available)
        n: Sample size (defaults to len(phi))
        var_oracle: Oracle uncertainty variance (from jackknife)

    Returns:
        EfficiencyStats with both IFR versions and variance components
    """
    if n is None:
        n = len(phi)

    # Ensure phi is centered (should already be, but be safe)
    phi_centered = phi - np.mean(phi)
    var_phi = float(np.var(phi_centered, ddof=1))

    if eif is None:
        # For IPS estimators without nuisance parameters, IF = EIF
        # This is a reasonable default for now
        logger.debug("No EIF provided, assuming IF = EIF (IPS-like estimator)")
        var_eif = var_phi
    else:
        # Ensure eif is centered
        eif_centered = eif - np.mean(eif)
        var_eif = float(np.var(eif_centered, ddof=1))

    # Compute IFR_main (without OUA)
    if var_phi > 0:
        ifr_main = min(max(var_eif / var_phi, 0.0), 1.0)
    else:
        logger.warning("IF variance is 0, setting IFR to 1")
        ifr_main = 1.0

    # Compute IFR_OUA (including oracle uncertainty)
    # IFR_OUA = Var(EIF) / (Var(phi) + n*Var_oracle)
    denom = var_phi + n * max(var_oracle, 0.0)
    if denom > 0:
        ifr_oua = min(max(var_eif / denom, 0.0), 1.0)
    else:
        ifr_oua = 1.0

    # Compute aESS versions
    aess_main = n * ifr_main
    aess_oua = n * ifr_oua

    return EfficiencyStats(
        ifr_main=ifr_main,
        ifr_oua=ifr_oua,
        aess_main=aess_main,
        aess_oua=aess_oua,
        var_phi=var_phi,
        var_eif=var_eif,
        var_oracle=var_oracle,
    )


def compute_eif_plugin_dr(
    weights: np.ndarray,
    rewards: np.ndarray,
    predictions: np.ndarray,
    psi: Optional[float] = None,
) -> np.ndarray:
    """Compute plug-in efficient influence function for DR estimators.

    For DR estimators, the efficient influence function under the working
    model (correct outcome model and propensity scores) is:

    EIF = W * (R - g(X)) + g(X) - ψ

    where:
    - W: importance weights
    - R: rewards (outcomes)
    - g(X): outcome predictions
    - ψ: the target parameter (estimate)

    Args:
        weights: Importance weights (should be mean-1)
        rewards: Rewards/outcomes used in estimation
        predictions: Outcome model predictions g(X)
        psi: Target parameter estimate (if None, uses mean of base)

    Returns:
        Efficient influence function values
    """
    # Normalize weights to mean 1 if not already
    weights_normalized = weights / np.mean(weights)

    # DR efficient influence function base
    base = weights_normalized * (rewards - predictions) + predictions

    # Subtract estimate to get influence function
    if psi is None:
        psi = float(np.mean(base))

    eif = base - psi

    return eif


def compute_eif_plugin_ips(
    weights: np.ndarray,
    rewards: np.ndarray,
) -> np.ndarray:
    """Compute plug-in efficient influence function for IPS estimators.

    For IPS estimators without nuisance parameters, the influence function
    IS the efficient influence function:

    EIF = W * R - ψ

    Args:
        weights: Importance weights (should be mean-1)
        rewards: Rewards/outcomes

    Returns:
        Efficient influence function values
    """
    # Normalize weights to mean 1 if not already
    weights_normalized = weights / np.mean(weights)

    # IPS influence function
    eif = weights_normalized * rewards

    # Center
    eif_centered = eif - np.mean(eif)

    return eif_centered


def compute_sampling_width(
    estimator: "BaseCJEEstimator",
    policy: str,
    alpha: float = 0.05,
    use_iic: bool = True,
    iic_kwargs: Optional[Dict[str, Any]] = None,
    compute_oua: bool = True,  # Enable by default now
) -> Tuple[float, SamplingVariance]:
    """Compute sampling width (Wvar) for uncertainty.

    Sampling width captures the statistical uncertainty from finite samples:
    Wvar = 2 × z_{1-α/2} × √(Var(φ)/n + Var_oracle)

    Args:
        estimator: Fitted CJE estimator
        policy: Target policy name
        alpha: Significance level (default 0.05 for 95% CI)
        use_iic: Whether to apply IIC variance reduction
        iic_kwargs: Optional IIC configuration
        compute_oua: Whether to compute oracle uncertainty

    Returns:
        Tuple of (Wvar, SamplingVariance with components)
    """
    # Get influence functions
    if_values = estimator.get_influence_functions(policy)
    if if_values is None:
        raise ValueError(f"No influence functions available for policy '{policy}'")

    # Make a copy to avoid modifying original
    phi = if_values.copy()

    # Apply IIC if requested
    if use_iic:
        try:
            # Check if IIC is available
            from ..calibration.iic import IsotonicInfluenceControl

            # Get judge scores from sampler
            judge_scores = estimator.sampler.get_judge_scores()
            if judge_scores is not None and len(judge_scores) == len(phi):
                iic = IsotonicInfluenceControl()
                phi_residualized, iic_diagnostics = iic.residualize(
                    influence=phi,
                    judge_scores=judge_scores,
                    policy=policy,
                    fold_ids=None,  # Could use folds if available
                )

                if iic_diagnostics.get("applied", False):
                    var_reduction = iic_diagnostics.get("var_reduction_pct", 0)
                    logger.info(
                        f"IIC reduced variance by {var_reduction:.1f}% for {policy}"
                    )
                    phi = phi_residualized
            else:
                logger.debug("Judge scores not available for IIC")
        except ImportError:
            logger.warning("IIC not available, skipping variance reduction")

    # Compute main variance
    var_main = float(np.var(phi, ddof=1))
    n = len(phi)

    # Compute OUA variance via oracle jackknife
    var_oracle = 0.0
    if compute_oua:
        # Optional hook on estimator: get leave-one-oracle-fold re-estimates
        jackknife_vals = None
        if hasattr(estimator, "get_oracle_jackknife"):
            try:
                jackknife_vals = estimator.get_oracle_jackknife(policy)
            except Exception as e:
                logger.debug(f"get_oracle_jackknife failed: {e}")

        if jackknife_vals is not None:
            jackknife_vals = np.asarray(jackknife_vals, dtype=float)
            K = len(jackknife_vals)
            if K >= 2:
                psi_bar = float(np.mean(jackknife_vals))
                var_oracle = (
                    (K - 1) / K * float(np.mean((jackknife_vals - psi_bar) ** 2))
                )
                logger.debug(
                    f"Oracle jackknife variance: {var_oracle:.6f} from {K} folds"
                )
            else:
                logger.debug(f"Not enough oracle folds for jackknife: {K}")

    # Total variance
    var_total = var_main / n + var_oracle

    # Compute width
    z_score = stats.norm.ppf(1 - alpha / 2)
    wvar = 2 * z_score * np.sqrt(var_total)

    return wvar, SamplingVariance(
        var_main=var_main,
        var_oracle=var_oracle,
        var_total=var_total,
    )


def compute_estimator_eif(
    estimator: "BaseCJEEstimator",
    policy: str,
) -> Optional[np.ndarray]:
    """Compute or retrieve efficient influence function for an estimator.

    This function attempts to get the EIF through various methods:
    1. Check if estimator has get_eif() method
    2. Use plug-in formulas based on estimator type
    3. Default to IF = EIF for simple estimators

    Args:
        estimator: Fitted estimator
        policy: Target policy

    Returns:
        EIF values or None if unavailable
    """
    # First check if estimator provides EIF directly
    if hasattr(estimator, "get_eif"):
        eif = estimator.get_eif(policy)
        if eif is not None:
            return eif

    # Try plug-in based on estimator type
    estimator_type = estimator.__class__.__name__

    if estimator_type in ["RawIPS", "SNIPS"]:
        # For pure IPS variants without calibration learning, IF = EIF
        return estimator.get_influence_functions(policy)

    elif estimator_type == "CalibratedIPS":
        # CalibratedIPS has first-stage learning (calibration), so IF ≠ EIF in general
        # Would need to implement get_eif() properly accounting for calibration
        logger.debug("CalibratedIPS EIF not implemented - returning None")
        return None

    elif estimator_type in ["DRCPOEstimator", "MRDREstimator", "TMLEEstimator"]:
        # For DR estimators, use plug-in formula if we have the components
        try:
            weights = estimator.get_weights(policy)

            # Try to get outcome predictions
            if hasattr(estimator, "_outcome_predictions"):
                predictions = estimator._outcome_predictions.get(policy)
            else:
                predictions = None

            # Get rewards
            rewards = estimator.sampler.get_rewards()

            if weights is not None and predictions is not None and rewards is not None:
                # Try to get the estimate
                psi_hat = None
                if hasattr(estimator, "get_estimate"):
                    try:
                        psi_hat = estimator.get_estimate(policy)
                    except:
                        pass
                return compute_eif_plugin_dr(weights, rewards, predictions, psi=psi_hat)
            else:
                logger.debug(
                    f"Missing components for EIF computation in {estimator_type}"
                )
        except Exception as e:
            logger.debug(f"Failed to compute plug-in EIF: {e}")

    # Default: assume IF = EIF
    logger.debug(f"Using IF as EIF for {estimator_type}")
    return estimator.get_influence_functions(policy)
