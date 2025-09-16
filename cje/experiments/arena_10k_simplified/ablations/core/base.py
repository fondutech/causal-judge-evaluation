"""Base class for ablation experiments."""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy import stats

from .schemas import ExperimentSpec, create_result

# Handle import based on execution context
try:
    from ..config import DR_CONFIG, CFBITS_CONFIG
except ImportError:
    # Fallback for different import paths
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DR_CONFIG as DR_CONFIG, CFBITS_CONFIG as CFBITS_CONFIG  # type: ignore[import, no-redef]

# No local diagnostics file needed!
# Import standard diagnostics from CJE
from cje.diagnostics.weights import effective_sample_size, hill_tail_index


# Local function for weight CV (not in CJE)
def weight_cv(weights: np.ndarray) -> float:
    """Coefficient of variation of weights."""
    weights = np.asarray(weights)
    weights = weights[np.isfinite(weights)]
    if len(weights) == 0:
        return float(np.nan)
    mean_w = np.mean(weights)
    if mean_w == 0:
        return float(np.nan)
    return float(np.std(weights) / mean_w)


# Add parent directories to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje import load_dataset_from_jsonl
from cje.calibration import calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators import CalibratedIPS
from cje.estimators.stacking import StackedDREstimator
from cje.estimators.dr_base import DRCPOEstimator
from cje.estimators.orthogonalized_calibrated_dr import OrthogonalizedCalibratedDRCPO
from cje.estimators.orthogonalized_ips import OrthogonalizedCalibratedIPS
from cje.estimators.tr_cpo import TRCPOEstimator
from cje.estimators.mrdr import MRDREstimator
from cje.estimators.tmle import TMLEEstimator
from cje.data.fresh_draws import load_fresh_draws_auto

logger = logging.getLogger(__name__)


class BaseAblation:
    """Base class for all ablation experiments.

    Provides common functionality:
    - Data loading and preparation
    - Oracle masking and calibration
    - Estimator creation and execution
    - Diagnostic computation
    """

    def __init__(self, name: str):
        """Initialize ablation.

        Args:
            name: Name of this ablation (e.g., "oracle_coverage")
        """
        self.name = name
        self.results: List[Dict[str, Any]] = []

    def prepare_dataset(
        self, spec: ExperimentSpec, seed: int
    ) -> Tuple[Any, int, Dict[int, Any]]:
        """Load and prepare dataset with oracle masking.

        Args:
            spec: Experiment specification
            seed: Random seed

        Returns:
            (dataset, n_oracle, original_oracle_labels)
        """
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Load dataset
        dataset = load_dataset_from_jsonl(spec.dataset_path)

        # Subsample if requested
        if spec.sample_size is not None:
            n_samples = min(spec.sample_size, len(dataset.samples))
            indices = sorted(random.sample(range(len(dataset.samples)), n_samples))
            dataset.samples = [dataset.samples[i] for i in indices]
        elif spec.sample_fraction is not None:
            n_samples = int(len(dataset.samples) * spec.sample_fraction)
            indices = sorted(random.sample(range(len(dataset.samples)), n_samples))
            dataset.samples = [dataset.samples[i] for i in indices]

        # Mask oracle labels if coverage < 1
        original_oracle_labels = {}
        n_oracle = len(dataset.samples)  # Default: all have oracle

        if spec.oracle_coverage is not None and spec.oracle_coverage < 1.0:
            # Find samples with oracle labels
            oracle_indices = [
                i
                for i, s in enumerate(dataset.samples)
                if s.metadata.get("oracle_label") is not None
            ]

            # Determine how many to keep
            n_keep = max(2, int(len(oracle_indices) * spec.oracle_coverage))
            keep_indices = set(
                random.sample(oracle_indices, min(n_keep, len(oracle_indices)))
            )

            # Mask labels not in keep set
            for i, sample in enumerate(dataset.samples):
                if i not in keep_indices and "oracle_label" in sample.metadata:
                    original_oracle_labels[i] = sample.metadata["oracle_label"]
                    sample.metadata = sample.metadata.copy()
                    sample.metadata["oracle_label"] = None

            n_oracle = len(keep_indices)

        return dataset, n_oracle, original_oracle_labels

    def create_estimator(
        self, spec: ExperimentSpec, sampler: PrecomputedSampler, cal_result: Any
    ) -> Any:
        """Create estimator based on specification.

        Args:
            spec: Experiment specification
            sampler: PrecomputedSampler with data
            cal_result: Calibration result

        Returns:
            Estimator instance
        """
        # Extract settings from spec.extra
        use_iic = spec.extra.get("use_iic", False) if spec.extra else False
        use_weight_calibration = (
            spec.extra.get("use_weight_calibration", False) if spec.extra else False
        )
        # Propagate oracle-calibrator uncertainty into SEs (ON by default unless explicitly disabled)
        oua = spec.extra.get("oua_jackknife", True) if spec.extra else True

        # Log parameter settings if needed
        # logger.info(f"Creating {spec.estimator} with use_iic={use_iic}, "
        #            f"use_weight_calibration(SIMCal)={use_weight_calibration}")

        # Stacking MC config knobs from spec.extra (optional)
        include_mc = (
            bool(spec.extra.get("include_mc_in_objective", True))
            if spec.extra
            else True
        )
        mc_lambda = float(spec.extra.get("mc_lambda", 1.0)) if spec.extra else 1.0

        estimator_map = {
            "raw-ips": lambda s: CalibratedIPS(
                s,
                calibrate_weights=False,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                use_iic=use_iic,  # Pass IIC setting
                oua_jackknife=oua,
                use_outer_cv=True,
                n_outer_folds=5,
                outer_cv_seed=42,
                honest_iic=use_iic,  # Enable honest IIC for raw-ips too
            ),  # No weight calibration, but still support OUA
            "calibrated-ips": lambda s: CalibratedIPS(
                s,
                calibrate_weights=True,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                use_iic=use_iic,  # Pass IIC setting
                oua_jackknife=oua,  # Always use calibration, enable OUA
                use_outer_cv=True,  # Enable outer CV for robust inference
                n_outer_folds=5,  # Use 5 folds for clustering
                outer_cv_seed=42,  # Consistent fold generation
                honest_iic=use_iic,  # Enable honest IIC when outer CV is used
            ),
            "orthogonalized-ips": lambda s: OrthogonalizedCalibratedIPS(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                use_iic=use_iic,  # Pass IIC setting
                oua_jackknife=oua,
                use_outer_cv=True,  # Enable outer CV for robust inference
                n_outer_folds=5,  # Use 5 folds for clustering
                outer_cv_seed=42,  # Consistent fold generation
                honest_iic=use_iic,  # Enable honest IIC when outer CV is used
            ),
            "dr-cpo": lambda s: DRCPOEstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                use_calibrated_weights=use_weight_calibration,  # Controlled by use_weight_calibration flag
                use_iic=use_iic,  # Pass IIC setting
                oua_jackknife=oua,
            ),
            "oc-dr-cpo": lambda s: OrthogonalizedCalibratedDRCPO(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                use_calibrated_weights=use_weight_calibration,  # Controlled by use_weight_calibration flag
                use_iic=use_iic,  # Pass IIC setting
                oua_jackknife=oua,
            ),
            "tr-cpo": lambda s: TRCPOEstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                use_iic=use_iic,  # Pass IIC setting
                oua_jackknife=oua,
                use_efficient_tr=False,  # Vanilla TR-CPO uses raw W
            ),
            "tr-cpo-e": lambda s: TRCPOEstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                use_iic=use_iic,  # Pass IIC setting
                oua_jackknife=oua,
                use_efficient_tr=True,  # Efficient TR-CPO uses m̂(S)
            ),
            "tr-cpo-e-anchored": lambda s: TRCPOEstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                use_iic=use_iic,  # Pass IIC setting
                oua_jackknife=oua,
                use_efficient_tr=True,  # Efficient TR-CPO uses m̂(S)
                anchor_on_simcal=True,  # Anchor on SIMCal weights for stability
                add_orthogonalizer=False,  # No orthogonalizer
            ),
            "tr-cpo-e-anchored-orthogonal": lambda s: TRCPOEstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                use_iic=use_iic,  # Pass IIC setting
                oua_jackknife=oua,
                use_efficient_tr=True,  # Efficient TR-CPO uses m̂(S)
                anchor_on_simcal=True,  # Anchor on SIMCal weights for stability
                add_orthogonalizer=True,  # Add orthogonalizer for variance reduction
            ),
            "calibrated-dr-cpo": lambda s: DRCPOEstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                use_calibrated_weights=True,  # Use SIMCal calibrated weights
                use_iic=use_iic,  # Pass IIC setting
                oua_jackknife=oua,
            ),
            "mrdr": lambda s: MRDREstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                use_calibrated_weights=use_weight_calibration,  # Controlled by use_weight_calibration flag
                use_iic=use_iic,  # Pass IIC setting
                oua_jackknife=oua,
            ),
            "tmle": lambda s: TMLEEstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],
                use_calibrated_weights=use_weight_calibration,  # Controlled by use_weight_calibration flag
                use_iic=use_iic,  # Pass IIC setting
                oua_jackknife=oua,
            ),
            "stacked-dr": lambda s: StackedDREstimator(
                s,
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],  # Use n_folds, not V_folds
                use_calibrated_weights=use_weight_calibration,  # Controlled by use_weight_calibration flag
                use_iic=use_iic,  # Pass IIC setting
                oua_jackknife=oua,
                covariance_regularization=1e-4,  # Add regularization for numerical stability
                include_mc_in_objective=include_mc,
                mc_lambda=mc_lambda,
                # Remove use_outer_split - it doesn't exist
            ),
            "stacked-dr-core": lambda s: StackedDREstimator(
                s,
                estimators=["dr-cpo", "tmle", "mrdr"],  # Only 3 core DR estimators
                reward_calibrator=cal_result.calibrator if cal_result else None,
                n_folds=DR_CONFIG["n_folds"],  # Use n_folds, not V_folds
                use_calibrated_weights=use_weight_calibration,  # Controlled by use_weight_calibration flag
                use_iic=use_iic,  # Pass IIC setting
                oua_jackknife=oua,
                covariance_regularization=1e-4,  # Add regularization for numerical stability
                include_mc_in_objective=include_mc,
                mc_lambda=mc_lambda,
            ),
        }

        if spec.estimator not in estimator_map:
            raise ValueError(f"Unknown estimator: {spec.estimator}")

        return estimator_map[spec.estimator](sampler)

    def _compute_rmse(
        self, estimates: Dict[str, float], truths: Dict[str, float]
    ) -> float:
        """Compute RMSE between estimates and oracle truths.

        Note: The 'unhelpful' policy is excluded from RMSE calculation because
        it has a very different reward distribution (mean ~0.14) compared to
        other policies (mean ~0.76). This causes systematic calibration bias.
        """
        if not estimates or not truths:
            return float(np.nan)

        squared_errors = []
        for policy in estimates:
            # Skip unhelpful policy - different reward distribution
            if policy == "unhelpful":
                continue

            if policy in truths:
                est = estimates[policy]
                truth = truths[policy]
                if np.isfinite(est) and np.isfinite(truth):
                    squared_errors.append((est - truth) ** 2)

        if not squared_errors:
            return float(np.nan)

        return float(np.sqrt(np.mean(squared_errors)))

    def _load_oracle_ground_truth(
        self, dataset_path: str, dataset: Any, target_policies: List[str]
    ) -> Dict[str, float]:
        """Load oracle ground truth values for comparison.

        Args:
            dataset_path: Path to dataset file
            dataset: Dataset object
            target_policies: List of target policies

        Returns:
            Dictionary mapping policy names to oracle mean values
        """
        oracle_means = {}
        data_dir = Path(dataset_path).parent
        responses_dir = data_dir / "responses"

        # Load base policy oracle labels from dataset
        base_oracle_values = []
        for sample in dataset.samples:
            if hasattr(sample, "metadata") and sample.metadata:
                oracle_val = sample.metadata.get("oracle_label")
                if oracle_val is not None:
                    base_oracle_values.append(oracle_val)

        if base_oracle_values:
            oracle_means["base"] = float(np.mean(base_oracle_values))

        # Load oracle labels for each target policy from response files
        for policy in target_policies:
            response_file = responses_dir / f"{policy}_responses.jsonl"
            if response_file.exists():
                oracle_values = []
                with open(response_file, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if (
                                "metadata" in data
                                and "oracle_label" in data["metadata"]
                            ):
                                oracle_val = data["metadata"]["oracle_label"]
                                if oracle_val is not None:
                                    oracle_values.append(oracle_val)
                        except json.JSONDecodeError:
                            continue

                if oracle_values:
                    oracle_means[policy] = float(np.mean(oracle_values))
                    # Store per-policy oracle counts for debiasing
                    try:
                        dataset_name = getattr(self, "name", "ablation")
                    except Exception:
                        dataset_name = "ablation"
                    # Attach to an attribute for later consumption in run_single
                    if not hasattr(self, "_oracle_counts_per_policy"):
                        self._oracle_counts_per_policy = {}
                    self._oracle_counts_per_policy[policy] = int(len(oracle_values))

        return oracle_means

    def compute_cfbits_metrics(
        self, estimator: Any, policy: str, result: Dict[str, Any]
    ) -> None:
        """Compute CF-bits metrics if enabled using structured CFBitsDiagnostics.

        Args:
            estimator: Fitted estimator
            policy: Target policy name
            result: Result dictionary to update
        """
        # Check if enabled
        cfg = result.get("spec", {}).get("extra", {})
        if not cfg.get("compute_cfbits", False):
            return

        try:
            from cje.cfbits import cfbits_report_fresh_draws, cfbits_report_logging_only
            from cje.diagnostics import CFBitsDiagnostics

            # Route to appropriate playbook by estimator name
            estimator_name = result["spec"]["estimator"]

            # Fresh draws scenario
            fresh_draws_estimators = [
                "dr-cpo",
                "mrdr",
                "tmle",
                "tr-cpo",
                "tr-cpo-e",
                "tr-cpo-e-anchored",
                "tr-cpo-e-anchored-orthogonal",
                "stacked-dr",
                "stacked-dr-core",
                "oc-dr-cpo",
            ]

            # Logging-only scenario
            logging_only_estimators = [
                "raw-ips",
                "calibrated-ips",
                "orthogonalized-ips",
            ]

            if estimator_name in fresh_draws_estimators:
                report = cfbits_report_fresh_draws(
                    estimator,
                    policy,
                    cfbits_cfg=CFBITS_CONFIG,  # Use new API with config dict
                )
                scenario = "fresh-draws"
            elif estimator_name in logging_only_estimators:
                report = cfbits_report_logging_only(
                    estimator,
                    policy,
                    cfbits_cfg=CFBITS_CONFIG,  # Use new API with config dict
                )
                scenario = "logging-only"
            else:
                logger.warning(f"Unknown estimator for CF-bits: {estimator_name}")
                return

            # Create structured CFBitsDiagnostics from report
            from cje.diagnostics import GateState

            # Map string gate state to enum
            gate_str = report.get("gates", {}).get("state", "CRITICAL")
            gate_map = {
                "GOOD": GateState.GOOD,
                "WARNING": GateState.WARNING,
                "CRITICAL": GateState.CRITICAL,
                "REFUSE": GateState.REFUSE,
            }
            gate_state = gate_map.get(gate_str, GateState.CRITICAL)

            # Extract gate reasons from report
            gate_reasons = []
            gates_dict = report.get("gates", {})
            if gates_dict.get("recommendation"):
                gate_reasons.append(gates_dict["recommendation"])

            cfbits_diag = CFBitsDiagnostics(
                policy=policy,
                estimator_type=estimator_name,
                scenario=scenario,
                # Core width metrics
                wid=report.get("identification", {}).get("wid"),
                wvar=report.get("sampling_width", {}).get("wvar"),
                w_tot=report.get("cfbits", {}).get("w_tot"),
                bits_tot=report.get("cfbits", {}).get("bits_tot"),
                # Efficiency metrics
                ifr_oua=report.get("efficiency", {}).get("ifr_oua"),
                aess_oua=report.get("efficiency", {}).get("aess_oua"),
                # σ(S) structural floors
                aessf_sigmaS=report.get("overlap", {}).get("aessf_sigmaS"),
                aessf_sigmaS_lcb=report.get("overlap", {}).get("aessf_lcb"),
                bc_sigmaS=report.get("overlap", {}).get("bc_sigmaS"),
                # Gate assessment
                gate_state=gate_state,
                gate_reasons=gate_reasons,
                # Budget recommendations
                labels_for_wid_reduction=report.get("gates", {}).get("recommended_n"),
                n_oracle_available=report.get("metadata", {}).get("n_samples"),
            )

            # Validate the diagnostics
            cfbits_diag.validate()

            # Store structured object (for compatibility with new system)
            result.setdefault("cfbits_diagnostics", {})[policy] = cfbits_diag

            # Store minimal summary for backward compatibility
            result["cfbits_summary"][policy] = {
                "bits_tot": cfbits_diag.bits_tot,
                "w_tot": cfbits_diag.w_tot,
                "wvar": cfbits_diag.wvar,
                "wid": cfbits_diag.wid,
                "ifr_oua": cfbits_diag.ifr_oua,
                "aess_oua": cfbits_diag.aess_oua,
                "aessf_lcb": cfbits_diag.aessf_sigmaS_lcb,
                "gate_state": cfbits_diag.gate_state.value,
                "gate_reasons": cfbits_diag.gate_reasons,
            }

            # Store gate info
            result["cfbits_gates"][policy] = {
                "state": cfbits_diag.gate_state.value,
                "reasons": cfbits_diag.gate_reasons,
                "labels_recommended": cfbits_diag.labels_for_wid_reduction,
            }

            # Store full data only if requested (to avoid bloat)
            if cfg.get("compute_cfbits_full", False):
                result["cfbits_data"][policy] = report  # Keep raw report for debugging

        except Exception as e:
            logger.warning(f"CF-bits computation failed for {policy}: {e}")
            # Graceful degradation - don't fail the experiment

    def compute_diagnostics(
        self, estimator: Any, result: Dict[str, Any], n_total: int
    ) -> None:
        """Compute and add diagnostics to result.

        Args:
            estimator: Fitted estimator
            result: Result dictionary to update
            n_total: Total number of samples
        """
        # Import CJE's diagnostic functions
        from cje.diagnostics.overlap import compute_overlap_metrics
        from cje.diagnostics.weights import hill_tail_index_stable

        # Get target policies
        policies = estimator.sampler.target_policies

        for policy in policies:
            try:
                # Get weights (method may vary by estimator)
                if hasattr(estimator, "get_weights"):
                    weights = estimator.get_weights(policy)
                elif hasattr(estimator, "_weights_cache"):
                    weights = estimator._weights_cache.get(policy)
                else:
                    weights = estimator.sampler.compute_importance_weights(policy)

                if weights is not None and len(weights) > 0:
                    # Compute basic diagnostics using local functions (simpler, direct)
                    ess = effective_sample_size(weights)
                    result["ess_absolute"][policy] = ess
                    result["ess_relative"][policy] = (
                        100.0 * ess / n_total if n_total > 0 else 0
                    )
                    result["tail_alpha"][policy] = hill_tail_index(weights)
                    result["weight_cv"][policy] = weight_cv(weights)

                    # Max weight (normalized)
                    weights_norm = weights / np.sum(weights)
                    result["max_weight"][policy] = float(np.max(weights_norm))

                    # Mass concentration (fraction of weights near zero)
                    # Use threshold of 1/(10*n) as "near zero"
                    threshold = 1.0 / (10 * len(weights))
                    result["mass_concentration"][policy] = float(
                        np.mean(weights_norm < threshold)
                    )

                    # Compute advanced overlap metrics from CJE diagnostics
                    overlap_metrics = compute_overlap_metrics(
                        weights,
                        target_ci_halfwidth=0.01,
                        n_samples=n_total,
                        compute_tail_index=True,
                        auto_tune_threshold=False,
                    )

                    # Store additional diagnostics
                    result.setdefault("hellinger_affinity", {})[policy] = float(
                        overlap_metrics.hellinger_affinity
                    )
                    result.setdefault("overlap_quality", {})[
                        policy
                    ] = overlap_metrics.overlap_quality
                    result.setdefault("can_calibrate", {})[policy] = bool(
                        overlap_metrics.can_calibrate
                    )
                    result.setdefault("recommended_method", {})[
                        policy
                    ] = overlap_metrics.recommended_method

            except Exception as e:
                logger.warning(f"Failed to compute diagnostics for {policy}: {e}")

            # Add CF-bits computation (non-fatal)
            self.compute_cfbits_metrics(estimator, policy, result)

    def run_single(self, spec: ExperimentSpec, seed: int) -> Dict[str, Any]:
        """Run single experiment with given seed.

        Args:
            spec: Experiment specification
            seed: Random seed

        Returns:
            Result dictionary
        """
        # Create result
        result = create_result(spec, seed)

        try:
            # Prepare data
            dataset, n_oracle, original_oracle_labels = self.prepare_dataset(spec, seed)
            result["n_samples"] = len(dataset.samples)
            result["n_oracle"] = n_oracle

            # ALWAYS calibrate rewards (judge → oracle) when oracle labels exist
            # This is NOT controlled by use_weight_calibration flag
            # Calibration mode can be configured via spec.extra["reward_calibration_mode"]
            reward_calibration_mode = (
                spec.extra.get("reward_calibration_mode", "monotone")
                if spec.extra
                else "monotone"
            )

            calibrated_dataset, cal_result = calibrate_dataset(
                dataset,
                judge_field="judge_score",
                oracle_field="oracle_label",
                enable_cross_fit=True,
                n_folds=DR_CONFIG["n_folds"] if n_oracle >= 50 else 3,
                calibration_mode=reward_calibration_mode,
                random_seed=seed,  # Pass the experiment seed for fold assignment
            )

            if cal_result:
                result["calibration_rmse"] = cal_result.calibration_rmse
                # R² may not be available in older versions
                if hasattr(cal_result, "calibration_r2"):
                    result["calibration_r2"] = cal_result.calibration_r2
                elif hasattr(cal_result, "r2"):
                    result["calibration_r2"] = cal_result.r2

                # Extract calibrated reward range to detect overlap issues
                # This catches the unhelpful policy problem (min ~0.4 instead of 0)
                if calibrated_dataset and calibrated_dataset.metadata:
                    cal_info = calibrated_dataset.metadata.get("calibration_info", {})
                    if "f_min" in cal_info:
                        result["calibrated_reward_min"] = cal_info["f_min"]
                    if "f_max" in cal_info:
                        result["calibrated_reward_max"] = cal_info["f_max"]

                    # Flag overlap issues: calibration can't extrapolate beyond observed range
                    # If min > 0.1 or max < 0.9, we have incomplete coverage of [0,1]
                    if "f_min" in cal_info and "f_max" in cal_info:
                        f_min = cal_info["f_min"]
                        f_max = cal_info["f_max"]
                        if f_min > 0.1 or f_max < 0.9:
                            result["reward_overlap_warning"] = True
                            logger.warning(
                                f"Calibrated reward range [{f_min:.3f}, {f_max:.3f}] "
                                f"does not cover full [0,1] oracle range - estimates may be biased"
                            )
                # Track which calibration mode was actually used
                if cal_result.calibrator and hasattr(
                    cal_result.calibrator, "selected_mode"
                ):
                    result["reward_calibration_used"] = (
                        cal_result.calibrator.selected_mode
                    )
                else:
                    result["reward_calibration_used"] = reward_calibration_mode

            # Create sampler and estimator
            sampler = PrecomputedSampler(calibrated_dataset)
            estimator = self.create_estimator(spec, sampler, cal_result)

            # Add fresh draws for DR methods
            if spec.estimator in [
                "dr-cpo",
                "oc-dr-cpo",
                "calibrated-dr-cpo",
                "mrdr",
                "tmle",
                "tr-cpo",
                "tr-cpo-e",
                "tr-cpo-e-anchored",
                "tr-cpo-e-anchored-orthogonal",
                "stacked-dr",
            ]:
                data_dir = Path(spec.dataset_path).parent

                # Get prompt IDs from the subsampled dataset
                dataset_prompt_ids = set()
                for sample in calibrated_dataset.samples:
                    if hasattr(sample, "prompt_id") and sample.prompt_id:
                        dataset_prompt_ids.add(sample.prompt_id)

                for policy in sampler.target_policies:
                    try:
                        # Load ALL fresh draws
                        all_fresh_draws = load_fresh_draws_auto(
                            data_dir, policy, verbose=False
                        )

                        # Filter to only include fresh draws matching our subsampled prompts
                        if dataset_prompt_ids:
                            from cje.data.fresh_draws import (
                                FreshDrawSample,
                                FreshDrawDataset,
                            )

                            filtered_samples: List[FreshDrawSample] = []
                            for fd_sample in all_fresh_draws.samples:
                                if (
                                    hasattr(fd_sample, "prompt_id")
                                    and fd_sample.prompt_id in dataset_prompt_ids
                                ):
                                    filtered_samples.append(fd_sample)

                            # Create filtered fresh draws dataset with required fields

                            # Count draws per prompt
                            draws_per_prompt_dict: Dict[str, int] = {}
                            for fd_sample in filtered_samples:
                                prompt_id = (
                                    fd_sample.prompt_id
                                    if hasattr(fd_sample, "prompt_id")
                                    else None
                                )
                                if prompt_id:
                                    draws_per_prompt_dict[prompt_id] = (
                                        draws_per_prompt_dict.get(prompt_id, 0) + 1
                                    )

                            # Get the most common draws per prompt value
                            draws_per_prompt = (
                                max(
                                    set(draws_per_prompt_dict.values()),
                                    key=list(draws_per_prompt_dict.values()).count,
                                )
                                if draws_per_prompt_dict
                                else 10
                            )

                            filtered_fresh_draws = FreshDrawDataset(
                                samples=filtered_samples,
                                target_policy=policy,  # Use the policy we're processing
                                draws_per_prompt=draws_per_prompt,
                            )

                            estimator.add_fresh_draws(policy, filtered_fresh_draws)
                            logger.info(
                                f"Added {len(filtered_samples)}/{len(all_fresh_draws.samples)} fresh draws for {policy}"
                            )
                        else:
                            # If no prompt IDs, use all fresh draws (fallback)
                            estimator.add_fresh_draws(policy, all_fresh_draws)

                    except FileNotFoundError:
                        logger.warning(f"No fresh draws for {policy}")

            # Run estimation
            estimation_result = estimator.fit_and_estimate()

            # Extract results
            for i, policy in enumerate(sampler.target_policies):
                result["estimates"][policy] = float(estimation_result.estimates[i])

                # Always store base standard errors if available
                if estimation_result.standard_errors is not None:
                    base_se = float(estimation_result.standard_errors[i])
                    result["standard_errors"][policy] = base_se

                    # Also store robust SEs separately if available (includes OUA)
                    has_robust_se = False
                    if (
                        hasattr(estimation_result, "robust_standard_errors")
                        and estimation_result.robust_standard_errors is not None
                    ):
                        robust_se = float(estimation_result.robust_standard_errors[i])
                        result.setdefault("robust_standard_errors", {})[
                            policy
                        ] = robust_se
                        has_robust_se = True

                    # Regular CIs always use base SE
                    se_for_ci = base_se

                    # Check if robust CIs are already computed (with t-critical values)
                    if (
                        hasattr(estimation_result, "robust_confidence_intervals")
                        and estimation_result.robust_confidence_intervals is not None
                        and i < len(estimation_result.robust_confidence_intervals)
                        and estimation_result.robust_confidence_intervals[i] is not None
                    ):
                        # Use pre-computed robust CI (has t-critical values)
                        ci = estimation_result.robust_confidence_intervals[i]
                        if isinstance(ci, (list, tuple)) and len(ci) == 2:
                            result["confidence_intervals"][policy] = (
                                float(ci[0]),
                                float(ci[1]),
                            )
                            # Also store as robust CI
                            result.setdefault("robust_confidence_intervals", {})[
                                policy
                            ] = (
                                float(ci[0]),
                                float(ci[1]),
                            )
                        else:
                            # Fallback to z-based CI
                            est = estimation_result.estimates[i]
                            result["confidence_intervals"][policy] = (
                                float(est - 1.96 * se_for_ci),
                                float(est + 1.96 * se_for_ci),
                            )
                            if has_robust_se:
                                # Use robust SE for robust CI
                                robust_se_val = (
                                    estimation_result.robust_standard_errors[i]
                                )
                                result.setdefault("robust_confidence_intervals", {})[
                                    policy
                                ] = (
                                    float(est - 1.96 * robust_se_val),
                                    float(est + 1.96 * robust_se_val),
                                )
                    else:
                        # Compute confidence interval from SE
                        # Use t-critical value if cluster DF is available
                        tcrit = 1.96  # Default to z-critical

                        # Try to extract cluster DF from metadata
                        md = getattr(estimation_result, "metadata", {}) or {}
                        seinfo = md.get("_se_diagnostics", {})
                        df = None

                        if isinstance(seinfo, dict) and policy in seinfo:
                            # Look for cluster counts in cluster_robust_detail
                            det = seinfo[policy].get("cluster_robust_detail", {})
                            G_out = det.get("G_outer")
                            G_in = det.get("G_inner")

                            # Use max(G_inner-1, G_outer-1) as degrees of freedom
                            cands = [
                                g - 1
                                for g in [G_out, G_in]
                                if isinstance(g, int) and g and g > 1
                            ]
                            if cands:
                                df = max(cands)

                        # Use t-distribution if we have valid DF
                        if df and df > 1:
                            tcrit = float(stats.t.ppf(0.975, df))

                        est = estimation_result.estimates[i]
                        result["confidence_intervals"][policy] = (
                            float(est - tcrit * se_for_ci),
                            float(est + tcrit * se_for_ci),
                        )
                        # Also store as robust CI if we have robust SE
                        if has_robust_se:
                            # Use robust SE for robust CI
                            robust_se_val = estimation_result.robust_standard_errors[i]
                            result.setdefault("robust_confidence_intervals", {})[
                                policy
                            ] = (
                                float(est - tcrit * robust_se_val),
                                float(est + tcrit * robust_se_val),
                            )

            # Extract metadata if available (filtering out verbose fields)
            if hasattr(estimation_result, "metadata") and estimation_result.metadata:
                # Filter out verbose/internal fields that bloat results
                filtered_metadata = {
                    k: v
                    for k, v in estimation_result.metadata.items()
                    if k
                    not in [
                        "if_sample_indices"
                    ]  # This is just [0, 1, 2, ...] - not needed
                }
                if filtered_metadata:  # Only store if non-empty after filtering
                    result["metadata"] = filtered_metadata

            # Extract DR-specific diagnostics if available
            if (
                hasattr(estimation_result, "diagnostics")
                and estimation_result.diagnostics
            ):
                diag = estimation_result.diagnostics

                # Extract overall status
                if hasattr(diag, "overall_status"):
                    result["overall_status"] = str(
                        diag.overall_status.name
                        if hasattr(diag.overall_status, "name")
                        else diag.overall_status
                    )

                # Extract outcome model R² for DR estimators
                if hasattr(diag, "outcome_r2_range"):
                    result["outcome_r2_min"], result["outcome_r2_max"] = (
                        diag.outcome_r2_range
                    )

                # Check for DR diagnostics
                if (
                    hasattr(diag, "dr_diagnostics_per_policy")
                    and diag.dr_diagnostics_per_policy is not None
                ):
                    for policy, policy_diag in diag.dr_diagnostics_per_policy.items():
                        if isinstance(policy_diag, dict):
                            # Extract outcome model R² per policy
                            if "r2_oof" in policy_diag:
                                result.setdefault("outcome_r2", {})[policy] = (
                                    policy_diag["r2_oof"]
                                )
                            if "orthogonality_score" in policy_diag:
                                result.setdefault("orthogonality_score", {})[policy] = (
                                    policy_diag["orthogonality_score"]
                                )
                            if "mc_variance_share" in policy_diag:
                                result.setdefault("mc_variance_share", {})[policy] = (
                                    policy_diag["mc_variance_share"]
                                )
                            if "draws_per_prompt" in policy_diag:
                                result["draws_per_prompt"] = policy_diag[
                                    "draws_per_prompt"
                                ]
                # Also check for orthogonality_scores directly
                if hasattr(diag, "orthogonality_scores"):
                    result["orthogonality_scores"] = diag.orthogonality_scores
                # Check for IIC diagnostics
                if hasattr(diag, "iic_diagnostics") and diag.iic_diagnostics:
                    result["iic_diagnostics"] = diag.iic_diagnostics

            # Also check metadata for IIC diagnostics (DR estimators store them there)
            if hasattr(estimation_result, "metadata") and estimation_result.metadata:
                if "iic_diagnostics" in estimation_result.metadata:
                    result["iic_diagnostics"] = estimation_result.metadata[
                        "iic_diagnostics"
                    ]
                # OC-DR-CPO stores orthogonality scores in metadata
                if "orthogonality_scores" in estimation_result.metadata:
                    result["orthogonality_scores"] = estimation_result.metadata[
                        "orthogonality_scores"
                    ]

            # Compute diagnostics
            self.compute_diagnostics(estimator, result, len(dataset.samples))

            # Restore oracle labels for ground truth computation
            if original_oracle_labels:
                for idx, oracle_label in original_oracle_labels.items():
                    dataset.samples[idx].metadata["oracle_label"] = oracle_label

            # Compute oracle truths
            oracle_truths = self._load_oracle_ground_truth(
                spec.dataset_path,
                dataset,
                list(sampler.target_policies),
            )
            result["oracle_truths"] = oracle_truths
            # If available, attach per-policy oracle counts to result
            if hasattr(self, "_oracle_counts_per_policy") and isinstance(
                self._oracle_counts_per_policy, dict
            ):
                result["n_oracle_per_policy"] = {
                    k: int(v) for k, v in self._oracle_counts_per_policy.items()
                }

            # Compute RMSE (excluding unhelpful policy which has different distribution)
            result["rmse_vs_oracle"] = self._compute_rmse(
                result["estimates"], oracle_truths
            )

            # Mean CI width
            if result["confidence_intervals"]:
                widths = [
                    ci[1] - ci[0] for ci in result["confidence_intervals"].values()
                ]
                result["mean_ci_width"] = float(np.mean(widths))

            result["success"] = True

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            result["error"] = str(e)
            result["success"] = False

        result["runtime_s"] = time.time() - result["start_ts"]

        # Convert numpy bools in diagnostic outputs to Python bools for JSON serialization
        # These come from the CJE library diagnostics
        if (
            "orthogonality_scores" in result
            and result["orthogonality_scores"] is not None
        ):
            for policy, scores in result["orthogonality_scores"].items():
                if isinstance(scores, dict) and "passes_test" in scores:
                    scores["passes_test"] = bool(scores["passes_test"])

        if "iic_diagnostics" in result and result["iic_diagnostics"] is not None:
            for policy, diag in result["iic_diagnostics"].items():
                if isinstance(diag, dict) and "mean_preserved" in diag:
                    diag["mean_preserved"] = bool(diag["mean_preserved"])

        # Convert all numpy types to Python types for JSON serialization
        result = self._convert_numpy(result)
        return result

    def _convert_numpy(self, obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(v) for v in obj]
        elif hasattr(obj, "item"):
            return obj.item()
        return obj

    def run_with_seeds(self, spec: ExperimentSpec) -> List[Dict[str, Any]]:
        """Run experiment with multiple seeds.

        Args:
            spec: Experiment specification

        Returns:
            List of results (one per seed)
        """
        results = []
        for i in range(spec.n_seeds):
            seed = spec.seed_base + i
            logger.info(f"Running {self.name} with seed {seed} ({i+1}/{spec.n_seeds})")
            result = self.run_single(spec, seed)
            results.append(result)

            # Log progress
            if result["success"]:
                logger.info(f"  ✓ RMSE: {result.get('rmse_vs_oracle', 'N/A'):.4f}")
            else:
                logger.warning(f"  ✗ Failed: {result.get('error', 'Unknown')}")

        return results

    def run_ablation(self) -> List[Dict[str, Any]]:
        """Run the complete ablation.

        Override this in subclasses to define the experiment grid.
        """
        raise NotImplementedError("Subclasses must implement run_ablation()")
