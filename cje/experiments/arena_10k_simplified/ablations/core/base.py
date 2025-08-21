"""Base class for ablation experiments."""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .schemas import ExperimentSpec, create_result
from .diagnostics import (
    effective_sample_size,
    hill_alpha,
    weight_cv,
    compute_rmse,
    simcal_distortion,
)
from .gates import check_gates, apply_mitigation_ladder, GateConfig

# Add parent directories to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from cje import load_dataset_from_jsonl
from cje.calibration import calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators import CalibratedIPS
from cje.estimators.dr_base import DRCPOEstimator
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
    - Gate checking and mitigation
    - Result caching
    """

    def __init__(self, name: str, cache_dir: Optional[Path] = None):
        """Initialize ablation.

        Args:
            name: Name of this ablation (e.g., "oracle_coverage")
            cache_dir: Directory for caching results
        """
        self.name = name
        self.cache_dir = cache_dir or Path(".ablation_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    def load_cached_result(
        self, spec: ExperimentSpec, seed: int
    ) -> Optional[Dict[str, Any]]:
        """Load cached result if it exists.

        Args:
            spec: Experiment specification
            seed: Random seed

        Returns:
            Cached result or None
        """
        cache_path = self.cache_dir / f"{spec.uid()}_{seed}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    result = json.load(f)
                logger.info(f"Loaded cached result for {spec.uid()}_{seed}")
                return result  # type: ignore[no-any-return]
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def save_result(
        self, spec: ExperimentSpec, seed: int, result: Dict[str, Any]
    ) -> None:
        """Save result to cache.

        Args:
            spec: Experiment specification
            seed: Random seed
            result: Result dictionary
        """
        cache_path = self.cache_dir / f"{spec.uid()}_{seed}.json"
        try:
            # Convert numpy types to Python types for JSON serialization
            import numpy as np

            def convert_numpy(obj: Any) -> Any:
                """Recursively convert numpy types to Python types."""
                import numpy as np

                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy(v) for v in obj)
                elif hasattr(obj, "item"):  # Catch any other numpy scalars
                    return obj.item()
                return obj

            result_converted = convert_numpy(result)

            with open(cache_path, "w") as f:
                json.dump(result_converted, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

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
        estimator_map = {
            "raw-ips": lambda s: CalibratedIPS(s, calibrate=False),
            "calibrated-ips": lambda s: CalibratedIPS(s),
            "dr-cpo": lambda s: DRCPOEstimator(
                s,
                calibrator=cal_result.calibrator if cal_result else None,
                n_folds=5,
                oracle_slice_config=(
                    spec.oracle_coverage < 1.0 if spec.oracle_coverage else False
                ),
            ),
            "mrdr": lambda s: MRDREstimator(
                s,
                calibrator=cal_result.calibrator if cal_result else None,
                n_folds=5,
                oracle_slice_config=(
                    spec.oracle_coverage < 1.0 if spec.oracle_coverage else False
                ),
            ),
            "tmle": lambda s: TMLEEstimator(
                s,
                calibrator=cal_result.calibrator if cal_result else None,
                n_folds=5,
                oracle_slice_config=(
                    spec.oracle_coverage < 1.0 if spec.oracle_coverage else False
                ),
            ),
        }

        if spec.estimator not in estimator_map:
            raise ValueError(f"Unknown estimator: {spec.estimator}")

        return estimator_map[spec.estimator](sampler)

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

        return oracle_means

    def compute_diagnostics(
        self, estimator: Any, result: Dict[str, Any], n_total: int
    ) -> None:
        """Compute and add diagnostics to result.

        Args:
            estimator: Fitted estimator
            result: Result dictionary to update
            n_total: Total number of samples
        """
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
                    # Compute diagnostics
                    ess = effective_sample_size(weights)
                    result["ess_absolute"][policy] = ess
                    result["ess_relative"][policy] = (
                        100.0 * ess / n_total if n_total > 0 else 0
                    )
                    result["tail_alpha"][policy] = hill_alpha(weights)
                    result["weight_cv"][policy] = weight_cv(weights)

                    # Max weight (normalized)
                    weights_norm = weights / np.sum(weights)
                    result["max_weight"][policy] = np.max(weights_norm)

                    # Check gates
                    gates = check_gates(weights, GateConfig(), n_total)
                    result["gate_status"][policy] = gates

            except Exception as e:
                logger.warning(f"Failed to compute diagnostics for {policy}: {e}")

    def run_single(self, spec: ExperimentSpec, seed: int) -> Dict[str, Any]:
        """Run single experiment with given seed.

        Args:
            spec: Experiment specification
            seed: Random seed

        Returns:
            Result dictionary
        """
        # Check cache
        cached = self.load_cached_result(spec, seed)
        if cached is not None:
            return cached

        # Create result
        result = create_result(spec, seed)

        try:
            # Prepare data
            dataset, n_oracle, original_oracle_labels = self.prepare_dataset(spec, seed)
            result["n_samples"] = len(dataset.samples)
            result["n_oracle"] = n_oracle

            # Calibrate
            calibrated_dataset, cal_result = calibrate_dataset(
                dataset,
                judge_field="judge_score",
                oracle_field="oracle_label",
                enable_cross_fit=True,
                n_folds=5 if n_oracle >= 50 else 3,
            )

            if cal_result:
                result["calibration_rmse"] = cal_result.calibration_rmse

            # Create sampler and estimator
            sampler = PrecomputedSampler(calibrated_dataset)
            estimator = self.create_estimator(spec, sampler, cal_result)

            # Add fresh draws for DR methods
            if spec.estimator in ["dr-cpo", "mrdr", "tmle"]:
                data_dir = Path(spec.dataset_path).parent
                for policy in sampler.target_policies:
                    try:
                        fresh_draws = load_fresh_draws_auto(
                            data_dir, policy, verbose=False
                        )
                        estimator.add_fresh_draws(policy, fresh_draws)
                    except FileNotFoundError:
                        logger.warning(f"No fresh draws for {policy}")

            # Run estimation
            estimation_result = estimator.fit_and_estimate()

            # Extract results
            for i, policy in enumerate(sampler.target_policies):
                result["estimates"][policy] = float(estimation_result.estimates[i])
                if estimation_result.standard_errors is not None:
                    result["standard_errors"][policy] = float(
                        estimation_result.standard_errors[i]
                    )
                    # Compute confidence interval from SE
                    est = estimation_result.estimates[i]
                    se = estimation_result.standard_errors[i]
                    result["confidence_intervals"][policy] = (
                        float(est - 1.96 * se),
                        float(est + 1.96 * se),
                    )

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

            # Compute RMSE
            result["rmse_vs_oracle"] = compute_rmse(result["estimates"], oracle_truths)

            # Mean CI width
            if result["confidence_intervals"]:
                widths = [
                    ci[1] - ci[0] for ci in result["confidence_intervals"].values()
                ]
                result["mean_ci_width"] = np.mean(widths)

            result["success"] = True

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            result["error"] = str(e)
            result["success"] = False

        result["runtime_s"] = time.time() - result["start_ts"]

        # Save to cache
        self.save_result(spec, seed, result)

        return result

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
