#!/usr/bin/env python3
"""Ablation study script using CJE's proper APIs and design patterns.

Key features:
1. Uses CJE's load_fresh_draws_auto instead of custom implementation
2. Composable functions for each step of the pipeline
3. Fail-fast error handling with clear error messages
4. Adaptive n_folds based on oracle sample size

Usage:
    # Quick test
    python ablation.py --estimators calibrated-ips
    
    # Full ablation
    python ablation.py --estimators raw-ips calibrated-ips dr-cpo mrdr tmle \
                       --oracle-coverages 0.05 0.1 0.2 0.5 1.0
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import random
import time

# Use CJE's high-level API where possible
from cje import (
    load_dataset_from_jsonl,
    calibrate_dataset,
    PrecomputedSampler,
    CalibratedIPS,
    DRCPOEstimator,
    MRDREstimator,
    TMLEEstimator,
)

# Use CJE's fresh draws utilities instead of custom implementation
from cje.data.fresh_draws import load_fresh_draws_auto
from cje.data.models import Dataset, EstimationResult

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Configuration
class AblationConfig:
    """Configuration for ablation experiments."""

    ORACLE_TRUTHS = {
        "clone": 0.7359,
        "parallel_universe_prompt": 0.7553,
        "premium": 0.7399,
        "unhelpful": 0.1440,
    }

    ESTIMATOR_CLASSES: Dict[str, Union[type, Callable]] = {
        "raw-ips": lambda s: CalibratedIPS(s, calibrate=False),
        "calibrated-ips": CalibratedIPS,
        "dr-cpo": DRCPOEstimator,
        "mrdr": MRDREstimator,
        "tmle": TMLEEstimator,
    }

    # Adaptive n_folds based on oracle count
    @staticmethod
    def get_n_folds(n_oracle: int) -> int:
        """Get appropriate number of CV folds based on oracle sample size."""
        if n_oracle < 10:
            return 2
        elif n_oracle < 50:
            return 3
        else:
            return 5


# Step 1: Data preparation functions (composable)
def prepare_dataset(
    data_path: str, sample_fraction: float = 1.0, seed: Optional[int] = None
) -> Dataset:
    """Load and optionally subsample dataset.

    Args:
        data_path: Path to CJE dataset
        sample_fraction: Fraction of dataset to use (0-1)
        seed: Random seed for subsampling

    Returns:
        Dataset object

    Raises:
        FileNotFoundError: If dataset doesn't exist
        ValueError: If dataset is invalid
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Load dataset - fail fast if not found
    dataset = load_dataset_from_jsonl(data_path)

    # Subsample if requested
    if sample_fraction < 1.0:
        n_samples = int(len(dataset.samples) * sample_fraction)
        if n_samples < 10:
            raise ValueError(
                f"Sample fraction {sample_fraction} yields only {n_samples} samples. Need at least 10."
            )

        indices = sorted(random.sample(range(len(dataset.samples)), n_samples))
        dataset.samples = [dataset.samples[i] for i in indices]

        logger.info(f"Subsampled to {n_samples} samples ({sample_fraction:.0%})")

    return dataset


def mask_oracle_labels(
    dataset: Dataset, oracle_coverage: float, seed: Optional[int] = None
) -> Tuple[Dataset, int]:
    """Mask oracle labels to simulate partial coverage.

    Args:
        dataset: Dataset with oracle labels
        oracle_coverage: Fraction to keep (0-1)
        seed: Random seed for masking

    Returns:
        Tuple of (modified dataset, number of oracle samples kept)

    Raises:
        ValueError: If no oracle labels found or coverage invalid
    """
    if not 0 < oracle_coverage <= 1:
        raise ValueError(f"Oracle coverage must be in (0, 1], got {oracle_coverage}")

    if seed is not None:
        random.seed(seed)

    # Find samples with oracle labels
    oracle_indices = [
        i
        for i, s in enumerate(dataset.samples)
        if s.metadata.get("oracle_label") is not None
    ]

    if not oracle_indices:
        raise ValueError("No oracle labels found in dataset")

    # Determine how many to keep
    n_keep = max(2, int(len(oracle_indices) * oracle_coverage))
    keep_indices = set(random.sample(oracle_indices, min(n_keep, len(oracle_indices))))

    # Mask labels not in keep set
    for i, sample in enumerate(dataset.samples):
        if i not in keep_indices and "oracle_label" in sample.metadata:
            # Create copy to avoid modifying original
            sample.metadata = sample.metadata.copy()
            sample.metadata["oracle_label"] = None

    logger.info(
        f"Kept {n_keep}/{len(oracle_indices)} oracle labels ({oracle_coverage:.0%} coverage)"
    )

    return dataset, n_keep


# Step 2: Calibration function
def calibrate_with_oracle(
    dataset: Dataset,
    n_oracle: int,
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
) -> Tuple[Dataset, Any]:
    """Calibrate dataset using oracle labels.

    Args:
        dataset: Dataset with judge scores and oracle labels
        n_oracle: Number of oracle samples (for adaptive n_folds)
        judge_field: Metadata field with judge scores
        oracle_field: Metadata field with oracle labels

    Returns:
        Tuple of (calibrated dataset, calibration result)
    """
    n_folds = AblationConfig.get_n_folds(n_oracle)

    calibrated_dataset, cal_result = calibrate_dataset(
        dataset,
        judge_field=judge_field,
        oracle_field=oracle_field,
        enable_cross_fit=True,
        n_folds=n_folds,
    )

    if cal_result:
        logger.info(
            f"Calibration RMSE: {cal_result.calibration_rmse:.3f}, n_folds: {n_folds}"
        )

    return calibrated_dataset, cal_result


# Step 3: Estimator creation
def create_estimator(
    sampler: PrecomputedSampler, estimator_name: str, cal_result: Optional[Any] = None
) -> Any:
    """Create estimator instance.

    Args:
        sampler: PrecomputedSampler with data
        estimator_name: Name of estimator to create
        cal_result: Calibration result (for DR estimators)

    Returns:
        Estimator instance

    Raises:
        ValueError: If estimator name unknown
    """
    if estimator_name not in AblationConfig.ESTIMATOR_CLASSES:
        raise ValueError(f"Unknown estimator: {estimator_name}")

    EstimatorClass = AblationConfig.ESTIMATOR_CLASSES[estimator_name]

    # Handle special case for raw-ips (lambda function)
    if estimator_name == "raw-ips":
        # EstimatorClass is a lambda that takes sampler
        return EstimatorClass(sampler)  # type: ignore

    # Create estimator with appropriate configuration
    if estimator_name in ["dr-cpo", "mrdr", "tmle"]:
        # DR estimators need calibrator and n_folds
        n_folds = 5  # Default for DR
        if cal_result and hasattr(cal_result, "calibrator"):
            return EstimatorClass(  # type: ignore
                sampler, calibrator=cal_result.calibrator, n_folds=n_folds
            )
        else:
            return EstimatorClass(sampler, n_folds=n_folds)  # type: ignore
    else:
        # Other IPS estimators
        return EstimatorClass(sampler)  # type: ignore


# Step 4: Fresh draws handling (using CJE's utilities)
def add_fresh_draws_for_dr(
    estimator: Any, sampler: PrecomputedSampler, data_dir: Path = Path("data copy")
) -> None:
    """Add fresh draws to DR estimator using CJE's utilities.

    Args:
        estimator: DR estimator instance
        sampler: PrecomputedSampler for target policies
        data_dir: Directory containing fresh draw files

    Raises:
        FileNotFoundError: If fresh draws not found
    """
    for policy in sampler.target_policies:
        try:
            # Use CJE's auto-loader instead of custom implementation
            fresh_draws = load_fresh_draws_auto(data_dir, policy, verbose=True)
            estimator.add_fresh_draws(policy, fresh_draws)
            logger.info(f"Added {len(fresh_draws.samples)} fresh draws for {policy}")
        except FileNotFoundError as e:
            # Fail fast - DR needs fresh draws
            raise FileNotFoundError(
                f"Fresh draws required for DR estimator but not found for policy '{policy}'. "
                f"Expected in {data_dir}/. Error: {e}"
            )


# Step 5: Main experiment runner (composed from pieces)
def run_experiment(
    data_path: str = "data copy/cje_dataset.jsonl",
    estimator: str = "calibrated-ips",
    oracle_coverage: float = 0.1,
    sample_fraction: float = 1.0,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single ablation experiment.

    This function composes the smaller functions above to run a complete
    experiment, following CJE's design patterns.

    Args:
        data_path: Path to CJE dataset
        estimator: Estimator name
        oracle_coverage: Fraction of oracle labels to use
        sample_fraction: Fraction of dataset to use
        seed: Random seed
        verbose: Enable verbose logging

    Returns:
        Dictionary with results and metadata
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    start_time = time.time()

    try:
        # Step 1: Load and prepare data
        dataset = prepare_dataset(data_path, sample_fraction, seed)

        # Step 2: Mask oracle labels
        dataset, n_oracle = mask_oracle_labels(dataset, oracle_coverage, seed)

        # Step 3: Calibrate
        calibrated_dataset, cal_result = calibrate_with_oracle(dataset, n_oracle)

        # Step 4: Create sampler
        sampler = PrecomputedSampler(calibrated_dataset)

        # Step 5: Create estimator
        estimator_obj = create_estimator(sampler, estimator, cal_result)

        # Step 6: Add fresh draws for DR
        if estimator in ["dr-cpo", "mrdr", "tmle"]:
            add_fresh_draws_for_dr(estimator_obj, sampler)

        # Step 7: Run estimation
        results = estimator_obj.fit_and_estimate()

        # Step 8: Package results
        runtime = time.time() - start_time

        return {
            "estimator": estimator,
            "oracle_coverage": oracle_coverage,
            "sample_fraction": sample_fraction,
            "n_samples": len(dataset.samples),
            "n_oracle": n_oracle,
            "estimates": {
                policy: float(results.estimates[i])
                for i, policy in enumerate(sampler.target_policies)
            },
            "standard_errors": {
                policy: (
                    float(results.standard_errors[i])
                    if results.standard_errors is not None
                    else None
                )
                for i, policy in enumerate(sampler.target_policies)
            },
            "oracle_truths": AblationConfig.ORACLE_TRUTHS,
            "runtime": runtime,
            "success": True,
            "error": None,
        }

    except Exception as e:
        # Fail fast with clear error
        logger.error(f"Experiment failed: {e}")
        return {
            "estimator": estimator,
            "oracle_coverage": oracle_coverage,
            "sample_fraction": sample_fraction,
            "success": False,
            "error": str(e),
            "runtime": time.time() - start_time,
        }


def run_ablation_grid(
    estimators: List[str],
    oracle_coverages: List[float],
    sample_fractions: List[float],
    data_path: str = "data copy/cje_dataset.jsonl",
    n_seeds: int = 5,
    output_dir: str = "ablation_results",
) -> List[Dict]:
    """Run grid of ablation experiments.

    Args:
        estimators: List of estimator names
        oracle_coverages: List of oracle coverage values
        sample_fractions: List of sample fraction values
        data_path: Path to dataset
        n_seeds: Number of random seeds
        output_dir: Directory for results

    Returns:
        List of experiment results
    """
    results = []
    total = len(estimators) * len(oracle_coverages) * len(sample_fractions) * n_seeds
    completed = 0

    for estimator in estimators:
        for coverage in oracle_coverages:
            for fraction in sample_fractions:
                for seed in range(n_seeds):
                    completed += 1
                    logger.info(
                        f"\n[{completed}/{total}] Running {estimator} "
                        f"(coverage={coverage:.0%}, fraction={fraction:.0%}, seed={seed})"
                    )

                    result = run_experiment(
                        data_path=data_path,
                        estimator=estimator,
                        oracle_coverage=coverage,
                        sample_fraction=fraction,
                        seed=seed,
                    )

                    results.append(result)

                    # Save intermediate results
                    output_path = Path(output_dir)
                    output_path.mkdir(exist_ok=True)

                    with open(output_path / "ablation_results.jsonl", "a") as f:
                        f.write(json.dumps(result) + "\n")

    # Save final results
    with open(output_path / "ablation_results_final.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nCompleted {len(results)} experiments")
    logger.info(f"Results saved to {output_path}")

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run CJE ablation experiments")

    parser.add_argument(
        "--estimators",
        nargs="+",
        default=["calibrated-ips"],
        choices=list(AblationConfig.ESTIMATOR_CLASSES.keys()),
        help="Estimators to evaluate",
    )

    parser.add_argument(
        "--oracle-coverages",
        nargs="+",
        type=float,
        default=[0.1],
        help="Oracle coverage fractions",
    )

    parser.add_argument(
        "--sample-fractions",
        nargs="+",
        type=float,
        default=[1.0],
        help="Dataset sample fractions",
    )

    parser.add_argument(
        "--data", default="data copy/cje_dataset.jsonl", help="Path to dataset"
    )

    parser.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds")

    parser.add_argument(
        "--output-dir", default="ablation_results", help="Output directory"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Run ablation
    results = run_ablation_grid(
        estimators=args.estimators,
        oracle_coverages=args.oracle_coverages,
        sample_fractions=args.sample_fractions,
        data_path=args.data,
        n_seeds=args.n_seeds,
        output_dir=args.output_dir,
    )

    # Print summary
    successful = sum(1 for r in results if r.get("success", False))
    print(f"\nSummary: {successful}/{len(results)} experiments successful")

    if successful < len(results):
        print("\nFailed experiments:")
        for r in results:
            if not r.get("success", False):
                print(
                    f"  - {r['estimator']} (coverage={r['oracle_coverage']:.0%}): {r.get('error', 'Unknown error')}"
                )


if __name__ == "__main__":
    main()
