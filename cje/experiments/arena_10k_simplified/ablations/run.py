#!/usr/bin/env python3
"""
Unified experiment runner using existing ablation infrastructure.

This leverages the BaseAblation class to run a complete parameter sweep
for the paper, testing all key design decisions systematically.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
from itertools import product
import numpy as np

# Add parent directories to path
import sys

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import configuration
from config import EXPERIMENTS, DR_CONFIG, DATA_PATH, RESULTS_PATH, CHECKPOINT_PATH

# Import existing ablation infrastructure
from ablations.core.base import BaseAblation
from ablations.core.schemas import ExperimentSpec, aggregate_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class UnifiedAblation(BaseAblation):
    """Unified ablation that tests all parameters systematically."""

    def __init__(self) -> None:
        super().__init__(name="unified")
        self.checkpoint_file = (
            CHECKPOINT_PATH
            if isinstance(CHECKPOINT_PATH, Path)
            else Path(CHECKPOINT_PATH)
        )
        self.results_file = (
            RESULTS_PATH if isinstance(RESULTS_PATH, Path) else Path(RESULTS_PATH)
        )
        self.completed_experiments = self._load_checkpoint()

    @staticmethod
    def _convert_numpy(obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: UnifiedAblation._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [UnifiedAblation._convert_numpy(v) for v in obj]
        elif hasattr(obj, "item"):
            return obj.item()
        return obj

    def _load_checkpoint(self) -> set:
        """Load completed experiment IDs from checkpoint."""
        completed = set()
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, "r") as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        if result.get("success"):
                            # Create ID from parameters
                            exp_id = self._generate_exp_id(result.get("spec", {}))
                            completed.add(exp_id)
                    except json.JSONDecodeError:
                        continue
            logger.info(
                f"Loaded {len(completed)} completed experiments from checkpoint"
            )
        return completed

    def _generate_exp_id(self, spec: Dict[str, Any]) -> str:
        """Generate unique experiment ID from specification."""
        # Create a deterministic string from key parameters
        key_params = [
            spec.get("estimator", ""),
            str(spec.get("sample_size", "")),
            str(spec.get("oracle_coverage", "")),
            str(spec.get("extra", {}).get("use_calibration", False)),
            str(spec.get("extra", {}).get("use_iic", False)),
            str(spec.get("extra", {}).get("reward_calibration_mode", "auto")),
            str(spec.get("extra", {}).get("weight_mode", "hajek")),
            str(spec.get("seed_base", 42)),
        ]
        return "_".join(key_params)

    def _save_checkpoint(self, result: Dict[str, Any]) -> None:
        """Append result to checkpoint file."""
        # Convert numpy types before saving
        result = self._convert_numpy(result)

        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Also save to main results file
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

    def create_estimator(
        self, spec: ExperimentSpec, sampler: Any, cal_result: Any
    ) -> Any:
        """Create estimator with unified parameter handling.

        This extends the base class method to handle our new parameters.
        """
        estimator_name = spec.estimator
        use_calibration = spec.extra.get("use_calibration", False)
        use_iic = spec.extra.get("use_iic", False)
        weight_mode = spec.extra.get("weight_mode", "hajek")

        # Handle IPS with calibration and weight_mode
        if estimator_name == "ips":
            from cje.estimators.calibrated_ips import CalibratedIPS

            return CalibratedIPS(
                sampler,
                calibrate=use_calibration,  # Now controlled by ablation parameter
                use_iic=use_iic,
                weight_mode=weight_mode,
                calibrator=(
                    cal_result.calibrator if use_calibration and cal_result else None
                ),
            )

        # Handle DR methods with our parameters
        if estimator_name in ["dr-cpo", "tmle", "mrdr"]:
            from cje.estimators.dr_base import DRCPOEstimator
            from cje.estimators.tmle import TMLEEstimator
            from cje.estimators.mrdr import MRDREstimator

            estimator_class = {
                "dr-cpo": DRCPOEstimator,
                "tmle": TMLEEstimator,
                "mrdr": MRDREstimator,
            }[estimator_name]

            kwargs = {
                "sampler": sampler,
                "n_folds": DR_CONFIG["n_folds"],
                "use_iic": use_iic,
                "weight_mode": weight_mode,
            }

            # Add calibrator if calibration is enabled
            if use_calibration and cal_result:
                kwargs["calibrator"] = cal_result.calibrator

            return estimator_class(**kwargs)

        elif estimator_name == "stacked-dr":
            from cje.estimators.stacking import StackedDREstimator

            return StackedDREstimator(
                sampler=sampler,
                estimators=["dr-cpo", "tmle", "mrdr"],
                V_folds=5,
                calibrator=(
                    cal_result.calibrator if use_calibration and cal_result else None
                ),
                weight_mode=weight_mode,
                parallel=False,
                use_iic=use_iic,
            )

        else:
            # Fall back to base implementation
            return super().create_estimator(spec, sampler, cal_result)

    def run_ablation(self) -> List[Dict[str, Any]]:
        """Run the complete unified ablation."""

        all_results = []
        total_experiments = 0
        skipped = 0
        completed = 0
        failed = 0

        # Generate all parameter combinations
        for estimator in EXPERIMENTS["estimators"]:
            for sample_size in EXPERIMENTS["sample_sizes"]:
                for oracle_coverage in EXPERIMENTS["oracle_coverages"]:

                    # All estimators now test both calibration modes
                    for use_calibration in EXPERIMENTS["use_calibration"]:
                        # IIC works for all estimators (IPS and DR)
                        # It's a general variance reduction technique for any asymptotically linear estimator
                        iic_values = EXPERIMENTS["use_iic"]

                        for use_iic in iic_values:
                            # Add reward calibration mode loop
                            for reward_calibration_mode in EXPERIMENTS[
                                "reward_calibration_mode"
                            ]:
                                # Add weight mode loop (hajek vs raw)
                                for weight_mode in EXPERIMENTS["weight_mode"]:
                                    # Create specification
                                    spec = ExperimentSpec(
                                        ablation="unified",
                                        dataset_path=str(DATA_PATH),
                                        estimator=estimator,
                                        sample_size=sample_size,
                                        oracle_coverage=oracle_coverage,
                                        n_seeds=1,  # Single seed
                                        seed_base=EXPERIMENTS["seed"],
                                        extra={
                                            "use_calibration": use_calibration,
                                            "use_iic": use_iic,
                                            "reward_calibration_mode": reward_calibration_mode,
                                            "weight_mode": weight_mode,
                                        },
                                    )

                                    # Check if already completed
                                    exp_id = self._generate_exp_id(spec.to_dict())
                                    if exp_id in self.completed_experiments:
                                        skipped += 1
                                        logger.debug(f"Skipping completed: {exp_id}")
                                        continue

                                    total_experiments += 1

                                    # Run experiment with single seed
                                    logger.info(
                                        f"\n[{completed + failed + 1}/{total_experiments}] "
                                        f"Running: {estimator} n={sample_size} "
                                        f"oracle={oracle_coverage:.0%} "
                                        f"cal={use_calibration} iic={use_iic} "
                                        f"mode={reward_calibration_mode} weight={weight_mode}"
                                    )

                                    try:
                                        # Run with single seed
                                        result = self.run_single(
                                            spec, EXPERIMENTS["seed"]
                                        )

                                        # Save result
                                        self._save_checkpoint(result)
                                        all_results.append(result)
                                        completed += 1

                                        # Log summary
                                        if result.get("success"):
                                            if "rmse_vs_oracle" in result:
                                                logger.info(
                                                    f"  ✓ RMSE: {result['rmse_vs_oracle']:.4f}"
                                                )
                                            else:
                                                logger.info("  ✓ Completed")

                                    except Exception as e:
                                        logger.error(f"  ✗ Failed: {e}")
                                        failed += 1

                                        # Save failed result
                                        failed_result = {
                                            "spec": spec.to_dict(),
                                            "success": False,
                                            "error": str(e),
                                        }
                                        self._save_checkpoint(failed_result)

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("UNIFIED ABLATION COMPLETE")
        logger.info(f"Completed: {completed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped (cached): {skipped}")
        logger.info(f"Results saved to: {self.results_file}")
        logger.info("=" * 60)

        return all_results


def main() -> None:
    """Run unified ablation experiments."""
    ablation = UnifiedAblation()
    results = ablation.run_ablation()

    # Save final aggregated results
    if results:
        summary_file = Path(RESULTS_PATH).parent / "unified_summary.json"
        with open(summary_file, "w") as f:
            # Convert numpy types before saving
            summary_data = UnifiedAblation._convert_numpy(
                {
                    "timestamp": time.time(),
                    "n_experiments": len(results),
                    "results": results,
                }
            )
            json.dump(summary_data, f, indent=2)
        logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
