#!/usr/bin/env python3
"""Test CF-bits integration with a minimal experiment."""

import json
import logging
from pathlib import Path

# Add parent directories to path
import sys

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import with proper paths
sys.path.insert(0, str(Path(__file__).parent))
from core.base import BaseAblation
from core.schemas import ExperimentSpec
from config import DATA_PATH, DR_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class TestCFBitsAblation(BaseAblation):
    """Minimal ablation to test CF-bits integration."""

    def __init__(self):
        super().__init__(name="test_cfbits")


def main():
    """Run a single experiment with CF-bits enabled."""

    # Test different estimators
    import sys

    estimator = sys.argv[1] if len(sys.argv) > 1 else "calibrated-ips"

    # Create minimal experiment spec
    spec = ExperimentSpec(
        ablation="test_cfbits",
        dataset_path=str(DATA_PATH),
        estimator=estimator,  # Configurable estimator
        sample_size=500,  # Small sample for speed
        oracle_coverage=0.10,  # 10% oracle coverage
        n_seeds=1,
        seed_base=42,
        extra={
            "use_weight_calibration": estimator
            != "raw-ips",  # Raw IPS doesn't use calibration
            "use_iic": False,
            "reward_calibration_mode": "monotone",
            "compute_cfbits": True,  # Enable CF-bits
            "compute_cfbits_full": False,  # Don't store full reports
        },
    )

    # Run the experiment
    ablation = TestCFBitsAblation()

    logger.info("Running test experiment with CF-bits enabled...")
    logger.info(f"Estimator: {spec.estimator}")
    logger.info(f"Sample size: {spec.sample_size}")
    logger.info(f"CF-bits enabled: {spec.extra['compute_cfbits']}")

    try:
        result = ablation.run_single(spec, seed=42)

        # Check if CF-bits were computed
        if result.get("success"):
            logger.info("✓ Experiment completed successfully")

            # Check CF-bits results
            if "cfbits_summary" in result and result["cfbits_summary"]:
                logger.info("✓ CF-bits computed successfully")

                # Display CF-bits metrics for each policy
                for policy, summary in result["cfbits_summary"].items():
                    if summary:
                        logger.info(f"\nCF-bits for {policy}:")
                        logger.info(f"  - Total bits: {summary.get('bits_tot')}")
                        logger.info(f"  - IFR (OUA): {summary.get('ifr_oua')}")
                        logger.info(f"  - aESS (OUA): {summary.get('aess_oua')}")
                        logger.info(f"  - A-ESSF LCB: {summary.get('aessf_lcb')}")
                        logger.info(f"  - Gate state: {summary.get('gate_state')}")
                        logger.info(f"  - Wvar: {summary.get('wvar')}")
                        logger.info(f"  - Wid: {summary.get('wid')}")
            else:
                logger.warning("⚠ CF-bits summary not found in results")

            # Check gates
            if "cfbits_gates" in result and result["cfbits_gates"]:
                logger.info("\nCF-bits gates:")
                for policy, gates in result["cfbits_gates"].items():
                    if gates:
                        logger.info(
                            f"  {policy}: {gates.get('state')} - {gates.get('reasons', [])}"
                        )

            # Save result for inspection
            output_file = Path("test_cfbits_result.json")
            with open(output_file, "w") as f:
                # Remove numpy arrays for JSON serialization
                result_json = json.dumps(result, default=str, indent=2)
                f.write(result_json)
            logger.info(f"\nFull result saved to: {output_file}")

        else:
            logger.error(f"✗ Experiment failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"✗ Error running experiment: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
