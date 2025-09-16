#!/usr/bin/env python3
"""Test script to verify CF-bits integration in ablations."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from ablations.core.base import BaseAblation
from ablations.core.schemas import ExperimentSpec
from config import DATA_PATH, CFBITS_CONFIG


class CFBitsTestAblation(BaseAblation):
    """Minimal ablation to test CF-bits integration."""

    def __init__(self):
        super().__init__(name="cfbits_test")

    def run_test(self):
        """Run a single experiment with CF-bits enabled."""
        spec = ExperimentSpec(
            ablation="cfbits_test",
            dataset_path=str(DATA_PATH),
            estimator="calibrated-ips",
            sample_size=500,
            oracle_coverage=0.1,
            n_seeds=1,
            seed_base=42,
            extra={
                "use_weight_calibration": True,
                "use_iic": False,
                "reward_calibration_mode": "monotone",
                "compute_cfbits": True,
                "cfbits_config": CFBITS_CONFIG,
            },
        )

        print(f"Running test with CF-bits enabled...")
        print(f"  Estimator: {spec.estimator}")
        print(f"  Sample size: {spec.sample_size}")
        print(f"  Oracle coverage: {spec.oracle_coverage}")
        print(f"  CF-bits config: {CFBITS_CONFIG}")

        # Run the experiment
        result = self.run_single(spec, seed=42)

        # Check CF-bits results - stored in cfbits_data, cfbits_summary, cfbits_gates
        if "cfbits_data" in result or "cfbits_summary" in result:
            print("\n✓ CF-bits integration successful!")
            print("\nCF-bits structure in results:")
            print(f"  - cfbits_data: {type(result.get('cfbits_data', None))}")
            print(f"  - cfbits_summary: {type(result.get('cfbits_summary', None))}")
            print(f"  - cfbits_gates: {type(result.get('cfbits_gates', None))}")

            # Check summary
            if result.get("cfbits_summary"):
                print("\nCF-bits summary:")
                for key, value in result["cfbits_summary"].items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")

            # Check gates
            if result.get("cfbits_gates"):
                print("\nCF-bits gates:")
                for key, value in result["cfbits_gates"].items():
                    print(f"  {key}: {value}")

            # Check per-policy data
            if result.get("cfbits_data") and isinstance(result["cfbits_data"], dict):
                print(f"\nCF-bits data for {len(result['cfbits_data'])} policies")
                for policy, data in result["cfbits_data"].items():
                    print(f"  - {policy}: {type(data)}")
        else:
            print("\n✗ CF-bits not found in results")
            print(f"Result keys: {list(result.keys())}")
            if "error" in result:
                print(f"Error: {result['error']}")

        return result


if __name__ == "__main__":
    ablation = CFBitsTestAblation()
    try:
        result = ablation.run_test()
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
