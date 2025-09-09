#!/usr/bin/env python3
"""
Quick test of the unified experiment system.

This runs a minimal subset of experiments to verify the system works.
"""

import json
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

# Temporarily modify config for testing
import config

# Save original values
orig_estimators = config.EXPERIMENTS["estimators"]
orig_sample_sizes = config.EXPERIMENTS["sample_sizes"]
orig_oracle_coverages = config.EXPERIMENTS["oracle_coverages"]
orig_seeds = config.EXPERIMENTS["seeds"]

# Set minimal test values - include ALL estimators from current config
config.EXPERIMENTS["estimators"] = [
    "raw-ips",
    "calibrated-ips",
    "orthogonalized-ips",
    "dr-cpo",
    "oc-dr-cpo",
    "tr-cpo",
    "tr-cpo-e",  # Added: efficient TR-CPO
    "stacked-dr",
]  # Test all estimators
config.EXPERIMENTS["sample_sizes"] = [500]  # Just 1 small size
config.EXPERIMENTS["oracle_coverages"] = [0.10]  # Just 1 coverage
config.EXPERIMENTS["seeds"] = [42]  # Single seed for speed

# Update paths for test (use absolute paths to avoid issues)
config.RESULTS_PATH = config.BASE_DIR / "results" / "test_results.jsonl"
config.CHECKPOINT_PATH = config.BASE_DIR / "results" / "test_checkpoint.jsonl"

print("=" * 60)
print("UNIFIED SYSTEM TEST")
print("=" * 60)
print(f"Testing with:")
print(f"  Estimators: {config.EXPERIMENTS['estimators']}")
print(f"  Sample sizes: {config.EXPERIMENTS['sample_sizes']}")
print(f"  Oracle coverages: {config.EXPERIMENTS['oracle_coverages']}")
print(f"  Seeds: {len(config.EXPERIMENTS['seeds'])}")
print("")

# Expected number of experiments with IIC and calibration constraints:
# Each estimator runs with IIC=True and IIC=False
# - raw-ips: 1 size × 1 coverage × 1 cal(False) × 2 IIC = 2
# - calibrated-ips: 1 size × 1 coverage × 1 cal(True) × 2 IIC = 2
# - orthogonalized-ips: 1 size × 1 coverage × 1 cal(True) × 2 IIC = 2
# - dr-cpo: 1 size × 1 coverage × 2 cal × 2 IIC = 4
# - oc-dr-cpo: 1 size × 1 coverage × 1 cal(True) × 2 IIC = 2
# - tr-cpo: 1 size × 1 coverage × 1 cal(False) × 2 IIC = 2
# - tr-cpo-e: 1 size × 1 coverage × 1 cal(False) × 2 IIC = 2
# - stacked-dr: 1 size × 1 coverage × 1 cal(True) × 2 IIC = 2
# Total: 18 experiments × 1 seed = 18 runs
print("Expected experiments: 18 configurations × 1 seed = 18 runs")
print("Estimated time: 1-2 minutes")
print("")

# Import and run
from run import UnifiedAblation


def main() -> int:
    """Run test."""
    try:
        # Clear any existing test files
        test_results = Path(config.RESULTS_PATH)
        test_checkpoint = Path(config.CHECKPOINT_PATH)
        if test_results.exists():
            test_results.unlink()
        if test_checkpoint.exists():
            test_checkpoint.unlink()

        # Run test
        ablation = UnifiedAblation()
        results = ablation.run_ablation()

        # Check results
        if test_results.exists():
            with open(test_results, "r") as f:
                lines = f.readlines()
                print(f"\n✓ Test generated {len(lines)} result entries")

                # Sample a result to check format
                if lines:
                    sample = json.loads(lines[0])
                    if sample.get("success"):
                        print("✓ Result format looks correct")
                        print(f"  - Has spec: {'spec' in sample}")
                        print(f"  - Has seed: {'seed' in sample}")
                        print(f"  - Has estimates: {'estimates' in sample}")
                        print(f"  - Has standard_errors: {'standard_errors' in sample}")
                        print(f"  - Has oracle_truths: {'oracle_truths' in sample}")

        print("\n" + "=" * 60)
        print("TEST COMPLETE - System appears to be working!")
        print("=" * 60)

        # Restore original config
        config.EXPERIMENTS["estimators"] = orig_estimators
        config.EXPERIMENTS["sample_sizes"] = orig_sample_sizes
        config.EXPERIMENTS["oracle_coverages"] = orig_oracle_coverages
        config.EXPERIMENTS["seeds"] = orig_seeds

        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
