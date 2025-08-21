#!/usr/bin/env python3
"""
Test if var_cap actually works in SIMCalibrator.
"""

import numpy as np
from cje.calibration.simcal import SIMCalibrator, SimcalConfig

# Create test data
np.random.seed(42)
n = 1000

# Create weights with high variance
raw_weights = np.random.lognormal(0, 2, n)
raw_weights = raw_weights / raw_weights.mean()  # Normalize to mean 1

# Create judge scores
judge_scores = np.random.uniform(0, 1, n)

# Create rewards for influence functions
rewards = np.random.uniform(0, 1, n)

print("Testing SIMCal var_cap enforcement")
print("=" * 50)
print(f"Raw weights variance: {np.var(raw_weights):.3f}")
print(f"Raw weights std: {np.std(raw_weights):.3f}")

# Test with different var_cap values
for var_cap in [None, 5.0, 2.0, 1.0, 0.5]:
    print(f"\nTesting var_cap={var_cap}")
    print("-" * 30)

    config = SimcalConfig(
        var_cap=var_cap,
        ess_floor=None,  # Disable ESS constraint to isolate var_cap
        include_baseline=True,
        baseline_shrink=0.0,
    )

    calibrator = SIMCalibrator(config)

    # Apply calibration
    calibrated_weights, info = calibrator.transform(
        raw_weights, judge_scores, rewards=rewards
    )

    final_var = np.var(calibrated_weights)
    final_std = np.std(calibrated_weights)

    print(f"  Final variance: {final_var:.3f}")
    print(f"  Final std: {final_std:.3f}")
    print(f"  Gamma (blend param): {info['gamma']:.3f}")
    print(f"  Variance reduced by: {(1 - final_var/np.var(raw_weights))*100:.1f}%")

    # Check if constraint is satisfied
    if var_cap is not None:
        if final_var <= var_cap * 1.01:  # Allow 1% tolerance for numerical error
            print(f"  ✓ Constraint satisfied: {final_var:.3f} <= {var_cap}")
        else:
            print(f"  ✗ CONSTRAINT VIOLATED: {final_var:.3f} > {var_cap}")

print("\n" + "=" * 50)
print("CONCLUSION:")
if var_cap is not None and final_var > var_cap * 1.01:
    print("✗ var_cap is NOT working properly")
else:
    print("✓ var_cap appears to be working")
