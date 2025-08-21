#!/usr/bin/env python3
"""
Test if auto-loading of fresh draws works.
"""

import sys
import random
import numpy as np
import logging
from pathlib import Path

# Enable debug logging to see what's happening
logging.basicConfig(level=logging.DEBUG)

sys.path.append(str(Path(__file__).parent))

from cje import load_dataset_from_jsonl
from cje.calibration import calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators.mrdr import MRDREstimator

print("=" * 60)
print("TEST: Auto-loading fresh draws")
print("=" * 60)

random.seed(42)
np.random.seed(42)

# Load minimal data
print("\n1. Loading data...")
dataset = load_dataset_from_jsonl("data/cje_dataset.jsonl")
dataset.samples = dataset.samples[:100]  # Use subset for speed
print(f"   Loaded {len(dataset.samples)} samples")

# Calibrate (required for DR)
print("\n2. Calibrating dataset...")
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
    enable_cross_fit=True,
    n_folds=5,
)
print(f"   Calibration complete")

# Create sampler
sampler = PrecomputedSampler(calibrated_dataset)
print(f"   Target policies: {sampler.target_policies}")

# Create MRDR estimator (should auto-load fresh draws)
print("\n3. Creating MRDR estimator...")
mrdr = MRDREstimator(
    sampler, calibrator=cal_result.calibrator, n_folds=5, oracle_slice_config=True
)

# Fit the estimator
print("\n4. Fitting estimator...")
mrdr.fit()
print("   ✓ Fit complete")

# Try to estimate (should auto-load fresh draws)
print("\n5. Running estimation (should auto-load fresh draws)...")
try:
    results = mrdr.fit_and_estimate()
    print("   ✓ Estimation successful!")

    # Check if fresh draws were loaded
    if mrdr._fresh_draws:
        print(f"\n   Fresh draws loaded for policies: {list(mrdr._fresh_draws.keys())}")
        for policy, fresh_data in mrdr._fresh_draws.items():
            print(f"   - {policy}: {len(fresh_data.samples)} samples")
    else:
        print("\n   ✗ No fresh draws were loaded")

    # Check if IIC diagnostics are present
    if hasattr(results, "metadata") and "iic_diagnostics" in results.metadata:
        print("\n   ✓ IIC diagnostics found in metadata!")
        for policy, diag in results.metadata["iic_diagnostics"].items():
            if diag.get("applied"):
                print(
                    f"   - {policy}: R²={diag.get('r_squared', 0):.3f}, "
                    f"SE reduction={diag.get('se_reduction', 0):.1%}"
                )
    else:
        print("\n   ✗ IIC diagnostics NOT found in metadata")

except Exception as e:
    print(f"   ✗ Estimation failed: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if mrdr._fresh_draws:
    print("✓ Fresh draws auto-loading WORKS!")
else:
    print("✗ Fresh draws auto-loading FAILED")
    print("  Make sure fresh draw files exist in data/responses/")
