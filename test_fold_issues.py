"""Test script to identify issues with fold management changes."""

import numpy as np
from cje.data import load_dataset_from_jsonl
from cje.calibration.dataset import calibrate_dataset
from cje.data import PrecomputedSampler
from cje.estimators.dr_base import DRCPOEstimator

# Create a simple test dataset
print("1. Creating test dataset...")
samples = []
for i in range(100):
    sample = {
        "prompt_id": f"prompt_{i}",
        "prompt": f"Test prompt {i}",
        "response": f"Response {i}",
        "base_policy_logprob": -10.0,
        "target_policy_logprobs": {"test_policy": -8.0},
        "metadata": {
            "judge_score": np.random.uniform(0, 1),
            "oracle_label": np.random.uniform(0, 1) if i < 20 else None
        }
    }
    samples.append(sample)

# Save to temp file
import json
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    for sample in samples:
        json.dump(sample, f)
        f.write('\n')
    temp_file = f.name

print(f"2. Loading dataset from {temp_file}...")
dataset = load_dataset_from_jsonl(temp_file, target_policies=["test_policy"])

print("3. Calibrating dataset with cross-fitting...")
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
    enable_cross_fit=True,
    n_folds=5
)

print("4. Creating PrecomputedSampler...")
sampler = PrecomputedSampler(calibrated_dataset)

print("5. Creating DR estimator...")
dr_estimator = DRCPOEstimator(
    sampler,
    calibrator=cal_result.calibrator,
    n_folds=5
)

print("6. Checking fold setup...")
print(f"   - Number of samples in dataset: {len(sampler.dataset.samples)}")
print(f"   - Number of valid samples: {sampler.n_valid_samples}")

# Check if any samples have cv_fold
has_cv_fold = any("cv_fold" in s.metadata for s in calibrated_dataset.samples)
print(f"   - Any samples have cv_fold metadata: {has_cv_fold}")

# Try to fit
print("\n7. Fitting DR estimator...")
try:
    dr_estimator.fit()
    print("   ✓ Fit successful!")
    
    # Check the internal state
    if hasattr(dr_estimator, '_promptid_to_fold'):
        print(f"   - Prompt ID to fold mapping: {len(dr_estimator._promptid_to_fold)} entries")
    
    # Test that folds are computed correctly
    test_data = sampler.get_data_for_policy("test_policy")
    if test_data and len(test_data) > 0:
        print(f"   - First sample cv_fold: {test_data[0].get('cv_fold', 'NOT FOUND')}")
    
except Exception as e:
    print(f"   ✗ Fit failed: {e}")
    import traceback
    traceback.print_exc()

# Clean up
import os
os.unlink(temp_file)

print("\nDone!")