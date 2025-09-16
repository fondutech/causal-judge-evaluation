#!/usr/bin/env python3
"""Simple test of CF-bits Wid computation with mock data."""

import numpy as np
from unittest.mock import Mock
from cje.cfbits.identification import compute_identification_width


def create_mock_estimator(judge_scores, oracle_labels, weights=None):
    """Create a mock estimator with specified data."""
    estimator = Mock()
    sampler = Mock()
    dataset = Mock()

    # Create samples with judge scores
    samples = []
    for i, score in enumerate(judge_scores):
        metadata = {"judge_score": score}

        # Add oracle label if provided
        if oracle_labels is not None and i < len(oracle_labels):
            metadata["oracle_label"] = oracle_labels[i]

        sample = Mock()
        sample.metadata = metadata
        samples.append(sample)

    dataset.samples = samples
    sampler.dataset = dataset

    # Set up weights
    if weights is None:
        weights = np.ones(len(judge_scores))
    weights = weights / np.mean(weights)  # Normalize to mean-1
    sampler.compute_importance_weights = Mock(return_value=weights)

    estimator.sampler = sampler
    return estimator


# Test case: Reasonable data that should work
judge_scores = list(np.linspace(0, 1, 100))
oracle_labels = list(np.random.rand(100))  # All samples have labels for simplicity
weights = np.exp(-np.array(judge_scores) * 0.5)  # Some weight variation

estimator = create_mock_estimator(judge_scores, oracle_labels, weights)

print("Testing CF-bits Wid computation with mock data...")
wid, diagnostics = compute_identification_width(
    estimator, "test_policy", alpha=0.05, n_bins=10
)

if wid is not None:
    print(f"\n✓ Wid computation successful!")
    print(f"  Wid value: {wid:.4f}")
    print(f"  Number of bins: {diagnostics.get('n_bins', 'N/A')}")
    print(f"  Oracle samples: {diagnostics.get('n_oracle', 'N/A')}")
    print(f"  Mass on unlabeled bins: {diagnostics.get('p_mass_unlabeled', 0):.3f}")
    print(f"  Violations found: {diagnostics.get('violations_found', 0)}")
    print(f"  ψ_min: {diagnostics.get('psi_min', 'N/A'):.4f}")
    print(f"  ψ_max: {diagnostics.get('psi_max', 'N/A'):.4f}")
else:
    print(f"\n✗ Wid computation failed")
    print(f"  Reason: {diagnostics.get('reason', 'Unknown')}")

# Test edge case: Sparse oracle labels
print("\n" + "=" * 50)
print("Testing with sparse oracle labels...")

sparse_oracle = list(np.random.rand(10)) + [None] * 90
estimator_sparse = create_mock_estimator(judge_scores, sparse_oracle)

wid_sparse, diag_sparse = compute_identification_width(
    estimator_sparse, "test_policy", alpha=0.05, n_bins=10
)

if wid_sparse is not None:
    print(f"✓ Wid with sparse labels: {wid_sparse:.4f}")
    print(f"  Mass on unlabeled bins: {diag_sparse.get('p_mass_unlabeled', 0):.3f}")
    print(f"  Should be higher uncertainty than dense case: {wid_sparse > wid}")
else:
    print(f"✗ Failed: {diag_sparse.get('reason', 'Unknown')}")

print("\nTest complete!")
