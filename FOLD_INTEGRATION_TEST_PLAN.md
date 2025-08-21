# Fold System Integration Test Plan

## Overview
While unit tests verify individual components, we need integration tests to ensure the unified fold system works correctly across the entire CJE pipeline.

## Test Categories

### 1. End-to-End Pipeline Consistency
**Purpose**: Verify same sample gets same fold throughout entire pipeline

```python
def test_fold_consistency_through_pipeline():
    """Test that a sample maintains same fold from calibration to estimation."""
    # Load dataset
    dataset = load_test_dataset()
    
    # Track a specific sample through pipeline
    target_prompt_id = dataset.samples[0].prompt_id
    expected_fold = get_fold(target_prompt_id)
    
    # Step 1: Calibration
    calibrated_dataset, cal_result = calibrate_dataset(
        dataset, 
        judge_field="judge_score",
        oracle_field="oracle_label",
        enable_cross_fit=True
    )
    
    # Verify calibrator assigns correct fold
    cal_fold_idx = [i for i, s in enumerate(dataset.samples) 
                    if s.prompt_id == target_prompt_id][0]
    assert cal_result.fold_ids[cal_fold_idx] == expected_fold
    
    # Step 2: Sampling
    sampler = PrecomputedSampler(calibrated_dataset)
    policy_folds = sampler.get_folds_for_policy("target_policy")
    
    # Find sample in filtered data
    policy_data = sampler.get_data_for_policy("target_policy")
    sample_idx = [i for i, d in enumerate(policy_data) 
                  if d["prompt_id"] == target_prompt_id][0]
    assert policy_folds[sample_idx] == expected_fold
    
    # Step 3: DR Estimation
    dr_estimator = DRCPOEstimator(sampler)
    dr_estimator.fit()
    
    # Verify DR uses correct fold
    dr_sample_idx = [i for i, s in enumerate(sampler.dataset.samples)
                     if s.prompt_id == target_prompt_id][0]
    assert dr_estimator.fold_assignments[dr_sample_idx] == expected_fold
```

### 2. Filtering Robustness
**Purpose**: Ensure folds remain consistent after filtering

```python
def test_fold_consistency_after_filtering():
    """Test that filtering doesn't change fold assignments."""
    dataset = load_test_dataset()
    
    # Get original folds
    original_folds = {
        s.prompt_id: get_fold(s.prompt_id) 
        for s in dataset.samples
    }
    
    # Apply various filters
    filters = [
        lambda s: s.reward > 0.5,
        lambda s: "specific" in s.prompt,
        lambda s: s.metadata.get("judge_score", 0) > 0.7
    ]
    
    for filter_fn in filters:
        filtered_samples = [s for s in dataset.samples if filter_fn(s)]
        filtered_dataset = Dataset(samples=filtered_samples, 
                                  target_policies=dataset.target_policies)
        
        # Verify each filtered sample keeps its fold
        for sample in filtered_dataset.samples:
            computed_fold = get_fold(sample.prompt_id)
            assert computed_fold == original_folds[sample.prompt_id]
```

### 3. Fresh Draws Inheritance
**Purpose**: Verify fresh draws inherit correct folds

```python
def test_fresh_draws_inherit_folds():
    """Test that fresh draws with same prompt_id get same fold."""
    # Create logged dataset
    logged_dataset = create_test_dataset()
    logged_folds = {
        s.prompt_id: get_fold(s.prompt_id)
        for s in logged_dataset.samples
    }
    
    # Create fresh draws with overlapping prompt_ids
    fresh_samples = []
    for sample in logged_dataset.samples[:10]:
        # Same prompt_id, different response
        fresh_sample = Sample(
            prompt_id=sample.prompt_id,
            prompt=sample.prompt,
            response=f"Fresh response for {sample.prompt_id}",
            reward=np.random.random()
        )
        fresh_samples.append(fresh_sample)
    
    fresh_dataset = FreshDrawDataset(samples=fresh_samples)
    
    # Verify fresh draws get same folds
    for sample in fresh_dataset.samples:
        fresh_fold = get_fold(sample.prompt_id)
        assert fresh_fold == logged_folds[sample.prompt_id]
```

### 4. Cross-Estimator Consistency
**Purpose**: Verify all estimators assign same folds

```python
def test_cross_estimator_fold_consistency():
    """Test that all estimators use identical fold assignments."""
    dataset = load_and_calibrate_test_dataset()
    sampler = PrecomputedSampler(dataset)
    
    # Track fold assignments from each estimator
    estimators = [
        CalibratedIPS(sampler),
        DRCPOEstimator(sampler),
        MRDREstimator(sampler),
        TMLEEstimator(sampler)
    ]
    
    fold_assignments = {}
    for estimator in estimators:
        estimator.fit()
        
        # Collect fold assignments
        est_name = estimator.__class__.__name__
        fold_assignments[est_name] = {}
        
        for i, sample in enumerate(dataset.samples):
            if hasattr(estimator, 'fold_assignments'):
                fold = estimator.fold_assignments[i]
            else:
                fold = get_fold(sample.prompt_id, 
                              estimator.n_folds, 
                              estimator.random_seed)
            fold_assignments[est_name][sample.prompt_id] = fold
    
    # Verify all estimators have identical assignments
    prompt_ids = list(fold_assignments[list(fold_assignments.keys())[0]].keys())
    for pid in prompt_ids:
        folds = [fold_assignments[est][pid] for est in fold_assignments]
        assert len(set(folds)) == 1, f"Inconsistent folds for {pid}: {folds}"
```

### 5. Oracle Balance Preservation
**Purpose**: Ensure oracle samples remain balanced after unified system

```python
def test_oracle_balance_preserved():
    """Test that oracle samples are evenly distributed across folds."""
    dataset = create_dataset_with_partial_oracle(
        n_samples=1000,
        oracle_fraction=0.1  # 100 oracle samples
    )
    
    oracle_mask = np.array([
        s.metadata.get("oracle_label") is not None 
        for s in dataset.samples
    ])
    
    # Use balanced fold assignment
    prompt_ids = [s.prompt_id for s in dataset.samples]
    folds = get_folds_with_oracle_balance(
        prompt_ids, oracle_mask, n_folds=5
    )
    
    # Check oracle distribution
    oracle_folds = folds[oracle_mask]
    for fold_id in range(5):
        count = np.sum(oracle_folds == fold_id)
        # Should be approximately 100/5 = 20 per fold
        assert 18 <= count <= 22, f"Fold {fold_id} has {count} oracle samples"
```

### 6. Determinism and Reproducibility
**Purpose**: Verify fold assignments are deterministic

```python
def test_fold_determinism_across_runs():
    """Test that fold assignments are reproducible across runs."""
    dataset = load_test_dataset()
    
    # Run 1
    folds_run1 = get_folds_for_dataset(dataset, n_folds=5, seed=42)
    
    # Run 2 (same parameters)
    folds_run2 = get_folds_for_dataset(dataset, n_folds=5, seed=42)
    
    # Should be identical
    np.testing.assert_array_equal(folds_run1, folds_run2)
    
    # Run 3 (different seed)
    folds_run3 = get_folds_for_dataset(dataset, n_folds=5, seed=99)
    
    # Should be different
    assert not np.array_equal(folds_run1, folds_run3)
    
    # But still valid (correct range)
    assert np.all((folds_run3 >= 0) & (folds_run3 < 5))
```

### 7. Performance Under Load
**Purpose**: Ensure fold system scales to large datasets

```python
def test_fold_performance_at_scale():
    """Test that fold assignment is fast for large datasets."""
    import time
    
    # Create large dataset
    n_samples = 100_000
    samples = []
    for i in range(n_samples):
        samples.append(Sample(
            prompt_id=f"prompt_{i}",
            prompt=f"Test {i}",
            response=f"Response {i}",
            reward=np.random.random()
        ))
    
    large_dataset = Dataset(samples=samples, target_policies=["policy"])
    
    # Time fold assignment
    start = time.time()
    folds = get_folds_for_dataset(large_dataset)
    elapsed = time.time() - start
    
    assert len(folds) == n_samples
    assert elapsed < 1.0, f"Took {elapsed:.2f}s for {n_samples} samples"
    
    # Verify correctness of sample
    for i in range(min(100, n_samples)):
        expected = get_fold(samples[i].prompt_id)
        assert folds[i] == expected
```

### 8. Edge Cases
**Purpose**: Handle edge cases gracefully

```python
def test_fold_edge_cases():
    """Test edge cases in fold assignment."""
    
    # Empty dataset
    empty_dataset = Dataset(samples=[], target_policies=[])
    empty_folds = get_folds_for_dataset(empty_dataset)
    assert len(empty_folds) == 0
    
    # Single sample
    single_sample = Sample(
        prompt_id="single",
        prompt="Test",
        response="Response",
        reward=1.0
    )
    single_dataset = Dataset(samples=[single_sample], target_policies=["p"])
    single_folds = get_folds_for_dataset(single_dataset)
    assert len(single_folds) == 1
    assert 0 <= single_folds[0] < 5
    
    # More folds than samples
    small_dataset = Dataset(samples=[single_sample] * 3, target_policies=["p"])
    many_folds = get_folds_for_dataset(small_dataset, n_folds=10)
    assert len(many_folds) == 3
    assert all(0 <= f < 10 for f in many_folds)
    
    # Special characters in prompt_id
    special_sample = Sample(
        prompt_id="test/with:special|chars<>",
        prompt="Test",
        response="Response",
        reward=1.0
    )
    special_fold = get_fold(special_sample.prompt_id)
    assert 0 <= special_fold < 5
```

## Test File Structure

```
cje/tests/
├── test_unified_folds.py          # Unit tests (existing)
└── test_fold_integration.py       # Integration tests (new)
    ├── test_pipeline_consistency
    ├── test_filtering_robustness
    ├── test_fresh_draws_inheritance
    ├── test_cross_estimator_consistency
    ├── test_oracle_balance_preservation
    ├── test_determinism_reproducibility
    ├── test_performance_at_scale
    └── test_edge_cases
```

## Implementation Priority

1. **High Priority** (Core functionality)
   - Pipeline consistency test
   - Cross-estimator consistency test
   - Filtering robustness test

2. **Medium Priority** (Important scenarios)
   - Fresh draws inheritance test
   - Oracle balance preservation test
   - Determinism test

3. **Low Priority** (Nice to have)
   - Performance test
   - Edge cases test

## Success Criteria

All integration tests should:
- Pass consistently (no flaky tests)
- Complete in < 10 seconds total
- Cover realistic usage patterns
- Catch the types of bugs we found during implementation

## Next Steps

1. Create `test_fold_integration.py` file
2. Implement high-priority tests first
3. Run with existing CJE test data
4. Add to CI/CD pipeline
5. Document any issues found