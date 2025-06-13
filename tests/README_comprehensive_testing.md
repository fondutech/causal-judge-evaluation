# CJE Comprehensive Testing Infrastructure

This directory contains a comprehensive testing infrastructure for the Causal Judge Evaluation (CJE) framework, designed to validate all theoretical guarantees and empirical claims from the research paper.

## 🎯 Testing Categories

### 1. Theoretical Guarantees (`test_cje_theoretical_guarantees.py`)
Tests the core mathematical properties claimed in the paper:

- **Unbiasedness**: Validates that CJE estimates are unbiased under assumptions B1-B3
- **Double Robustness**: Tests that estimator remains consistent when either outcome model OR propensity model is correct
- **Single-Rate Efficiency**: Validates that only one nuisance needs n^{-1/4} convergence rate
- **Asymptotic Normality**: Tests that √n(V̂ - V) → N(0, σ²_eff)
- **Calibration Properties**: Validates monotonicity preservation and centering (E[w] = 1)

### 2. Empirical Validation (`test_cje_empirical_validation.py`)
Tests the empirical claims and benchmarks from the paper:

- **Arena-Hard Reproduction**: Simulates and validates the Arena-Hard benchmark results
- **Confidence Interval Shrinkage**: Tests CI shrinkage claims vs IPS baseline
- **Compute Efficiency**: Validates 6x speedup claims vs decode+judge
- **Variance Reduction**: Tests variance reduction compared to IPS/DR baselines

### 3. Property-Based Testing (`test_cje_property_based.py`)
Uses Hypothesis for property-based testing of mathematical invariants:

- **Estimator Invariants**: Properties that should hold for any valid estimator output
- **Calibration Properties**: Monotonicity and centering properties under all conditions
- **Numerical Stability**: Behavior with extreme weights, near-zero probabilities
- **Boundary Conditions**: Edge cases like minimal data, identical policies

### 4. Integration Testing (`test_cje_integration_comprehensive.py`)
Tests the complete end-to-end CJE pipeline:

- **Algorithm 1 Implementation**: Full cross-fitted CJE estimator workflow
- **Paper Workflow**: Complete Log → Calibrate → Estimate pipeline
- **Deployment Checklist**: Validates production deployment requirements
- **Diagnostics**: ESS calculation, weight distribution monitoring

## 🚀 Running Tests

### Quick Development Tests
```bash
# Run fast tests only (small sample sizes)
pytest tests/test_cje_* -m "not slow" -v

# Run specific category
pytest tests/test_cje_theoretical_guarantees.py -v
```

### Comprehensive Validation
```bash
# Run all tests including slow ones
pytest tests/test_cje_* --run-slow -v

# Run only theoretical guarantee tests
pytest tests/test_cje_* --theoretical-only -v

# Run only empirical validation tests  
pytest tests/test_cje_* --empirical-only -v

# Run tests that directly validate paper claims
pytest tests/test_cje_* --paper-validation -v
```

### Property-Based Testing
```bash
# Run property-based tests with more examples
pytest tests/test_cje_property_based.py --hypothesis-max-examples=50 -v
```

## 📊 Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.theoretical` - Tests theoretical guarantees
- `@pytest.mark.empirical` - Tests empirical claims
- `@pytest.mark.property` - Property-based tests
- `@pytest.mark.integration` - End-to-end tests
- `@pytest.mark.slow` - Tests that take significant time
- `@pytest.mark.paper_validation` - Direct paper claim validation
- `@pytest.mark.robustness` - Edge cases and robustness

## 🔬 What Each Test Validates

### Theoretical Guarantees
| Test | Paper Section | Validates |
|------|---------------|-----------|
| `test_unbiasedness_simple_bandit` | Section 5.2 | E[V̂] = V under assumptions |
| `test_isotonic_weight_calibration_centering` | Section 4.2 | E[w_calibrated] = 1 |
| `test_confidence_interval_coverage` | Section 5.3 | 95% CI has 95% coverage |
| `test_heavy_tailed_weights` | Section 6.4 | Robustness to extreme weights |

### Empirical Claims
| Test | Paper Section | Validates |
|------|---------------|-----------|
| `test_arena_hard_pipeline_integration` | Section 7 | Full Arena-Hard workflow |
| `test_compute_speedup_simulation` | Table 2 | 6x speedup vs decode+judge |
| `test_variance_reduction_vs_ips` | Section 7.3 | CI shrinkage vs IPS |

### Implementation Correctness
| Test | Algorithm/Section | Validates |
|------|------------------|-----------|
| `test_algorithm_1_implementation` | Algorithm 1 | Cross-fitted CJE estimator |
| `test_deployment_checklist_validation` | Section 6.5 | Production deployment |
| `test_weight_distribution_diagnostics` | Section 7.4 | ESS and clipping diagnostics |

## 🏗️ Test Infrastructure

### Fixtures (`conftest.py`)
- `mock_arena_data` - Synthetic Arena-Hard-like data
- `small_test_config` / `full_test_config` - Test size configurations
- `random_seed` - Reproducible randomness
- `performance_tracker` - Performance monitoring

### Custom Test Data Generators
- `create_simple_bandit_scenario()` - Known ground truth scenarios
- `create_production_like_logs()` - Realistic production data simulation
- `valid_log_entry()` / `valid_dataset()` - Hypothesis strategies

## 📈 Expected Test Results

### Theoretical Tests
- **Unbiasedness**: Estimates should be within 5-10% of true values
- **Coverage**: 95% confidence intervals should have 90-100% coverage
- **Monotonicity**: All calibration procedures preserve monotonicity
- **Centering**: Calibrated weights have mean 1.0 ± 0.1

### Empirical Tests  
- **Speedup**: CJE should be 2-6x faster than decode+judge
- **Variance**: DR should reduce variance vs IPS (allow some noise)
- **Integration**: End-to-end pipeline should complete successfully

### Property Tests
- **Invariants**: All estimator outputs should be finite and well-formed
- **Stability**: Results should be stable across parameter variations
- **Robustness**: Should handle edge cases gracefully

## 🐛 Troubleshooting

### Common Issues

**Property tests failing due to extreme generated data:**
```bash
# Reduce hypothesis search space
pytest tests/test_cje_property_based.py --hypothesis-max-examples=20
```

**Slow tests timing out:**
```bash
# Run with smaller sample sizes
pytest tests/test_cje_* -m "not slow" --tb=short
```

**Theoretical tests failing due to high variance:**
- Check that sample sizes are sufficient (n ≥ 100)
- Verify random seeds are set consistently
- Consider increasing tolerance for finite-sample effects

**Integration tests failing:**
- Ensure all dependencies are installed
- Check that mock components are properly configured
- Verify calibration succeeds (requires oracle labels)

### Test Configuration

Modify test parameters in `conftest.py`:
```python
# For faster development
small_test_config = {
    "n_samples": 50,    # Reduce for speed
    "n_trials": 5,      # Reduce for speed  
    "k_folds": 3,       # Minimum for cross-validation
}
```

## 📚 References

This testing infrastructure validates the claims from:

**"Causal Judge Evaluation (CJE): Unbiased, Calibrated & Cost-Efficient Off-Policy Metrics for LLM Systems"**

Key validated claims:
- Theorem 5.2: Single-rate efficiency
- Algorithm 1: Cross-fitted CJE estimator  
- Table 2: Compute cost comparison
- Section 7: Arena-Hard empirical results
- Section 6.5: Deployment checklist

## ✅ Release Readiness Checklist

Before public release, ensure:

- [ ] All theoretical guarantee tests pass
- [ ] Arena-Hard integration test passes  
- [ ] Property-based tests pass with default settings
- [ ] Performance claims validated (within reasonable bounds)
- [ ] Edge case robustness tests pass
- [ ] Documentation matches implementation

Run the full suite:
```bash
pytest tests/test_cje_* --run-slow --paper-validation -v
```

Look for the final summary:
```
🎉 All CJE tests passed! Repository ready for release.
``` 