# CJE Test Suite

## Overview

Comprehensive test suite for the Causal Judge Evaluation framework, ensuring correctness of causal inference methods, calibration algorithms, and diagnostic tools. The suite combines unit tests, integration tests, and end-to-end validation using real Arena 10K data.

## When to Use

### Use **Unit Tests** when:
- Developing new calibration methods
- Adding estimator functionality
- Modifying core algorithms
- Need fast feedback during development

### Use **Integration Tests** when:
- Testing complete workflows
- Validating estimator interactions
- Checking data flow through pipelines
- Verifying cross-component behavior

### Use **Arena Sample Tests** when:
- Validating against real data
- Testing production scenarios
- Benchmarking performance
- Ensuring backward compatibility

## File Structure

```
tests/
├── conftest.py                    # Shared fixtures and utilities
├── run_all_tests.py              # Test runner script
│
├── Core Tests                    
│   ├── test_simple.py            # Minimal judge calibration
│   ├── test_pipeline.py          # End-to-end workflows
│   ├── test_integration.py       # Full system integration
│   └── test_data_models.py       # Data model validation
│
├── API Tests
│   ├── test_analysis.py          # analyze_dataset() with Arena data
│   ├── test_cli.py               # CLI commands and parsing
│   └── test_export.py            # JSON/CSV export formats
│
├── Estimator Tests
│   ├── test_dr_basic.py          # DR estimation fundamentals
│   ├── test_dr_diagnostics.py    # All estimator diagnostics
│   ├── test_stacked_simcal.py    # SIMCal weight calibration
│   └── test_custom_outcome_model.py # Custom outcome models
│
├── Diagnostic Tests
│   ├── test_new_diagnostics.py   # IPSDiagnostics, DRDiagnostics
│   ├── test_stability_diagnostics.py # Tail index, ESS, stability
│   └── test_robust_inference.py  # Inference robustness checks
│
├── Utility Tests
│   ├── test_fresh_draws.py       # Fresh draw loading
│   ├── test_teacher_forcing.py   # Chat templates, log prob
│   ├── test_validation.py        # Dataset validation
│   └── test_edge_cases.py        # Missing values, extremes
│
├── Documentation Tests
│   └── test_documentation_examples.py # README code validation
│
└── data/                          # Test datasets
    ├── arena_sample/              # Real Arena 10K subset
    └── *.jsonl                    # Synthetic test data
```

## Core Concepts

### 1. Test Categories
Tests are organized by functionality and marked with pytest markers:
- **@pytest.mark.unit** - Fast, isolated component tests
- **@pytest.mark.integration** - Multi-component workflow tests
- **@pytest.mark.slow** - Tests requiring API calls or heavy computation

### 2. Arena Sample Data
Real subset from Arena 10K evaluation:
- 100 samples with actual judge scores and oracle labels
- 4 target policies: clone, premium, parallel_universe_prompt, unhelpful
- Fresh draws for each policy enabling DR estimation
- Ground truth for validation

### 3. Fixture Architecture
Shared fixtures in `conftest.py` provide consistent test data:
- **basic_dataset**: Simple 20-sample dataset with all fields
- **dataset_with_oracle**: 50% oracle coverage for calibration testing
- **dataset_for_dr**: Cross-validation folds for DR testing
- **synthetic_fresh_draws**: Mock fresh draws for DR without files

### 4. Assertion Helpers
Standard validation functions ensure consistency:
- **assert_valid_estimation_result**: Validates EstimationResult structure
- **assert_weights_calibrated**: Checks weight calibration properties
- **assert_dataset_valid**: Comprehensive dataset validation
- **assert_diagnostics_complete**: Verifies diagnostic completeness

## Common Interface

### Running Tests

```bash
# Run all tests
poetry run pytest cje/tests/

# Run by category
poetry run pytest -m unit          # Fast unit tests only
poetry run pytest -m integration   # Integration tests
poetry run pytest -m "not slow"    # Skip slow tests

# Run specific modules
poetry run pytest cje/tests/test_analysis.py -v      # High-level API
poetry run pytest cje/tests/test_dr_diagnostics.py   # DR diagnostics
poetry run pytest cje/tests/test_simple.py::test_judge_calibration  # Single test

# With coverage
poetry run pytest --cov=cje --cov-report=html cje/tests/
```

### Writing New Tests

```python
import pytest
from cje import analyze_dataset

class TestNewFeature:
    """Test new feature functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.test_data = Path(__file__).parent / "data" / "basic_test_data.jsonl"
    
    @pytest.mark.unit
    def test_feature_basic(self, basic_dataset):
        """Test basic feature behavior."""
        result = your_feature(basic_dataset)
        assert_valid_result(result)
    
    @pytest.mark.integration
    def test_feature_with_real_data(self):
        """Test with Arena sample data."""
        result = analyze_dataset(
            "data/arena_sample/dataset.jsonl",
            your_new_parameter=True
        )
        assert result.metadata["your_feature"] == expected_value
```

## Key Design Decisions

### 1. **Real Data Testing**
Arena sample data provides ground truth validation:
- Catches regressions in production scenarios
- Validates against known-good results
- Tests all estimators with same data

### 2. **Modular Test Organization**
Tests grouped by functionality, not implementation:
- Easy to find relevant tests
- Clear what each file tests
- Parallel test execution friendly

### 3. **Shared Fixtures**
Common data patterns centralized in conftest.py:
- Consistent test data across modules
- Reduced boilerplate
- Easy to add new data patterns

### 4. **Progressive Complexity**
Tests build from simple to complex:
- `test_simple.py` - Minimal functionality
- `test_pipeline.py` - Component integration
- `test_analysis.py` - Full system with real data

## Common Issues

### "FileNotFoundError for test data"
Ensure running from project root:
```bash
cd /path/to/causal-judge-evaluation
poetry run pytest cje/tests/
```

### "Slow test execution"
Skip slow tests during development:
```bash
poetry run pytest -m "not slow" cje/tests/
```

### "Import errors"
Install package in development mode:
```bash
poetry install
# or
pip install -e .
```

## Performance

- **Unit tests**: < 1 second each
- **Integration tests**: 1-5 seconds each
- **Full suite**: ~30 seconds without slow tests
- **With slow tests**: ~2 minutes (includes API calls)

Test execution tips:
- Use `-x` to stop on first failure
- Use `-k pattern` to run tests matching pattern
- Use `--lf` to run last failed tests
- Use `-n auto` for parallel execution (requires pytest-xdist)

## Summary

The CJE test suite provides comprehensive validation through 145+ tests covering all estimators, calibration methods, and diagnostic tools. It combines fast unit tests for development, integration tests for workflow validation, and real Arena data tests for production confidence, ensuring the framework produces correct, unbiased causal estimates.