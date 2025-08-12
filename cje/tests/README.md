# CJE Tests

Simple tests to verify the pipeline works end-to-end.

## Running Tests

```bash
# Run all tests
poetry run pytest cje/tests/

# Or run individual tests:
poetry run pytest cje/tests/test_simple.py      # Quick in-memory test
poetry run pytest cje/tests/test_pipeline.py    # Full pipeline test
poetry run pytest cje/tests/test_edge_cases.py  # Edge cases and error handling
poetry run pytest cje/tests/test_integration.py # Integration with test data

# Include slow tests (API-dependent tests)
poetry run pytest --run-slow cje/tests/
```

## Test Data

Run `poetry run python cje/tests/data/create_test_data.py` to generate test datasets:
- `basic_test_data.jsonl` - Simple data with all fields
- `missing_values_data.jsonl` - Data with some missing log probs
- `extreme_weights_data.jsonl` - Edge cases for weight calculations
- `judge_calibration_data.jsonl` - Data with oracle labels for calibration
- `chat_data.jsonl` - Chat format examples

## What's Tested

### Basic Functionality (test_simple.py, test_pipeline.py)
- Judge score calibration with oracle labels
- Data loading from JSONL and in-memory
- CJE estimation with calibrated weights
- Chat format conversion

### Edge Cases (test_edge_cases.py)
- Missing log probabilities (some samples have None)
- Extreme importance weights
- Partial missing data (drops invalid samples)

### Integration (test_integration.py)
- Full workflow from raw judge scores to final estimates
- Weight diagnostics
- End-to-end pipeline validation