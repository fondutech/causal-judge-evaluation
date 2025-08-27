# CJE Examples

Simple, practical examples of using CJE.

## Quick Start

The simplest way to use CJE:

```python
from cje import analyze_dataset
results = analyze_dataset("your_data.jsonl")
print(f"Estimate: {results.estimates[0]:.3f}")
```

## Example Scripts

- `basic_workflows.py` - Three common CJE workflows (oracle labels, judge calibration, log prob computation)
- `oracle_slice_demo.py` - Demonstrates how confidence intervals widen with less oracle data
- `minimal_example.py` - Absolute minimal usage example
- `stacked_dr_example.py` - Uses the Stacked DR estimator combining DR-CPO, TMLE, and MRDR
- `overlap_metrics_demo.py` - Demonstrates overlap metrics like Hellinger affinity

## Running Examples

```bash
# Run basic workflows
python examples/basic_workflows.py

# Run oracle slice demo  
python examples/oracle_slice_demo.py
```

## Key Lessons

1. **Start simple**: Use `analyze_dataset()` for most cases
2. **Add complexity only when needed**: Fresh draws, custom calibration, etc.
3. **Trust the defaults**: CJE's defaults are tuned for production use