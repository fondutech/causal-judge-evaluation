#!/usr/bin/env python3
"""Minimal CJE usage example - this is all you need for most cases."""

from cje import analyze_dataset

# That's it! One import, one function call
results = analyze_dataset("data/example.jsonl")

# View results
print(f"Policy estimate: {results.estimates[0]:.3f}")
print(f"Standard error: {results.standard_errors[0]:.3f}")
print(
    f"95% CI: [{results.estimates[0] - 1.96*results.standard_errors[0]:.3f}, "
    f"{results.estimates[0] + 1.96*results.standard_errors[0]:.3f}]"
)

# Check diagnostics (optional)
if results.diagnostics:
    print(f"\nEffective sample size: {results.diagnostics.weight_ess:.1%}")
    if results.diagnostics.weight_ess < 0.1:
        print("⚠️ Warning: Low ESS - results may be unstable")
