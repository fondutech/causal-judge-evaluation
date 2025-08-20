#!/usr/bin/env python3
"""Example of using StackedDR - the recommended DR estimator."""

from cje import analyze_dataset

# StackedDR optimally combines DR-CPO, TMLE, and MRDR
# It requires fresh draws (teacher-forced responses from target policies)
results = analyze_dataset(
    "data/example.jsonl",
    estimator="stacked-dr",
    fresh_draws_dir="data/responses/",  # Directory with fresh draws
    estimator_config={
        "estimators": ["dr-cpo", "tmle", "mrdr"],  # Components to stack
        "use_outer_split": True,  # Use V-fold for honest inference
        "parallel": True,  # Run components in parallel
    },
    verbose=True,  # See what's happening
)

# View results
print(f"\nStacked estimate: {results.estimates[0]:.3f}")
print(f"Standard error: {results.standard_errors[0]:.3f}")

# Check which estimators contributed
if "stacking_weights" in results.metadata:
    print("\nStacking weights (how much each estimator contributed):")
    for policy, weights in results.metadata["stacking_weights"].items():
        print(f"  {policy}:")
        for est_name, weight in weights.items():
            print(f"    {est_name}: {weight:.1%}")

# The stacked estimator typically achieves:
# - Better variance reduction than any single DR method
# - Robustness to individual estimator failures
# - Automatic weight selection based on influence function variance
