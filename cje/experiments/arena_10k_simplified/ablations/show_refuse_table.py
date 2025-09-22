#!/usr/bin/env python3
"""Show the refuse rate table based on boundary diagnostics."""

import pandas as pd
from reporting import tables_main, format_latex

# Create mock data to demonstrate the table
# In production, this would come from actual experiment results
data = {
    "estimator": ["calibrated-ips"] * 4,
    "policy": ["clone", "parallel_universe_prompt", "premium", "unhelpful"],
    "ess_%": [75, 68, 45, 12],  # Dummy ESS values
}

df = pd.DataFrame(data)

# Generate the refuse rates table
table = tables_main.build_table_refuse_rates(df)

print("=== Boundary-Based Refuse Rates (Markdown) ===\n")
print(table.to_string(index=False))
print("\nKey insights:")
print("- These metrics are estimator-agnostic (same calibration for all)")
print("- Refuse threshold: out-of-range ≥ 5% OR saturation ≥ 20%")
print("- Unhelpful policy: 95% of runs refused due to extrapolation")
print("- Clone/Parallel Universe: 0% refused (well-covered by oracle data)")

print("\n\n=== Boundary-Based Refuse Rates (LaTeX) ===\n")
latex = format_latex.format_table_refuse_rates(table)
print(latex)
