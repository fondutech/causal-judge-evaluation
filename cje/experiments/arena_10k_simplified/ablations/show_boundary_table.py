#!/usr/bin/env python3
"""Quick script to show the boundary diagnostic table."""

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

# Generate the boundary diagnostic table
table = tables_main.build_table_boundary_diagnostics(df)

print("=== Boundary Diagnostic Table (Markdown) ===\n")
print(table.to_string(index=False))

print("\n\n=== Boundary Diagnostic Table (LaTeX) ===\n")
latex = format_latex.format_table_boundary_diagnostics(table)
print(latex)
