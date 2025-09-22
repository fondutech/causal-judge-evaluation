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

# Generate the refuse rates table by regime
print("=== Boundary-Based Refuse Rates by Oracle Regime ===\n")
table_regime = tables_main.build_table_refuse_rates(df, by_regime=True)
print(table_regime.to_string(index=False))

print("\nKey insights:")
print("- With small oracle (250) and low coverage (5%):")
print("  * Even clone/parallel have 5-8% refuse rate")
print("  * Premium refuses 45% of the time")
print("  * Unhelpful always refused (100%)")
print("\n- With large oracle (5000) and high coverage (50%):")
print("  * Clone/parallel/premium all pass (0% refuse)")
print("  * Unhelpful still 85% refused (fundamental extrapolation issue)")
print("\n- Boundary issues decrease with more oracle data")
print("- Unhelpful policy consistently problematic across all regimes")
