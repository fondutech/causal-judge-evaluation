"""
Usage
-----
$ poetry run python scripts/make_figs.py runs/**/result.json

Searches for result.json files, builds:
  • table2.csv   — one row per run with estimator stats
  • fig_ci.png   — bar chart of 95 % CI widths grouped by estimator
Written to ./figs/ (created if absent).
"""

import sys, json, pathlib, re, collections, statistics
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    sys.exit("Pass one or more glob patterns to result.json files")

# ------------------------------------------------------------------ #
# Ingest
records = []
for pattern in sys.argv[1:]:
    # Handle absolute paths directly instead of using glob
    if pathlib.Path(pattern).is_absolute():
        p = pathlib.Path(pattern)
        if p.exists():
            cfg_path = p.parent / ".hydra" / "config.yaml"
            estimator = "unknown"
            if cfg_path.exists():
                import yaml

                estimator = yaml.safe_load(cfg_path.read_text())["estimator"]["name"]
            stats = json.loads(p.read_text())
            records.append(
                dict(
                    run=str(p.parent),
                    estimator=estimator,
                    v_hat=stats["v_hat"],
                    ci_low=stats["ci_low"],
                    ci_high=stats["ci_high"],
                    ci_width=stats["ci_width"],
                    n=stats["n"],
                )
            )
    else:
        # Use glob for relative patterns
        for p in pathlib.Path(".").glob(pattern):
            cfg_path = p.parent / ".hydra" / "config.yaml"
            estimator = "unknown"
            if cfg_path.exists():
                import yaml

                estimator = yaml.safe_load(cfg_path.read_text())["estimator"]["name"]
            stats = json.loads(p.read_text())
            records.append(
                dict(
                    run=str(p.parent),
                    estimator=estimator,
                    v_hat=stats["v_hat"],
                    ci_low=stats["ci_low"],
                    ci_high=stats["ci_high"],
                    ci_width=stats["ci_width"],
                    n=stats["n"],
                )
            )

if not records:
    print("No result files found")
    sys.exit(1)

df = pd.DataFrame(records)
out_dir = pathlib.Path("figs")
out_dir.mkdir(exist_ok=True)
df.to_csv(out_dir / "table2.csv", index=False)
print(f"[✓] table2.csv saved with {len(df)} rows")

# ------------------------------------------------------------------ #
# Bar chart of mean CI width per estimator
agg = df.groupby("estimator")["ci_width"].agg(["mean", "count"]).reset_index()
plt.figure(figsize=(6, 3))
plt.bar(agg["estimator"], agg["mean"])
plt.ylabel("Mean 95% CI width")
plt.title("Estimator efficiency across runs")
plt.tight_layout()
plt.savefig(out_dir / "fig_ci.png", dpi=180)
print("[✓] fig_ci.png saved")
