"""
Aggregate variance across prompt templates.
Usage:
    poetry run python scripts/ci_summary.py runs/202505*/result.json
"""

import sys, json, glob, statistics, pathlib, re
from typing import Dict, Any


def ci_width(res: Dict[str, Any]) -> float:
    return float(res["ci_width"])


rows = []
for path in glob.glob(sys.argv[1]):
    tpl = re.search(r"prompt-sweep-[^/]+/(.*?)/", path) or re.search(
        r"result.json", path
    )
    template = tpl.group(1) if tpl else "unknown"
    res = json.loads(pathlib.Path(path).read_text())
    rows.append((template, ci_width(res)))

for t in sorted({r[0] for r in rows}):
    vals = [w for tpl, w in rows if tpl == t]
    print(f"{t:20}  mean={statistics.mean(vals):.4f}  n={len(vals)}")
