# Causal Judge Evaluation (CJE)

[![Docs](https://readthedocs.org/projects/causal-judge-evaluation/badge/?version=latest)](https://causal-judge-evaluation.readthedocs.io/en/latest/)
[![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Shape‑constrained, unbiased off‑policy metrics for LLM systems and beyond. Turn routine judge scores into reliable counterfactual policy estimates with variance control.

Quickstart
----------

Install from source and run your first estimate.

```bash
git clone https://github.com/fondutech/causal-judge-evaluation.git
cd causal-judge-evaluation
pip install -e .
```

Python API:

```python
from cje import analyze_dataset

results = analyze_dataset("logs.jsonl", estimator="calibrated-ips")
print(f"Estimate: {results.estimates[0]:.3f} ± {1.96*results.standard_errors[0]:.3f}")
```

CLI:

```bash
python -m cje analyze logs.jsonl --estimator calibrated-ips -o results.json
```

At a Glance
-----------

```
┌──────────────┐      ┌────────────────────────────────┐      ┌─────────────────────────────┐      ┌──────────────────────┐
│  Logs (JSONL)│  ─→  │ Calibrate f: S → R             │  ─→  │ Estimate (IPS or DR)        │  ─→  │   Diagnostics & CIs  │
│ X, A, S,     │      │ (oracle slice, cross‑fit)      │      │ W = exp(lp_t − lp_b)        │      │ ESS, tails, compare  │
│ log pπ₀, pπ′ │      │                                │      │SIMCal → W_c (stable weights)│      │ policies, gates      │
└──────────────┘      └────────────────────────────────┘      └─────────────────────────────┘      └──────────────────────┘
```

Where: X=prompt, A=response, S=judge score, lp=log probability.

Choosing an Estimator
---------------------

- Logged data only → `calibrated-ips` (default non‑DR; SIMCal‑stabilized IPS)
- Have fresh draws → `stacked-dr` (default DR; calibrated stacked DR ensemble)
- Not sure → `--estimator auto` (chooses DR if `--fresh-draws-dir` is set, else IPS)

Data Format (minimal)
---------------------

```json
{
  "prompt": "...",
  "response": "...",
  "base_policy_logprob": -35.7,
  "target_policy_logprobs": {"A": -33.1, "B": -34.2},
  "metadata": {"judge_score": 0.85, "oracle_label": 0.90}
}
```

Documentation
-------------

- Quickstart and tutorials: https://causal-judge-evaluation.readthedocs.io/en/latest/
- How it works (pipeline schematic): https://causal-judge-evaluation.readthedocs.io/en/latest/explanation/how_it_works.html
- Examples: `examples/`
- Arena experiment: `cje/experiments/arena_10k_simplified/`

Development
-----------

- Install: `make install` (Poetry managed), or `pip install -e .`
- Tests: `make test` (end‑to‑end focus), coverage available
- Lint/format: `make lint`, `make format`
- Docs: `make docs` and `make docs-serve`

License
-------

MIT License – see [LICENSE](LICENSE).
