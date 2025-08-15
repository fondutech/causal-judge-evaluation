# CLAUDE.md

Context and guardrails for working with CJE.

## Core Principle

**Think Before Doing** - I have good judgment AFTER gathering context.
I make poor decisions when I rush to implement.

## Before ANY Implementation

**STOP and gather context:**
1. Search for existing patterns in the codebase
2. Read related files completely
3. Understand why things are the way they are
4. Consider implications and edge cases
5. THEN propose an approach

I often have good insights after proper investigation.
I often make mistakes when I skip this step.

## Questions to Ask Myself

Before implementing:
- Have I searched for how this is done elsewhere in the codebase?
- Do I understand WHY the current design exists?
- Am I about to reinvent something that already exists?
- Is this the simplest solution that works?

During implementation:
- Am I more than 3 files deep? (Stop and check if on track)
- Am I breaking something that already works?
- Would a smaller change suffice?

After implementing:
- Can the user maintain this?
- Did I explain the key decisions?
- Are the tests still passing?

## What NOT to Do

**Never** add these to CJE:
- Workflow orchestration (users compose tools)
- State management (each run is independent)
- Retry logic (fail fast and clear)
- Magic values (None/NaN for failures)
- Clever abstractions (YAGNI)

**Never** do these as Claude:
- Implement before understanding context
- Create new files when editing would work
- Make docs unless explicitly asked
- Commit without explicit request
- Fix what isn't broken

## Understanding CJE

**What CJE Does**: Unbiased off-policy evaluation of LLMs using causal inference.
Answers: "What would our metrics be if we deployed policy π' instead of π₀?"

**The Core Problem**: Offline "LLM-as-judge" scores are correlational - computed under 
logging policy π₀, they don't answer the counterfactual question. CJE recasts judge-based 
evaluation as calibrated causal inference.

**Core Flow**:
```
logs.jsonl → calibrate_dataset() → estimator.fit_and_estimate() → results
                    ↓                         ↓
            (maps judge→oracle)    (applies SIMCal + computes estimates)
```

**Why SIMCal Works** (theoretical guarantees):
- Mean preservation: Calibration never changes E[W] = 1
- Variance reduction: Monotone projection always reduces variance (majorization)
- Variance safety: Cap ensures Var(W_calibrated) ≤ ρ·Var(W_baseline)
- √n inference: DR achieves efficiency bound when assumptions hold

**Key Components**:
- `data/` - Dataset, Sample models with validation
- `calibration/` - Isotonic regression, SIMCal, judge calibration
- `estimators/` - IPS, CalibratedIPS, DR, MRDR, TMLE
- `diagnostics/` - IPSDiagnostics, DRDiagnostics, reliability gates
- `experiments/arena_10k_simplified/` - Full pipeline example

**Three Calibrations** (keep these straight):
1. **Reward**: Isotonic f: judge → oracle on small slice (preserves mean)
2. **Weight**: SIMCal - projects onto monotone functions, weakly reduces variance by majorization
3. **Outcome**: Cross-fitted g(X) predictions for DR orthogonality

**Estimator Hierarchy**:
```
BaseCJEEstimator
├── RawIPS (baseline, no calibration)
├── CalibratedIPS (SIMCal weights, production default)
└── DREstimator (abstract)
    ├── DRCPOEstimator (basic DR)
    ├── MRDREstimator (multiply robust)
    └── TMLEEstimator (targeted learning)
```

**Data Requirements**:
- Always: `prompt`, `response`, `base_policy_logprob`, `target_policy_logprobs`
- For calibration: `metadata.judge_score` 
- For oracle calibration: Some samples with `metadata.oracle_label`
- For DR: Fresh draws via `add_fresh_draws()`

**Key Concepts**:
- **ESS (Effective Sample Size)**: (Σw)²/Σw² - measures effective overlap
- **SIMCal**: OOF stacking of {baseline, ↑, ↓} candidates + variance cap ρ
  - Projects weights onto monotone functions of judge score S
  - Weakly reduces variance by majorization (always increases ESS)
  - Blend → reproject enforces Var(W) ≤ ρ·Var(baseline)
- **Oracle slice**: Small random subsample with ground truth for reward calibration
- **Cross-fitting**: Train on k-1 folds, predict on kth (orthogonality for DR)
- **Influence functions**: Per-sample contributions, enable √n inference
- **Refusal gates**: Return NaN when ESS < threshold or tail index < 2

**Key Assumptions** (simplified):
- **(D2) Overlap**: π₀(a|x) > 0 whenever π'(a|x) > 0
- **(J1) Oracle slice**: Simple random subsample with ground truth Y
- **(J2-M) Judge monotone sufficiency**: E[Y|S] is monotone, E[W|S] is monotone
- **(R3) DR rates**: Either weights or outcome model converges at n^(-1/4)

## Technical Context

```bash
source set_secrets.sh  # API setup
poetry run pytest cje/ # Run tests
python analyze_dataset.py --data data.jsonl --estimator calibrated-ips
```

**Common Issues**:
- "ESS too low" → Policies too different, need DR with fresh draws
- "NaN estimates" → Check diagnostics.summary(), likely catastrophic overlap
- "Import errors" → Wrong directory or missing poetry install
- "No module named cje" → Need `pip install -e .` or `poetry install`

## Navigation Patterns

**To find examples of X**:
```bash
# Find how something is used
grep -r "SIMCal" cje/ --include="*.py"

# Find test examples
grep -r "test.*calibrat" cje/tests/

# Find where a class is defined
grep -r "class.*CalibratedIPS" cje/
```

**Key files to check**:
- `analyze_dataset.py` - Main entry point, shows full flow
- `cje/tests/test_integration.py` - End-to-end examples
- `experiments/arena_10k_simplified/` - Production pipeline

## My Best Practices

1. **Research first** - grep/read before implementing
2. **Start minimal** - Can always add more
3. **Test immediately** - Run tests after changes
4. **Explain changes** - User should understand what I did
5. **Stay focused** - Return to user's actual goal

## When to Be Skeptical

CJE results are suspect when:
- ESS is very low (< 10% typical, < 1% critical)
- Judge scores drift over time (check Kendall τ)
- Policies are very different from logging policy
- Oracle slice is small or non-random
- DR orthogonality score CI doesn't contain 0

Always check `diagnostics.summary()` before trusting estimates.

## Remember

The user knows their problem better than I do.
My job is execution, not strategy.
Simple and working beats perfect and complex.

**I think better when I slow down and gather context.**

When the user says "let's just leave things be and focus on simplicity" - listen.