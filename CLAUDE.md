# CLAUDE.md

Context and guardrails for working with CJE.

## CRITICAL: Search Before You Build

**THE #1 MISTAKE I MAKE: Reimplementing something that already exists**

Before writing ANY new function or utility:
1. **SEARCH for existing implementations** - grep the entire codebase
2. **CHECK experiments/ folders** - they often have utilities you can reuse
3. **LOOK in tests/** - they show how existing functions are used
4. **READ related modules completely** - the function you need is probably there

The codebase already has utilities for:
- Oracle ground truth handling
- Label restoration after masking
- Dataset calibration
- Fresh draws loading
- Weight diagnostics
- Visualization generation

## Core Principle

**Think Before Doing** - I have good judgment AFTER gathering context.
I make poor decisions when I rush to implement.

## Before ANY Implementation

**STOP and search the codebase:**
1. **READ THE README.md** in the directory you're working in - it explains the module's design
2. **CHECK PARENT AND SIBLING READMEs** - they explain how modules interact
3. **GREP for keywords** related to what you're about to build
4. **LOOK for similar patterns** in existing code
5. **READ the imports** of related files - they reveal existing utilities
6. **ONLY THEN consider implementing** something new

**CRITICAL: Directory READMEs are your map!**
- Each module has a README explaining its purpose and design
- Reading the README takes 2 minutes and prevents hours of mistakes
- The README tells you what already exists and where to find it

I often have good insights after proper investigation.
I often make mistakes when I skip this step.

## Questions to Ask Myself

Before implementing ANYTHING:
- **Have I read the README.md in this directory?**
- **Do I understand this module's purpose from its README?**
- **Have I searched for existing implementations?**
- **Have I checked the experiments/ folders?** (they have many utilities)
- Am I about to reinvent something that already exists?
- Would importing an existing function be simpler?

During implementation:
- Am I creating a utility that might already exist?
- Should I be importing instead of implementing?
- Am I more than 3 files deep? (Stop and check if on track)
- Would a smaller change suffice?
- **If I'm changing behavior, does the README need updating?**

After implementing:
- **Did I update the directory README if my changes affect the module's description?**
- **Are the README's examples still accurate?**
- Could I have reused existing code?
- Can the user maintain this?
- Did I explain why I didn't use existing utilities (if applicable)?

## What NOT to Do

**Never** add these to CJE:
- Duplicate utilities (search first!)
- Workflow orchestration (users compose tools)
- State management (each run is independent)
- Retry logic (fail fast and clear)
- Magic values (None/NaN for failures)
- Clever abstractions (YAGNI)

**Never** do these as Claude:
- **Implement before searching for existing solutions**
- **Create new utilities without checking experiments/ and utils/**
- **Assume something doesn't exist without grep**
- Create new files when editing would work
- Make docs unless explicitly asked
- Commit without explicit request
- Fix what isn't broken

## Lessons from Fold Management Implementation

### Key Insights
1. **Investigation depth matters** - Found 5 systems not 3 through thorough search
2. **Simple solutions scale** - `hash(prompt_id) % n_folds` solved everything
3. **Plans are thinking tools** - Planning forced consideration of edge cases
4. **Test the full pipeline** - Unit tests aren't enough, integration matters
5. **Index-based systems are fragile** - Always use stable identifiers

### Critical Integration Points to Check
When making changes that affect data flow:
- **Check if components extract data from filtered datasets** (indexing issues)
- **Verify backward compatibility needs** (even for "breaking" changes)
- **Test with realistic data** (filtered, missing values, fresh draws)
- **Run existing tests during implementation** (not just at the end)

### Red Flags That Need Investigation
- Multiple components doing similar things differently
- Index-based operations on filtered data
- State stored that could be computed on-demand
- Position-dependent logic in data pipelines

## Critical Data Handling Principles

### Never Fabricate Missing Data
**Fail or filter, never fill.** When data is missing:
- **FAIL**: Raise clear error with what's missing and why it's needed
- **FILTER**: Skip the record with appropriate logging
- **NEVER**: Insert default/dummy values (no 0.5 for missing scores!)

Common violations to avoid:
- Using `or` with numeric fields: `score or 0.5` treats 0.0 as falsy
- Asserting "reasonable" defaults for missing values
- Silently replacing None/NaN with made-up values

Correct patterns:
```python
# BAD: Fabricates data
judge_score = data.get("judge_score") or 0.5  # 0.0 becomes 0.5!

# GOOD: Explicit handling
if "judge_score" in data and data["judge_score"] is not None:
    judge_score = data["judge_score"]
else:
    raise ValueError(f"Missing judge_score for {record_id}")
```

The only exception: Explicitly documented default behaviors (e.g., draw_idx=0 for backwards compatibility).

## Documentation Standards

### Module README Structure
When creating READMEs for CJE modules (estimators/, calibration/, diagnostics/, etc.):

1. **Overview** (2-3 sentences)
2. **When to Use** (decision guide)
3. **File Structure** (simple tree)
4. **Core Concepts** (brief explanations)
5. **Common Interface** (main usage pattern)
6. **Key Design Decisions** (architectural choices)
7. **Common Issues** (troubleshooting)
8. **Performance** (if relevant)
9. **Summary** (2-3 sentences)

Target lengths:
- Simple modules: ~250 lines
- Complex modules: ~350 lines
- Never exceed 400 lines (move details to docstrings)

### Documentation Principles
- Focus on WHY not just WHAT
- Avoid stale-prone implementation details
- Cross-reference instead of duplicating
- Keep math practical with interpretations

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

## Codebase Structure

```
cje/                          # Core library package
├── calibration/              # Judge → Oracle calibration
├── data/                     # Data models and sampling
│   ├── models.py            # Dataset, Sample, EstimationResult
│   ├── precomputed_sampler.py # Filtering, caching, sampling
│   ├── fresh_draws.py       # Fresh draw loading
│   └── folds.py             # Unified fold management (single source of truth)
├── diagnostics/              # Weight and estimation diagnostics  
├── estimators/               # IPS, DR, MRDR, TMLE implementations
├── experiments/              
│   └── arena_10k_simplified/ # Production pipeline example
├── interface/                # High-level API (analyze_dataset)
├── teacher_forcing/          # Fresh draw generation
├── tests/                    # Comprehensive test suite
├── utils/                    # Export and analysis utilities
└── visualization/            # Plotting and visual diagnostics
```

**Why SIMCal Works** (theoretical guarantees):
- Mean preservation: Calibration never changes E[W] = 1
- Variance reduction: Monotone projection always reduces variance (majorization)
- Variance safety: Cap ensures Var(W_calibrated) ≤ ρ·Var(W_baseline)
- √n inference: DR achieves efficiency bound when assumptions hold

**Key Components**:
- `data/` - Dataset, Sample models with validation
- `calibration/` - Isotonic regression, SIMCal, judge calibration, oracle slice augmentation
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
- **Oracle slice augmentation**: Adds (L/p)×m̂(S)×(Y-f̂(S)) for honest CIs accounting for calibration uncertainty
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

## Finding Existing Code and Understanding Modules

**ALWAYS START WITH READMEs:**
1. **Read the README.md in the directory you're working in** - it's your primary guide
2. **Check parent directory README** - understand the broader context
3. **Look at sibling module READMEs** - see how modules work together

**Where to look for existing implementations:**
1. The README in the current directory (it lists what's available!)
2. `experiments/arena_10k_simplified/` - production pipeline with many utilities
3. `cje/tests/` - Shows how things are meant to be used
4. Module-specific utilities in each package (data/, calibration/, diagnostics/)

**How to search effectively:**
- **Start with README.md files** - they're the map to the codebase
- Use grep to search for keywords related to your task
- Look at imports in similar files - they reveal what already exists
- Check the main pipeline files to see established patterns
- Read test files to understand intended usage

**README files are documentation contracts:**
- They describe what the module does and how to use it
- If you change the module's behavior, you MUST update the README
- If the README says something exists, it should exist
- If you add significant functionality, document it in the README

## My Best Practices

1. **READ READMEs FIRST** - Every directory has a README explaining what's there
2. **SEARCH SECOND** - grep for existing implementations before writing code
3. **Import, don't implement** - If it exists, import it. Don't recreate it.
4. **Check experiments/** - Most utilities you need are already there
5. **Start minimal** - Can always add more
6. **Test immediately** - Run tests after changes
7. **Update READMEs** - Keep documentation accurate when code changes
8. **Explain reuse** - Tell user when I'm reusing existing code
9. **Stay focused** - Return to user's actual goal

**My implementation checklist:**
- [ ] Did I read the README.md in this directory?
- [ ] Did I understand the module's design from its README?
- [ ] Did I search for existing implementations?
- [ ] Did I check experiments/arena_10k_simplified/?
- [ ] Am I reusing existing utilities where possible?
- [ ] If I changed behavior, did I update the README?
- [ ] If creating new code, did I explain why existing utilities won't work?

## When to Be Skeptical

CJE results are suspect when:
- ESS is very low (< 10% typical, < 1% critical)
- Judge scores drift over time (check Kendall τ)
- Policies are very different from logging policy
- Oracle slice is small or non-random
- DR orthogonality score CI doesn't contain 0

Always check `diagnostics.summary()` before trusting estimates.

## Rules for Updating This Document

**When to update CLAUDE.md:**
- After learning a key pattern that prevents mistakes
- When discovering an important principle about the codebase
- To document high-level architecture decisions
- To capture lessons learned from repeated errors

**What belongs in CLAUDE.md:**
- High-level principles and patterns
- Common mistakes to avoid
- Where to find things (directories, not specific functions)
- Design philosophy and architectural decisions
- Workflow patterns and best practices

**What does NOT belong:**
- Code-level implementation details
- Specific function names or signatures
- Low-level technical specifics
- Anything that will become stale quickly
- Details better kept in docstrings or READMEs

**Keep it maintainable:**
- Focus on timeless principles over specifics
- Point to where details live, don't duplicate them
- Keep sections concise and scannable
- Remove outdated information promptly

## Remember

The user knows their problem better than I do.
My job is execution, not strategy.
Simple and working beats perfect and complex.

**I think better when I slow down and gather context.**

When the user says "let's just leave things be and focus on simplicity" - listen.