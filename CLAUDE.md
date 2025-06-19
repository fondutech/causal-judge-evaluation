# CLAUDE.md

Core guidance for Claude Code when working with the CJE (Causal Judge Evaluation) repository.

## 🎯 Information Hygiene Rules

### STARTUP PROTOCOL
At session start, immediately:
1. Check for session notes older than today → Delete
2. Run `python scripts/hygiene_check.py` → Fix issues
3. Run `cje check-deps` → Verify dependencies
4. Look for `# type: ignore` comments → Track count

### AUTO-CLEANUP TRIGGERS
When you see these, fix immediately without asking:
- Commands/features that don't exist (e.g., `cje results`)
- References to deleted files or modules
- Silent import failures (try/except ImportError)
- Documentation that contradicts implementation
- Session notes older than current session

### KNOWLEDGE LIFECYCLE
1. **Add**: Document new learnings concisely
2. **Consolidate**: Merge related information 
3. **Expire**: Delete outdated content aggressively
4. **Goal**: This file gets SHORTER over time

## Core Requirements

### Before ANY Task Completion
```bash
make lint  # MUST pass both black and mypy
```

### Essential Commands
```bash
make dev-setup          # Initial setup with Poetry
poetry run pytest              # Run unit tests (fast)
poetry run pytest --run-slow   # Include slow tests
poetry run pytest --integration-only  # Integration tests only
cje run --cfg-path configs --cfg-name arena_test  # Run experiment
cje check-deps          # Check optional dependencies
python scripts/hygiene_check.py  # Check codebase health
```

## Architecture Overview

**Pipeline**: Data → Log Probs → Judge → Calibrate → Estimate → Results

**Key Modules**:
- `cje/uncertainty/`: Clean uncertainty quantification
- `cje/estimators/`: DRCPO/MRDR/IPS implementations
- `cje/judge/providers/`: Unified provider system (consolidated)

**Critical Details**:
- Teacher forcing required for unbiased log probabilities
- Gamma calibration AFTER isotonic (not before)
- Variance shrinkage: λ* = Cov[w²v, w(r-μ)²] / E[w²v²]

## Recent Changes

**Provider Consolidation** (June 19, 2025):
- Single provider system: XProvider (not UnifiedXProvider)
- Each provider supports both regular and structured output
- Removed 14 redundant files

**Uncertainty Module**:
- Mandatory uncertainty in `cje/uncertainty/`
- Use `examples/clean_uncertainty_api.py` for reference
- Config: `configs/uncertainty_example.yaml`

**Documentation Notes**:
- `cje results` command does NOT exist (use Python API)
- Old uncertainty config options removed from unified.py
- New module not in generated API docs yet