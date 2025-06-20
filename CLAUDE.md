# CLAUDE.md

Core guidance for Claude Code when working with the CJE (Causal Judge Evaluation) repository.

## 🎯 Information Hygiene Rules

### STARTUP PROTOCOL
At session start, immediately:
1. Check git status and recent commits for context
2. Run `python scripts/hygiene_check.py` to check codebase health
3. Run `make lint` to verify code quality

### AUTO-CLEANUP TRIGGERS
When you see these, fix immediately without asking:
- Commands/features that don't exist (e.g., `cje results`)
- References to deleted files or modules
- Documentation that contradicts implementation
- Outdated comments or docstrings

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
make dev-setup                 # Initial setup with Poetry
poetry run pytest              # Run unit tests (fast)
poetry run pytest --run-slow   # Include slow tests
cje run --cfg-path configs --cfg-name arena_test  # Run experiment
```

## Architecture Overview

**Pipeline**: Data → Log Probs → Judge → Calibrate → Estimate → Results

**Key Design Principles**:
- All judges return `JudgeScore(mean, variance)` - no exceptions
- Single source of truth - no duplicate implementations
- Uncertainty is built-in, not bolted-on

**Critical Implementation Details**:
- Teacher forcing required for unbiased log probabilities (two-pass generation)
- Cross-fitting prevents overfitting in calibration (default k=5)
- Log ratio clipping at ±20.0 prevents numerical issues
- All scores stored as `{"mean": x, "variance": y}` dictionaries

## Current State (December 2024)

**Judge System**:
- Unified architecture where ALL judges return `JudgeScore` objects
- Three uncertainty methods: deterministic (var=0), structured, monte_carlo
- No backward compatibility needed (no existing users)
- Clean provider abstraction with capability tracking

**Known Issues**:
- `cje/uncertainty/` module exists but isn't integrated
- Some tests reference this unimplemented module
- Needs proper documentation and integration

**Documentation**:
- Just cleaned up to reflect actual implementation
- Removed references to planned but unimplemented features
- API docs now correctly reference existing modules

## Key Insights

**Simplicity Wins**: User explicitly stated "we don't have any users" - this allowed removal of ~700 lines of backward compatibility code.

**Clean Architecture**: The judge system is well-designed with proper uncertainty quantification built in from the start.

**Provider Support**: Fireworks and Together support full teacher forcing; OpenAI/Anthropic are judge-only.

## What NOT to Reference

These were removed or never implemented:
- `UNIFIED_JUDGE_SUMMARY.md` (deleted)
- `unified_judge_migration.md` (deleted) 
- `scripts/migrate_to_unified_judges.py` (deleted)
- `cje/utils/score_storage.py` (deleted)
- Any `*_unified.py` modules (never existed)