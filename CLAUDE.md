# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 PRIMARY DIRECTIVE
Continuously update this file to reflect current understanding, user preferences, and project state. This is not optional - it's a core responsibility to ensure knowledge persistence across sessions.

## Meta-Objectives and User Preferences

### Core Philosophy
The user values:
- **Simplicity over complexity**: Remove duplicate implementations, maintain single sources of truth
- **Clear organization**: Each component should have an obvious home, no confusing duplicates
- **Documentation accuracy**: Documentation should reflect actual implementation, not aspirational features
- **Proactive maintenance**: Continuously identify and remove stale code/docs without being asked
- **Information hygiene**: Actively remove outdated content rather than just adding new - this file should get shorter over time as old context becomes irrelevant

### Key Principles
- **One way to do things**: No duplicate implementations or alternative approaches
- **Simplicity wins**: When in doubt, choose the simpler solution
- **Documentation reflects reality**: No aspirational features or future plans

## IMPORTANT: Code Quality Requirements

**BEFORE COMPLETING ANY TASK**, you MUST run the following commands to ensure code quality:
```bash
make lint        # Runs both black formatting and mypy type checking
```

Both black and mypy MUST pass without errors before considering any task complete. If there are errors:
1. Fix all black formatting issues (usually automatic)
2. Fix all mypy type errors
3. Run `make lint` again to verify all issues are resolved

## Essential Commands

### Development Setup
```bash
make dev-setup    # Install dependencies with Poetry and set up pre-commit hooks
```

### Running Tests
```bash
# Basic test run
poetry run pytest

# Include slow tests
poetry run pytest --run-slow

# Run specific test categories
poetry run pytest --theoretical-only    # Theoretical guarantees only
poetry run pytest --empirical-only      # Empirical validation only
poetry run pytest --paper-validation    # Paper claim validation

# Run a single test
poetry run pytest tests/test_file.py::test_function_name
```

### Linting and Formatting
```bash
make lint        # Run both black formatting and mypy type checking
make format      # Format code with black only
make mypy        # Type check only

# Or directly:
black cje/ examples/ tests/
mypy cje/ examples/ --exclude ".*test.*" --ignore-missing-imports --no-strict-optional
```

### Documentation
```bash
make docs        # Build documentation
make docs-serve  # Serve documentation locally at http://localhost:8000
make docs-clean  # Clean rebuild documentation
```

### Running CJE Experiments
```bash
# Full experiment pipeline
cje run --cfg-path configs --cfg-name arena_test

# Individual pipeline stages
cje log           # Compute log probabilities
cje judge         # Generate judge scores
cje calibrate     # Calibrate judge
cje estimate      # Run causal estimation

# Arena experiment
python examples/run_arena_experiment.py --config arena_experiment
```

## High-Level Architecture

### Core Pipeline Flow
The CJE pipeline follows this sequence:
1. **Data Loading** (`cje/data/`) - Loads datasets (ChatbotArena, CSV, JSONL)
2. **Log Probability Computation** (`cje/loggers/`) - Computes policy log probabilities
3. **Judge Scoring** (`cje/judge/`) - Scores responses using LLM judges
4. **Calibration** (`cje/calibration/`) - Isotonic calibration with cross-fitting
5. **Causal Estimation** (`cje/estimators/`) - DRCPO/MRDR/IPS estimation
6. **Results** (`cje/results/`) - Policy rankings and diagnostics

### Key Design Patterns

**Provider Abstraction**: All LLM interactions go through provider interfaces in `cje/judge/providers/`. When adding new LLM providers, implement both regular and structured variants.

**Configuration System**: Uses Hydra with unified config schema in `cje/config/unified.py`. All experiments are driven by YAML configs in `configs/`.

**Caching Strategy**: Extensive caching for expensive operations:
- LLM API calls cached in `cje/utils/inference_cache.py`
- Pipeline results cached based on work_dir in config
- Checkpointing support for long-running experiments

**Weight Diagnostics**: Built-in importance weight quality checks in `cje/utils/weight_diagnostics.py`. Always monitor ESS (Effective Sample Size) warnings.

### Research Components

The `cje/research/` module contains experimental features:
- `arena_experiment.py` - Full arena-style evaluation orchestrator
- `phase_manager.py` - Manages multi-phase research experiments
- `validation.py` - Gold standard validation runners

### Critical Implementation Details

**Cross-Fitting**: The k-fold cross-validation in calibration prevents overfitting. Default k=5, but can be adjusted based on data size.

**Log Ratio Clipping**: Default clip value of 20.0 prevents extreme importance weights. This is configurable in diagnostics section.

**Oracle Mode**: When enabled, uses a stronger model for ground truth labels. Useful for validation but increases API costs.

**Trajectory Support**: The codebase supports both single-turn and multi-turn trajectory evaluation via `cje/data/trajectory_dataset.py` and `cje/estimators/trajectory_drcpo.py`.

**Weight Calibration Bug Fix**: Fixed critical bug where re-scaling after capping could push weights above the cap again. The implementation now accepts small bias (E[w] ≠ 1) to maintain variance control, which is preferable to violating the cap constraint. See `cje/estimators/calibration.py` for the corrected implementation.

### Paper Implementation Notes

This codebase implements the CJE paper (Landesberg 2025) with extensions:

**Core Paper Components**:
- Calibrated DR-CPO estimator (Section 4.2) → `cje/estimators/drcpo.py`
- Isotonic calibration (Section 2.2) → `cje/calibratrunion/isotonic.py`
- MRDR variant (Section 4.3) → `cje/estimators/mrdr.py`
- Cross-fitted algorithm (Algorithm 1) → Implemented in all estimators
- Single-rate efficiency (Theorem 5.2) → Preserved in implementation

**Key Extensions**:
- Multi-policy vectorized evaluation (paper focuses on single π')
- Comprehensive provider support beyond paper examples
- Arena research framework in `cje/research/`
- Production features: caching, checkpointing, progress tracking

**Key Implementation Features**:
- Pi0 data generation scripts in `scripts/generate_pi0_data.py`
- Arena research experiments with gold standard validation
- Fixed weight calibration accepts small bias (E[w] ≠ 1) to maintain variance control
- Comprehensive visualization and weight diagnostics

### Current Focus Areas

**Arena 10K Oracle Experiment** (June 2024):
- Located in `experiments/arena_10k_oracle/`
- Designed for paper validation with human labels via crowdsourcing
- Key insight: Experiment has natural breakpoints for offline human labeling
- Scripts 01-04 implemented, 05-07 marked as TBD
- Uses Fireworks API for Llama models, ~$1k total budget

**Structural Preferences**:
- `experiments/` for self-contained experiments with own scripts/configs/docs
- `examples/` for usage examples of the main CJE pipeline
- No duplicate documentation - keep docs with implementation
- Empty directories should be removed
- Output files belong in `outputs/` or experiment-specific directories

### Documentation Notes

**Package Status**: CJE is not yet published to PyPI. All installation must be done via development setup with Poetry.

**Config References**: Use `arena_test` as the default config example for quick testing (20 samples), or `arena_research_experiment` for full runs (10k samples).

**Common Documentation Issues**:
- The `cje results` command does not exist - results are accessed via Python API or saved JSON files
- Installation should always use `poetry install` after cloning, never `pip install cje`
- When updating docs, ensure ReadTheDocs rebuilds by pushing to GitHub
- Removed `README_simple.md` which referenced non-existent pip install

**Pre-commit Hooks**: The repository uses strict pre-commit hooks including:
- Black formatting (automatic fixes)
- Mypy type checking (may block commits)
- To bypass temporarily: `git commit --no-verify`

**Type Annotations**: When adding new code:
- Always add type hints to function signatures
- Use `Optional[T]` instead of `T = None` for optional types
- Import types from `typing` module
- For numpy arrays in matplotlib, use `.tolist()` to avoid mypy issues

### Proactive Knowledge Management

**IMPORTANT**: This is a core expectation, not optional. Claude should proactively:

**1. After EVERY significant task**:
- Update CLAUDE.md if any new patterns, preferences, or context emerged
- Document any decisions made (e.g., "User preferred X over Y because...")
- Add any discovered constraints or gotchas
- Remove references to deleted/modified code
- **IMPORTANT**: Also remove outdated information - if something is no longer true or relevant, delete it rather than just adding corrections

**2. Start of each session**:
- Review recent commits for context
- Check if CLAUDE.md reflects current state
- Look for incomplete work from previous sessions
- Scan for new TODOs or FIXMEs in code

**3. Unprompted maintenance** (do without being asked):
- When you notice inconsistencies, fix them
- When you see stale docs, update them
- When you find redundancy, consolidate it
- When you see unclear naming, suggest improvements

**4. Knowledge capture triggers**:
- User expresses a preference → Document it
- User corrects an approach → Note the preferred way
- Task reveals hidden complexity → Explain it for future
- Error occurs repeatedly → Document the solution

**5. What to document**:
- Design decisions and their rationale
- Preferred tools/libraries/approaches
- Common pitfalls and their solutions
- Performance considerations discovered
- API quirks or limitations encountered
- Workflow preferences (e.g., "always run make lint")

**Example proactive behaviors**:
- "I noticed config files in multiple places, shall I consolidate them?"
- "This documentation references a deleted module, updating it now..."
- "Based on your previous preference for simplicity, I suggest..."
- "I found duplicate test utilities, removing redundancy..."

**Red flags to watch for**:
- Documentation mentioning non-existent files
- Multiple ways to do the same thing
- Unused dependencies or imports
- Output files in repository root
- Empty or nearly-empty directories
- Commented-out code without explanation
- Old session notes that are no longer relevant
- Accumulating "historical" sections that don't inform current work

### Information Consolidation Rules

**This file should get SHORTER over time, not longer**:
- When adding new information, check if it makes something else obsolete
- Consolidate similar sections rather than adding new ones
- Remove session notes older than 2-3 sessions unless they contain critical context
- Replace verbose explanations with concise rules once patterns are established
- Delete "historical" information that no longer affects current work

**Example**: Instead of keeping "June 2024: Removed simple_cli.py because...", just note in philosophy: "Single implementation principle - no duplicate approaches"

### Session Handoff Protocol

**At the end of each session**, create a brief "handoff" section here with:
- Date and summary of work completed
- Any unfinished tasks or open questions
- Decisions that need user confirmation
- Areas that need attention next time
- **What old information can be removed next session**

**Example**:
```
## Session Notes - June 14, 2024
- Completed: Removed duplicate implementations (simple_cli.py, core.py, etc.)
- Completed: Cleaned up documentation, removed stale READMEs
- Pending: Arena 10K experiment scripts 05-07 still TBD
- Note: User prefers simplicity - rejected complex multi-path implementations
- Check next: Test if all examples still work after simplification
```

## Session Summary - June 14, 2025

### Major Work Completed
- **Codebase simplification**: Removed ~1,600 lines of duplicate code (simple_cli.py, core.py, etc.)
- **Documentation cleanup**: Removed stale files, updated all references
- **Paper analysis**: Reviewed CJE paper, noted 30-line claim vs 10k+ line reality
- **Design decision**: Rejected variance-aware judge selection as too complex

### Key Learnings Consolidated into Principles Above
- Single implementation principle 
- Proactive maintenance expected
- Information hygiene (remove old content)

### Still Pending
- Arena 10K Oracle experiment scripts 05-07
- Test examples post-simplification
- Consider consolidating provider variants

### To Remove Next Session
- This session summary (after 2-3 more sessions)
- Old references to June 2024 throughout the file
- Detailed file lists from cleanup (already captured in principles)