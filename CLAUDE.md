# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT**: Also reference ~/.claude_memory.md for personal preferences and cross-project context.

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

**Automated Oracle Mode**: When enabled, uses a stronger model (e.g., GPT-4o) instead of human labels for ground truth validation. Useful for validation but increases API costs. Note: This is different from human oracle labels collected via crowdsourcing.

**Trajectory Support**: The codebase supports both single-turn and multi-turn trajectory evaluation via `cje/data/trajectory_dataset.py` and `cje/estimators/trajectory_drcpo.py`.

**Weight Calibration Bug Fix**: Fixed critical bug where re-scaling after capping could push weights above the cap again. The implementation now accepts small bias (E[w] ≠ 1) to maintain variance control, which is preferable to violating the cap constraint. See `cje/estimators/calibration.py` for the corrected implementation.

**Teacher Forcing Implementation**: CRITICAL - Two-pass generation is REQUIRED for consistent log probability computation between π₀ and target policies:
1. Generate responses using chat completions API (natural generation)
2. Score responses using completions API with echo=True (teacher forcing)

This is NOT an optimization issue - it's a fundamental requirement for causal identification. Using single-pass generation would introduce bias because π₀ and π' would be scored differently. The `generate_with_consistent_logp` method implements this correctly.

**Token Extraction Fix (June 2025)**: Fixed critical bug in `_teacher_forcing_logprob` where tokenization context differences (e.g., `']</s>'` vs `'] </s>'`) caused extraction of wrong tokens. Now uses direct response search with divergence-based fallback in `_extract_response_logprobs_by_divergence`. This resolved the "Cabbages" -21.625 logprob issue where `</s>` tokens were being extracted instead of response tokens.

**Llama 4 Template Fix (June 2025)**: Resolved incorrect log probabilities by implementing proper Llama 4 prompt template. The issue was not a Fireworks API bug but incorrect template usage:
- Llama 3 uses: `<s>[INST] ... [/INST] response</s>`
- Llama 4 uses: `<|begin_of_text|><|header_start|>...<|header_end|>...<|eot|>`
After implementing automatic template detection and correct formatting, all logprobs are now correct (0.0 for forced responses like "Cabbages"). The Fireworks API works correctly when using the proper template format.

Currently only Fireworks (confirmed working with Llama 4 template) and Together (unconfirmed) support the required completions API. See `docs/guides/teacher_forcing.rst` for details.

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

**Arena 10K Oracle Experiment** (June 2025):
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

- Detailed file lists from cleanup (already captured in principles)

## Session Notes - June 15, 2025

### Completed Work
- **Legacy code removal**: Removed all legacy clip parameter from weight processing pipeline
  - Removed from: unified.py, multi_target_sampler.py, all estimators (IPS, DRCPO, MRDR, base_crossfit)
  - Updated config files (arena_test.yaml, arena_research_experiment.yaml)
  - Updated all documentation references
- **Backward compatibility cleanup**: 
  - Removed confidence_intervals() alias method
  - Removed legacy metadata format handling
  - Fixed IPS estimator to use proper base class and interface
- **Code quality**: All mypy and black checks pass

### Key Changes
- Weight clipping now handled via log_ratio_clip in diagnostics config (hard clipping at ±20)
- Removed soft clipping approach in favor of numerical stabilization + hard clipping
- IPS estimator refactored to match standard Estimator interface

### Next Session
- Remove this session summary after 2-3 sessions
- Consider removing supports_logprobs from provider_registry (kept for now as it's part of provider capability tracking)
- Test all examples still work after IPS refactoring

## Session Notes - June 16, 2025

### Completed Work
- **Arena 10K Oracle Experiment Pipeline**:
  - Fixed logprob=0 issues caused by duplicate entries in checkpoint files
  - Generated 72 logging policy responses with proper teacher-forcing logprobs
  - Scored all responses with judge (avg: 0.746)
  - Generated all 3 target policies (pi_hot, pi_cot, pi_concise) with consistent logprobs
  - Exported data for human labeling (18 calibration, 54 evaluation samples)

- **Critical Teacher Forcing Token Extraction Fix**:
  - Fixed deeper bug where wrong tokens were extracted due to tokenization context differences
  - Issue: Template had space in `[/INST] </s>` vs `[/INST]</s>` causing token boundary misalignment
  - Root cause: Tokenizer creates different tokens based on context (e.g., `']</s>'` as single token vs separate)
  - Solution: Implemented `_extract_response_logprobs_by_divergence` using direct response search
  - Impact: "Cabbages" logprob corrected from -21.625 to ~-1.8 (was extracting `</s>` tokens)
  
### Key Learnings
- **Two-pass generation is REQUIRED**: User emphasized this is for causal identification, not optimization
- Fireworks supports batch completions API (600 RPM limit)
- Token extraction must account for tokenization context differences
- Created multiple script versions (04_generate_targets_fast.py, minimal, all) - needs consolidation
- Checkpoint handling needs improvement to prevent duplicates

### Important Refactors Identified
1. Consolidate multiple generation scripts into one canonical version
2. Improve checkpoint handling to track per-policy, per-sample progress
3. Re-run all generations with fixed token extraction
4. Better error handling for API timeouts

### Next Steps
- Re-generate all experiment data with corrected logprob extraction
- The checkpoint files contain incorrect logprobs and need regeneration