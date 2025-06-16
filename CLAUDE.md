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

**Completions Template System**: The `cje/loggers/completions_templates.py` module provides templates for converting chat conversations to continuous strings required by completions API endpoints. Currently provides:
- Llama 3: `<|begin_of_text|><|start_header_id|>...<|end_header_id|>...<|eot_id|>`
- Llama 4: `<|begin_of_text|><|header_start|>...<|header_end|>...<|eot|>`

**IMPORTANT**: Users must explicitly specify the correct `completions_template_format` for their model. There is NO auto-detection. Using the wrong template will result in incorrect log probabilities. Example:
```python
runner = APIPolicyRunner(
    provider="fireworks",
    model_name="llama-v3p3-70b-instruct",
    completions_template_format="llama3"  # REQUIRED!
)
```

**Teacher Forcing Validation**: The `cje/loggers/template_validation.py` module prevents silent failures by validating template configuration before experiments:
```python
runner = APIPolicyRunner("fireworks", "llama-v3p3-70b-instruct")
runner.validate_teacher_forcing()  # Raises error if template mismatch detected
```
Validation tests known high-probability responses (e.g., "4" for "2+2?") and provides detailed diagnostics when log probabilities are suspiciously low (< -20), indicating template mismatches or provider incompatibility.

**Provider Support for Teacher Forcing**:
- **Fireworks AI**: ✅ Fully supports completions API with echo=True for teacher forcing (all models)
- **Together AI**: ⚠️ Mixed support - Llama 3.x models work, but Llama 4 models do NOT support echo=True
  - ✅ Working: Llama 3.3, 3.2, 3.1, 3.0 models 
  - ❌ Not working: Llama 4 models (returns "Echo not yet supported for this model" error)
- **OpenAI**: ❌ Deprecated completions API, no echo support in chat completions
- **Anthropic**: ❌ No completions API

Both Fireworks AI and Together AI (for Llama 3.x models) support teacher forcing. See usage guide at `experiments/arena_10k_oracle/scripts/COMPLETIONS_TEMPLATES_USAGE.md`.

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


## Session Notes - June 16, 2025 (continued)

### Additional Completed Work
- **Completions Template System Simplification**:
  - Removed all model name auto-detection logic per user request
  - Simplified API to require explicit `completions_template_format` specification
  - Created comprehensive validation system to detect template misconfigurations
  - Added `validate_teacher_forcing()` method to APIPolicyRunner
  
- **Documentation Updates**:
  - Created comprehensive guide at `/docs/guides/completions_templates.rst`
  - Updated teacher_forcing.rst to reference new template system
  - Updated configuration_reference.rst with completions_template_format examples
  - Added completions_templates to guides index for discoverability
  
- **Code Cleanup**:
  - All legacy development files already cleaned up (test_simple_completions.py, etc.)
  - Kept only `test_simplified_templates.py` as the canonical test
  - All code passes black formatting and mypy type checking
  
### Key Design Decision
- User explicitly requested: "The user will be responsible for providing the right completions format for the model they are using"
- This simplified the codebase significantly and made errors more explicit

## Session Notes - June 17, 2025 (Today)

### Completed Work
- **Completions Template System**: Refactored prompt templates to clarify they're specifically for completions API
  - Renamed `prompt_templates.py` → `completions_templates.py`
  - Renamed `PromptTemplate` → `CompletionsTemplate` throughout
  - Added clear documentation that these are for converting chat to continuous strings for teacher forcing
  - Created usage guide at `experiments/arena_10k_oracle/scripts/COMPLETIONS_TEMPLATES_USAGE.md`
  
- **Llama 4 Template Fix Resolution**: 
  - Root cause: Wrong prompt template for Llama 4 models (was using Llama 3 format)
  - Solution: Automatic template detection based on model name
  - Result: "Cabbages" logprob fixed from -21.625 to reasonable values (~0.0)
  - Confirmed Fireworks API works correctly with proper template

### Key Architectural Decisions
- Templates are named "CompletionsTemplate" to distinguish from other prompt templates
- Auto-detection for common providers (Fireworks, Together, OpenAI, Anthropic)
- Extensible system allows custom templates via config or code
- All linting checks pass (black, mypy)

### To Remove Next Session
- Session notes from June 15 (legacy clip removal details - already captured in principles)
- Detailed Arena 10K experiment issues from June 16 (resolved by template fix)

### Updates Today
- **Simplified completions template system**: 
  - Removed all auto-detection - users must explicitly specify `completions_template_format`
  - Provides Llama 3 and Llama 4 templates out of the box
  - Clear error messages if wrong format is specified
  - Updated all config files to include `completions_template_format`
- **Comprehensive validation system** to prevent silent failures:
  - `cje/loggers/template_validation.py`: Tests known high-probability responses
  - Provides detailed diagnostics when validation fails
  - Integrated into APIPolicyRunner with `validate_teacher_forcing()` method
- **Documentation**: Created clear guide at `COMPLETIONS_TEMPLATE_GUIDE.md`
- All code passes black formatting and mypy type checking