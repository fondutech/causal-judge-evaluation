# CLAUDE.md

Core guidance for Claude Code when working with the CJE repository.

## 🎯 Project Philosophy

**Do One Thing Well** - The Unix philosophy guides this codebase:
- Each tool has a single, well-defined purpose
- Complex workflows are the user's responsibility to orchestrate
- Tools compose naturally without hidden coupling
- Prefer explicit pipelines over "smart" automation

Core principles:
- Clear separation of concerns (one responsibility per module)
- Type safety with Pydantic models (explicit contracts)
- Explicit error handling (no magic fallbacks or silent failures)
- Simple, composable abstractions (avoid framework-itis)
- YAGNI (You Aren't Gonna Need It) - don't build what isn't needed

## 📝 Documentation Principles

- Keep documentation minimal and focused on core concepts
- **CRITICAL: Avoid content that will go stale** - if it depends on other files, it will drift
- Focus on principles and patterns rather than specific code
- Update README.md for user-facing changes, keep CLAUDE.md for timeless guidance

### Directory-Level Documentation

Every major directory within `cje/` that contains non-trivial core implementations should have its own README.md. These READMEs are **for internal developers** (not end users) and should follow these principles:

1. **Architecture-focused**: Describe the overall design and how components fit together
2. **Interface-focused**: Document the common interfaces and contracts, not implementation details  
3. **Change-resilient**: Focus on patterns and principles that won't change with every PR
4. **Self-contained**: Minimize references to other documentation that might drift
5. **Developer-oriented**: Include troubleshooting, common issues, and testing guidance

#### ⚠️ Anti-Patterns to Avoid

**DO NOT** include in READMEs:
- Specific thresholds or magic numbers (they change)
- Lists of all files with descriptions (use `ls` instead)
- Cross-references to specific line numbers
- Implementation details like variable names or internal methods
- Anything you wouldn't want to update when refactoring

**INSTEAD** focus on:
- Conceptual overview and design rationale
- Common patterns used throughout the module
- Interface contracts that rarely change
- Troubleshooting guidance based on symptoms, not code

#### Examples

❌ **Bad**: "The ESS threshold is set to 0.30 in line 238 of calibrated_ips.py"
✅ **Good**: "Estimators use ESS thresholds to ensure reliability"

❌ **Bad**: "The `_weights_cache` dictionary stores computed weights"
✅ **Good**: "Weights are cached to avoid recomputation"

❌ **Bad**: "See utils/diagnostics.py for the implementation"
✅ **Good**: "Diagnostics follow a three-tier status system"

Current directories with READMEs:
- `cje/calibration/` - Three distinct calibration approaches (reward, weight, DR)
- `cje/data/` - Data models and validation patterns
- `cje/diagnostics/` - Three-tier diagnostic architecture
- `cje/estimators/` - Estimator hierarchy and selection guide
- `cje/experiments/` - Experiment orchestration patterns

Each README should:
- Explain the directory's single responsibility
- Describe key design decisions and trade-offs
- Show usage patterns without implementation details
- Reference interfaces, not concrete implementations

## 📁 Repository Structure

```
cje/                      # Production implementation
├── calibration/          # Calibration utilities (isotonic, judge calibration)
├── data/                 # Data models, loading, validation
├── estimators/           # IPS, DR, MRDR, TMLE estimators
├── utils/                # Utilities (diagnostics, export, fresh draws)
├── visualization/        # Plotting and dashboard generation
├── teacher_forcing/      # Log probability computation
├── experiments/          # Arena experiment pipeline
└── tests/                # Comprehensive test suite
```

## 🚀 Quick Start

```python
from cje import analyze_dataset

results = analyze_dataset("data.jsonl", estimator="calibrated-ips")
```

For detailed API usage, see the main README.

## 🔧 Testing

```bash
poetry run pytest cje/
```

For specific experiment commands, see the README in each experiment directory.

## 🔑 API Keys

Set environment variables for API access:
- `OPENAI_API_KEY` - Judge and oracle evaluation
- `FIREWORKS_API_KEY` - Response generation and log probabilities

## 📊 Data Format

CJE expects JSONL with log probabilities and optional judge scores. Fields like `prompt_id` and `reward` are auto-generated as needed. See data format documentation for details.

## 🏗️ Key Architectural Decisions

1. **Do One Thing**: Each script/function has exactly one responsibility
2. **Clean Separation**: Data generation vs analysis are separate steps
3. **Optional Everything**: Rewards, prompt_id, oracle labels - all optional with sensible defaults
4. **Explicit Failures**: Use `None` for failures, never magic values
5. **No Hidden State**: Tools don't remember previous runs or modify global state
6. **User Orchestrates**: Complex workflows are shell scripts, not hidden automation
7. **Metadata Collection**: Non-core fields go in metadata automatically
8. **Transparent Filtering**: Use `sampler.n_valid_samples` to see samples after filtering
9. **Three Isotonic Mappings**: Global f_all for rewards, cross-fitted f^(-k) for DR, stacked SIMCal for weights
10. **DR via Inheritance**: DR inherits from CalibratedIPS to reuse weight machinery

## 🚨 Code Review Red Flags

- **Overengineering**: Workflow engines, state management, retry logic
- **Hidden coupling**: Tools depending on each other's output formats
- **Magic values**: Using -100.0 or similar as fallbacks
- **Mixed concerns**: Calibration during data generation
- **Multiple responsibilities**: Classes doing more than one thing
- **Unnecessary abstractions**: Code only used once
- **"Smart" tools**: Trying to do everything or hide complexity
- **Hidden state**: Global configuration or stateful libraries

## 🔬 Calibration Strategy

CJE uses three distinct calibration approaches:
1. **Reward Calibration**: Maps judge scores to oracle labels
2. **Weight Calibration**: Stacked SIMCal prevents weight explosion
3. **DR Outcome Models**: Cross-fitted for orthogonality

The implementation details are in `cje/calibration/`.

## 🤖 Doubly Robust (DR) Design

**Key insight**: DR inherits from CalibratedIPS to reuse weight machinery.

Fresh draws are the user's responsibility:
- Generate responses with your pipeline
- Score them with your judge
- Format as simple JSONL
- Attach to estimator with `add_fresh_draws()`

**Asymmetry is intentional**: Logged data is authoritative, fresh draws augment it. We provide the math, you provide the data pipeline.

## 🎨 Design Principles

1. **Do One Thing Well**
   - Each tool solves exactly one problem
   - Composition happens in shell scripts, not library code
   - If you need two things done, use two tools

2. **YAGNI (You Aren't Gonna Need It)**
   - Don't create abstractions for single use cases
   - Inline code that's only called from one place
   - Remove layers that don't add value

3. **Explicit is Better than Implicit**
   - No magic strings or hidden behavior
   - Clear function signatures and return types
   - Obvious data flow

4. **Fail Fast and Clearly**
   - Return None or raise exceptions, never magic values
   - Helpful error messages that guide users
   - Don't hide failures

5. **Users Are Smart**
   - Don't try to "protect" users from complexity
   - Give them the tools, let them build the workflows
   - Document the pieces, not prescriptive processes

## 🚫 What We Don't Do

CJE is a library, not a framework. We explicitly avoid:

1. **Workflow Orchestration**: Use shell scripts, Make, or Airflow - not our job
2. **Retry Logic**: Use systemd, cron, or bash loops - not our job
3. **State Management**: Use files, databases, or queues - not our job
4. **Progress Tracking**: Use tqdm in your scripts - not our job
5. **Configuration Management**: Use environment vars or config files - not our job
6. **Data Validation Workflows**: We validate structure, you validate semantics
7. **End-to-End Pipelines**: We provide pieces, you build pipelines

If you find yourself wanting CJE to "manage" something, stop. That's your job.
The library provides the math and data structures. You provide the glue.

Remember: The goal is to be **simple, correct, and maintainable** - not clever.