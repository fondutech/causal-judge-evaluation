# Naming and Organization Proposal

## Current State

### Existing Structure
```
arena_10k_simplified/
├── pipeline_steps/          # Data generation pipeline (existing, working well)
│   ├── prepare_arena_data.py
│   ├── generate_responses.py
│   ├── compute_logprobs.py
│   ├── add_scores_with_resume.py
│   └── prepare_cje_data.py
├── pipeline/                # Analysis pipeline (our new refactoring target)
│   ├── data_loader.py      # ✅ Created
│   ├── reward_handler.py   # ✅ Created
│   └── estimator_factory.py # ✅ Created
└── analyze_dataset.py      # Monolithic file we're refactoring
```

## Problems with Current Approach

1. **Confusing dual "pipeline" directories** - unclear which does what
2. **Inconsistent naming patterns**:
   - `pipeline_steps/` uses verbs: `generate_`, `compute_`, `prepare_`
   - `pipeline/` uses nouns: `data_loader`, `reward_handler`
3. **Unclear purpose** from directory names alone
4. **Module names don't indicate action** (what does `data_loader` do? Load? Validate? Transform?)

## Proposed Reorganization

### Option 1: Clear Separation by Purpose (RECOMMENDED)
```
arena_10k_simplified/
├── data_generation/         # Generate dataset (was pipeline_steps/)
│   ├── prepare_arena_data.py
│   ├── generate_responses.py
│   ├── compute_logprobs.py
│   ├── add_scores.py
│   └── prepare_cje_data.py
├── analysis/                # Analyze dataset (was pipeline/)
│   ├── loading.py          # Load and validate data
│   ├── calibration.py      # Handle rewards and calibration
│   ├── estimation.py       # Create and run estimators
│   ├── results.py          # Format and display results
│   ├── diagnostics.py      # Compute and display diagnostics
│   ├── visualization.py    # Generate plots and dashboards
│   └── export.py           # Export results to various formats
├── analyze.py              # Thin orchestrator (was analyze_dataset.py)
└── generate.py             # Orchestrator for data generation
```

### Option 2: Action-Based Organization
```
arena_10k_simplified/
├── generate/                # All generation tasks
│   ├── arena_data.py
│   ├── responses.py
│   ├── logprobs.py
│   ├── scores.py
│   └── cje_dataset.py
├── analyze/                 # All analysis tasks
│   ├── load_data.py
│   ├── calibrate_rewards.py
│   ├── setup_estimator.py
│   ├── run_estimation.py
│   ├── display_results.py
│   ├── display_diagnostics.py
│   ├── create_plots.py
│   └── export_results.py
├── run_analysis.py
└── run_generation.py
```

### Option 3: Workflow-Based Organization
```
arena_10k_simplified/
├── workflows/
│   ├── generation/
│   │   ├── prepare.py
│   │   ├── generate.py
│   │   ├── score.py
│   │   └── package.py
│   └── analysis/
│       ├── ingest.py
│       ├── process.py
│       ├── estimate.py
│       ├── report.py
│       └── export.py
├── analyze.py
└── generate.py
```

## Recommendation: Option 1

**Why Option 1 is best:**

1. **Clear separation**: `data_generation/` vs `analysis/` immediately tells you the purpose
2. **Preserves working code**: Just rename `pipeline_steps/` → `data_generation/`
3. **Descriptive module names**: `calibration.py` is clearer than `reward_handler.py`
4. **Follows CLAUDE.md principles**: 
   - "Clean Separation: Data generation vs analysis are separate steps"
   - "Do One Thing Well"
5. **Easy migration**: Minimal changes to existing working code

## Module Naming Principles

### Use Nouns for Modules (What it handles)
- ✅ `calibration.py` - Handles calibration
- ✅ `diagnostics.py` - Handles diagnostics  
- ❌ `calibrate_rewards.py` - Verb implies single function
- ❌ `reward_handler.py` - Generic "handler" is vague

### Use Verbs for Functions (What it does)
- ✅ `def load_data()` - Clear action
- ✅ `def create_estimator()` - Clear action
- ❌ `def data_loader()` - Noun as function is awkward
- ❌ `def handle_rewards()` - Generic "handle" is vague

### Be Specific
- ✅ `visualization.py` - Clear domain
- ✅ `export.py` - Clear purpose
- ❌ `utils.py` - Too generic
- ❌ `helpers.py` - Too vague

## Migration Plan

### Phase 1: Rename Existing
```bash
# Rename directories
mv pipeline_steps/ data_generation/
mv pipeline/ analysis/

# Rename existing modules for clarity
mv analysis/data_loader.py analysis/loading.py
mv analysis/reward_handler.py analysis/calibration.py
mv analysis/estimator_factory.py analysis/estimation.py
```

### Phase 2: Update Imports
- Update imports in moved files
- Update any references in analyze_dataset.py
- Update test files

### Phase 3: Continue Extraction
Extract remaining functions into properly named modules:
- `results.py` - Display results
- `diagnostics.py` - Display diagnostics
- `visualization.py` - Generate plots
- `export.py` - Export to CSV/JSON

### Phase 4: Create Thin Orchestrator
Rename and slim down:
- `analyze_dataset.py` → `analyze.py` (~100 lines)
- Create `generate.py` if needed for data generation orchestration

## Benefits of This Approach

1. **Self-documenting structure** - Directory names explain purpose
2. **No confusion** between generation and analysis pipelines
3. **Consistent naming** - Nouns for modules, verbs for functions
4. **Preserves working code** - Minimal changes to data_generation
5. **Clear boundaries** - Each module has one responsibility
6. **Easy to test** - Clear what each module should do
7. **Future-proof** - Easy to add new analysis or generation steps

## Decision Points

1. **Should we do this reorganization now?**
   - PRO: Clean foundation for refactoring
   - CON: More changes before we see results
   - RECOMMENDATION: Yes, do it now (1 hour of work saves days of confusion)

2. **Should we rename analyze_dataset.py to analyze.py?**
   - PRO: Shorter, clearer
   - CON: Breaking change for users
   - RECOMMENDATION: Yes, but keep alias for compatibility

3. **Should we create generate.py orchestrator?**
   - PRO: Symmetry with analyze.py
   - CON: Not needed yet (YAGNI)
   - RECOMMENDATION: No, wait until needed