# Analyze Dataset Refactoring Plan

## Current State Analysis

### The Problem
`analyze_dataset.py` has grown to 1,157 lines and handles 12+ distinct responsibilities:
1. Command-line argument parsing
2. Data loading and validation  
3. Oracle label masking for partial coverage
4. Reward calibration and assignment
5. Estimator factory and configuration
6. Fresh draws addition for DR estimators
7. Estimation execution
8. Results display and formatting
9. Diagnostic interpretation and display
10. Visualization generation (plots and dashboard)
11. Export to various formats
12. Error handling and recovery

This violates the Unix philosophy from CLAUDE.md: "Do one thing well."

### Already Completed
✅ Created `pipeline/` directory structure
✅ Extracted `data_loader.py` - handles data loading only
✅ Extracted `reward_handler.py` - handles rewards/calibration only

## Architectural Vision

### Design Principles (from CLAUDE.md)
1. **Do One Thing Well**: Each module has exactly one responsibility
2. **Explicit is Better**: Clear interfaces, no hidden behavior
3. **Composition Over Frameworks**: Tools compose naturally
4. **User Orchestrates**: Complex workflows are user's responsibility
5. **YAGNI**: Don't build what isn't needed

### Target Architecture
```
analyze_dataset.py (thin orchestrator ~100 lines)
    ├── pipeline/
    │   ├── data_loader.py        ✅ [DONE] Load and validate data
    │   ├── reward_handler.py      ✅ [DONE] Handle rewards/calibration
    │   ├── estimator_factory.py   [TODO] Create and configure estimators
    │   ├── fresh_draws.py         [TODO] Add fresh draws to DR estimators
    │   ├── runner.py              [TODO] Execute estimation
    │   ├── results_formatter.py   [TODO] Format and display results
    │   ├── diagnostics_display.py [TODO] Display diagnostics
    │   ├── visualizer.py          [TODO] Generate plots and dashboard
    │   └── exporter.py            [TODO] Export to CSV/JSON
    └── cli.py                     [TODO] Command-line interface
```

## Implementation Phases

### Phase 1: Extract Core Components (Current)
Extract remaining functions into pipeline modules without changing interfaces.

**1.1 Extract Estimator Factory** (`estimator_factory.py`)
```python
def create_estimator(args, sampler, cal_result=None):
    """Create estimator based on args."""
    # Move setup_estimator() logic here
    
def add_fresh_draws(estimator, args, sampler):
    """Add fresh draws to DR estimator if needed."""
    # Move add_fresh_draws_to_estimator() logic here
```

**1.2 Extract Results Display** (`results_formatter.py`)
```python
def format_results(results, args, config):
    """Format estimation results for display."""
    # Move display_results() and compute_base_statistics()
```

**1.3 Extract Diagnostics Display** (`diagnostics_display.py`)
```python
def display_diagnostics(results, args):
    """Display diagnostic information."""
    # Move display_weight_diagnostics() and display_dr_diagnostics()
```

**1.4 Extract Visualization** (`visualizer.py`)
```python
def generate_plots(dataset, results, args):
    """Generate visualization plots."""
    # Move plot generation logic
    
def generate_dashboard(dataset, results, args):
    """Generate HTML dashboard."""
    # Move dashboard generation logic
```

**1.5 Extract Export** (`exporter.py`)
```python
def export_results(results, args):
    """Export results to CSV/JSON."""
    # Move export logic
```

**1.6 Extract Runner** (`runner.py`)
```python
def run_estimation(estimator, args):
    """Run estimation and handle errors."""
    # Core estimation execution
```

### Phase 2: Refactor Main Orchestrator
Reduce analyze_dataset.py to ~100 lines that just orchestrates the pipeline:

```python
def main():
    # 1. Parse arguments
    args = parse_args()
    
    # 2. Load data
    dataset = pipeline.load_data(args.data)
    
    # 3. Handle rewards
    dataset, cal_result = pipeline.handle_rewards(dataset, args)
    
    # 4. Create estimator
    estimator = pipeline.create_estimator(args, dataset, cal_result)
    
    # 5. Add fresh draws if needed
    pipeline.add_fresh_draws(estimator, args)
    
    # 6. Run estimation
    results = pipeline.run_estimation(estimator)
    
    # 7. Display results
    pipeline.display_results(results, args)
    
    # 8. Generate visualizations
    if not args.no_plots:
        pipeline.generate_visualizations(dataset, results, args)
    
    # 9. Export if requested
    if args.export:
        pipeline.export_results(results, args)
```

### Phase 3: Clean Architecture (Optional Future)
- Move CLI parsing to separate `cli.py`
- Create `config.py` for configuration management
- Add `__main__.py` for package execution
- Consider plugin architecture for custom estimators

## Key Decisions

### 1. Module Boundaries
**Decision**: Strict single responsibility per module
- No module does two things
- No "utility" modules with mixed concerns
- Each module exports 1-3 main functions

### 2. Data Flow
**Decision**: Explicit parameter passing
- No global state
- No hidden dependencies
- All data flows through function parameters

### 3. Error Handling
**Decision**: Let errors bubble up
- Modules raise descriptive errors
- Orchestrator handles recovery/retry
- No silent failures

### 4. Configuration
**Decision**: Pass config explicitly
- No global config object
- Args passed down as needed
- Each module validates its own config

### 5. Testing Strategy
**Decision**: Test modules independently
- Each module has its own test file
- Mock dependencies for unit tests
- Integration test for full pipeline

## Migration Strategy

### Step 1: Extract Without Breaking (NOW)
- Copy functions to new modules
- Import from new modules in analyze_dataset.py
- Verify nothing breaks

### Step 2: Clean Up Interfaces
- Simplify function signatures
- Remove unnecessary parameters
- Add type hints

### Step 3: Refactor Orchestrator
- Replace inline code with module calls
- Remove extracted functions
- Simplify main flow

### Step 4: Add Tests
- Unit tests for each module
- Integration test for pipeline
- Regression tests for key scenarios

## Success Metrics

1. **Line Count**: analyze_dataset.py < 150 lines
2. **Module Size**: Each module < 300 lines
3. **Single Responsibility**: Each module does ONE thing
4. **Test Coverage**: >80% for critical paths
5. **No Regressions**: All existing functionality works

## Risks and Mitigations

### Risk 1: Breaking Existing Workflows
**Mitigation**: Keep exact same CLI interface, test extensively

### Risk 2: Performance Regression
**Mitigation**: Profile before/after, optimize only if needed

### Risk 3: Over-Engineering
**Mitigation**: Follow YAGNI, only extract what exists

### Risk 4: Losing Context
**Mitigation**: Preserve existing comments, add module docstrings

## Next Immediate Steps

1. ✅ Create planning document (THIS FILE)
2. Extract estimator_factory.py
3. Extract results_formatter.py
4. Extract diagnostics_display.py
5. Test each extraction
6. Continue with remaining modules

## Notes

- Keep imports at module level (not function level) for clarity
- Use consistent naming: `handle_X`, `create_Y`, `display_Z`
- Each module should be independently testable
- Don't create abstractions for single use cases (YAGNI)
- Preserve all existing functionality exactly