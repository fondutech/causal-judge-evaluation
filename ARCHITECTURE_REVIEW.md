# CJE Architecture Review - Post Phase 5 Week 1

## 🔍 Zooming Out: The Big Picture

### What We've Built (100 Python files)
```
cje/
├── calibration/       # ✅ Core calibration (isotonic, judge, SIMCal)
├── data/             # ✅ Models, loaders, samplers
├── diagnostics/      # 🆕 Unified diagnostic system (Phase 5)
├── estimators/       # ✅ IPS, DR, MRDR, TMLE
├── utils/
│   └── diagnostics/  # 📦 Phase 1-4 implementations (7 modules)
├── visualization/    # 🎨 Plotting and dashboards
├── teacher_forcing/  # ✅ Log probability computation
└── experiments/      # 🧪 Arena pipeline
```

### The Journey: 5 Phases of Evolution

**Phase 1**: Core DR Diagnostics ✅
- Hill tail index
- Orthogonality scores
- DM-IPS decomposition

**Phase 2**: Stability & Reliability ✅
- Kendall τ drift detection
- Calibration diagnostics
- EIF normality tests

**Phase 3**: Robust Inference ✅
- Stationary bootstrap (4000 iterations)
- FDR control
- Cluster-robust SEs

**Phase 4**: Automated Gates ✅
- 5 gate types
- Configurable thresholds
- Stop/ship decisions

**Phase 5 Week 1**: Unification ✅
- DiagnosticSuite as single source of truth
- DiagnosticRunner consolidates computation
- Removed redundant tail_ratio_99_5

## 🔬 Zooming In: Current State Analysis

### ✅ What's Working Well

1. **Clean Separation of Concerns**
   - `cje/utils/diagnostics/`: Low-level diagnostic functions
   - `cje/diagnostics/`: High-level orchestration
   - `cje/estimators/`: Integration points

2. **Type Safety**
   - Pydantic models for data structures
   - Dataclasses for diagnostic results
   - Mypy passing on all new code

3. **Backward Compatibility**
   - Legacy IPSDiagnostics still works
   - Old tail_ratio fields deprecated but not removed
   - Existing scripts continue to function

### ⚠️ Integration Gaps Discovered

1. **Incomplete Migration**
   ```bash
   # 58 references to tail_ratio still exist
   grep -r "tail_ratio" cje/ --include="*.py" | wc -l
   # Output: 58
   ```
   - Visualization code still uses tail_ratio
   - Some tests reference old fields
   - Documentation not updated

2. **Orphaned Visualizations**
   - `cje/visualization/` unaware of DiagnosticSuite
   - Dashboard code still expects old diagnostic format
   - Weight plots use deprecated tail_ratio_99_5

3. **Estimator Inconsistency**
   - CalibratedIPS updated ✅
   - DREstimator updated ✅
   - But RawIPS, TMLE not fully integrated
   - Some estimators compute diagnostics twice

4. **CLI Integration Missing**
   - analyze_dataset.py doesn't use DiagnosticSuite
   - No --diagnostic-level flag implemented
   - Gate display hardcoded, not using new display module

### 🔴 Critical Issues

1. **Double Computation**
   ```python
   # In CalibratedIPS._build_diagnostics()
   w_diag = compute_weight_diagnostics(weights)  # Old way
   # Then in BaseCJEEstimator.fit_and_estimate()
   runner.run(self, result)  # Computes again!
   ```

2. **Stability Never Runs**
   ```python
   # DiagnosticConfig defaults:
   check_stability: bool = False  # Never enabled!
   ```
   Users never see drift detection unless manually enabled.

3. **Lost Influence Functions**
   ```python
   # DR estimators store in metadata["dr_influence"]
   # But DiagnosticRunner looks for result.influence_functions
   # Mismatch causes robust inference to fail silently
   ```

## 📊 Usage Patterns Analysis

### Who Uses What
- **analyze_dataset.py**: Uses old IPSDiagnostics + gates
- **Visualizations**: Use old diagnostic format exclusively  
- **Tests**: Mix of old and new patterns
- **DiagnosticSuite**: Only used in new diagnostic module (18 refs)

### Data Flow Issues
```
User → analyze_dataset → Estimator → [Computes diagnostics twice]
                              ↓
                     Old IPSDiagnostics
                              ↓
                     Visualization (broken for new fields)
                              ↓
                     DiagnosticSuite (orphaned)
```

## 🎯 Recommendations

### Immediate Fixes (Week 1 Cleanup)
1. **Remove double computation** in CalibratedIPS
2. **Enable stability by default** in DiagnosticConfig
3. **Fix influence function mismatch** in DR estimators
4. **Update analyze_dataset.py** to use DiagnosticSuite

### Week 2 Priorities (Original Plan)
1. **Update visualization module** to use DiagnosticSuite
2. **Implement CLI verbosity levels**
3. **Create unified dashboard**

### Week 3: Full Integration
1. **Update all estimators** consistently
2. **Migrate all tests** to new patterns
3. **Remove deprecated code paths**
4. **Update documentation**

### Week 4: Polish
1. **Performance optimization** (lazy loading)
2. **Caching strategy** for expensive diagnostics
3. **Historical tracking** for trend analysis

## 📈 Metrics

### Code Quality
- **Duplication**: High (diagnostic computation in 3+ places)
- **Coupling**: Medium (DiagnosticSuite not well integrated)
- **Cohesion**: Good within modules, poor across modules
- **Test Coverage**: ~40% for diagnostic modules

### Performance Impact
- **Bootstrap**: 4000 iterations × N policies = slow
- **Double computation**: 2x overhead on every run
- **No caching**: Recomputes on every access

### User Experience
- **Discoverability**: Poor (features hidden behind flags)
- **Actionability**: Good (recommendations clear)
- **Consistency**: Poor (different diagnostic formats)

## 🤔 Philosophical Questions

1. **Should diagnostics be mandatory or optional?**
   - Currently optional (run_diagnostics=True by default but configurable)
   - Paper suggests they're essential for valid inference
   - Trade-off: Performance vs correctness

2. **How much backward compatibility to maintain?**
   - Currently maintaining all old fields
   - Increases complexity significantly
   - When to deprecate vs remove?

3. **Where should diagnostics live?**
   - Currently split between utils/ and diagnostics/
   - Should consolidate or keep separated?

4. **What's the right abstraction level?**
   - DiagnosticSuite tries to be universal
   - But IPS vs DR have different needs
   - Over-abstraction vs duplication

## 🚀 Next Steps

### Option A: Continue Phase 5 as Planned
- Week 2: Dashboard and CLI
- Week 3: Testing and docs
- Week 4: Deprecation

### Option B: Fix Critical Issues First
- Fix double computation
- Enable stability checks
- Fix influence functions
- Then continue Phase 5

### Option C: Simplify Architecture
- Remove DiagnosticSuite abstraction
- Keep diagnostics in estimators
- Focus on visualization integration

## 💡 Key Insights

1. **Perfect is the enemy of good** - Our pursuit of unified diagnostics created more complexity
2. **Integration > Features** - Orphaned features are worse than missing features  
3. **Gradual migration is hard** - Supporting both old and new increases complexity exponentially
4. **Defaults matter** - Features off by default might as well not exist

## 📝 Recommendation

**Go with Option B**: Fix critical issues first, then continue Phase 5. The architecture is sound but needs proper integration. The unified diagnostic system is the right direction, but we must:

1. Fix the three critical issues immediately
2. Update analyze_dataset.py to be the reference implementation
3. Then proceed with visualization updates
4. Deprecate old code only after new code is fully integrated

The key is making DiagnosticSuite the **actual** source of truth, not just the **intended** source of truth.