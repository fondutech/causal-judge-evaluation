# Current Diagnostic Flow Analysis

## Current State (After DR Fix)

### For DR Estimators (dr-cpo, tmle, mrdr):
1. `dr_base.estimate()` creates DRDiagnostics directly ✅
2. Returns EstimationResult with diagnostics populated
3. `base_estimator.fit_and_estimate()` ALSO runs DiagnosticRunner 
4. DiagnosticRunner creates DiagnosticSuite (duplicate work!)
5. Compat layer tries to create DRDiagnostics from suite (but already exists)
6. Result has BOTH:
   - `result.diagnostics` (DRDiagnostics we created)
   - `result.diagnostic_suite` (DiagnosticSuite from runner)

### For IPS Estimators (raw-ips, calibrated-ips):
1. `estimate()` returns EstimationResult with diagnostics=None
2. `base_estimator.fit_and_estimate()` runs DiagnosticRunner
3. DiagnosticRunner creates DiagnosticSuite  
4. Compat layer creates IPSDiagnostics from suite
5. Result has BOTH:
   - `result.diagnostics` (IPSDiagnostics from compat)
   - `result.diagnostic_suite` (DiagnosticSuite from runner)

## The Problem

We have **THREE diagnostic representations**:
1. **IPSDiagnostics/DRDiagnostics** - What we actually use for display
2. **DiagnosticSuite** - Intermediate representation nobody needs
3. **Metadata dicts** - Legacy format in results

## Safe Removal Strategy

### Phase 1: Make IPS Create Direct Diagnostics (Like DR)
- Modify calibrated_ips.py to create IPSDiagnostics in estimate()
- Modify raw_ips.py to create IPSDiagnostics in estimate()
- Both already have _build_diagnostics() methods!

### Phase 2: Disable DiagnosticRunner in base_estimator
- Add flag to skip DiagnosticRunner if diagnostics already exist
- Or simply remove the DiagnosticRunner code path

### Phase 3: Remove DiagnosticSuite Files
- Delete diagnostics/suite.py
- Delete diagnostics/runner.py  
- Delete data/diagnostics_compat.py
- Update imports

### Phase 4: Clean Up analyze_dataset.py
- Remove suite checks
- Use only result.diagnostics

## Risk Assessment

### Low Risk Changes:
✅ IPS estimators already have _build_diagnostics() methods
✅ DR already works without suite
✅ Display functions work with IPSDiagnostics/DRDiagnostics

### Medium Risk Areas:
⚠️ analyze_dataset.py has some suite checks
⚠️ Some conditional logic based on suite presence

### Mitigation:
- Make changes incrementally
- Test after each phase
- Keep git commits atomic for easy revert

## Decision: Proceed with Phase 1

Start by making IPS estimators create diagnostics directly, just like we did for DR.
This is safe because:
1. The methods already exist (_build_diagnostics)
2. We can test immediately
3. It doesn't break anything (suite still runs)
4. We can verify everything works before removing suite