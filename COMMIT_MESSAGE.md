refactor: Remove unused gates system and simplify diagnostics

## Summary
Removed 1,500+ lines of unused diagnostic gates code and simplified the
diagnostic system. The gates feature was never enabled by default and 
provided no value to users.

## Changes

### Removed (1,500+ lines)
- Deleted `cje/utils/diagnostics/gates.py` (785 lines)
- Deleted `cje/tests/test_gates.py` (541 lines)  
- Removed gate-related parameters from all estimators
- Removed gate-related CLI arguments from analyze_dataset.py
- Removed unused DiagnosticSuite fields: `gate_report`, `computation_time`, `estimator_type`

### Simplified
- Removed `run_gates` and `gate_config` parameters throughout
- Removed `_run_gates()` and `_suite_to_gate_format()` from DiagnosticRunner
- Removed `_run_diagnostic_gates()` from base_estimator
- Cleaned up imports and references

### Fixed
- Stability diagnostics now enabled by default (was disabled)
- Removed double computation of diagnostics
- Added Hill tail index to replace deprecated tail_ratio_99_5
- Fixed influence function access patterns

### Documentation
- Added comprehensive documentation of the refactor journey
- Documented lessons learned about over-engineering
- Created architecture decision records

## Impact
- ✅ Zero user impact (feature was never used)
- ✅ Simpler, cleaner codebase
- ✅ Reduced maintenance burden
- ✅ All tests pass

## Lessons Learned
This refactor revealed that we only needed 35 lines of simple fixes
instead of the 2,500+ lines of complex systems we initially built.
Key lesson: YAGNI (You Aren't Gonna Need It) is usually right.

Fixes: None (pure refactor)
Breaking changes: None (unused feature)