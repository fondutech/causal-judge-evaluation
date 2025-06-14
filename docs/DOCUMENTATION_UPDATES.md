# Documentation Updates Needed

This file tracks documentation updates needed to align with recent implementation changes.

## High Priority Updates

### 1. Weight Calibration Bug Fix
- **Location**: `docs/theory/mathematical_foundations.rst` and `docs/guides/weight_processing.rst`
- **Update**: Document that the implementation now accepts small bias (E[w] ≠ 1) to maintain variance control rather than violating cap constraints
- **Reference**: `cje/estimators/calibration.py` changes

### 2. Codebase Simplification Updates (COMPLETED)
- **Status**: ✅ COMPLETED
- **Changes Made**:
  - Removed duplicate CLI (`simple_cli.py`)
  - Removed alternative pipeline (`core.py`)
  - Removed duplicate config system (`simple.py`)
  - Removed unused data loader (`unified_loader.py`)
  - Removed duplicate providers directory
  - Updated all documentation to reflect simplified structure

### 3. API Reference Cleanup (COMPLETED)
- **Status**: ✅ COMPLETED  
- **Removed**:
  - `docs/api/generated/cje.providers.rst`
  - References to all deleted modules

### 4. Removed Components (COMPLETED)
- **Status**: ✅ COMPLETED
- **Confirmed removed**:
  - `cje.utils.aws_secrets` (already removed in previous cleanup)
  - References to `ArenaWorkflowImports` class

## Medium Priority Updates

### 5. Configuration Guide Updates
- **Location**: `docs/guides/configuration_reference.rst`
- **Update**: Add section on simplified configuration using dataclasses
- **Show**: Migration from Hydra configs to simple YAML configs

### 6. Weight Stabilization Methods
- **Location**: `docs/guides/weight_processing.rst`
- **Add**: Documentation for SWITCH and log-exp stabilization methods
- **Reference**: `cje.utils.weight_stabilization.py`

### 7. Visualization Guide
- **Location**: Create new `docs/guides/visualization.rst`
- **Content**: Document new visualization utilities in `cje.results.visualization.py`

## Low Priority Updates

### 8. Examples Documentation
- **Update**: `docs/tutorials/` to include examples using simplified API
- **Add**: Simple example using `CJEConfig` and `run_cje()`

### 9. CLI Documentation
- **Location**: `docs/guides/user_guide.rst`
- **Add**: Section on simplified CLI commands (init, validate, etc.)

## Code Examples to Add

### Simple Configuration Example
```python
from cje.config.simple import CJEConfig, PolicyConfig

config = CJEConfig(
    logging_policy=PolicyConfig(name="baseline", model_name="gpt-3.5-turbo"),
    target_policies=[
        PolicyConfig(name="improved", model_name="gpt-4"),
    ],
)
```

### Unified Provider Example
```python
from cje.providers.unified import UnifiedProvider

provider = UnifiedProvider("openai", "gpt-4")
response, logp = provider.complete("Hello", return_logprobs=True)
```

### Visualization Example
```python
from cje.results.visualization import plot_policy_comparison

fig = plot_policy_comparison(result, ["baseline", "improved"])
fig.savefig("comparison.png")
```

## Notes
- All mypy type checking now passes
- Pre-commit hooks are configured but may need AWS CodeArtifact auth
- The simplified API is in beta and may change