# Documentation Updates Needed

This file tracks documentation updates needed to align with recent implementation changes.

## High Priority Updates

### 1. Weight Calibration Bug Fix
- **Location**: `docs/theory/mathematical_foundations.rst` and `docs/guides/weight_processing.rst`
- **Update**: Document that the implementation now accepts small bias (E[w] ≠ 1) to maintain variance control rather than violating cap constraints
- **Reference**: `cje/estimators/calibration.py` changes

### 2. Simplified API Documentation
- **Location**: Create new `docs/guides/simplified_api.rst`
- **Content**:
  - Document `cje.core.run_cje()` function
  - Document `cje.config.simple.CJEConfig` dataclass
  - Document `cje.providers.unified.UnifiedProvider`
  - Document `cje.cli.simple_cli` commands
  - Example usage patterns

### 3. API Reference Updates
- **Add new modules**:
  - `cje.core`
  - `cje.config.simple`
  - `cje.providers.unified`
  - `cje.data.unified_loader`
  - `cje.estimators.base_crossfit`
  - `cje.results.visualization`
  - `cje.utils.weight_stabilization`
  - `cje.cli.simple_cli`

### 4. Removed Components
- **Remove references to**:
  - `cje.utils.aws_secrets` (removed)
  - `cje.utils.simple_errors` (removed)
  - `JudgeConfigError` (replaced with ValueError)
  - `ArenaWorkflowImports` class

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