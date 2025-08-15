# Contributing to CJE

Thank you for your interest in contributing to CJE! This document outlines our development philosophy, standards, and processes.

## 🎯 Core Philosophy

We follow the Unix philosophy: **Do One Thing Well**. Before contributing, please read [CLAUDE.md](CLAUDE.md) to understand our architectural principles.

### What We Value
- ✅ **Simplicity** over cleverness
- ✅ **Explicit** over implicit
- ✅ **Composable** over monolithic
- ✅ **Clear errors** over magic fallbacks
- ✅ **YAGNI** - You Aren't Gonna Need It

### What We Avoid
- ❌ Overengineering and unnecessary abstractions
- ❌ Hidden coupling between components
- ❌ Magic values (-100.0, -999, etc.)
- ❌ "Smart" tools that hide complexity
- ❌ Framework-style orchestration

## 🛠️ Development Setup

```bash
# Clone the repository
git clone https://github.com/causal-judge-evaluation/cje.git
cd cje

# Install with poetry
poetry install

# Install pre-commit hooks
poetry run pre-commit install

# Run tests to verify setup
poetry run pytest
```

## 📝 Code Standards

### Type Safety
All code must be type-safe:
```python
# Good ✅
def compute_weights(scores: np.ndarray, target_mean: float = 1.0) -> np.ndarray:
    ...

# Bad ❌
def compute_weights(scores, target_mean=1.0):
    ...
```

### Error Handling
Be explicit about failures:
```python
# Good ✅
if not result.is_valid:
    return None  # Clear failure signal

# Bad ❌
if not result.is_valid:
    return -100.0  # Magic value
```

### Pydantic Models
Use Pydantic for data validation:
```python
# Good ✅
class EstimationResult(BaseModel):
    estimates: List[float]
    standard_errors: List[float]
    
    @validator('estimates')
    def check_finite(cls, v):
        if not all(np.isfinite(v)):
            raise ValueError("Estimates must be finite")
        return v
```

### Single Responsibility
Each function/class should do ONE thing:
```python
# Good ✅
def calibrate_weights(weights: np.ndarray) -> np.ndarray:
    """Just calibrate weights."""
    ...

def compute_variance(weights: np.ndarray) -> float:
    """Just compute variance."""
    ...

# Bad ❌
def calibrate_and_analyze_weights(weights: np.ndarray) -> Dict[str, Any]:
    """Does calibration, variance, diagnostics, plotting..."""
    ...
```

## 🧪 Testing Requirements

### All Changes Need Tests
- Every new feature must have tests
- Every bug fix must have a regression test
- Maintain or increase coverage (currently >80%)

### Test Structure
```python
def test_specific_behavior():
    """Test one specific behavior."""
    # Arrange
    data = create_test_data()
    
    # Act
    result = function_under_test(data)
    
    # Assert
    assert result.value == expected_value
    assert result.status == LogProbStatus.SUCCESS
```

### Running Tests
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest cje/tests/test_calibration.py

# Run with coverage
poetry run pytest --cov=cje --cov-report=html

# Run only fast tests
poetry run pytest -m "not slow"
```

## 📋 Pull Request Process

### 1. Before Opening a PR

- [ ] Read [CLAUDE.md](CLAUDE.md) 
- [ ] Ensure your code follows our philosophy
- [ ] All tests pass (`poetry run pytest`)
- [ ] Type checking passes (`poetry run mypy cje`)
- [ ] Code is formatted (`poetry run black cje`)
- [ ] Documentation is updated if needed

### 2. PR Guidelines

#### Title
Use conventional commit format:
- `feat: Add new estimator`
- `fix: Correct variance calculation`
- `docs: Update README`
- `refactor: Simplify calibration logic`
- `test: Add regression test for edge case`
- `perf: Optimize weight computation`

#### Description Template
```markdown
## Summary
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Follows YAGNI principle
- [ ] No magic values
- [ ] Single responsibility maintained
```

### 3. Code Review Criteria

Your PR will be reviewed for:

1. **Correctness**: Does it work as intended?
2. **Simplicity**: Is it as simple as possible?
3. **Philosophy**: Does it follow our principles?
4. **Tests**: Are edge cases covered?
5. **Documentation**: Are public APIs documented?

## 🚫 What NOT to Submit

### Don't Add These
- ❌ Workflow orchestration (use shell scripts)
- ❌ Retry logic (user's responsibility)
- ❌ State management systems
- ❌ Configuration frameworks
- ❌ Unnecessary abstractions
- ❌ "Clever" one-liners that sacrifice clarity

### Red Flags in Code Review
- Multiple responsibilities in one class/function
- Hidden dependencies between components
- Magic values instead of explicit None/errors
- Overly generic abstractions used once
- Complex inheritance hierarchies
- Global state or singletons

## 💡 Specific Contributions We Want

### High Priority
- Performance optimizations (with benchmarks)
- Additional outcome models for DR estimators
- Better diagnostic visualizations
- Notebook examples for common use cases
- Support for additional judge formats

### Medium Priority
- Additional statistical tests
- Alternative weight calibration methods
- Integration with popular ML frameworks
- CLI improvements

### Low Priority (Probably YAGNI)
- Web UI
- Database integrations
- Distributed computing support
- Real-time monitoring

## 📚 Documentation

### Docstring Format
```python
def calibrate_weights(
    weights: np.ndarray,
    target_mean: float = 1.0,
    variance_cap: Optional[float] = None
) -> np.ndarray:
    """Calibrate importance weights using isotonic regression.
    
    Args:
        weights: Raw importance weights
        target_mean: Target mean for calibration (default: 1.0)
        variance_cap: Optional variance cap for stability
        
    Returns:
        Calibrated weights with target mean
        
    Raises:
        ValueError: If weights contain NaN or negative values
    """
```

### When to Add Documentation
- Public APIs: Always
- Private methods: When non-obvious
- Complex algorithms: Add references to papers
- Magic numbers: Explain why this specific value

## 🔒 Security

- Never commit API keys or secrets
- Don't add code that could be used maliciously
- Sanitize any user inputs
- Be cautious with file system operations

## 🤝 Getting Help

- Open an issue for bugs
- Discussions for design questions
- Check existing issues before creating new ones
- Include minimal reproducible examples

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## ⚙️ Repository Settings (For Maintainers)

### Branch Protection Rules for `main`
- ✅ Require pull request reviews (1 approval minimum)
- ✅ Dismiss stale PR approvals when new commits are pushed
- ✅ Require status checks to pass:
  - `pytest`
  - `mypy`
  - `black`
- ✅ Require branches to be up to date before merging
- ✅ Include administrators in restrictions
- ❌ Do NOT auto-delete head branches (keep history)

### GitHub Actions Required Checks
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install poetry
      - run: poetry install
      - run: poetry run pytest
      - run: poetry run mypy cje
      - run: poetry run black --check cje
```

### Merge Settings
- ✅ Allow squash merging (default)
- ✅ Default to PR title for squash commits
- ❌ Disable merge commits (keep history clean)
- ❌ Disable rebase merging (avoid rewriting history)

### Issue Templates
Create `.github/ISSUE_TEMPLATE/bug_report.md`:
```markdown
---
name: Bug Report
about: Report incorrect behavior
labels: bug
---

**Describe the bug**
Clear description of what went wrong

**To Reproduce**
```python
# Minimal code to reproduce
```

**Expected behavior**
What should happen

**Actual behavior**
What actually happens

**Environment:**
- CJE version:
- Python version:
- OS:
```

### PR Template
Create `.github/pull_request_template.md`:
```markdown
## Summary
<!-- Brief description -->

## Motivation
<!-- Why is this needed? -->

## Changes
- 
- 

## Testing
<!-- How was this tested? -->

## Checklist
- [ ] Tests pass (`poetry run pytest`)
- [ ] Type checking passes (`poetry run mypy cje`)
- [ ] Code formatted (`poetry run black cje`)
- [ ] Follows YAGNI principle
- [ ] No magic values
- [ ] Single responsibility maintained
- [ ] Documentation updated if needed
```

## 🙏 Thank You!

We appreciate your contributions to making CJE better. Remember: simple, correct, and maintainable always beats clever and complex.