# CJE Migration Guide: Removing Dangerous Fallbacks

## Overview

This guide shows how to migrate CJE from dangerous fallback values to explicit error handling.

## Core Principles

1. **Never use fallback values** - They corrupt results silently
2. **Always return Result types** - Force explicit error handling
3. **Log everything** - Full context for debugging
4. **Retry intelligently** - Different strategies for different errors
5. **Fail loudly** - Make problems visible immediately

## File-by-File Migration

### 1. `cje/utils/error_handling.py`

```python
# ❌ OLD - REMOVE THESE
FALLBACK_LOG_PROB = -100.0  # Corrupts importance weights!
FALLBACK_PROBABILITY = 1e-10
FALLBACK_SCORE = 0.0
FALLBACK_RESPONSE = "ERROR_FALLBACK_RESPONSE"

def safe_call(func, *args, fallback=None, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return fallback  # Silent corruption!

# ✅ NEW - EXPLICIT HANDLING
@dataclass
class Result:
    success: bool
    value: Optional[Any] = None
    error: Optional[ErrorContext] = None
    
    def unwrap(self) -> Any:
        if not self.success:
            raise RuntimeError(f"Unwrap failed: {self.error}")
        return self.value

@with_retry(max_attempts=3)
def safe_operation(func, *args, **kwargs) -> Result:
    # Returns Result - forces explicit handling
    # Never returns arbitrary values
```

### 2. `cje/loggers/api_policy.py`

```python
# ❌ OLD
def log_prob(self, context: str, response: str) -> float:
    try:
        result = self._compute_via_api(context, response)
        return result
    except Exception as e:
        logger.warning(f"Failed: {e}")
        return 0.0  # WRONG! This means P=1.0!

# ✅ NEW 
def compute_log_prob(self, context: str, response: str) -> LogProbResult:
    for attempt in range(self.max_retries):
        try:
            value = self._compute_via_api(context, response)
            # Validate
            if value > 0:
                raise ValueError(f"Invalid log prob: {value}")
            return LogProbResult(
                status=LogProbStatus.SUCCESS,
                value=value,
                attempts=attempt + 1
            )
        except Exception as e:
            if self._should_retry(e) and attempt < self.max_retries - 1:
                time.sleep(self._get_retry_delay(attempt))
                continue
            # All retries failed
            return LogProbResult(
                status=self._classify_error(e),
                error=str(e),
                attempts=attempt + 1
            )
```

### 3. `cje/loggers/multi_target_sampler.py`

```python
# ❌ OLD
def compute_log_probs(self, context: str, response: str) -> List[float]:
    logps = []
    for runner in self._runners:
        logp = safe_call(
            runner.log_prob,
            context,
            response,
            fallback=FALLBACK_LOG_PROB  # -100.0!
        )
        logps.append(logp)
    return logps

# ✅ NEW
def compute_log_probs(self, context: str, response: str) -> List[LogProbResult]:
    results = []
    for runner in self._runners:
        result = runner.compute_log_prob(context, response)
        results.append(result)
        
        if not result.is_valid:
            logger.warning(
                f"Policy {runner.name} failed: {result.status} - {result.error}"
            )
    return results

def compute_importance_weights(
    self,
    base_idx: int,
    results: List[LogProbResult]
) -> List[Optional[float]]:
    """Compute weights - None for invalid results."""
    base = results[base_idx]
    if not base.is_valid:
        return [None] * len(results)
        
    weights = []
    for i, result in enumerate(results):
        if i == base_idx:
            weights.append(1.0)
        elif result.is_valid:
            log_ratio = np.clip(result.value - base.value, -20, 20)
            weights.append(np.exp(log_ratio))
        else:
            weights.append(None)  # Explicit None
    return weights
```

### 4. `cje/loggers/trajectory_sampler.py`

```python
# ❌ OLD
logp = safe_call(
    runner.log_prob,
    state,
    action,
    fallback=FALLBACK_LOG_PROB
)
logps.append(float(logp or FALLBACK_LOG_PROB))

# ✅ NEW
result = runner.compute_log_prob(state, action)
if result.is_valid:
    logps.append(result.value)
else:
    # Option 1: Skip this trajectory
    logger.warning(f"Skipping trajectory due to: {result.error}")
    continue
    
    # Option 2: Mark as None
    logps.append(None)
    
    # Option 3: Raise exception
    raise RuntimeError(f"Log prob failed: {result.error}")
```

### 5. Usage in Experiments

```python
# ❌ OLD PATTERN
def run_experiment(data):
    for sample in data:
        try:
            logp = compute_log_prob(sample)
        except:
            logp = -100.0  # NEVER DO THIS!
        importance_weight = exp(logp - base_logp)

# ✅ NEW PATTERN  
def run_experiment(data):
    sampler = IdealMultiTargetSampler(policies, base_policy_name="p0")
    
    # Process with proper error handling
    batch_result = sampler.process_batch(data)
    
    # Check results explicitly
    print(f"Complete: {batch_result.num_complete}")
    print(f"Partial: {batch_result.num_partial}")
    print(f"Failed: {batch_result.num_failed}")
    
    # Filter valid results
    valid_results = [
        r for r in batch_result.results 
        if r.all_valid
    ]
    
    # Or handle partially valid
    for result in batch_result.results:
        if result.any_valid:
            # Use what we have
            valid_weights = result.get_valid_weights()
            # ...
```

## Testing the Migration

### 1. Unit Tests

```python
def test_no_fallback_values():
    """Ensure no fallback values are used."""
    policy = APIPolicy("test", "model", mock_client)
    mock_client.fail_next = True
    
    result = policy.compute_log_prob("context", "response")
    
    assert not result.is_valid
    assert result.value is None  # NOT -100.0!
    assert result.error is not None

def test_importance_weights_with_failures():
    """Test weight computation with failures."""
    results = [
        LogProbResult(SUCCESS, value=-10.0),  # Base
        LogProbResult(SUCCESS, value=-12.0),  # Target 1
        LogProbResult(API_ERROR, error="Failed"),  # Target 2
    ]
    
    weights = compute_importance_weights(0, results)
    
    assert weights[0] == 1.0  # Base
    assert weights[1] == exp(2.0)  # Valid
    assert weights[2] is None  # Failed - not some fake value!
```

### 2. Integration Tests

```python
def test_batch_processing_with_failures():
    """Test that batches continue despite failures."""
    sampler = IdealMultiTargetSampler(policies, "base")
    
    # Include some bad data
    samples = [
        ("s1", "context", "response"),
        ("s2", "", ""),  # Empty - will fail
        ("s3", "context", "response"),
    ]
    
    batch_result = sampler.process_batch(samples)
    
    assert batch_result.num_samples == 3
    assert batch_result.num_failed <= 1
    assert batch_result.num_complete >= 2
```

### 3. Smoke Tests

```bash
# Check for any remaining fallback values
grep -r "FALLBACK_LOG_PROB\|= -100\.0\|= 0\.0.*#.*log" cje/

# Check for safe_call with fallback
grep -r "safe_call.*fallback" cje/

# Check for dangerous patterns
grep -r "except.*:\s*return\s*[\-0-9]" cje/
```

## Rollout Plan

### Phase 1: Add New APIs (Week 1)
- Add `LogProbResult` and `Result` types
- Add new methods alongside old ones
- Start using in new code

### Phase 2: Update Core Components (Week 2)
- Update `multi_target_sampler.py`
- Update `api_policy.py`
- Add comprehensive logging

### Phase 3: Remove Old Code (Week 3)
- Delete `FALLBACK_*` constants
- Remove old `safe_call` function
- Update all callers

### Phase 4: Monitor and Fix (Week 4)
- Monitor error rates
- Fix any edge cases
- Update documentation

## Benefits After Migration

1. **Data Integrity**: No more silent corruption
2. **Debuggability**: Clear error messages with context
3. **Reliability**: Smart retries reduce transient failures
4. **Transparency**: Know exactly what failed and why
5. **Type Safety**: Can't accidentally use invalid values

## Common Pitfalls to Avoid

1. **Don't create new fallbacks**
   ```python
   # ❌ BAD
   DEFAULT_LOGP = -50.0
   
   # ✅ GOOD
   result = compute_log_prob(...)
   if not result.is_valid:
       # Handle explicitly
   ```

2. **Don't ignore Results**
   ```python
   # ❌ BAD
   result = compute_log_prob(...)
   value = result.value  # Might be None!
   
   # ✅ GOOD
   result = compute_log_prob(...)
   if result.is_valid:
       value = result.value
   else:
       # Handle failure
   ```

3. **Don't suppress errors**
   ```python
   # ❌ BAD
   try:
       return compute_all_logps()
   except:
       return []  # Hide the error
   
   # ✅ GOOD
   results = compute_all_logps()
   valid_results = [r for r in results if r.is_valid]
   if len(valid_results) < len(results):
       logger.warning(f"{len(results) - len(valid_results)} failures")
   ```

## Success Metrics

After migration, you should see:
- Zero instances of -100.0 or 0.0 in log probability data
- Clear error messages in logs when failures occur
- Ability to trace every failed computation
- No mysterious importance weight explosions
- Happy users who trust their results!

## Questions?

The new design makes it nearly impossible to accidentally corrupt data. If you're unsure about how to handle a specific case, the answer is always: **make failures explicit, never use fallback values**.