# Telemetry Implementation: Quick Start

## Immediate Changes (5 Minutes)

### 1. Add to base_estimator.py

```python
# In cje/estimators/base_estimator.py, line 86-87
# Add these lines before the compatibility layer code:

            # Populate legacy fields for backward compatibility
            if not hasattr(result, "diagnostics") or result.diagnostics is None:
+               # Telemetry and deprecation warning
+               import logging
+               import warnings
+               logger = logging.getLogger(__name__)
+               
+               # Log usage for analysis
+               logger.info(
+                   "TELEMETRY: Legacy diagnostics requested | "
+                   f"estimator={self.__class__.__name__} | "
+                   f"has_suite={result.diagnostic_suite is not None}"
+               )
+               
+               # Warn users about deprecation
+               warnings.warn(
+                   "Accessing result.diagnostics is deprecated and will be removed in v2.0.0. "
+                   "Please use result.diagnostic_suite instead. "
+                   "See migration guide: https://github.com/cje/docs/migration.md",
+                   DeprecationWarning,
+                   stacklevel=3
+               )
+               
                # Create legacy diagnostics from suite
                from ..data.diagnostics_compat import (
```

### 2. Add to analyze_dataset.py

```python
# In cje/experiments/arena_10k_simplified/analyze_dataset.py
# Add at the top of display_dr_diagnostics() function (line 505):

def display_dr_diagnostics(results: Any, args: Any) -> None:
    """Display DR diagnostics if available."""
+   # Telemetry: Track which diagnostic path is used
+   import logging
+   logger = logging.getLogger(__name__)
+   
+   has_suite = hasattr(results, "diagnostic_suite") and results.diagnostic_suite is not None
+   has_legacy = hasattr(results, "diagnostics") and results.diagnostics is not None
+   logger.debug(f"TELEMETRY: DR display | suite={has_suite} | legacy={has_legacy}")
    
    # MINIMAL CHANGE: Check for DiagnosticSuite first (new path)
    if hasattr(results, "diagnostic_suite") and results.diagnostic_suite is not None:
```

### 3. Environment Variable Control

```python
# In cje/estimators/base_estimator.py, at the top of the file:
import os

# Control compatibility behavior via environment
LEGACY_DIAGNOSTICS_ENABLED = os.getenv("CJE_LEGACY_DIAGNOSTICS", "true").lower() == "true"
LEGACY_DIAGNOSTICS_WARN = os.getenv("CJE_LEGACY_WARN", "true").lower() == "true"

# Then in fit_and_estimate(), line 86:
            # Populate legacy fields for backward compatibility
-           if not hasattr(result, "diagnostics") or result.diagnostics is None:
+           if LEGACY_DIAGNOSTICS_ENABLED and (not hasattr(result, "diagnostics") or result.diagnostics is None):
+               if LEGACY_DIAGNOSTICS_WARN:
+                   # Show warnings
+               # Create legacy diagnostics...
```

## Telemetry Collection Script

Create `scripts/analyze_telemetry.py`:

```python
#!/usr/bin/env python
"""Analyze diagnostic telemetry from logs."""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

def parse_logs(log_file):
    """Parse telemetry from log file."""
    
    # Patterns to match
    patterns = {
        'legacy_requested': r'TELEMETRY: Legacy diagnostics requested',
        'estimator_type': r'estimator=(\w+)',
        'has_suite': r'has_suite=(True|False)',
        'dr_display': r'TELEMETRY: DR display.*suite=(True|False).*legacy=(True|False)',
        'suite_used': r'TELEMETRY: Using DiagnosticSuite path',
        'legacy_used': r'TELEMETRY: Using legacy diagnostics path',
    }
    
    stats = defaultdict(Counter)
    
    with open(log_file) as f:
        for line in f:
            if 'TELEMETRY' in line:
                # Track legacy requests
                if 'Legacy diagnostics requested' in line:
                    stats['legacy_requests']['total'] += 1
                    
                    # Extract estimator type
                    match = re.search(patterns['estimator_type'], line)
                    if match:
                        stats['legacy_by_estimator'][match.group(1)] += 1
                
                # Track display paths
                if 'DR display' in line:
                    match = re.search(patterns['dr_display'], line)
                    if match:
                        suite_available = match.group(1) == 'True'
                        legacy_available = match.group(2) == 'True'
                        
                        if suite_available and legacy_available:
                            stats['display_paths']['both_available'] += 1
                        elif suite_available:
                            stats['display_paths']['suite_only'] += 1
                        elif legacy_available:
                            stats['display_paths']['legacy_only'] += 1
    
    return stats

def print_report(stats):
    """Print telemetry report."""
    
    print("=" * 60)
    print("DIAGNOSTIC TELEMETRY REPORT")
    print("=" * 60)
    
    # Legacy usage
    total_legacy = stats['legacy_requests']['total']
    print(f"\nLegacy Diagnostics Requests: {total_legacy}")
    
    if stats['legacy_by_estimator']:
        print("\nBy Estimator Type:")
        for estimator, count in sorted(stats['legacy_by_estimator'].items()):
            pct = (count / total_legacy * 100) if total_legacy > 0 else 0
            print(f"  {estimator:20} {count:5} ({pct:5.1f}%)")
    
    # Display paths
    if stats['display_paths']:
        print("\nDisplay Path Availability:")
        total_displays = sum(stats['display_paths'].values())
        for path, count in sorted(stats['display_paths'].items()):
            pct = (count / total_displays * 100) if total_displays > 0 else 0
            print(f"  {path:20} {count:5} ({pct:5.1f}%)")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if total_legacy == 0:
        print("✅ No legacy usage detected - safe to remove compatibility layer!")
    elif total_legacy < 10:
        print("⚠️  Low legacy usage - consider accelerated removal timeline")
    else:
        print("❌ Significant legacy usage - maintain compatibility layer")
        print("   Focus on migrating these estimators:")
        for estimator, count in sorted(stats['legacy_by_estimator'].items(), 
                                      key=lambda x: x[1], reverse=True)[:3]:
            print(f"   - {estimator} ({count} uses)")

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "cje.log"
    
    if not Path(log_file).exists():
        print(f"Log file not found: {log_file}")
        print("\nTo enable telemetry logging:")
        print("  export CJE_LOG_LEVEL=INFO")
        print("  python analyze_dataset.py ... 2>&1 | tee cje.log")
        sys.exit(1)
    
    stats = parse_logs(log_file)
    print_report(stats)
```

## Usage Instructions

### 1. Enable Telemetry Logging

```bash
# Run with logging enabled
export CJE_LOG_LEVEL=INFO
python analyze_dataset.py --data data.jsonl --estimator tmle 2>&1 | tee run.log

# Or with Python logging config
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cje_telemetry.log'),
        logging.StreamHandler()
    ]
)
```

### 2. Collect Data for 2-4 Weeks

```bash
# Aggregate logs periodically
cat logs/week1/*.log > telemetry_week1.log
python scripts/analyze_telemetry.py telemetry_week1.log
```

### 3. Make Data-Driven Decision

Based on telemetry results:

| Legacy Usage | Recommendation | Timeline |
|--------------|---------------|----------|
| 0-5% | Remove immediately | 1 week |
| 5-20% | Accelerated removal | 4 weeks |
| 20-50% | Standard removal | 8-12 weeks |
| >50% | Keep permanently | ∞ |

## Testing the Telemetry

```python
# Quick test script
from cje import load_dataset_from_jsonl, PrecomputedSampler, CalibratedIPS
import logging

logging.basicConfig(level=logging.INFO)

dataset = load_dataset_from_jsonl("test_data.jsonl")
sampler = PrecomputedSampler(dataset)
estimator = CalibratedIPS(sampler)
result = estimator.fit_and_estimate()

# This should trigger telemetry if accessing legacy format
if result.diagnostics:
    print("Legacy diagnostics available")
    # Should see: "TELEMETRY: Legacy diagnostics requested"
    # Should see: DeprecationWarning
```

## Gradual Rollout

### Week 1: Silent Telemetry
```python
logger.info("TELEMETRY: ...")  # Just log, no warnings
```

### Week 2: Soft Warnings
```python
if SHOW_WARNINGS:
    warnings.warn(..., DeprecationWarning)  # Opt-in warnings
```

### Week 3: Default Warnings
```python
warnings.warn(..., DeprecationWarning)  # Everyone sees warnings
```

### Week 4: Analyze and Decide
```bash
python scripts/analyze_telemetry.py all_logs.log
# Make go/no-go decision based on data
```

## Key Benefits

1. **Zero Risk**: Just logging, no behavior change
2. **Real Data**: Actual usage patterns, not guesses
3. **Gradual**: Can roll back at any point
4. **Transparent**: Users see deprecation timeline
5. **Automated**: Script analyzes logs automatically

## Next Steps

1. **Add telemetry code** (5 minutes)
2. **Deploy to dev/staging** (test for 1 day)
3. **Deploy to production** (collect for 2-4 weeks)
4. **Analyze results** (make decision)
5. **Proceed or pivot** based on data

This approach ensures we make informed decisions based on actual usage rather than assumptions.