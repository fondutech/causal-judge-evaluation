Diagnostics and Gates
=====================

CJE provides comprehensive diagnostics to audit assumptions and detect potential issues. This guide explains how to interpret these diagnostics and use them as quality gates.

Overview
--------

Every CJE estimator returns diagnostic information through the EstimationResult:

.. code-block:: python

   from cje import analyze_dataset
   
   results = analyze_dataset("data.jsonl", estimator="calibrated-ips")
   
   # Access diagnostics for IPS-based estimators
   if hasattr(results, 'diagnostics') and results.diagnostics:
       print(results.diagnostics.summary())  # Human-readable summary
   
   # Access detailed per-policy info from metadata
   if 'diagnostics' in results.metadata:
       policy_diag = results.metadata['diagnostics']['policy_name']

Diagnostics are organized into categories based on estimator type:

1. **IPSDiagnostics**: For RawIPS and CalibratedIPS (weight diagnostics, ESS, calibration)
2. **DRDiagnostics**: For DR-based estimators (orthogonality, influence functions, cross-fit R²)
3. **Metadata diagnostics**: Additional info stored in results.metadata

Weight Diagnostics
------------------

Effective Sample Size (ESS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most important weight diagnostic. ESS measures how many "effective" samples remain after importance weighting:

.. math::

   ESS = \frac{(\sum w_i)^2}{\sum w_i^2}

**Interpretation:**

.. list-table::
   :header-rows: 1

   * - ESS Fraction
     - Status
     - Interpretation
     - Action
   * - >20%
     - 🟢 Green
     - Excellent overlap
     - Proceed confidently
   * - 10-20%
     - 🟡 Amber
     - Adequate overlap
     - Monitor, consider more data
   * - 5-10%
     - 🟠 Orange
     - Marginal overlap
     - Use with caution
   * - <5%
     - 🔴 Red
     - Poor overlap
     - Do not trust results

**Example:**

.. code-block:: python

   # Check ESS for each policy
   for policy in results.metadata['target_policies']:
       diag = results.metadata['diagnostics'][policy]
       ess_frac = diag['weights']['ess_fraction']
       
       if ess_frac < 0.05:
           print(f"⚠️ WARNING: {policy} has ESS={ess_frac:.1%}")
           print("   Results may be unreliable!")

Weight Distribution Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key statistics about the importance weight distribution:

.. code-block:: python

   weight_stats = diagnostics['policy_name']['weights']
   
   print(f"Max weight: {weight_stats['max_weight']:.1f}")
   print(f"Weight CV: {weight_stats['cv']:.2f}")  # Coefficient of variation
   print(f"Tail ratio: {weight_stats['tail_ratio']:.1f}")  # top 5% / bottom 95%

**Red Flags:**
- Max weight > 100
- CV > 5
- Tail ratio > 10

Tail Index (Hill Estimator)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimates the tail heaviness of the weight distribution:

.. math::

   \hat{\alpha} = \frac{1}{k} \sum_{i=1}^k \log\left(\frac{w_{(i)}}{w_{(k+1)}}\right)

Where α is the tail index (lower = heavier tails).

**Interpretation:**

.. list-table::
   :header-rows: 1

   * - Tail Index
     - Tail Type
     - Implication
   * - α > 3
     - Light tails
     - Stable estimation
   * - 2 < α ≤ 3
     - Moderate tails
     - Some instability
   * - 1 < α ≤ 2
     - Heavy tails
     - High variance
   * - α ≤ 1
     - Very heavy tails
     - Infinite variance

Extreme Weight Detection
~~~~~~~~~~~~~~~~~~~~~~~~

Identifies samples with extreme importance weights:

.. code-block:: python

   # From experiment_config.py thresholds
   EXTREME_HIGH = 100.0
   EXTREME_LOW = 0.01
   
   extreme_samples = []
   for i, w in enumerate(weights):
       if w > EXTREME_HIGH or w < EXTREME_LOW:
           extreme_samples.append({
               'index': i,
               'weight': w,
               'prompt': dataset.samples[i].prompt[:50]
           })
   
   if extreme_samples:
       print(f"Found {len(extreme_samples)} extreme weights")
       for s in extreme_samples[:5]:  # Show first 5
           print(f"  Sample {s['index']}: w={s['weight']:.2f}")

Calibration Diagnostics
-----------------------

Judge-Oracle Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~

Measures how well judge scores predict oracle labels:

.. code-block:: python

   cal_diag = results.metadata.get('calibration_diagnostics', {})
   
   print(f"Kendall τ: {cal_diag['kendall_tau']:.3f}")
   print(f"Spearman ρ: {cal_diag['spearman_rho']:.3f}")
   print(f"R²: {cal_diag['r_squared']:.3f}")

**Quality Thresholds:**
- τ > 0.7: Excellent judge
- 0.5 < τ ≤ 0.7: Good judge
- 0.3 < τ ≤ 0.5: Moderate judge
- τ ≤ 0.3: Poor judge

Isotonic Calibration Quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluates the isotonic regression fit:

.. code-block:: python

   # Check calibration monotonicity
   violations = cal_diag.get('monotonicity_violations', 0)
   if violations > 0:
       print(f"⚠️ {violations} monotonicity violations detected")
   
   # Check calibration coverage
   coverage = cal_diag.get('oracle_coverage', 0)
   print(f"Oracle coverage: {coverage:.1%} of data")
   
   # Check extrapolation warnings
   extrap_frac = cal_diag.get('extrapolation_fraction', 0)
   if extrap_frac > 0.1:
       print(f"⚠️ {extrap_frac:.1%} of predictions require extrapolation")

Calibration Reliability Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualize calibration quality:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Get calibration curve data
   judge_bins = cal_diag['reliability_curve']['judge_bins']
   oracle_means = cal_diag['reliability_curve']['oracle_means']
   calibrated_means = cal_diag['reliability_curve']['calibrated_means']
   
   plt.figure(figsize=(8, 6))
   plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
   plt.plot(judge_bins, oracle_means, 'o-', label='Raw judge')
   plt.plot(judge_bins, calibrated_means, 's-', label='Calibrated')
   plt.xlabel('Judge Score')
   plt.ylabel('Oracle Label')
   plt.legend()
   plt.title('Calibration Reliability Diagram')

SIMCal Diagnostics
~~~~~~~~~~~~~~~~~~

Specific to weight calibration:

.. code-block:: python

   simcal_diag = policy_diag.get('simcal', {})
   
   # Variance reduction achieved
   var_reduction = simcal_diag.get('variance_reduction', 0)
   print(f"Variance reduction: {var_reduction:.1%}")
   
   # Stacking weights (how candidates were combined)
   stacking = simcal_diag.get('stacking_weights', {})
   print(f"Baseline weight: {stacking.get('baseline', 0):.1%}")
   print(f"Increasing weight: {stacking.get('increasing', 0):.1%}")
   print(f"Decreasing weight: {stacking.get('decreasing', 0):.1%}")
   
   # Which projection was most useful?
   if stacking.get('increasing', 0) > 0.5:
       print("→ Increasing projection dominant (positive correlation)")
   elif stacking.get('decreasing', 0) > 0.5:
       print("→ Decreasing projection dominant (negative correlation)")
   else:
       print("→ Mixed or baseline projection")

Doubly Robust Diagnostics
-------------------------

For DR estimators (DR-CPO, MRDR, TMLE):

Orthogonality Score
~~~~~~~~~~~~~~~~~~~

Tests if the DR correction term is centered at zero:

.. code-block:: python

   dr_diag = results.diagnostics  # DRDiagnostics object
   
   # Check orthogonality (should contain 0)
   orth_score = dr_diag.orthogonality_score
   orth_ci = dr_diag.orthogonality_ci
   
   if orth_ci[0] <= 0 <= orth_ci[1]:
       print("✓ Orthogonality satisfied")
   else:
       print(f"⚠️ Orthogonality violated: {orth_score:.3f} [{orth_ci[0]:.3f}, {orth_ci[1]:.3f}]")

Influence Function Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examines the empirical influence functions:

.. code-block:: python

   # Get influence functions
   influence = results.metadata.get('dr_influence', {})
   
   for policy, ifs in influence.items():
       # Check for outliers
       z_scores = (ifs - ifs.mean()) / ifs.std()
       outliers = np.sum(np.abs(z_scores) > 3)
       
       print(f"{policy}:")
       print(f"  Mean IF: {ifs.mean():.3e} (should be ≈0)")
       print(f"  Std IF: {ifs.std():.3f}")
       print(f"  Outliers: {outliers} ({outliers/len(ifs):.1%})")

Component Breakdown
~~~~~~~~~~~~~~~~~~~

Decomposes DR estimate into IPS and DM components:

.. code-block:: python

   breakdown = dr_diag.component_breakdown
   
   for policy in breakdown:
       ips_contrib = breakdown[policy]['ips_contribution']
       dm_contrib = breakdown[policy]['dm_contribution']
       total = breakdown[policy]['total']
       
       print(f"{policy}:")
       print(f"  IPS component: {ips_contrib:.3f} ({ips_contrib/total:.1%})")
       print(f"  DM component: {dm_contrib:.3f} ({dm_contrib/total:.1%})")
       print(f"  Total estimate: {total:.3f}")

Cross-Fit Performance
~~~~~~~~~~~~~~~~~~~~~

Evaluates outcome model performance across folds:

.. code-block:: python

   # Check R² across folds
   for fold, r2 in enumerate(dr_diag.fold_r_squared):
       print(f"Fold {fold}: R² = {r2:.3f}")
   
   # Large variance across folds suggests instability
   r2_std = np.std(dr_diag.fold_r_squared)
   if r2_std > 0.1:
       print(f"⚠️ High R² variance across folds: {r2_std:.3f}")

Quality Gates
-------------

Implement automated quality gates:

.. code-block:: python

   def check_quality_gates(results):
       """Return (passed, warnings, failures)."""
       warnings = []
       failures = []
       
       # ESS gate (critical)
       for policy in results.metadata['target_policies']:
           ess = results.metadata['diagnostics'][policy]['weights']['ess_fraction']
           if ess < 0.05:
               failures.append(f"{policy}: ESS={ess:.1%} < 5%")
           elif ess < 0.10:
               warnings.append(f"{policy}: ESS={ess:.1%} < 10%")
       
       # Tail index gate
       for policy in results.metadata['target_policies']:
           tail_idx = results.metadata['diagnostics'][policy]['weights'].get('tail_index')
           if tail_idx and tail_idx < 2:
               warnings.append(f"{policy}: Heavy tails (α={tail_idx:.2f})")
       
       # Calibration gate
       cal_diag = results.metadata.get('calibration_diagnostics', {})
       if cal_diag.get('kendall_tau', 1) < 0.3:
           warnings.append(f"Poor judge quality (τ={cal_diag['kendall_tau']:.2f})")
       
       # DR orthogonality gate
       if hasattr(results.diagnostics, 'orthogonality_ci'):
           ci = results.diagnostics.orthogonality_ci
           if not (ci[0] <= 0 <= ci[1]):
               warnings.append("DR orthogonality violated")
       
       passed = len(failures) == 0
       return passed, warnings, failures
   
   # Use in production
   passed, warnings, failures = check_quality_gates(results)
   
   if not passed:
       print("❌ QUALITY GATES FAILED:")
       for f in failures:
           print(f"   - {f}")
       raise ValueError("Results do not meet quality standards")
   
   if warnings:
       print("⚠️ Quality warnings:")
       for w in warnings:
           print(f"   - {w}")

Diagnostic Dashboards
---------------------

CJE can generate comprehensive diagnostic dashboards:

.. code-block:: python

   from cje import analyze_dataset
   
   # Generate all diagnostic plots
   results = analyze_dataset(
       "data.jsonl",
       estimator="calibrated-ips",
       output_dir="diagnostics/",
       generate_plots=True  # Creates diagnostic dashboard
   )

This creates:
- Weight distribution histograms
- ESS tracking plots
- Calibration reliability diagrams
- Q-Q plots for influence functions
- Component breakdown charts
- Cross-validation performance

Interpreting Status Codes
--------------------------

CJE uses a traffic light system:

.. list-table::
   :header-rows: 1

   * - Status
     - Symbol
     - Meaning
     - Action
   * - Green
     - 🟢
     - All checks passed
     - Proceed with confidence
   * - Amber
     - 🟡
     - Minor issues detected
     - Review warnings, proceed with caution
   * - Red
     - 🔴
     - Critical issues found
     - Do not use results, investigate issues

Common Issues and Solutions
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Issue
     - Solution
   * - Low ESS (<5%)
     - • Collect more data
       • Use tighter variance cap
       • Consider different policies with better overlap
   * - Heavy tails (α < 2)
     - • Use CalibratedIPS instead of RawIPS
       • Apply weight clipping
       • Increase variance cap in SIMCal
   * - Poor judge quality (τ < 0.3)
     - • Get better judge model
       • Increase oracle coverage
       • Use direct rewards if available
   * - Orthogonality violated
     - • Check outcome model specification
       • Ensure cross-fitting is enabled
       • Verify fresh draws quality
   * - High extrapolation
     - • Increase oracle coverage
       • Ensure oracle labels span judge range
       • Check for distribution shift

Best Practices
--------------

1. **Always check ESS first** - It's the most important diagnostic
2. **Set up automated gates** - Don't rely on manual inspection
3. **Log all diagnostics** - Track trends over time
4. **Use visual diagnostics** - Plots reveal patterns numbers miss
5. **Compare estimators** - If diagnostics are poor, try different estimators
6. **Monitor in production** - Diagnostics can detect data drift

Next Steps
----------

- See :doc:`simcal` for understanding weight calibration diagnostics
- See :doc:`theory` for theoretical justification of gates
- See :doc:`estimators` for choosing estimators based on diagnostics
- See :doc:`api/utils` for diagnostic utility functions