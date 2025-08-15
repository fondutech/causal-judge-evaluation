Visualization Guide
===================

CJE provides comprehensive visualization tools to understand weight distributions, calibration quality, and policy comparisons.

Overview
--------

Visualizations are generated automatically when running analysis, or can be created programmatically:

.. code-block:: python

   from cje import analyze_dataset
   
   # Automatic visualization generation
   results = analyze_dataset(
       "data.jsonl",
       estimator="calibrated-ips",
       plot_dir="plots/"  # Enables visualization
   )

All visualizations are saved to the specified directory and can be customized via configuration.

Weight Diagnostics Dashboard
----------------------------

The weight dashboard provides a comprehensive view of importance weight behavior across policies.

Components
~~~~~~~~~~

1. **Weight Distributions**: Histograms showing raw and calibrated weight distributions
2. **ESS Tracking**: Effective sample size across policies
3. **Concentration Curves**: Cumulative weight mass vs fraction of samples
4. **Tail Behavior**: Log-log plots for tail analysis

Generation
~~~~~~~~~~

.. code-block:: python

   from cje.visualization import create_weight_dashboard
   
   # After fitting estimator
   create_weight_dashboard(
       estimator=estimator,
       sampler=sampler,
       dataset=dataset,
       output_path="plots/weight_dashboard.png"
   )

Interpretation
~~~~~~~~~~~~~~

- **Ideal**: Uniform weights (horizontal line at 1.0)
- **Good**: Moderate variation, ESS > 30%
- **Warning**: Heavy concentration, ESS 10-30%
- **Critical**: Extreme concentration, ESS < 10%

Calibration Analysis
--------------------

Visualizes the judge-to-oracle calibration quality.

Reliability Diagram
~~~~~~~~~~~~~~~~~~~

Shows how well calibrated judge scores are:

.. code-block:: python

   from cje.visualization import plot_calibration_analysis
   
   plot_calibration_analysis(
       dataset=dataset,
       calibration_result=cal_result,
       output_path="plots/calibration.png"
   )

Components:
- **Diagonal line**: Perfect calibration
- **Blue curve**: Raw judge scores vs oracle
- **Orange curve**: Calibrated scores vs oracle
- **Shaded region**: 95% confidence interval

Isotonic Fit
~~~~~~~~~~~~

Visualizes the isotonic regression mapping:

.. code-block:: python

   from cje.visualization import plot_isotonic_fit
   
   plot_isotonic_fit(
       judge_scores=judge_scores[oracle_mask],
       oracle_labels=oracle_labels[oracle_mask],
       calibration_map=calibrator,
       output_path="plots/isotonic_fit.png"
   )

Shows:
- Scatter plot of judge vs oracle
- Fitted isotonic curve
- Confidence bands
- Extrapolation regions (if any)

Policy Comparison Plots
-----------------------

Compare estimated values across policies with uncertainty.

Forest Plot
~~~~~~~~~~~

.. code-block:: python

   from cje.visualization import plot_policy_comparison
   
   plot_policy_comparison(
       results=results,
       output_path="plots/policy_forest.png"
   )

Features:
- Point estimates with 95% CI
- Baseline reference line
- Statistical significance indicators
- Sorted by estimate value

Pairwise Comparisons
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cje.visualization import plot_pairwise_differences
   
   plot_pairwise_differences(
       results=results,
       baseline="base",
       output_path="plots/pairwise.png"
   )

Shows all policy differences from baseline with:
- Difference estimates
- Confidence intervals
- Significance testing (Bonferroni corrected)

DR Component Analysis
---------------------

For doubly robust estimators, visualize the decomposition.

Component Breakdown
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cje.visualization import plot_dr_components
   
   plot_dr_components(
       dr_diagnostics=results.diagnostics,
       output_path="plots/dr_breakdown.png"
   )

Displays:
- IPS component contribution
- Direct method contribution
- Total estimate
- Stacked bar chart by policy

Influence Functions
~~~~~~~~~~~~~~~~~~~

Q-Q plots and histograms of empirical influence functions:

.. code-block:: python

   from cje.visualization import plot_influence_functions
   
   plot_influence_functions(
       influence_dict=results.metadata['dr_influence'],
       output_path="plots/influence.png"
   )

Helps identify:
- Outliers in influence functions
- Normality violations
- Heavy tails

Cross-Validation Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cje.visualization import plot_cv_performance
   
   plot_cv_performance(
       fold_diagnostics=results.diagnostics.fold_diagnostics,
       output_path="plots/cv_performance.png"
   )

Shows:
- R² across folds
- RMSE variation
- Fold-specific estimates
- Stability assessment

Extreme Weight Analysis
-----------------------

Detailed analysis of samples with extreme importance weights.

Distribution Plots
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cje.visualization import plot_extreme_weights
   
   plot_extreme_weights(
       weights_dict={"policy": weights},
       threshold_high=100,
       threshold_low=0.01,
       output_path="plots/extreme_weights.png"
   )

Identifies:
- Samples with w > threshold_high
- Samples with w < threshold_low
- Distribution of extreme weights
- Prompt characteristics of extremes

Sample Investigation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cje import analyze_extreme_weights
   
   json_report, text_report = analyze_extreme_weights(
       dataset=dataset,
       sampler=sampler,
       raw_weights_dict=raw_weights,
       calibrated_weights_dict=cal_weights,
       n_extreme=10,
       output_dir="plots/"
   )

Generates detailed report with:
- Top/bottom weighted samples
- Prompt and response analysis
- Policy likelihood comparisons
- Recommendations for handling

SIMCal Visualization
--------------------

Visualize the Stacked SIMCal weight calibration process.

Candidate Projections
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cje.visualization import plot_simcal_candidates
   
   plot_simcal_candidates(
       raw_weights=weights,
       candidates=["baseline", "increasing", "decreasing"],
       judge_scores=judge_scores,
       output_path="plots/simcal_candidates.png"
   )

Shows:
- Raw weights vs judge scores
- Three candidate projections
- Selected combination
- Variance reduction achieved

Stacking Weights
~~~~~~~~~~~~~~~~

.. code-block:: python

   from cje.visualization import plot_stacking_weights
   
   plot_stacking_weights(
       stacking_weights=simcal_diagnostics['stacking_weights'],
       output_path="plots/stacking.png"
   )

Pie chart showing contribution of each candidate.

Dashboard Generation
--------------------

Generate all visualizations at once:

.. code-block:: python

   from cje.visualization import generate_full_dashboard
   
   generate_full_dashboard(
       results=results,
       dataset=dataset,
       estimator=estimator,
       sampler=sampler,
       output_dir="dashboard/"
   )

Creates:
- ``index.html``: Interactive dashboard
- ``weight_analysis.png``: Weight diagnostics
- ``calibration.png``: Calibration plots
- ``policy_comparison.png``: Forest plots
- ``diagnostics_summary.json``: Machine-readable diagnostics

Customization
-------------

Plot Styling
~~~~~~~~~~~~

.. code-block:: python

   from cje.visualization import set_plot_style
   
   # Use CJE default style
   set_plot_style("default")
   
   # Or customize
   set_plot_style({
       "figure.figsize": (12, 8),
       "font.size": 12,
       "axes.labelsize": 14,
       "lines.linewidth": 2
   })

Color Schemes
~~~~~~~~~~~~~

.. code-block:: python

   from cje.visualization import set_color_palette
   
   # Predefined palettes
   set_color_palette("colorblind")  # Colorblind-friendly
   set_color_palette("pastel")      # Soft colors
   set_color_palette("vibrant")     # High contrast
   
   # Custom palette
   set_color_palette({
       "base": "#2E86AB",
       "target": "#A23B72",
       "good": "#73AB84",
       "warning": "#F18F01",
       "critical": "#C73E1D"
   })

Export Formats
~~~~~~~~~~~~~~

All plots support multiple formats:

.. code-block:: python

   # High-resolution PNG
   plot_func(..., output_path="plot.png", dpi=300)
   
   # Vector format (publication-ready)
   plot_func(..., output_path="plot.pdf")
   plot_func(..., output_path="plot.svg")
   
   # Interactive HTML
   plot_func(..., output_path="plot.html", interactive=True)

Interactive Visualizations
--------------------------

For exploratory analysis, use interactive plots:

.. code-block:: python

   from cje.visualization import create_interactive_dashboard
   
   dashboard = create_interactive_dashboard(results, dataset)
   dashboard.show()  # Opens in browser
   
   # Or save
   dashboard.save("dashboard.html")

Features:
- Hover tooltips with details
- Zoom and pan
- Linked brushing across plots
- Export selected data

Performance Considerations
--------------------------

For large datasets:

.. code-block:: python

   # Subsample for visualization
   from cje.visualization import plot_weights_subsampled
   
   plot_weights_subsampled(
       weights=weights,
       n_samples=10000,  # Subsample size
       seed=42,
       output_path="plots/weights_sample.png"
   )
   
   # Use hexbin for dense scatter plots
   from cje.visualization import plot_calibration_hexbin
   
   plot_calibration_hexbin(
       judge_scores=judge_scores,
       oracle_labels=oracle_labels,
       gridsize=50,
       output_path="plots/calibration_hex.png"
   )

Best Practices
--------------

1. **Always generate weight dashboard** - Most informative single visualization
2. **Check calibration plot** - Ensures judge quality
3. **Include confidence intervals** - Show uncertainty in estimates
4. **Use colorblind-friendly palettes** - Ensure accessibility
5. **Save high-resolution versions** - For publications (300+ DPI)
6. **Generate both static and interactive** - Static for reports, interactive for exploration
7. **Document outliers** - Use extreme weight analysis for anomalies

Common Issues
-------------

**Large file sizes**
   Use PNG with compression or reduce DPI for drafts

**Overlapping labels**
   Adjust figure size or use rotation/abbreviations

**Too many policies**
   Focus on top policies or use grouped/faceted plots

**Slow rendering**
   Subsample data or use rasterization for dense plots

Next Steps
----------

- See :doc:`diagnostics` for interpreting visualizations
- See :doc:`api/visualization` for full API reference
- See :doc:`examples` for complete visualization workflows