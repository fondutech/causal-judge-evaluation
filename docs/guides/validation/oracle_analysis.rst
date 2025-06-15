Oracle Analysis Guide
====================

Oracle analysis provides high-precision evaluation by comparing CJE estimates against ground truth utility labels from powerful models. This guide covers oracle validation techniques and interpretation of oracle analysis results.

Overview
--------

Oracle analysis answers the critical question: "How accurate are my CJE estimates?" by comparing them against expensive but highly accurate "oracle" utility labels from models like GPT-4o.

**Key Benefits:**

- **Validation**: Verify CJE estimates are unbiased and well-calibrated
- **Debug**: Identify when judge calibration or weight processing fails
- **Confidence**: Build trust in CJE methodology for your specific use case
- **Optimization**: Compare different estimators, judges, and hyperparameters

Oracle vs. Proxy Evaluation
----------------------------

.. list-table:: Oracle vs Proxy Comparison
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - Proxy Judge
     - Oracle Judge
   * - **Cost**
     - Low ($0.0005/sample)
     - High ($0.01-0.03/sample)
   * - **Speed**
     - Fast (seconds)
     - Slow (minutes)
   * - **Coverage**
     - Full dataset
     - Sparse sampling (10-30%)
   * - **Quality**
     - Good correlation
     - Ground truth reference
   * - **Use Case**
     - Production estimation
     - Validation & debugging

Oracle Validation Workflow
---------------------------

1. **Sparse Oracle Labeling**
   
   Label 10-30% of samples with expensive oracle model:

   .. code-block:: python

      # Automatic in arena analysis
      oracle_fraction = 0.25  # 25% oracle coverage
      oracle_model = "gpt-4o"  # High-quality model

2. **Judge Calibration**
   
   Train cheap proxy judge to match oracle on labeled samples:

   .. code-block:: python

      # Isotonic regression: proxy_score → oracle_score
      calibration_fn = fit_isotonic_regression(proxy_scores, oracle_scores)
      calibrated_scores = calibration_fn(all_proxy_scores)

3. **CJE Estimation**
   
   Run CJE with calibrated judge on full dataset:

   .. code-block:: python

      v_hat_cje = run_cje_estimation(calibrated_scores, importance_weights)

4. **Oracle Ground Truth**
   
   Compute true utility on oracle-labeled samples:

   .. code-block:: python

      v_oracle = np.mean(oracle_scores_subset)

5. **Validation Comparison**
   
   Compare CJE estimate vs oracle truth:

   .. code-block:: python

      absolute_error = abs(v_hat_cje - v_oracle)
      relative_error = absolute_error / v_oracle
      ci_coverage = (v_oracle >= ci_low) and (v_oracle <= ci_high)

Running Oracle Analysis
-----------------------

Automatic Oracle Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Most convenient - built into arena analysis:

.. code-block:: bash

   python scripts/run_arena_analysis.py \
       --max-samples 1000 \
       --oracle-fraction 0.25 \
       --oracle-model "gpt-4o" \
       --proxy-model "gpt-3.5-turbo"

**Automatic Features:**

- Oracle sample selection (random or stratified)
- Judge calibration curve generation
- Coverage testing (CI validation)
- Error analysis and reporting
- Visual diagnostic plots

Manual Oracle Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For custom datasets, you can implement oracle analysis manually:

.. code-block:: python

   from cje.judge import JudgeFactory
   from cje.calibration import IsotonicCalibrator
   import numpy as np

   # Setup oracle and proxy judges
   oracle_judge = JudgeFactory.create("openai", model="gpt-4o")
   proxy_judge = JudgeFactory.create("openai", model="gpt-3.5-turbo")

   # Score subset with oracle
   oracle_indices = np.random.choice(len(dataset), size=int(0.25 * len(dataset)))
   oracle_scores = []
   proxy_scores = []
   
   for idx in oracle_indices:
       oracle_scores.append(oracle_judge.score(dataset[idx]))
       proxy_scores.append(proxy_judge.score(dataset[idx]))
   
   # Fit calibration
   calibrator = IsotonicCalibrator()
   calibrator.fit(proxy_scores, oracle_scores)
   
   # Apply to all data
   all_proxy_scores = [proxy_judge.score(d) for d in dataset]
   calibrated_scores = calibrator.transform(all_proxy_scores)

Interpreting Oracle Results
---------------------------

Key Metrics
~~~~~~~~~~~

**Absolute Error**
   ``|v_hat - v_oracle|`` - How far off is the estimate?

**Relative Error**
   ``|v_hat - v_oracle| / v_oracle`` - Percentage error

**CI Coverage**
   Does confidence interval contain oracle truth?

**Judge Correlation**
   Spearman correlation between proxy and oracle scores

Example Output
~~~~~~~~~~~~~~

.. code-block:: json

   {
     "results": {
       "gpt-4": {
         "v_hat": 0.742,           // CJE estimate
         "se": 0.023,              // Standard error
         "ci_low": 0.697,          // 95% CI lower bound
         "ci_high": 0.787,         // 95% CI upper bound
         "oracle_truth": 0.758,    // Oracle ground truth
         "absolute_error": 0.016,  // |0.742 - 0.758| = 0.016
         "relative_error": 0.021,  // 2.1% error
         "ci_coverage": true       // Oracle within CI
       }
     },
     "judge_calibration": {
       "spearman_correlation": 0.834,  // Strong correlation
       "calibration_slope": 0.89,      // Slight miscalibration
       "coverage_rate": 0.94           // 94% CI coverage
     }
   }

Quality Assessment
~~~~~~~~~~~~~~~~~~

**🟢 Excellent Results:**

- Absolute error < 0.05
- Relative error < 5%
- CI coverage ≥ 90%
- Judge correlation ≥ 0.8

**🟡 Good Results:**

- Absolute error < 0.10
- Relative error < 10%
- CI coverage ≥ 85%
- Judge correlation ≥ 0.7

**🔴 Poor Results:**

- Absolute error > 0.10
- Relative error > 15%
- CI coverage < 80%
- Judge correlation < 0.6

Diagnostic Plots
----------------

Judge Calibration Curve
~~~~~~~~~~~~~~~~~~~~~~~~

Shows proxy vs oracle score relationship:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot calibration curve
   plt.figure(figsize=(8, 6))
   plt.scatter(proxy_scores, oracle_scores, alpha=0.5)
   plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
   plt.xlabel('Proxy Judge Score')
   plt.ylabel('Oracle Judge Score')
   plt.title('Judge Calibration Analysis')
   plt.legend()
   plt.show()

**Interpretation:**

- **Diagonal line**: Perfect calibration
- **S-curve**: Systematic bias in proxy judge
- **Scattered points**: High noise, low correlation

CI Coverage Analysis
~~~~~~~~~~~~~~~~~~~~

Analyze confidence interval performance:

.. code-block:: python

   # Check CI coverage
   n_covered = 0
   for i, policy in enumerate(target_policies):
       estimate = estimates[i]
       ci_low, ci_high = confidence_intervals[i]
       oracle_value = oracle_values[i]
       
       if ci_low <= oracle_value <= ci_high:
           n_covered += 1
   
   coverage_rate = n_covered / len(target_policies)
   print(f"CI Coverage: {coverage_rate:.2%}")

Error Analysis
~~~~~~~~~~~~~~

Analyze estimation errors:

.. code-block:: python

   # Calculate error components
   errors = {
       'absolute': abs(cje_estimate - oracle_mean),
       'relative': abs(cje_estimate - oracle_mean) / oracle_mean,
       'squared': (cje_estimate - oracle_mean) ** 2
   }
   
   print(f"Absolute Error: {errors['absolute']:.3f}")
   print(f"Relative Error: {errors['relative']:.2%}")
   print(f"MSE: {errors['squared']:.6f}")

**Components:**

- **Judge error**: Proxy-oracle miscalibration
- **Weight error**: Importance sampling variance
- **Model error**: Outcome model bias
- **Sample error**: Finite sample noise

Troubleshooting Poor Oracle Results
-----------------------------------

Low Judge Correlation (< 0.6)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Causes:**

- Proxy judge too weak for task
- Different evaluation criteria
- Insufficient oracle samples

**Solutions:**

- Use stronger proxy judge (e.g., GPT-4 instead of GPT-3.5)
- Increase oracle fraction to 30-50%
- Check prompt alignment between judges

High Absolute Error (> 0.10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Causes:**

- Poor judge calibration
- Extreme importance weights
- Insufficient data

**Solutions:**

- Increase oracle sample size
- Use more similar policies (lower weight variance)
- Try different estimators (SNIPS, MRDR)

Poor CI Coverage (< 80%)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Causes:**

- Underestimated uncertainty
- Biased estimation
- Non-normal error distribution

**Solutions:**

- Bootstrap confidence intervals
- Cross-validation for uncertainty
- Check weight diagnostics

Advanced Oracle Techniques
---------------------------

Stratified Oracle Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement stratified sampling for better coverage:

.. code-block:: python

   # Stratify by score quantiles
   proxy_scores = [proxy_judge.score(d) for d in dataset]
   quantiles = np.percentile(proxy_scores, [0, 25, 50, 75, 100])
   
   # Sample from each stratum
   oracle_indices = []
   for i in range(len(quantiles) - 1):
       stratum_mask = (proxy_scores >= quantiles[i]) & (proxy_scores < quantiles[i+1])
       stratum_indices = np.where(stratum_mask)[0]
       n_samples = int(0.25 * len(stratum_indices))
       oracle_indices.extend(np.random.choice(stratum_indices, n_samples))

Multi-Oracle Validation
~~~~~~~~~~~~~~~~~~~~~~~

Use multiple oracle models for robustness:

.. code-block:: python

   oracle_judges = [
       JudgeFactory.create("openai", model="gpt-4o"),
       JudgeFactory.create("anthropic", model="claude-3-opus"),
       JudgeFactory.create("openai", model="gpt-4")
   ]

   # Average oracle scores
   oracle_scores = []
   for idx in oracle_indices:
       scores = [judge.score(dataset[idx]) for judge in oracle_judges]
       oracle_scores.append(np.mean(scores))

Best Practices
--------------

**Oracle Model Selection:**

- Use strongest available model (GPT-4o, Claude-3-Opus)
- Consistent with proxy judge provider when possible
- Consider cost vs quality tradeoffs

**Sample Size Planning:**

- Minimum 100 oracle samples for reliable correlation
- 500+ oracle samples for precise error estimation
- Scale oracle fraction with dataset complexity

**Validation Frequency:**

- Every major model update
- Before production deployment
- Monthly for drift detection
- After significant data distribution changes

**Quality Control:**

- Monitor judge correlation trends
- Track CI coverage rates
- Alert on significant error increases
- Regular oracle model updates

Integration with CI/CD
----------------------

Automated Oracle Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # .github/workflows/oracle-validation.yml
   name: Oracle Validation
   on:
     push:
       paths: ['cje/**', 'configs/**']
   
   jobs:
     validate:
       runs-on: ubuntu-latest
       steps:
         - name: Run Oracle Analysis
           run: |
             python scripts/run_arena_analysis.py \
               --max-samples 500 \
               --oracle-fraction 0.3 \
               --validation-mode

Quality Gates
~~~~~~~~~~~~~

.. code-block:: python

   # Deployment quality gates
   # Compare CJE estimate with oracle ground truth
   oracle_mean = np.mean(oracle_scores)
   cje_estimate = estimator.estimate().v_hat[0]
   
   relative_error = abs(cje_estimate - oracle_mean) / oracle_mean
   
   if relative_error > 0.15:
       raise ValueError("Oracle validation failed - high estimation error")
   
   # Check if CI contains oracle truth
   ci_low, ci_high = estimator.confidence_interval()
   if not (ci_low <= oracle_mean <= ci_high):
       raise ValueError("Poor confidence interval calibration")

This comprehensive oracle analysis ensures your CJE estimates are accurate, well-calibrated, and suitable for production deployment. 