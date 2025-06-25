Oracle Evaluation in CJE
========================

Oracle evaluation provides ground truth validation for CJE estimates. This guide covers both types of oracle supported in CJE: automated (AI-based) and human (crowdsourced).

.. important::
   CJE uses "oracle" in two distinct contexts - automated AI oracles and human crowdsourced oracles. Always be clear which type you mean.

Oracle Types Overview
---------------------

.. list-table:: Oracle Type Comparison
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - Automated Oracle (AI)
     - Human Oracle
   * - **What it is**
     - Stronger AI model as ground truth
     - Human annotators via crowdsourcing
   * - **Speed**
     - Minutes to hours
     - Days to weeks
   * - **Cost**
     - $0.01-0.03 per label
     - $0.08-0.30 per label
   * - **Quality**
     - Strong model approximation
     - True human judgment
   * - **Integration**
     - Fully automated in CJE
     - Manual export/import process
   * - **Use case**
     - Development, testing, validation
     - Final validation, research papers

Automated Oracle (AI-Based)
---------------------------

Uses powerful models like GPT-4o or Claude-3-Opus as ground truth reference.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

   # In config files (e.g., example_eval.yaml)
   oracle:
     enabled: true
     provider: "openai"
     model_name: "gpt-4o"
     temperature: 0.0
     fraction: 0.25  # Label 25% of samples

Usage Example
~~~~~~~~~~~~~

.. code-block:: bash

   # Automatic oracle validation in arena analysis
   python scripts/run_arena_analysis.py \
       --max-samples 1000 \
       --oracle-fraction 0.25 \
       --oracle-model "gpt-4o" \
       --proxy-model "gpt-3.5-turbo"

Manual Implementation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cje.judge import JudgeFactory
   from cje.calibration import IsotonicCalibrator
   
   # Setup judges
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

Human Oracle (Crowdsourced)
---------------------------

Collects ground truth labels from human annotators via platforms like Surge AI or MTurk.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

   # In experiments like arena_10k_oracle
   oracle:
     enabled: false  # Human labels imported separately
     provider: "human"
     platform: "surge"
     cost_per_vote: 0.08
     votes_per_sample: 3

Workflow
~~~~~~~~

1. **Export data for labeling**:

   .. code-block:: python

      # Prepare samples for human annotation
      export_for_crowdsourcing(
          samples=dataset,
          output_file="labeling_task.json",
          platform="surge"
      )

2. **Coordinate with platform**:
   
   - Upload task definition
   - Set quality requirements
   - Monitor annotation progress

3. **Import results**:

   .. code-block:: python

      # Load human labels
      human_labels = load_crowdsourced_labels("surge_results.json")
      
      # Merge with CJE data
      for sample, label in zip(dataset, human_labels):
          sample['oracle_score'] = label['rating'] / 10.0

4. **Analyze in CJE**:

   .. code-block:: python

      # Use human labels as ground truth
      oracle_truth = np.mean([s['oracle_score'] for s in dataset])
      cje_estimate = estimator.estimate().v_hat[0]
      
      error = abs(cje_estimate - oracle_truth)
      print(f"CJE vs Human Oracle: {error:.3f}")

See ``experiments/arena_10k_oracle`` for a complete human oracle example.

Oracle Validation Workflow
--------------------------

Regardless of oracle type, the validation workflow follows these steps:

1. **Sparse Oracle Labeling**
   
   Label subset of data with expensive oracle:

   .. code-block:: python

      oracle_fraction = 0.25  # 25% coverage
      oracle_scores = get_oracle_scores(dataset, fraction=oracle_fraction)

2. **Judge Calibration**
   
   Train cheap proxy to match oracle:

   .. code-block:: python

      # Isotonic regression: proxy → oracle
      calibration_fn = fit_isotonic_regression(proxy_scores, oracle_scores)
      calibrated_scores = calibration_fn(all_proxy_scores)

3. **CJE Estimation**
   
   Run CJE with calibrated judge:

   .. code-block:: python

      v_hat_cje = run_cje_estimation(calibrated_scores, importance_weights)

4. **Validation**
   
   Compare CJE estimate vs oracle truth:

   .. code-block:: python

      absolute_error = abs(v_hat_cje - v_oracle)
      relative_error = absolute_error / v_oracle
      ci_coverage = (v_oracle >= ci_low) and (v_oracle <= ci_high)

Interpreting Results
--------------------

Key Metrics
~~~~~~~~~~~

**Absolute Error**
   ``|v_hat - v_oracle|`` - Raw difference from truth

**Relative Error**
   ``|v_hat - v_oracle| / v_oracle`` - Percentage error

**CI Coverage**
   Does 95% CI contain oracle truth?

**Judge Correlation**
   Spearman ρ between proxy and oracle

Quality Thresholds
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Metric
     - 🟢 Excellent
     - 🟡 Good
     - 🔴 Poor
     - Action if Poor
   * - Absolute Error
     - < 0.05
     - < 0.10
     - > 0.10
     - Check calibration
   * - Relative Error
     - < 5%
     - < 10%
     - > 15%
     - Increase oracle %
   * - CI Coverage
     - ≥ 90%
     - ≥ 85%
     - < 80%
     - Bootstrap CIs
   * - Correlation
     - ≥ 0.8
     - ≥ 0.7
     - < 0.6
     - Stronger proxy

Example Output
~~~~~~~~~~~~~~

.. code-block:: json

   {
     "results": {
       "gpt-4": {
         "v_hat": 0.742,
         "se": 0.023,
         "ci_low": 0.697,
         "ci_high": 0.787,
         "oracle_truth": 0.758,
         "absolute_error": 0.016,
         "relative_error": 0.021,
         "ci_coverage": true
       }
     },
     "calibration": {
       "correlation": 0.834,
       "coverage_rate": 0.94
     }
   }

Diagnostic Plots
----------------

Judge Calibration Curve
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   plt.figure(figsize=(8, 6))
   plt.scatter(proxy_scores, oracle_scores, alpha=0.5)
   plt.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
   plt.xlabel('Proxy Judge Score')
   plt.ylabel('Oracle Judge Score')
   plt.title('Judge Calibration Analysis')

**Interpretation:**
- Diagonal: Perfect calibration
- S-curve: Systematic bias
- Scatter: High noise

Best Practices
--------------

**Choosing Oracle Type**

1. Use **automated oracle** for:
   - Development and testing
   - Rapid experimentation
   - Continuous validation
   - Cost-sensitive applications

2. Use **human oracle** for:
   - Research publications
   - Production validation
   - Subjective tasks
   - Gold standard benchmarks

**Sample Size Guidelines**

- Automated: 100-500 oracle samples (25-50% of data)
- Human: 50-200 samples (cost permitting)
- Minimum 100 for reliable correlation

**Common Pitfalls**

1. **Config confusion**: Human oracle has ``enabled: false`` because labels are imported
2. **Naming ambiguity**: Always specify "automated" or "human" oracle
3. **Cost underestimation**: Human oracle can be 10x more expensive
4. **Coverage bias**: Ensure oracle samples are representative

Advanced Techniques
-------------------

Stratified Oracle Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sample from score quantiles
   quantiles = np.percentile(proxy_scores, [0, 25, 50, 75, 100])
   
   oracle_indices = []
   for i in range(len(quantiles) - 1):
       stratum = (scores >= quantiles[i]) & (scores < quantiles[i+1])
       indices = np.where(stratum)[0]
       n_samples = int(0.25 * len(indices))
       oracle_indices.extend(np.random.choice(indices, n_samples))

Multi-Oracle Ensemble
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use multiple oracles for robustness
   oracles = [
       JudgeFactory.create("openai", model="gpt-4o"),
       JudgeFactory.create("anthropic", model="claude-3-opus")
   ]
   
   oracle_scores = []
   for sample in dataset:
       scores = [oracle.score(sample) for oracle in oracles]
       oracle_scores.append(np.mean(scores))

Troubleshooting
---------------

**Low Correlation (< 0.6)**
   - Try stronger proxy judge
   - Increase oracle coverage to 50%
   - Check prompt alignment

**High Error (> 10%)**
   - Verify calibration quality
   - Check for extreme weights
   - Consider different estimator

**Poor Coverage (< 80%)**
   - Use bootstrap CIs
   - Increase sample size
   - Check distributional assumptions

See Also
--------

- :doc:`/guides/troubleshooting` - General debugging
- :doc:`/api/estimators` - Estimator details
- ``experiments/arena_10k_oracle`` - Human oracle example