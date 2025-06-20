Uncertainty-Aware Evaluation
============================

*Complete guide to using CJE's uncertainty quantification features*

CJE treats uncertainty as a first-class citizen in evaluation. Every judge score includes both a mean and variance, enabling more robust policy comparisons and better understanding of confidence in results.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

**What is Uncertainty-Aware Evaluation?**

Traditional LLM evaluation treats judge scores as point estimates. However, judges can be uncertain about their assessments - a response might be borderline between good and great, or a judge might struggle with ambiguous cases. CJE's uncertainty-aware evaluation:

1. **Quantifies judge confidence** in each score
2. **Propagates uncertainty** through the causal estimation pipeline
3. **Improves estimate robustness** via variance-based weight shrinkage
4. **Provides richer insights** through variance decomposition

**Key Benefits:**

- More accurate confidence intervals that reflect both sampling and judge uncertainty
- Automatic down-weighting of uncertain samples to improve effective sample size (ESS)
- Calibration of judge confidence to match true uncertainty
- Detailed diagnostics showing sources of variance

Quick Start
-----------

With the unified judge system (June 2025), ALL judges now return uncertainty estimates:

.. code-block:: python

   from cje.judge.factory_unified import JudgeFactory
   from cje.uncertainty import UncertaintyAwareDRCPO, UncertaintyEstimatorConfig

   # 1. Create a judge (all judges now return JudgeScore with mean+variance)
   judge = JudgeFactory.create(
       provider="openai",
       model="gpt-4o",
       template="comprehensive_judge",
       uncertainty_method="structured"  # or "deterministic" or "monte_carlo"
   )

   # 2. Score samples (always returns JudgeScore objects)
   samples = [{"context": "...", "response": "..."}]
   judge_scores = judge.score_batch(samples)
   # Each score has .mean and .variance attributes

   # 3. Configure estimator with uncertainty features
   estimator_config = UncertaintyEstimatorConfig(
       k_folds=5,
       use_variance_shrinkage=True,
       shrinkage_method="adaptive",
       target_ess_fraction=0.8,
   )
   estimator = UncertaintyAwareDRCPO(estimator_config)

   # 4. Run evaluation
   result = estimator.fit(
       judge_scores=judge_scores,
       oracle_rewards=oracle_rewards,
       importance_weights=weights,
       policy_names=["Policy A", "Policy B"],
   )

   # 5. Analyze results with uncertainty
   print(result.summary())

Setting Up Uncertainty-Aware Judges
-----------------------------------

With the unified judge system, ALL judges now return `JudgeScore` objects with uncertainty. You choose the uncertainty estimation method:

1. Deterministic (Zero Variance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For traditional point estimates:

.. code-block:: python

   from cje.judge.factory_unified import JudgeFactory
   
   # Creates a judge that always returns variance=0
   judge = JudgeFactory.create(
       provider="openai",
       model="gpt-4o",
       template="comprehensive_judge",
       uncertainty_method="deterministic",
       temperature=0.0
   )
   
   score = judge.score("Context", "Response")
   # score.mean = 0.75, score.variance = 0.0

2. Structured Output with Confidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model estimates its own uncertainty:

.. code-block:: python

   # Default method - model returns score + confidence
   judge = JudgeFactory.create(
       provider="openai",
       model="gpt-4o",
       template="comprehensive_judge",
       uncertainty_method="structured"  # Default
   )
   
   score = judge.score("Context", "Response")
   # score.mean = 0.75, score.variance = 0.02 (model-estimated)

The judge prompts the model to return both score and confidence, converting confidence to variance.

3. Monte Carlo Sampling
~~~~~~~~~~~~~~~~~~~~~~~

Sample multiple times to estimate uncertainty empirically:

.. code-block:: python

   judge = JudgeFactory.create(
       provider="anthropic",
       model="claude-3-sonnet",
       uncertainty_method="monte_carlo",
       temperature=0.7,  # Higher for diversity
       mc_samples=10     # Number of samples
   )
   
   score = judge.score("Context", "Response")
   # score.mean = 0.73, score.variance = 0.03 (empirical)

This approach:
- Scores each sample multiple times
- Computes mean and variance from the samples
- More expensive but works with any model

4. Custom Uncertainty Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For advanced use cases, implement custom uncertainty estimation:

.. code-block:: python

   class BERTUncertaintyJudge(UncertaintyAwareJudge):
       """Custom judge using BERT confidence scores."""
       
       def score_single(self, sample: Dict) -> JudgeScore:
           # Get BERT embeddings and classification
           logits = self.bert_model(sample["text"])
           probs = torch.softmax(logits, dim=-1)
           
           # Score is expected value
           score = (probs * self.class_values).sum()
           
           # Variance from probability distribution
           variance = (probs * (self.class_values - score)**2).sum()
           
           return JudgeScore(mean=float(score), variance=float(variance))

Judge Templates for Uncertainty
-------------------------------

CJE includes specialized templates that prompt judges to express uncertainty:

**uncertainty_aware_judge**:

.. code-block:: text

   You are evaluating an AI assistant's response. Provide:
   1. A quality score from 0 to 1
   2. Your confidence in this score (0-1)
   3. Brief reasoning
   
   Consider:
   - Helpfulness and relevance
   - Accuracy and truthfulness
   - Clarity and coherence
   
   Be explicit about uncertainty when:
   - The query is ambiguous
   - The response quality is borderline
   - You lack domain expertise
   
   Return as JSON:
   {
       "score": <float 0-1>,
       "confidence": <float 0-1>,
       "reasoning": "<explanation>"
   }

**comprehensive_judge_with_aspects**:

.. code-block:: text

   Evaluate the response on multiple aspects:
   
   1. Relevance (0-1): How well does it address the query?
   2. Accuracy (0-1): Is the information correct?
   3. Clarity (0-1): Is it well-written and clear?
   4. Completeness (0-1): Does it fully answer the question?
   
   For each aspect, provide:
   - Score (0-1)
   - Confidence (0-1)
   
   Overall score is the weighted average.

Calibration and Variance Adjustment
-----------------------------------

CJE provides two types of calibration for uncertainty-aware evaluation:

1. Isotonic Calibration (Bias Correction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maps biased judge scores to unbiased values:

.. code-block:: python

   from cje.uncertainty.calibration import calibrate_variance_gamma
   
   # Calibrate judge scores using oracle labels
   iso_model, gamma = calibrate_variance_gamma(
       judge_scores=judge_scores,  # List[JudgeScore]
       oracle_rewards=oracle_rewards,  # Ground truth
   )
   
   # Apply calibration to new scores
   calibrated_scores = [
       JudgeScore(
           mean=iso_model.transform([s.mean])[0],
           variance=s.variance * gamma
       )
       for s in new_scores
   ]

2. Gamma Calibration (Confidence Adjustment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Corrects systematic over/under-confidence:

.. code-block:: python

   # Gamma > 1: Judge is overconfident (underestimates uncertainty)
   # Gamma < 1: Judge is underconfident (overestimates uncertainty)
   
   # Computed during isotonic calibration
   gamma = sum((y_true - y_calibrated)**2) / sum(variances)

**Important**: Gamma is computed AFTER isotonic calibration to measure only irreducible uncertainty, not bias.

Variance-Based Weight Shrinkage
-------------------------------

High-uncertainty samples can dominate importance-weighted estimates. CJE automatically shrinks weights for uncertain samples:

.. code-block:: python

   # Optimal shrinkage formula
   w_shrunk = w / (1 + lambda * v)
   
   # Where lambda is chosen to minimize variance:
   lambda_optimal = Cov[w²v, w(r-μ)] / E[w²v²]

Configuration options:

.. code-block:: python

   estimator_config = UncertaintyEstimatorConfig(
       use_variance_shrinkage=True,
       shrinkage_method="adaptive",  # or "optimal", "fixed"
       shrinkage_lambda=0.1,         # For "fixed" method
       target_ess_fraction=0.8,      # For "adaptive" method
   )

Shrinkage methods:

- **"optimal"**: Uses the theoretical optimal lambda (can be unstable)
- **"adaptive"**: Maintains minimum ESS constraint (recommended)
- **"fixed"**: User-specified lambda value

Multi-Policy Evaluation
-----------------------

Uncertainty-aware evaluation excels at multi-policy comparison:

.. code-block:: python

   # Evaluate multiple policies simultaneously
   result = estimator.fit(
       judge_scores=scores,
       oracle_rewards=rewards,
       importance_weights=weights,  # Shape: (n_samples, n_policies)
       policy_names=["GPT-3.5", "GPT-4", "Claude-3", "Gemini"],
   )
   
   # Rich comparison features
   comparison = result.pairwise_comparison("GPT-4", "Claude-3")
   print(f"GPT-4 vs Claude-3:")
   print(f"  Difference: {comparison['difference']:.4f}")
   print(f"  95% CI: [{comparison['ci_lower']:.4f}, {comparison['ci_upper']:.4f}]")
   print(f"  P-value: {comparison['p_value']:.4f}")
   print(f"  Significant: {comparison['is_significant']}")
   
   # Ranking with uncertainty
   ranking = result.rank_policies()
   for rank, policy in ranking:
       print(f"{rank}. {policy.name}: {policy.estimate.mean:.4f} ± {policy.estimate.se:.4f}")

Understanding Results
---------------------

Uncertainty-aware results provide rich diagnostics:

1. Variance Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~

Understand where uncertainty comes from:

.. code-block:: python

   policy = result.get_policy("GPT-4")
   decomp = policy.estimate.variance_decomposition
   
   print(f"Variance sources for {policy.name}:")
   print(f"  Efficient influence function: {decomp.eif_pct:.1f}%")
   print(f"  Judge uncertainty: {decomp.judge_pct:.1f}%")
   print(f"  Cross-term: {decomp.cross_pct:.1f}%")
   
   # High judge % suggests need for better judge or more samples

2. Diagnostic Reports
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate comprehensive diagnostics
   diagnostics = result.diagnostics()
   
   print("Calibration quality:")
   print(f"  Gamma: {diagnostics.gamma:.3f}")
   print(f"  Isotonic R²: {diagnostics.isotonic_r2:.3f}")
   
   print("\nWeight statistics:")
   print(f"  ESS %: {diagnostics.ess_percentage:.1f}%")
   print(f"  Max weight: {diagnostics.max_weight:.2f}")
   
   print("\nUncertainty concentration:")
   print(f"  Top 10% variance contribution: {diagnostics.top_10pct_var:.1f}%")

3. Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~

CJE provides multiple CI types:

.. code-block:: python

   policy = result.get_policy("GPT-4")
   
   # Standard CI (sampling uncertainty only)
   ci_standard = policy.estimate.confidence_interval(include_judge_var=False)
   
   # Full CI (sampling + judge uncertainty)
   ci_full = policy.estimate.confidence_interval(include_judge_var=True)
   
   # Bootstrap CI (non-parametric)
   ci_bootstrap = policy.estimate.bootstrap_ci(n_bootstrap=1000)

Configuration Examples
----------------------

Minimal Uncertainty Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # configs/uncertainty_minimal.yaml
   dataset:
     name: "./data.csv"
   
   logging_policy:
     provider: "openai"
     model_name: "gpt-3.5-turbo"
     temperature: 0.7
   
   target_policies:
     - name: "improved"
       provider: "openai"
       model_name: "gpt-4o"
       temperature: 0.7
   
   # Judge configuration (all judges now support uncertainty)
   judge:
     provider: "openai"
     model_name: "gpt-4o"
     template: "comprehensive_judge"
     temperature: 0.0
     uncertainty_method: "structured"  # or "deterministic" or "monte_carlo"
   
   # Estimator with uncertainty features
   estimator:
     name: "DRCPO"
     k: 5
     # Uncertainty features enabled by default in new implementation

Production Uncertainty Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # configs/uncertainty_production.yaml
   dataset:
     name: "ChatbotArena"
     split: "train"
     sample_limit: 5000
   
   logging_policy:
     provider: "fireworks"
     model_name: "llama-3-70b-instruct"
     temperature: 0.5
   
   target_policies:
     - name: "gpt4_helpful"
       provider: "openai"
       model_name: "gpt-4o"
       temperature: 0.3
       system_prompt: "You are a helpful, thorough assistant."
       mc_samples: 5
     
     - name: "claude_concise"
       provider: "anthropic"
       model_name: "claude-3-opus"
       temperature: 0.3
       system_prompt: "You are a concise, direct assistant."
       mc_samples: 5
   
   judge:
     provider: "openai"
     model_name: "gpt-4o"
     template: "comprehensive_judge"
     temperature: 0.0
     uncertainty_method: "structured"  # Get model's confidence estimates
   
   estimator:
     name: "DRCPO"
     k: 10
   
   # Advanced uncertainty settings
   uncertainty:
     variance_shrinkage:
       enabled: true
       method: "adaptive"
       target_ess_fraction: 0.85
     
     calibration:
       min_oracle_samples: 100
       confidence_level: 0.95
     
     diagnostics:
       save_plots: true
       verbose: true

Common Patterns and Best Practices
----------------------------------

1. **Start Simple**: Begin with structured output judges before trying complex approaches

2. **Calibration Data**: Ensure sufficient oracle samples (>100) for reliable calibration

3. **Monitor Gamma**: 
   - γ ≈ 1: Well-calibrated judge
   - γ > 2: Judge is overconfident
   - γ < 0.5: Judge is underconfident

4. **Variance Bounds**: Remember variance ∈ [0, 0.25] for scores in [0, 1]

5. **Shrinkage Trade-offs**: 
   - More shrinkage → Higher ESS, more bias
   - Less shrinkage → Lower ESS, less bias

6. **Multi-Policy Tips**:
   - Use same judge configuration across all policies
   - Ensure sufficient samples for reliable pairwise comparisons
   - Consider multiple testing correction for many comparisons

Troubleshooting
---------------

For uncertainty-specific issues, see the :doc:`troubleshooting` guide's uncertainty section. Common issues:

- **High gamma (>3)** → Judge is overconfident, try MC sampling
- **Low ESS** → Use adaptive shrinkage or more samples
- **Unstable variance** → Try fixed shrinkage method
- **Calibration failures** → Need more oracle samples (>100)


See Also
--------

- :doc:`/examples/clean_uncertainty_api` - Complete code example
- :doc:`/api/uncertainty` - API reference
- :doc:`/theory/uncertainty_theory` - Mathematical foundations
- :doc:`custom_components` - Building custom uncertainty-aware judges