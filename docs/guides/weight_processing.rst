Weight Processing Pipeline
=========================

**Turning Raw Probabilities into Reliable Policy Estimates**

This guide explains CJE's sophisticated weight processing pipeline—why each step exists, what problems it solves, and how the components work together to produce stable, unbiased causal estimates.

Why Weight Processing Matters
-----------------------------

In off-policy evaluation, we estimate how well a target policy π' would perform using only data from a different behavior policy π₀. The core challenge: some responses are much more likely under π' than π₀, leading to extreme importance weights that can make estimates unreliable.

**The Fundamental Problem**:

- If π'(response|context) >> π₀(response|context), the weight explodes → high variance
- If π'(response|context) << π₀(response|context), the weight vanishes → that sample contributes nothing
- A few extreme weights can dominate the entire estimate

**Our Solution**: A carefully designed pipeline that stabilizes weights while preserving the unbiasedness guarantees of causal inference.

Pipeline Overview
-----------------

.. code-block:: text

   Raw Log Probs → Hard Clipping → Soft Stabilization → Exponentiation → Cross-Fold Calibration → DR Estimation
        |              |                   |                  |                |                    |
    log π'(s|x)    Prevent         Normalize each      w = exp(·)      Achieve            v̂ = μ̂ + w(r-μ̂)
    log π₀(s|x)    overflow        policy fairly                       E[w] = 1              
        |              |                   |                  |                |                    |
        └─────────────────────────────────────────────────────────────────────────────────────────┘
                                              |
                                      📊 Diagnostic Monitoring
                                      • ESS computation & warnings
                                      • Overlap analysis
                                      • Consistency checking

Stage 1: Computing Log Importance Ratios
----------------------------------------

**What**: Calculate log(π'/π₀) for each response

**Why**: Working in log space prevents numerical underflow for tiny probabilities

**How**:

.. code-block:: python

   # For each context-response pair
   log_ratio = log_prob_target - log_prob_behavior
   
   # Shape: (n_samples, n_policies)
   # Range: Unbounded (can be ±∞ for extreme mismatches)

**Potential Issues**:

- Chat API vs Completions API can give different probabilities for the same text
- Teacher forcing bugs show up here as inconsistent weights for identical policies

Stage 2: Hard Clipping (Overflow Prevention)
--------------------------------------------

**What**: Clip log ratios to [-20, +20]

**Why**: 

- exp(20) ≈ 485 million - already extreme
- exp(50) causes numerical overflow
- exp(-50) causes numerical underflow

**How**:

.. code-block:: python

   # Default: log_ratio_clip = 20.0
   if abs(log_ratio) > log_ratio_clip:
       log_ratio = clip(log_ratio, -20, +20)

**Effect**: 

- Prevents computational crashes
- Limits maximum weight ratio to ~485M:1
- Applied globally to all policies

**Trade-off**: Introduces bias for truly extreme policy differences, but prevents infinite variance

Stage 3: Soft Stabilization (The Key Innovation)
------------------------------------------------

**What**: Subtract 75th percentile of each policy's log weights

**Why Previous Approaches Failed**:

1. **Global max subtraction**: Creates winner-take-all where one policy dominates
2. **No stabilization**: Allows numerical instability
3. **Aggressive clipping**: Destroys weight diversity

**Our Solution**:

.. code-block:: python

   # Per-policy normalization (not global!)
   for each policy k:
       percentile_75 = percentile(log_weights[:, k], 75)
       stabilized_log_weights[:, k] = log_weights[:, k] - percentile_75

**Why This Works**:

- **Preserves relative differences**: Weights maintain their ordering within each policy
- **Fair comparison**: Each policy normalized independently, preventing one from dominating due to scale
- **Numerical stability**: Brings log weights into reasonable range before exponentiation
- **Triggered adaptively**: Only applies when |log_weight| > 10

**Example**:

Before stabilization:
- Policy A weights: [1e-10, 1e-8, 1e-6, 1e-4]  (all tiny)
- Policy B weights: [1e4, 1e6, 1e8, 1e10]      (all huge)

After stabilization:
- Policy A weights: [0.01, 0.1, 1, 10]         (reasonable range)
- Policy B weights: [0.01, 0.1, 1, 10]         (same range, fair comparison)

Stage 4: Exponentiation
-----------------------

**What**: Convert log weights back to weights: w = exp(log_weight)

**Why float64**: 

- float32 overflows at exp(~89)
- float64 handles up to exp(~709)
- Critical for numerical stability

**How**:

.. code-block:: python

   # Cast to float64 before exp to prevent overflow
   weights = np.exp(log_weights.astype(np.float64))

Stage 5: ESS Monitoring & Diagnostics
-------------------------------------

**What**: Compute Effective Sample Size and flag issues

**Why ESS Matters**:

ESS measures how many "effective" samples you have after importance weighting:

- ESS = 100%: Perfect overlap, all samples equally useful
- ESS = 10%: Only 10% of your samples effectively contribute
- ESS < 5%: Estimates dominated by very few samples (unreliable)

**How**:

.. code-block:: python

   # For each policy
   ESS = (sum(weights))² / sum(weights²)
   ESS_percent = 100 * ESS / n_samples
   
   if ESS_percent < 5:
       print("🚨 CRITICAL: Estimates will be unreliable!")
   elif ESS_percent < 15:
       print("⚠️  WARNING: Estimates may be noisy")

**Diagnostics Provided**:

1. **Per-policy ESS**: Not averaged - each policy assessed independently
2. **Overlap analysis**: Quantifies distribution alignment
3. **Consistency checking**: Flags when identical policies have non-unit weights

Stage 6: Cross-Fold Isotonic Calibration
----------------------------------------

**What**: Transform weights to have exact mean = 1.0 per fold

**Why This Is Critical**:

Doubly-robust estimation requires E[w] = 1. Without calibration:
- Raw weights often have mean ≠ 1 due to finite sample effects
- This introduces bias even with perfect outcome models
- Variance can be unnecessarily high

**How Isotonic Regression Works**:

.. code-block:: python

   # For each cross-validation fold
   1. Sort weights: [0.1, 0.5, 2.0, 10.0, 50.0]
   2. Create target sequence with same mean=1.0
   3. Fit monotonic function: f(raw_weight) → calibrated_weight
   4. Apply to all weights in fold
   5. Rescale to ensure exact mean = 1.0

**Key Properties**:

- **Monotonic**: Preserves weight ordering (higher stays higher)
- **Exact calibration**: Achieves E[w] = 1 precisely
- **Cross-fit**: Prevents overfitting via k-fold procedure

**Theoretical Guarantee**: This calibration maintains the √n convergence rate of doubly-robust estimators while reducing finite-sample bias.

Stage 7: Doubly-Robust Estimation
---------------------------------

**What**: Combine calibrated weights with outcome predictions

**The DR Formula**:

.. code-block:: python

   # For each sample i and policy k
   ψᵢᵏ = μ̂ᵏ(xᵢ) + wᵢᵏ * (rᵢ - μ̂(xᵢ, yᵢ))
   
   # Policy value estimate
   v̂ᵏ = mean(ψᵏ)

**Why DR Works**:

1. **Outcome model term** μ̂ᵏ(x): Provides low variance baseline
2. **Correction term** w(r - μ̂): Fixes bias from imperfect outcome model
3. **Double robustness**: Consistent if either weights OR outcome model is correct

Diagnostic Tools & Monitoring
-----------------------------

CJE provides comprehensive diagnostics to catch issues early:

**Weight Distribution Analysis**:

.. code-block:: python

   from cje.utils.weight_diagnostics import diagnose_weights_with_overlap
   
   diagnostics = diagnose_weights_with_overlap(
       weights, behavior_logprobs, target_logprobs
   )
   
   # Automatic status flags:
   # 🟢 GOOD: ESS > 10%, good overlap
   # 🟡 WARNING: ESS 5-10%, moderate issues  
   # 🔴 CRITICAL: ESS < 5%, poor overlap

**Visual Diagnostics**:

1. **Weight distributions**: Histograms with reference lines
2. **ESS comparison**: Bar charts across policies
3. **Overlap visualization**: Log-ratio percentile plots
4. **Diagnostic dashboard**: Complete HTML report

**Consistency Checking**:

For identical policies (same model, prompt, temperature), weights should = 1.0:

.. code-block:: python

   if config_matches_behavior(policy_config):
       if abs(mean_weight - 1.0) > 0.1:
           # Red flag: Teacher forcing or computation bug!

When Things Go Wrong
--------------------

**Symptom**: ESS < 5% (CRITICAL)

**Causes & Solutions**:

1. **Poor overlap**: Target policy very different from behavior
   - Solution: Collect more diverse behavior data
   - Solution: Use less extreme target policies

2. **Teacher forcing bugs**: Inconsistent probability computation
   - Diagnostic: Check if identical policies have weight ≈ 1.0
   - Solution: Fix API usage (chat vs completions)

3. **Extreme prompts**: Massive distribution shift
   - Solution: Use MRDR (model-regularized) estimator
   - Solution: Increase sample size

**Symptom**: Numerical instability

**Solutions**:

1. Enable all stabilization (default)
2. Reduce log_ratio_clip if needed
3. Check for log probability computation bugs

Configuration Examples
----------------------

**Default (Recommended)**:

.. code-block:: yaml

   estimator:
     name: DRCPO
     stabilize_weights: true      # Soft stabilization
     calibrate_weights: true      # Isotonic calibration
     calibrate_outcome: true      # Outcome calibration

**Conservative Mode** (Extreme datasets):

.. code-block:: yaml

   estimator:
     name: DRCPO
     log_ratio_clip: 15          # More aggressive clipping
     clip: 1000                  # Legacy weight clipping

**Research Mode** (Theoretical purity):

.. code-block:: yaml

   estimator:
     name: DRCPO
     stabilize_weights: false    # No stabilization
     calibrate_weights: false    # No calibration
     clip: null                  # No clipping

Key Takeaways
-------------

1. **Each stage addresses a specific failure mode** - from numerical overflow to finite-sample bias
2. **Soft stabilization is the key innovation** - preserves diversity while ensuring stability  
3. **Calibration is essential** - transforms theoretical guarantees into practical reliability
4. **Diagnostics prevent silent failures** - ESS and overlap metrics flag issues early
5. **The pipeline is configurable** - adjust for your specific needs

The result: A robust system that turns wild importance weights into reliable policy value estimates, with clear diagnostics when things go wrong.