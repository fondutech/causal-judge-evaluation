Weight Processing Pipeline
=========================

**Complete Pipeline: Raw Log Probabilities → Final Estimates**

This guide describes the full weight processing pipeline in CJE, documenting every transformation from raw log probabilities to final policy value estimates, plus the new diagnostic and visualization capabilities for catching weight issues early.

Pipeline Overview
-----------------

.. code-block:: text

   Raw Log Probs → Hard Clipping → Soft Stabilization → Exponentiation → Cross-Fold Calibration → DR Estimation
        |              |                   |                  |                |                    |
    log π'(s|x)    log-ratio         percentile-based      w = exp(·)    isotonic           v̂ = E[ψ]
    log π₀(s|x)      ±20.0           subtraction                        regression              
        |              |                   |                  |                |                    |
        └─────────────────────────────────────────────────────────────────────────────────────────┘
                                              |
                                      📊 Weight Diagnostics
                                      • ESS computation & warnings
                                      • Identical policy consistency  
                                      • Visual diagnostic plots
                                      • Automatic status flagging
                                      • Early bug detection

🔍 Weight Diagnostics & Visualization
-------------------------------------

Automatic Weight Health Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting with CJE v2024+, all arena analysis automatically includes importance weight diagnostics to catch consistency issues, low ESS, and teacher forcing problems early.

**Location**: ``cje/utils/weight_diagnostics.py``  
**Integration**: Automatic in ``arena_analysis_python.py`` and ``ArenaAnalyzer``

Key Diagnostic Metrics
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass
   class WeightDiagnostics:
       policy_name: str
       min_weight: float
       max_weight: float  
       mean_weight: float
       ess_fraction: float              # Effective Sample Size as % of N
       extreme_weight_count: int        # Weights > 1000 or < 0.001
       zero_weight_count: int           # Exactly zero weights
       consistency_flag: str            # "GOOD", "WARNING", "CRITICAL"
       weight_coefficient_variation: float

Automatic Status Flags
~~~~~~~~~~~~~~~~~~~~~~

- **🟢 GOOD**: ESS ≥ 10%, <10% extreme weights, mean ≈ expected for identical policies
- **🟡 WARNING**: 1% ≤ ESS < 10%, or 10-50% extreme weights  
- **🔴 CRITICAL**: ESS < 1%, >50% extreme weights, or mean weight far from expected

Identical Policy Consistency Checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Critical Feature**: Automatically detects when "identical" policies (e.g., scout = logging) don't have weights ≈ 1.0:

.. code-block:: python

   expected_weight = 1.0 if "scout" in policy_name.lower() else None
   weight_error = abs(mean_weight - expected_weight)
   if weight_error > 0.1:
       return "CRITICAL"  # Indicates teacher forcing or policy configuration issues

**Real-World Example**: Caught teacher forcing bugs where scout policy had weights of 10^-30 instead of 1.0, revealing computation inconsistencies.

Arena Analysis Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Automatic Display**: All arena analysis now shows weight diagnostics:

.. code-block:: text

   📊 Importance Weight Summary

   | Policy | ESS | Mean Weight | Status | Issues |
   |--------|-----|-------------|--------|--------|
   | llama4_scout | 100.0% | 1.0000 | ✅ GOOD | None |
   | llama4_maverick | 1.0% | 0.0101 | ❌ CRITICAL | Low ESS (1.0%), 97 extreme |

   ✅ All importance weights look healthy

Interactive Diagnostic Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from examples.arena_interactive import ArenaAnalyzer

   analyzer = ArenaAnalyzer()
   analyzer.run_analysis('arena_analysis')    # Includes automatic diagnostics

   # Additional weight-specific methods:
   analyzer.plot_weight_diagnostics()         # Visual diagnostic dashboard  
   analyzer.quick_weight_check('scout')       # Quick visual check for one policy
   analyzer.diagnose_weights()                # Get detailed diagnostic objects

Standalone Diagnostic Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cje.utils.weight_diagnostics import analyze_arena_weights
   from cje.utils.weight_plots import create_weight_diagnostic_dashboard

   # Analyze any arena data
   diagnostics = analyze_arena_weights(arena_data)

   # Create visual dashboard (saves PNG files)
   create_weight_diagnostic_dashboard(arena_data, "diagnostics_output/")

Visual Diagnostics
~~~~~~~~~~~~~~~~~~

**Weight Distribution Plots**: 

- Histogram of log₁₀(weights) with expected=1.0 reference line
- Scatter plot of weights vs sample index (detect patterns)
- Color-coded titles by diagnostic status (green/orange/red)

**ESS Comparison Charts**:

- Bar chart comparing ESS across policies
- Warning/critical threshold reference lines  
- Percentage labels and status color-coding

**Diagnostic Dashboard**: Complete set of plots automatically saved as PNG files

Success Story: Teacher Forcing Bug Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The weight diagnostics caught a critical teacher forcing implementation bug:

**🔴 Before Fix**:

- Scout policy weights: 10^-30 to 10^20 (should be ≈1.0)  
- ESS: 5.3% (critical)
- 91% extreme weights
- Status: CRITICAL with clear guidance

**🟢 After Fix**:

- Scout policy weights: Exactly 1.0 (perfect)
- ESS: 100% (perfect)  
- 0% extreme weights
- Status: GOOD

**Key Insight**: Weight inconsistency served as the perfect "canary in the coal mine" 🐤, revealing fundamental teacher forcing computation problems that would have been hard to detect otherwise.

⚙️ Stage 1: Raw Log Probability Computation
-------------------------------------------

**Location**: ``MultiTargetSampler.importance_weights_matrix()``  
**Input**: ``(contexts, responses, logp_behavior)``  
**Output**: ``log_weights_matrix`` (raw log importance ratios)

.. code-block:: python

   # Compute log importance weights: log π'(s|x) - log π₀(s|x)
   log_weights_matrix = logp_matrix - logp_behavior_array[:, np.newaxis]

**Shape**: ``(n_samples, n_policies)``  
**Range**: Unbounded (can be ±∞ for pathological cases)

⚙️ Stage 2: Hard Log-Ratio Clipping ✂️
--------------------------------------

**Location**: ``MultiTargetSampler.importance_weights_matrix()`` (lines 285-295)  
**Purpose**: Prevent astronomical weights that cause overflow/underflow

Default Parameters
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   log_ratio_clip = 20.0  # ± 20 log units
   # Max weight ratio: exp(20) ≈ 485,165,195 (485M)

Logic
~~~~~

.. code-block:: python

   if np.any(np.abs(log_weights_matrix) > log_ratio_clip):
       console.print("✂️  Hard clipping log ratios to ±20.0 (prevents exp overflow)")
       log_weights_matrix = np.clip(log_weights_matrix, -log_ratio_clip, log_ratio_clip)

**Effect**: Caps extreme log ratios before they can cause numerical issues

⚙️ Stage 3: Soft Stabilization 🎯
---------------------------------

**Location**: ``MultiTargetSampler.importance_weights_matrix()`` (lines 310-340)  
**Purpose**: Prevent winner-take-all while preserving weight diversity and treating policies fairly

Default Parameters
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   stabilization_threshold = 10.0  # Trigger when |log_weight| > 10
   percentile_for_subtraction = 75  # Use 75th percentile per policy (not global)

Logic
~~~~~

.. code-block:: python

   if stabilize and np.any(np.abs(log_weights_matrix) > 10):
       console.print("🔧 Applying soft numerical stabilization (preserves weight diversity)")
       
       # Softer approach: subtract 75th percentile per policy instead of global max
       # This prevents winner-take-all while treating each policy fairly
       percentile_75_per_policy = np.percentile(log_weights_matrix, 75, axis=0)
       stabilized_log_weights = log_weights_matrix - percentile_75_per_policy
       
       # More generous clipping bounds to preserve diversity
       if clip is not None:
           log_clip = np.log(clip)
           max_stabilized = np.max(stabilized_log_weights)
           stabilized_log_weights = np.clip(
               stabilized_log_weights, 
               max_stabilized - log_clip,  # Preserves relative ratios
               max_stabilized              # Upper bound at current max
           )

**Effect**: Prevents single weights from dominating while keeping weight diversity

⚙️ Stage 4: Exponentiation & Legacy Clipping
--------------------------------------------

**Location**: ``MultiTargetSampler.importance_weights_matrix()`` (lines 345-365)

Default Parameters
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   clip = None  # Legacy clipping disabled by default (redundant with log-space protection)

Logic
~~~~~

.. code-block:: python

   # Exponentiate stabilized weights (cast to float64 to prevent overflow)
   # float32 overflows at exp(≈88.7), but float64 handles up to exp(≈700)
   weights_matrix = np.exp(stabilized_log_weights.astype(np.float64))

   # Legacy clipping (disabled by default - hard log-ratio clipping provides protection)
   # When enabled, applied in log-space to preserve relative ratios
   if clip is not None:
       # Applied during stabilization in log-space, not here
       pass

   # Final safety check (prevent negatives - should be unnecessary)
   weights_matrix = np.maximum(weights_matrix, 0)

**Output**: Raw importance weights matrix ``(n_samples, n_policies)``

⚙️ Stage 5: ESS Diagnostic & Guard-Rails 🚨
-------------------------------------------

**Location**: ``MultiTargetSampler.importance_weights_matrix()`` (lines 370-390)

Default Thresholds
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   critical_ess_threshold = 5.0   # % of samples
   warning_ess_threshold = 15.0   # % of samples

Logic
~~~~~

.. code-block:: python

   # Compute ESS per policy (not averaged!)
   ess_values = [ESS_k for each policy k]
   ess_percentages = [100 * ess_k / n_samples for ess_k in ess_values]

   # Per-policy guard-rails
   for k, ess_pct in enumerate(ess_percentages):
       if ess_pct < 5.0:
           console.print(f"🚨 CRITICAL: {policy_name}: {ess_pct:.1f}% - estimates unreliable!")
       elif ess_pct < 15.0:
           console.print(f"⚠️  WARNING: {policy_name}: {ess_pct:.1f}% - estimates may be noisy")

   if all_healthy:
       console.print(f"✅ All policies have healthy ESS (min: {min(ess_percentages):.1f}%)")

**Output**: Diagnostic warnings + weight statistics

⚙️ Stage 6: Cross-Fold Isotonic Weight Calibration 📊
-----------------------------------------------------

**Location**: ``DRCPO._process_fold()`` → ``calibrate_weights_isotonic()``  
**Purpose**: Achieve exact target mean (1.0) while preserving monotonicity

Default Parameters
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   target_mean = 1.0                     # Target mean for calibrated weights  
   max_calibrated_weight = 500.0         # Hard cap for calibrated weights
   min_samples_for_calibration = 10      # Minimum samples per fold

Algorithm
~~~~~~~~~

.. code-block:: python

   def calibrate_weights_isotonic(weights, fold_indices, target_mean=1.0):
       for fold in unique_folds:
           fold_weights = weights[fold_mask]
           
           # Fit isotonic regression: calibrated_weight = f(raw_weight)
           # Map to exponentially spaced targets that achieve target_mean
           iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip')
           iso_reg.fit(sorted_weights, exp_targets)
           
           # Apply calibration
           calibrated_fold = iso_reg.predict(fold_weights)
           
           # Ensure exact target mean
           achieved_mean = np.mean(calibrated_fold)
           if achieved_mean > 1e-12:
               calibrated_fold = calibrated_fold * (target_mean / achieved_mean)
           
           # Apply hard cap
           calibrated_fold = np.minimum(calibrated_fold, max_calibrated_weight)
           
           # 🔧 CRITICAL: Re-scale after capping to maintain E[w]=target_mean
           # Capping high weights lowers the mean, introducing finite-sample bias
           capped_mean = np.mean(calibrated_fold)
           if capped_mean > 1e-12:
               calibrated_fold = calibrated_fold * (target_mean / capped_mean)

**Effect**: Transforms raw weights to have exactly mean=1.0 while preserving order

.. warning::
   **Critical Theoretical Note**: The re-scaling after capping is essential to maintain DR's unbiasedness guarantee. Without it, capping high weights introduces finite-sample bias by lowering E[w] below 1.0, which violates the theoretical foundation of doubly-robust estimation.

⚙️ Stage 7: Outcome Model Calibration (Optional)
------------------------------------------------

**Location**: ``DRCPO._process_fold()`` → ``calibrate_outcome_model_isotonic()``  
**Purpose**: Calibrate outcome model predictions against true rewards

**Default**: ``calibrate_outcome = True``

.. code-block:: python

   if self.calibrate_outcome:
       # Calibrate outcome model predictions against training rewards
       calibration_fn, diagnostics = calibrate_outcome_model_isotonic(
           train_preds, train_rewards
       )
       mu_hat_test_calibrated = calibration_fn(mu_hat_test)

**Effect**: Corrects systematic bias in outcome model predictions

⚙️ Stage 8: Doubly-Robust Estimation
------------------------------------

**Location**: ``DRCPO._process_fold()``  
**Formula**: Final DR estimate per policy

.. code-block:: python

   # EIF components: μ_πᵏ(x) + wᵏ * (r - μ(x,y))
   # Uses calibrated weights and/or calibrated outcome model
   eif_test = mu_pi_test + W_test_calibrated * (
       r_test[:, np.newaxis] - mu_hat_test_calibrated[:, np.newaxis]
   )

   # Final estimate: v̂ᵏ = (1/n) Σᵢ ψᵢᵏ
   v_hat = np.mean(eif_all, axis=0)

🎛️ Configuration Options
------------------------

In Code Constants
~~~~~~~~~~~~~~~~~

Easy to modify:

.. code-block:: python

   # Stage 2: Hard clipping
   log_ratio_clip = 20.0  # in MultiTargetSampler.importance_weights_matrix()

   # Stage 3: Soft stabilization  
   stabilization_threshold = 10.0    # Trigger threshold
   percentile_for_subtraction = 75   # Use 75th percentile per policy (axis=0)

   # Stage 5: ESS guard-rails
   critical_ess_threshold = 5.0      # % for critical warning
   warning_ess_threshold = 15.0      # % for warning

In YAML Config
~~~~~~~~~~~~~~

.. code-block:: yaml

   # Estimator configuration with weight processing options
   estimator:
     name: "DRCPO"                   # Doubly-robust (recommended)
     k: 5                            # Cross-validation folds
     clip: null                      # Stage 4: Legacy clipping disabled (default)
     stabilize_weights: true         # Stage 3: Enable/disable stabilization
     calibrate_weights: true         # Stage 6: Enable/disable weight calibration  
     calibrate_outcome: true         # Stage 7: Enable/disable outcome calibration

   # Weight diagnostic configuration (automatic in arena analysis)
   diagnostics:
     ess_warning_threshold: 10.0     # ESS % warning threshold  
     ess_critical_threshold: 1.0     # ESS % critical threshold
     extreme_weight_threshold: 1000  # Define "extreme" weights
     save_diagnostic_plots: true     # Auto-save weight distribution plots
     identical_policy_tolerance: 0.1 # Tolerance for identical policy weight checking

Conservative Mode
~~~~~~~~~~~~~~~~~

For extreme datasets:

.. code-block:: yaml

   # Estimator configuration (conservative mode)
   estimator:
     name: "DRCPO"                   # Doubly-robust (recommended)
     k: 5                            # Cross-validation folds
     clip: 5000.0                    # Enable legacy clipping with high threshold

Research Mode
~~~~~~~~~~~~~

Maximum theoretical purity:

.. code-block:: yaml

   # Estimator configuration (research mode)
   estimator:
     name: "DRCPO"                   # Doubly-robust (recommended)
     k: 5                            # Cross-validation folds
     clip: null                      # No weight clipping (default)
     stabilize_weights: false        # Disable stabilization
     calibrate_weights: false        # Disable calibration

📊 Typical Output Flow
---------------------

Normal Case
~~~~~~~~~~~

No interventions needed:

.. code-block:: text

   Computing importance weights for 2 policies...
   ✅ ESS looks healthy (25.3%)
   ✓ Isotonic weight calibration enabled for DRCPO
   ✓ Cross-validation complete!

Extreme Case
~~~~~~~~~~~~

All interventions triggered:

.. code-block:: text

   Computing importance weights for 2 policies...
   ✂️  Hard clipping log ratios to ±20.0 (prevents exp overflow)
      • Original range: [-19.3, 723.2]
      • Clipped range: [-19.3, 20.0]
   🔧 Applying soft numerical stabilization (preserves weight diversity)
      • Original log weight range: [-19.3, 20.0]  
      • Stabilized log weight range: [-1.3, 7.9]
      📊 ESS per policy: ['28.6', '5.3'] / 100
      📊 ESS percentages: ['28.6%', '5.3%'] (avg: 16.9%)
   ⚠️  LOW ESS warnings:
      • llama4_scout: 5.3% - estimates may be noisy
      💡 Consider: More samples or different target policies
      ✅ Preserved weight differences across policies
   ✓ Isotonic weight calibration enabled for DRCPO
   ✓ Cross-validation complete!

🏆 Key Design Principles
-----------------------

1. **Fail Safe**: System degrades gracefully under extreme conditions
2. **Preserve Signal**: Clipping/stabilization maintains relative policy differences  
3. **Exact Calibration**: Isotonic regression achieves exact target statistics
4. **Actionable Warnings**: Users get clear guidance when ESS is low
5. **Research Friendly**: All interventions can be disabled for theoretical work
6. **Numerical Safety**: float64 casting prevents silent overflow corruption

🔧 Advanced Customization
------------------------

To expose more parameters in YAML config:

.. code-block:: python

   # In MultiTargetSampler.importance_weights_matrix()
   log_ratio_clip = cfg.get('log_ratio_clip', 20.0)
   stabilization_percentile = cfg.get('stabilization_percentile', 75)
   ess_warning_threshold = cfg.get('ess_warning_threshold', 15.0)

Then in YAML:

.. code-block:: yaml

   # Estimator configuration (advanced parameters)
   estimator:
     name: "DRCPO"                    # Doubly-robust (recommended)
     k: 5                             # Cross-validation folds
     log_ratio_clip: 30               # More aggressive clipping
     stabilization_percentile: 80     # Use 80th percentile  
     ess_warning_threshold: 20        # Higher warning threshold

This pipeline ensures robust, reliable policy evaluation while maintaining theoretical soundness and providing clear diagnostics at every stage.

🛠️ Teacher Forcing Consistency
------------------------------

Importance Weight as Diagnostic Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Key Insight**: For identical policies (e.g., scout = logging), importance weights should be exactly 1.0. Deviations indicate fundamental computation issues.

**Diagnostic Philosophy**: 

- ❌ **Wrong**: "Weights look inconsistent, let me manually set scout weights = behavior weights"
- ✅ **Right**: "Weights look inconsistent, this reveals a bug in teacher forcing computation"

**Example Detection**:

.. code-block:: python

   if policy_is_identical_to_behavior(policy_name):
       expected_weight = 1.0
       weight_error = abs(mean_weight - expected_weight)
       if weight_error > 0.1:
           print(f"🚨 DIAGNOSTIC: {policy_name} should have weights ≈ 1.0, got {mean_weight:.2e}")
           print("   This indicates teacher forcing computation inconsistency") 