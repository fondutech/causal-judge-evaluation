Configuration Reference
======================

*Complete guide to CJE-Core configuration with all parameters and examples*

This reference consolidates all configuration examples from across the documentation to reduce duplication and provide a single source of truth.

.. contents:: Quick Navigation
   :local:
   :depth: 2

🚀 Quick Start Configurations
-----------------------------

Minimal Configuration
~~~~~~~~~~~~~~~~~~~~

The absolute minimum to get started:

.. code-block:: yaml

   # config/minimal.yaml
   # Dataset configuration
   dataset:
     name: "./my_data.csv"          # Path to your data file
   
   # Logging policy (what generated the historical data)
   logging_policy:
     provider: "openai"
     model_name: "gpt-3.5-turbo"
     temperature: 0.7
   
   # Target policies (what we want to evaluate)
   target_policies:
     - name: "test"
       provider: "openai"
       model_name: "gpt-4o-mini"
       temperature: 0.7
       mc_samples: 5               # Monte Carlo samples per context
   
   # Judge configuration
   judge:
     provider: "openai"
     model_name: "gpt-4o-mini"
     template: "quick_judge"
     temperature: 0.0              # Deterministic for consistency
   
   # Estimator configuration
   estimator:
     name: "DRCPO"                 # Doubly-robust (recommended)
     k: 5                          # Cross-validation folds
     clip: 20.0                    # Importance weight clipping

Production Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Recommended settings for production use:

.. code-block:: yaml

   # config/production.yaml
   # Dataset configuration
   dataset:
     name: "./data/production.jsonl"
     split: "test"
   
   # Logging policy (what generated the historical data)
   logging_policy:
     provider: "openai"
     model_name: "gpt-3.5-turbo"
     temperature: 0.7
     system_prompt: "You are a helpful assistant."
   
   # Target policies (what we want to evaluate)
   target_policies:
     - name: "enhanced"
       provider: "openai"
       model_name: "gpt-4o"
       temperature: 0.7
       mc_samples: 5               # Monte Carlo samples per context
       system_prompt: "You are an expert assistant with deep knowledge."
   
   # Judge configuration
   judge:
     provider: "openai"
     model_name: "gpt-4o-mini"
     template: "comprehensive_judge"
     temperature: 0.0              # Deterministic for consistency
     
   # Estimator configuration
   estimator:
     name: "DRCPO"                 # Doubly-robust (recommended)
     k: 5                          # Cross-validation folds
     clip: 20.0                    # Importance weight clipping
     n_jobs: -1                    # Use all CPU cores
   
   # Paths configuration
   paths:
     work_dir: "./outputs/production_run"

Arena Configuration
~~~~~~~~~~~~~~~~~~

For large-scale ChatBot Arena-style analysis:

.. code-block:: yaml

   # config/arena.yaml
   # Dataset configuration
   dataset:
     name: "ChatbotArena"          # Built-in dataset
     split: "train"
   
   # Logging policy (what generated the historical data)
   logging_policy:
     provider: "fireworks"
     model_name: "llama-3-8b-instruct"
     temperature: 0.7
   
   # Target policies (what we want to evaluate)
   target_policies:
     - name: "gpt4"
       provider: "openai"
       model_name: "gpt-4o"
       temperature: 0.7
       mc_samples: 3               # Monte Carlo samples per context
     - name: "claude"  
       provider: "anthropic"
       model_name: "claude-3-sonnet-20240229"
       temperature: 0.7
       mc_samples: 3               # Monte Carlo samples per context
   
   # Judge configuration
   judge:
     provider: "openai"
     model_name: "gpt-4o"
     template: "comprehensive_judge"
     temperature: 0.0              # Deterministic for consistency
     
   # Estimator configuration
   estimator:
     name: "MRDR"                  # Model-regularized doubly-robust
     k: 10                         # More folds for larger dataset
     clip: 20.0                    # Importance weight clipping
   
   # Paths configuration
   paths:
     work_dir: "./outputs/arena_analysis"

🔌 Available Providers
----------------------

CJE supports the following providers for models and judges:

.. list-table:: Provider Reference
   :header-rows: 1
   :widths: 15 30 30 25

   * - Provider ID
     - Description
     - Example Models
     - Required Environment
   * - ``openai``
     - OpenAI API
     - gpt-4-turbo, gpt-3.5-turbo
     - ``OPENAI_API_KEY``
   * - ``anthropic``
     - Anthropic API
     - claude-3-sonnet, claude-3-haiku
     - ``ANTHROPIC_API_KEY``
   * - ``google``
     - Google AI/Gemini
     - gemini-pro, gemini-1.5-pro
     - ``GOOGLE_API_KEY``
   * - ``fireworks``
     - Fireworks AI
     - llama-v3-8b-instruct
     - ``FIREWORKS_API_KEY``
   * - ``together``
     - Together AI
     - mixtral-8x7b-instruct
     - ``TOGETHER_API_KEY``
   * - ``hf``
     - HuggingFace local
     - Any HF model
     - Local GPU/CPU
   * - ``mock``
     - Testing/development
     - mock-model
     - None (testing only)

📋 Complete Parameter Reference
------------------------------

Dataset Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   dataset:
     # Required: Data source
     name: "./path/to/data.jsonl"     # File path, CSV, or built-in name
     
     # Optional parameters
     split: "test"                    # Dataset split (train/test/validation)
     max_samples: 1000               # Limit for testing
     shuffle: true                   # Randomize order
     seed: 42                        # Reproducible shuffling

**Supported formats:**
- **JSONL files**: ``.jsonl`` extension
- **CSV/TSV files**: ``.csv``, ``.tsv`` extensions  
- **Built-in datasets**: ``"ChatbotArena"``, ``"PairwiseComparison"``

Paths Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   paths:
     work_dir: "./outputs/experiment"    # Main output directory
     cache_dir: "./cache"               # Cache location (optional)
     
     # Advanced: Override specific paths
     logs_path: "./custom/logs.jsonl"   # Custom log file location
     judge_path: "./custom/judge.jsonl" # Custom judge scores location

Policy Configuration
~~~~~~~~~~~~~~~~~~~

**Logging Policy (π₀):**

.. code-block:: yaml

   logging_policy:
     # Model specification
     model_name: "gpt-3.5-turbo"
     provider: "openai"
     
     # Prompting
     system_prompt: "You are a helpful assistant."
     message_template: "{context}"  # Optional custom template
     
     # Generation parameters
     temperature: 0.7
     max_tokens: 150
     top_p: 1.0
     
     # Advanced
     cache_key: "logging_v1"        # For caching consistency

**Target Policies (π'):**

.. code-block:: yaml

   target_policies:
     - name: "enhanced"              # Required: Policy identifier
       model_name: "gpt-4o"          # Required: Model name
       provider: "openai"            # Required: Provider
       
       # Prompting (same as logging_policy)
       system_prompt: "You are an expert assistant."
       message_template: "{context}"
       
       # Generation parameters
       temperature: 0.3
       max_tokens: 200
       top_p: 0.9
       
       # Evaluation settings
       mc_samples: 5                 # Monte Carlo samples per example
       cache_key: "target_v1"        # Unique cache identifier

**Multiple Target Policies:**

.. code-block:: yaml

   target_policies:
     - name: "conservative"
       model_name: "gpt-4o-mini"
       provider: "openai"
       temperature: 0.1
       mc_samples: 5
       
     - name: "creative"
       model_name: "gpt-4o"
       provider: "openai"
       temperature: 0.9
       mc_samples: 5
       
     - name: "claude_baseline"
       model_name: "claude-3-sonnet-20240229"
       provider: "anthropic"
       temperature: 0.7
       mc_samples: 3

Judge Configuration  
~~~~~~~~~~~~~~~~~~

**OpenAI Judge:**

.. code-block:: yaml

   judge:
     provider: "openai"
     model_name: "gpt-4o-mini"
     
     # Template selection
     template: "comprehensive_judge"   # Built-in template
     
     # Or custom template
     custom_template: |
       Rate the response on a scale of 0-1:
       Context: {context}
       Response: {response}
       Score:
     
     # Generation parameters
     temperature: 0.0                # Low temperature for consistency
     max_tokens: 10

**Anthropic Judge:**

.. code-block:: yaml

   judge:
     provider: "anthropic"
     model_name: "claude-3-haiku-20240307"
     template: "quick_judge"
     temperature: 0.0

**Skip Judge (Use Ground Truth):**

.. code-block:: yaml

   judge:
     skip: true                      # Use ground truth labels from data
     provider: "openai"              # Still required but not used
     model_name: "gpt-3.5-turbo"     # Still required but not used

**Local Model Judge:**

.. code-block:: yaml

   judge:
     provider: "hf"                   # HuggingFace local models
     model_name: "microsoft/deberta-v3-large-mnli"
     device: "cuda"                   # or "cpu", "mps", etc.
     torch_dtype: "auto"              # or "float16", "bfloat16"
     batch_size: 16                   # Batch processing size

Estimator Configuration
~~~~~~~~~~~~~~~~~~~~~~

**IPS (Inverse Propensity Scoring):**

.. code-block:: yaml

   estimator:
     name: "IPS"
     clip: 20.0                     # Importance weight clipping (exp(20) ≈ 485M)
                                    # Theory: Prevents variance explosion from extreme weights
     seed: 42                       # Random seed for reproducibility

**SNIPS (Self-Normalized IPS):**

.. code-block:: yaml

   estimator:
     name: "SNIPS"
     clip: 20.0                     # Weight clipping threshold 
                                    # Theory: SNIPS normalizes weights, reducing bias vs IPS
     seed: 42

**DR-CPO (Doubly Robust - Cross Policy Optimization):**

.. code-block:: yaml

   estimator:
     name: "DRCPO"                  # Implements Algorithm 1 from CJE paper
     k: 5                          # Cross-fitting folds for nuisance estimation
                                    # Theory: Prevents overfitting bias in outcome model
     clip: 20.0                    # Log-ratio clipping before exponentiation
                                    # Theory: exp(20) ≈ 485M max weight, prevents overflow
     calibrate_weights: true       # Isotonic calibration ensuring E[w] = 1
                                    # Theory: CRITICAL for single-rate efficiency (Theorem 5.2)
     calibrate_outcome: true       # Additional outcome model calibration
                                    # Theory: Beyond paper baseline, reduces systematic bias
     n_jobs: -1                    # Parallel processing (-1 = all cores)
     seed: 42

**MRDR (Multiple Robust Doubly Robust):**

.. code-block:: yaml

   estimator:
     name: "MRDR"                   # Variance-optimized outcome model selection
     k: 10                         # More folds for better robustness
                                    # Theory: MRDR benefits from more cross-validation
     clip: 20.0                    # Same clipping as DR-CPO
     calibrate_weights: true       # Weight calibration (same as DR-CPO)
     calibrate_outcome: true       # Outcome calibration (implementation enhancement)
     n_jobs: -1
     seed: 42
     
     # Advanced MRDR parameters
     regularization: 0.01          # Weighted regression regularization
                                    # Theory: Prevents overfitting in weighted least squares
     max_iter: 1000               # Maximum optimization iterations

Provider Configuration
~~~~~~~~~~~~~~~~~~~~~

**OpenAI:**

.. code-block:: yaml

   # In any model configuration
   provider: "openai"
   model_name: "gpt-4o"           # or gpt-4o-mini, gpt-3.5-turbo, etc.
   
   # Optional OpenAI-specific parameters
   api_key: "your-api-key"        # Or set OPENAI_API_KEY env var
   organization: "your-org-id"    # Optional
   api_base: "https://custom.api" # For custom endpoints

**Anthropic:**

.. code-block:: yaml

   provider: "anthropic"
   model_name: "claude-3-sonnet-20240229"
   
   # Optional parameters
   api_key: "your-api-key"        # Or set ANTHROPIC_API_KEY env var

**Google (Gemini):**

.. code-block:: yaml

   provider: "google"
   model_name: "gemini-pro"
   
   api_key: "your-api-key"        # Or set GOOGLE_API_KEY env var

**Fireworks:**

.. code-block:: yaml

   provider: "fireworks"
   model_name: "accounts/fireworks/models/llama-v2-7b-chat"
   
   api_key: "your-api-key"        # Or set FIREWORKS_API_KEY env var

**Together AI:**

.. code-block:: yaml

   provider: "together"
   model_name: "meta-llama/Llama-2-7b-chat-hf"
   
   api_key: "your-api-key"        # Or set TOGETHER_API_KEY env var

🔄 Common Use Case Configurations
--------------------------------

System Prompt Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~

Testing different communication styles:

.. code-block:: yaml

   # config/prompt_engineering.yaml
   dataset:
     name: "./data/customer_queries.csv"
   
   logging_policy:
     model_name: "gpt-4o-mini"
     provider: "openai"
     system_prompt: "You are a helpful customer support agent."
   
   target_policies:
     - name: "professional"
       model_name: "gpt-4o-mini"
       provider: "openai"
       system_prompt: "You are a professional customer support specialist with 10 years of experience."
       mc_samples: 5
       
     - name: "friendly"
       model_name: "gpt-4o-mini"
       provider: "openai"
       system_prompt: "You are a friendly and enthusiastic customer support agent who loves helping people."
       mc_samples: 5
   
   judge:
     provider: "openai"
     model_name: "gpt-4o-mini"
     template: "customer_service_judge"
   
   estimator:
     name: "DRCPO"
     k: 5

Model Comparison
~~~~~~~~~~~~~~~

Comparing different models:

.. code-block:: yaml

   # config/model_comparison.yaml
   dataset:
     name: "./data/benchmark.jsonl"
   
   logging_policy:
     model_name: "gpt-3.5-turbo"
     provider: "openai"
   
   target_policies:
     - name: "gpt4_upgrade"
       model_name: "gpt-4o"
       provider: "openai"
       mc_samples: 5
       
     - name: "claude_alternative"
       model_name: "claude-3-sonnet-20240229"  
       provider: "anthropic"
       mc_samples: 5
   
   judge:
     provider: "openai"
     model_name: "gpt-4o-mini"
   
   estimator:
     name: "DRCPO"

Parameter Tuning
~~~~~~~~~~~~~~~

Optimizing generation parameters:

.. code-block:: yaml

   # config/parameter_tuning.yaml
   dataset:
     name: "./data/creative_tasks.jsonl"
   
   logging_policy:
     model_name: "gpt-4o"
     provider: "openai"
     temperature: 0.7
   
   target_policies:
     - name: "low_temp"
       model_name: "gpt-4o"
       provider: "openai"
       temperature: 0.1
       mc_samples: 5
       
     - name: "high_temp"
       model_name: "gpt-4o"
       provider: "openai"
       temperature: 1.2
       mc_samples: 5
       
     - name: "nucleus_sampling"
       model_name: "gpt-4o"
       provider: "openai"
       temperature: 0.8
       top_p: 0.9
       mc_samples: 5
   
   estimator:
     name: "SNIPS"    # Faster for parameter sweeps

Multi-Provider Comparison
~~~~~~~~~~~~~~~~~~~~~~~~

Comparing models from different providers:

.. code-block:: yaml

   # config/provider_comparison.yaml
   dataset:
     name: "./data/comparison.jsonl"
   
   logging_policy:
     model_name: "gpt-3.5-turbo"
     provider: "openai"
   
   target_policies:
     - name: "gpt4_upgrade"
       model_name: "gpt-4o"
       provider: "openai"
       mc_samples: 5
       
     - name: "claude_alternative"
       model_name: "claude-3-sonnet-20240229"
       provider: "anthropic"
       mc_samples: 5
       
     - name: "fireworks_option"
       model_name: "accounts/fireworks/models/llama-v2-7b-chat"
       provider: "fireworks"
       mc_samples: 5
   
   judge:
     provider: "openai"
     model_name: "gpt-4o-mini"
   
   estimator:
     name: "DRCPO"

⚙️ Advanced Configuration
------------------------

Environment Variables
~~~~~~~~~~~~~~~~~~~

CJE supports environment variable configuration:

.. code-block:: bash

   # API Keys
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export GOOGLE_API_KEY="your-google-key"
   export TOGETHER_API_KEY="your-together-key"
   
   # Cache and output locations
   export CJE_CACHE_DIR="./cache"
   export CJE_OUTPUT_DIR="./outputs"
   
   # Performance tuning
   export CJE_N_JOBS="8"          # Parallel processing
   export CJE_BATCH_SIZE="32"     # Batch size for API calls

Hydra Configuration Override
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Override any parameter from the command line:

.. code-block:: bash

   # Override single parameters
   cje run --cfg-path configs --cfg-name base estimator.name=SNIPS
   
   # Override nested parameters
   cje run --cfg-path configs --cfg-name base judge.model_name=gpt-4o
   
   # Override multiple parameters
   cje run --cfg-path configs --cfg-name base \
     estimator.name=MRDR \
     estimator.k=10 \
     judge.temperature=0.0

Advanced Caching
~~~~~~~~~~~~~~~

Configure caching for different components:

.. code-block:: yaml

   # Cache configuration
   cache:
     # Global cache settings
     enabled: true
     cache_dir: "./cache"
     
     # Component-specific caching
     logs:
       enabled: true
       ttl: 86400              # 24 hours in seconds
       
     judge:
       enabled: true
       ttl: 604800             # 1 week
       
     models:
       enabled: true
       ttl: null               # Never expire

Multi-GPU Configuration
~~~~~~~~~~~~~~~~~~~~~

For large-scale processing with multiple GPUs:

.. code-block:: yaml

   # Distributed processing
   distributed:
     enabled: true
     devices: ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
     
   # Model parallelism for large models
   target_policies:
     - name: "large_model"
       model_name: "meta-llama/Llama-2-70b-chat-hf"
       provider: "hf"
       device_map: "auto"        # Automatic device placement
       load_in_8bit: true        # Enable quantization

🚨 Validation & Troubleshooting
------------------------------

Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~

Always validate your configuration before running:

.. code-block:: bash

   # Validate configuration
   cje validate config --cfg-path configs --cfg-name my_experiment
   
   # Validate data file
   cje validate data my_data.jsonl --verbose
   
   # Quick data check
   cje validate quick my_data.jsonl

Common Configuration Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Missing Required Fields:**

.. code-block:: yaml

   # ❌ Missing provider
   target_policies:
     - name: "test"
       model_name: "gpt-4o"
       # provider: "openai"  # Required!

**Invalid Parameter Values:**

.. code-block:: yaml

   # ❌ Invalid estimator name
   estimator:
     name: "InvalidEstimator"   # Should be IPS, SNIPS, DRCPO, or MRDR

**Inconsistent Configuration:**

.. code-block:: yaml

   # ❌ MRDR with k=1 (should be ≥2)
   estimator:
     name: "MRDR"
     k: 1                      # Should be ≥2 for cross-fitting

Configuration Debugging
~~~~~~~~~~~~~~~~~~~~~~

Debug configuration issues with verbose output:

.. code-block:: bash

   # Enable debug logging
   cje run --cfg-path configs --cfg-name debug \
     hydra.verbose=true \
     hydra.job.chdir=false

   # Resolve and print final configuration
   cje config --cfg-path configs --cfg-name my_experiment

Performance Tuning
~~~~~~~~~~~~~~~~~

Optimize configuration for performance:

.. code-block:: yaml

   # Fast configuration for testing
   target_policies:
     - name: "fast_test"
       model_name: "gpt-4o-mini"    # Faster than gpt-4o
       mc_samples: 1                # Minimum samples
   
   estimator:
     name: "IPS"                    # Fastest estimator
     n_jobs: -1                     # Use all cores
   
   dataset:
     max_samples: 100               # Limit for testing

   # Production configuration for accuracy
   target_policies:
     - name: "production"
       model_name: "gpt-4o"
       mc_samples: 10               # More samples for precision
   
   estimator:
     name: "MRDR"                   # Most robust
     k: 10                          # More folds
   
   judge:
     provider: "openai"
     model_name: "gpt-4o"           # Best judge quality

📚 Configuration Examples Repository
-----------------------------------

All configuration examples are maintained in the ``configs/`` directory:

.. code-block:: text

   configs/
   ├── minimal.yaml              # Minimal working configuration
   ├── production.yaml           # Production-ready settings
   ├── arena.yaml               # Arena analysis configuration
   ├── debugging.yaml           # Debug and development settings
   ├── performance/
   │   ├── fast.yaml            # Optimized for speed
   │   └── accurate.yaml        # Optimized for accuracy
   └── examples/
       ├── prompt_engineering.yaml
       ├── model_comparison.yaml
       └── parameter_tuning.yaml

See the repository for the latest examples and templates you can copy and modify for your use cases. 

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

   estimator:
     log_ratio_clip: 30               # More aggressive clipping
     stabilization_percentile: 80     # Use 80th percentile  
     ess_warning_threshold: 20        # Higher warning threshold

📋 Paper Implementation Guidelines
---------------------------------

*From the CJE paper (Section 6.5 "Deployment Checklist" and Section 4.5 "Implementation Tips")*

Essential Settings
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Logging policy configuration (CRITICAL for theoretical guarantees)
   logging_policy:
     temperature: 0.3               # Minimum for overlap guarantee (paper requirement)
                                    # Theory: temp ≥ 0.3 ensures π'(s|x) > 0 ⇒ π₀(s|x) > 0

   # Calibration configuration
   oracle_slice: 0.25               # 25% oracle data for judge calibration (paper default)
                                    # Theory: Balances calibration accuracy vs evaluation data

   # Weight processing
   estimator:
     clip: 20.0                     # Paper default: clip(100) in tail smoother
                                    # Implementation: exp(20) ≈ 485M for numerical stability
     calibrate_outcome: true        # Implementation enhancement (beyond paper)

Production Deployment Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*From Paper Section 6.5:*

1. **Logging Setup**
   
   .. code-block:: yaml
   
      logging_policy:
        temperature: 0.3             # ≥ 0.3 for support overlap
        logprobs: true               # Essential for exact importance weights
   
2. **Nightly Calibration Jobs**
   
   .. code-block:: yaml
   
      monitoring:
        mse_threshold: 0.1           # Alert if calibration MSE > 0.1
        ess_threshold: 0.1           # Alert if ESS < 10%

3. **Diagnostic Persistence**
   
   .. code-block:: yaml
   
      diagnostics:
        save_per_policy_ess: true    # Monitor effective sample size
        save_clipped_mass: true      # Monitor weight clipping frequency
        save_weight_means: true      # Monitor weight consistency

4. **Launch Gate Configuration**
   
   .. code-block:: yaml
   
      inference:
        confidence_level: 0.95       # Ship π' when CI_lower(π') > CI_upper(π₀)

Theoretical Parameter Guidance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Based on paper theoretical results:*

.. code-block:: yaml

   # Cross-fitting (affects convergence rates)
   estimator:
     k: 5                           # Paper default
                                    # Theory: k=5 balances bias-variance for n^{-1/4} rates
     k: 10                          # For datasets ≤ 5k samples (paper recommendation)

   # Clipping (affects robustness)
   estimator:
     clip: 20.0                     # Conservative (exp(20) ≈ 485M)
     clip: 100.0                    # Paper's "tail smoother default clip(100)"
                                    # Theory: Higher clip preserves more signal but risks variance

   # Outcome model complexity (affects single-rate property)
   outcome_model:
     model_type: "ridge"            # Paper: "start small (ridge or tree-based)"
     complexity: "adaptive"         # Increase only if CI coverage suffers

Cost-Performance Trade-offs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

*From Paper Section 6.4 "Compute Cost":*

.. code-block:: yaml

   # Fast configuration (minimize GPU time)
   estimator:
     name: "SNIPS"                  # Skip outcome model fitting
     mc_samples: 1                  # Reduce target policy sampling

   # Accurate configuration (maximize precision)
   estimator:
     name: "MRDR"                   # Variance-optimal outcome model
     mc_samples: 5                  # More samples for μ_π'(x) estimation
     k: 10                          # More cross-fitting folds

   # Balanced configuration (paper recommendation)
   estimator:
     name: "DRCPO"                  # Doubly-robust with good variance
     mc_samples: 2                  # Adequate for most use cases
     k: 5                           # Standard cross-fitting

Oracle Data Requirements
~~~~~~~~~~~~~~~~~~~~~~~

*From paper calibration methodology:*

.. list-table:: Oracle Data Guidelines
   :header-rows: 1
   :widths: 20 30 50

   * - Dataset Size
     - Oracle Fraction
     - Rationale
   * - < 1k samples
     - 30-40%
     - Need sufficient calibration data per fold
   * - 1k-10k samples  
     - 25% (paper default)
     - Balances calibration vs evaluation
   * - > 10k samples
     - 15-20%
     - Large n allows smaller oracle fraction

.. code-block:: yaml

   # Adaptive oracle sizing
   oracle_configuration:
     min_oracle_per_fold: 10        # Minimum for isotonic regression
     target_oracle_fraction: 0.25   # Paper default
     adaptive_sizing: true          # Adjust based on dataset size

Research vs Production Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Research Mode** *(maximum theoretical purity)*:

.. code-block:: yaml

   estimator:
     stabilize_weights: false       # Disable numerical interventions
     calibrate_outcome: false       # Paper baseline (weight calibration only)
     clip: null                     # No weight clipping

**Production Mode** *(robust deployment)*:

.. code-block:: yaml

   estimator:
     stabilize_weights: true        # Enable numerical stabilization
     calibrate_outcome: true        # Additional robustness layer
     clip: 20.0                     # Conservative clipping for stability

📋 Complete Configuration Example
--------------------------------

Here's a comprehensive configuration file showing all available options:

.. code-block:: yaml

   # config/complete_example.yaml
   # This shows ALL configuration options with their default values
   
   # Path configuration
   paths:
     work_dir: "./outputs/experiment"  # Where to save results
   
   # Dataset configuration
   dataset:
     name: "ChatbotArena"              # Dataset name or path
     split: "train"                    # train/test/validation
     sample_limit: 1000                # Optional: limit samples
     seed: 42                          # Random seed for sampling
   
   # Logging policy (required) - what generated the historical data
   logging_policy:
     provider: "openai"                # Required: provider name
     model_name: "gpt-3.5-turbo"      # Required: model identifier
     temperature: 0.7                  # Sampling temperature
     top_p: 1.0                       # Nucleus sampling (1.0 = disabled)
     max_new_tokens: 512              # Max tokens to generate
     system_prompt: null              # Optional system prompt
     api_key: null                    # Optional: override env var
     base_url: null                   # Optional: custom endpoint
   
   # Target policies (required) - what we want to evaluate
   target_policies:
     - name: "improved_model"         # Policy identifier
       provider: "openai"             # Required: provider name
       model_name: "gpt-4-turbo"      # Required: model identifier
       temperature: 0.7               # Sampling temperature
       top_p: 1.0                     # Nucleus sampling
       max_new_tokens: 512            # Max tokens to generate
       system_prompt: null            # Optional system prompt
       mc_samples: 5                  # Monte Carlo samples per context
       api_key: null                  # Optional: override env var
       base_url: null                 # Optional: custom endpoint
   
   # Judge configuration (required)
   judge:
     provider: "openai"               # Required: provider name
     model_name: "gpt-4-turbo"        # Required: model identifier
     template: "quick_judge"          # Template name
     temperature: 0.0                 # Low temp for consistency
     max_tokens: 100                  # Max tokens for judgment
     max_retries: 3                   # Retry attempts
     timeout: 30                      # Timeout in seconds
     api_key: null                    # Optional: override env var
     base_url: null                   # Optional: custom endpoint
     skip: false                      # Skip judging (use ground truth)
   
   # Estimator configuration (required)
   estimator:
     name: "DRCPO"                    # IPS/SNIPS/DRCPO/MRDR
     k: 5                             # Cross-validation folds
     clip: 20.0                       # Log-ratio clipping threshold
     seed: 42                         # Random seed
     n_jobs: -1                       # Parallel jobs (-1 = all cores)
     # Advanced options
     outcome_model_cls: null          # Custom outcome model class
     outcome_model_kwargs: {}         # Outcome model parameters
     featurizer: null                 # Custom featurizer
     calibrate_weights: true          # Isotonic weight calibration
     calibrate_outcome: false         # Outcome model calibration
     stabilize_weights: true          # Numerical stabilization
   
   # Oracle configuration (optional)
   oracle:
     enabled: false                   # Enable oracle labeling
     provider: "openai"               # Oracle provider
     model_name: "gpt-4-turbo"        # Oracle model
     template: "quick_judge"          # Oracle template
     temperature: 0.0                 # Oracle temperature
     max_tokens: 100                  # Max tokens
     logging_policy_oracle_fraction: 0.25  # Fraction for calibration
     seed: 42                         # Random seed
   
   # Research configuration (optional)
   research:
     enabled: false                   # Enable research mode
     gold_validation:
       enabled: false                 # Create validation set
       samples_per_target: 100        # Samples per target policy
       create_ab_pairs: true          # Create A/B comparisons
       shuffle_pairs: true            # Randomize pair order
     diagnostics:
       enabled: true                  # Enable diagnostics
       mean_bias_threshold: 0.2       # Bias threshold
       spearman_threshold: 0.6        # Correlation threshold
       clipped_mass_threshold: 0.01   # Weight clipping threshold
       ess_threshold: 0.25            # ESS threshold
   
   # Weight diagnostics (optional)
   diagnostics:
     log_ratio_clip: 20.0            # Hard clipping for log ratios
     ess_warning_threshold: 15.0     # ESS warning (% of n)
     ess_critical_threshold: 5.0     # ESS critical (% of n)
     identical_policy_tolerance: 0.1 # Tolerance for policy comparison
     save_diagnostic_plots: true     # Save weight distribution plots

This configuration framework ensures both theoretical fidelity and production robustness, with clear guidance on when to deviate from paper defaults. 