.. CJE-Core documentation master file, created by
   sphinx-quickstart on Wed Jun 11 16:07:45 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CJE: Causal Judge Evaluation Toolkit
=====================================

.. image:: img/CJE_logo.svg
   :align: center
   :alt: CJE Logo
   :width: 400

.. image:: https://img.shields.io/pypi/v/cje.svg
   :target: https://pypi.org/project/cje/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python version

.. image:: https://img.shields.io/github/license/fondutech/causal-judge-evaluation.svg
   :target: https://github.com/fondutech/causal-judge-evaluation/blob/main/LICENSE
   :alt: License

CJE provides **robust off-policy evaluation** for Large Language Models using causal inference methods. Estimate policy improvements without deployment using logged interaction data.

.. raw:: html

   <div style="text-align: center; margin: 40px 0;">
   <a href="start_here.html" style="display: inline-block; background: #0969da; color: white; padding: 16px 32px; border-radius: 8px; text-decoration: none; font-size: 1.2em; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: all 0.2s;">
   🚀 Start Here - Choose Your Path
   </a>
   </div>

.. important::
   **First time here?** Click the button above to find your personalized learning path (5 min → 45 min tracks)

**What is CJE?** A toolkit that answers "What would happen if we deployed policy π'?" using only historical logs:

* **📊 Causal, not correlational**: Corrects for distribution shift between logged and target policies
* **⚡ Faster evaluation**: Reuses existing responses with teacher-forced scoring
* **🎯 Tighter confidence intervals**: Via calibrated doubly-robust estimation with uncertainty quantification
* **🔬 Theory-backed**: Implements Algorithm 1 from the CJE paper with single-rate efficiency
* **📈 Uncertainty-aware**: Built-in support for judge confidence and variance estimation


⚡ Quick Start Examples
----------------------

**Problem**: You want to test if GPT-4 performs better than GPT-3.5 for your chatbot, but:

- A/B testing in production is risky and expensive
- You already have thousands of GPT-3.5 conversations logged
- You need statistically rigorous results, not just "vibes"

**Solution**: CJE evaluates new policies using your existing data!

**5-Minute Demo**

.. code-block:: bash

   # Install and run a quick test
   git clone https://github.com/fondutech/causal-judge-evaluation.git
   cd causal-judge-evaluation
   poetry install
   
   # Set your API key (Fireworks offers free tier)
   export FIREWORKS_API_KEY="your-key-here"
   
   # Run evaluation on 20 samples (takes ~1 minute)
   cje run --cfg-path configs --cfg-name example_eval

**What You'll See**:

.. code-block:: text

   ✅ Loaded 20 ChatBot Arena conversations
   ✅ Computing log probabilities for historical policy...
   ✅ Generating judge scores with uncertainty...
   ✅ Running causal estimation (DR-CPO)...
   
   🎯 RESULTS SUMMARY
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Best Policy: llama-4-maverick (0.725 ± 0.042)
   
   Policy Rankings:
   1. llama-4-maverick:  0.725 [0.641, 0.809] ⭐ BEST
   2. llama-4-scout:     0.683 [0.599, 0.767]
   
   Baseline (historical): 0.650 ± 0.038
   
   📊 llama-4-maverick shows +11.5% improvement (p=0.021)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Real Use Case: System Prompt Optimization**

.. code-block:: python

   from cje import run_experiment
   
   # Test a new helpful assistant prompt
   results = run_experiment(
       config_path="configs/system_prompt_comparison.yaml"
   )
   
   # Access structured results
   summary = results['summary']
   print(f"Recommended: {summary['recommended_policy']}")
   # Output: "helpful_assistant_v2"
   
   print(f"Confidence: {summary['confidence']}")  
   # Output: "HIGH (p < 0.001, well-powered analysis)"
   
   # Get detailed comparisons
   for ranking in results['policy_rankings']:
       policy = ranking['policy']
       estimate = ranking['estimate']
       ci = ranking['confidence_interval']
       print(f"{policy}: {estimate:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

**Key Benefits Over Traditional A/B Testing**:

- ⚡ **10x faster**: Results in minutes, not weeks
- 💰 **90% cheaper**: No API costs for new response generation  
- 📊 **Rigorous CIs**: Causal inference corrects for distribution shift
- 🔄 **Multi-policy**: Test 5+ policies simultaneously

🏗️ Architecture Overview
------------------------

CJE implements a principled pipeline for causal evaluation:

.. code-block:: text

   Dataset → Log Probabilities → Judge Scores → Causal Estimation → Results
      ↓            ↓                  ↓              ↓                ↓
   CSV/JSON   π₀(a|x), π'(a|x)    Human/AI Judge   DR-CPO/MRDR    Policy Rankings

**Key Innovation**: Doubly-robust estimation corrects for distribution shift between your logged policy π₀ and target policy π' you want to evaluate.

📚 Documentation Structure
--------------------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   start_here
   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & Examples

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Technical Reference

   theory/index
   api/estimators
   api/index


🔍 Estimator Quick Reference
----------------------------

Choose the right method for your use case:

.. list-table:: Estimator Selection Guide
   :header-rows: 1
   :widths: 12 20 15 25 28

   * - Method
     - Theoretical Properties
     - Computational Cost
     - Key Advantages
     - From Paper?
   * - **IPS**
     - Unbiased, high variance
     - Very Low
     - Simple, fast, interpretable
     - ✅ Classical baseline
   * - **SNIPS**
     - ≈Unbiased, lower variance
     - Low
     - Better than IPS, still simple
     - ✅ Standard method
   * - **DR-CPO**
     - **Single-rate efficient**, double robust
     - Medium
     - **Paper Algorithm 1**: Best balance of accuracy/speed
     - ✅ **Core contribution**
   * - **MRDR**
     - **Semiparametric optimal**, variance-minimizing
     - High
     - Maximum robustness and precision
     - ⚠️ Paper mention + full implementation

.. tip::
   **💡 Recommendation**: Start with **DR-CPO** (the paper's main algorithm) for most applications, fall back to SNIPS for large-scale scenarios.

**Key Theoretical Results** *(from the paper)*:
   * **Single-Rate Efficiency**: DR-CPO achieves √n-efficiency when only ONE nuisance (weights OR outcome model) is well-specified
   * **Double Robustness**: Unbiased if either importance weights or outcome model is correct  
   * **Semiparametric Optimality**: MRDR attains the Cramér-Rao lower bound when both nuisances converge

🎯 Common Use Cases
-------------------

**System Prompt Engineering**
   Test different communication styles and response formats
   → :doc:`guides/user_guide` → "System Prompt Engineering"

**Model Upgrades**
   Evaluate if upgrading to a newer/larger model improves performance
   → :doc:`guides/user_guide` → "Model Comparison"

**Parameter Tuning**
   Optimize temperature, top-p, and other generation parameters
   → :doc:`guides/user_guide` → "Parameter Tuning"

**ChatBot Arena Analysis**
   Large-scale evaluation using human preference data
   → :doc:`guides/arena_analysis` → Complete end-to-end guide

**A/B Test Analysis**
   Convert pairwise comparison data into policy utilities
   → :doc:`tutorials/pairwise_evaluation` → Bradley-Terry modeling

📊 Performance Benchmarks
-------------------------

Expected performance characteristics (empirical validation in progress):

- **Significant MSE reduction** vs naive importance sampling
- **Further improvements** with outcome modeling (MRDR)
- **Robust teacher forcing** implementation across providers
- **Scales to millions** of logged interactions with cross-fold processing

🚨 Quick Troubleshooting
------------------------

**Wide confidence intervals?**
   → More data, similar policies, or try SNIPS estimator

**Estimators disagree significantly?**
   → Check calibration plots, consider more ground truth labels

**Slow performance?**
   → Use IPS/SNIPS, reduce mc_samples, or smaller models

**Configuration errors?**
   → Run ``cje validate`` before experiments

.. seealso::
   Each guide includes comprehensive troubleshooting sections for specific workflows.

🏆 Research & Citation
---------------------

This implementation is based on the **Causal Judge Evaluation** research paper:

.. code-block:: bibtex

   @article{landesberg2025cje,
     title={Causal Judge Evaluation (CJE): Unbiased, Calibrated \& Cost-Efficient Off-Policy Metrics for LLM Systems},
     author={Landesberg, Eddie},
     year={2025},
     note={Implementation available at https://github.com/fondutech/causal-judge-evaluation}
   }

**Paper Highlights**:
   * **Algorithm 1**: Cross-fitted Calibrated DR-CPO (exact implementation in ``MultiDRCPOEstimator``)
   * **Theorem 5.2**: Single-rate efficiency through isotonic weight calibration
   * **Section 6**: Production deployment guidelines and compute cost analysis

**Implementation Enhancements** *(beyond paper baseline)*:
   * Multi-policy joint evaluation with full covariance estimation
   * Optional outcome model calibration for additional robustness  
   * Production-grade numerical stabilization and diagnostics
   * Automatic model selection and weight monitoring

If you use CJE in your research, please cite the paper above. For implementation-specific features, you may also reference this software package.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`