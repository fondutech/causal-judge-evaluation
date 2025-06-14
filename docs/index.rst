.. CJE-Core documentation master file, created by
   sphinx-quickstart on Wed Jun 11 16:07:45 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CJE: Causal Judge Evaluation Toolkit
=====================================

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

.. important::
   **📄 Paper Implementation**: This toolkit implements the **Causal Judge Evaluation** methodology from the research paper, with additional production-ready enhancements:
   
   * **Core Algorithm**: Exact implementation of Algorithm 1 (Calibrated DR-CPO) 
   * **Theoretical Guarantees**: Single-rate efficiency, semiparametric optimality, double robustness
   * **Implementation Enhancements**: Multi-policy evaluation, outcome calibration, numerical stabilization
   * **Production Features**: Automatic diagnostics, weight monitoring, robust error handling

🔬 **Theory-to-Practice Pipeline**
   Paper Algorithm 1 → ``MultiDRCPOEstimator`` → ``name: "DRCPO"`` config

.. note::
   **New to CJE?** → Start with :doc:`quickstart` for a 5-minute introduction.
   
   **Experienced User?** → Jump to :doc:`guides/index` for advanced workflows.
   
   **From the Paper?** → See :doc:`theory/mathematical_foundations` for theory-implementation mapping.

🚀 Learning Paths by User Type
------------------------------

Choose your path based on your role and goals:

**📊 Data Scientists & Analysts**
   Start here if you work with spreadsheets, CSV files, or Pandas DataFrames
   
   → :doc:`quickstart` → :doc:`guides/user_guide` → :doc:`tutorials/pairwise_evaluation`

**🤖 ML Engineers & Practitioners**
   Start here if you deploy models and need production-ready evaluation
   
   → :doc:`installation` → :doc:`guides/arena_analysis` → :doc:`guides/weight_processing`

**🔬 Researchers & Academics**
   Start here if you need theoretical understanding and custom implementations
   
   → :doc:`theory/index` → :doc:`guides/custom_components` → :doc:`api/estimators`

**⚙️ Platform Engineers**
   Start here if you're integrating CJE into larger systems
   
   → :doc:`api/index` → :doc:`guides/aws_setup` → :doc:`guides/custom_components`

⚡ Quick Start Examples
----------------------

**5-Minute Test Run**

.. code-block:: bash

   # Clone and install
   git clone https://github.com/fondutech/causal-judge-evaluation.git
   cd causal-judge-evaluation
   poetry install
   
   # Run test
   cje run --cfg-path configs --cfg-name arena_test

**Compare Two System Prompts**

.. code-block:: python

   from cje.pipeline import run_pipeline
   
   # Run with configuration files (Hydra-based)
   results = run_pipeline(
       cfg_path="configs",
       cfg_name="my_experiment"
   )
   
   # Results contain the complete experiment output
   print(f"Results: {results}")
   
   # For programmatic usage, use estimators directly:
   from cje.estimators import get_estimator
   estimator = get_estimator("DRCPO", sampler=sampler)
   estimator.fit(data)
   estimate_result = estimator.estimate()
   print(f"Policy estimates: {estimate_result.v_hat}")

**Arena-Style Evaluation**

.. code-block:: python

   from examples.arena_interactive import ArenaAnalyzer
   
   analyzer = ArenaAnalyzer()
   analyzer.quick_test()  # Run with sample data
   analyzer.plot_estimates()  # Visualize results

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

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics

   contributing
   changelog
   license

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

Recent evaluation on ChatBot Arena data (33M interactions):

- **85% MSE reduction** vs naive importance sampling
- **92% MSE reduction** with outcome modeling (MRDR)
- **100% detection rate** for teacher forcing bugs in testing
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

