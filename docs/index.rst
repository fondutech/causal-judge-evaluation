CJE: Causal Judge Evaluation
============================

.. image:: img/CJE_logo.svg
   :align: center
   :alt: CJE Logo
   :width: 400

**Fast, accurate LLM evaluation using AI judges with causal inference.**

CJE helps you answer "What would happen if we deployed this new model/prompt?" using only historical data. No risky A/B tests, no expensive human evaluation.

Quick Start (5 minutes)
-----------------------

**1. Install CJE**

.. code-block:: bash

   git clone https://github.com/fondutech/causal-judge-evaluation.git
   cd causal-judge-evaluation
   poetry install
   
   # Set API key (Fireworks offers free tier)
   export FIREWORKS_API_KEY="your-key-here"

**2. Run Your First Evaluation**

.. code-block:: bash

   # Test evaluation on example data
   cje run --cfg-path configs --cfg-name example_eval

**3. Use in Python**

.. code-block:: python

   from cje.config.unified import simple_config
   
   config = simple_config(
       dataset_name="./data/test.jsonl",
       logging_model="gpt-3.5-turbo",
       target_model="gpt-4",
       judge_model="gpt-4o",
       estimator_name="DRCPO"
   )
   results = config.run()
   print(f"Target policy score: {results['results']['DRCPO']['estimates'][0]:.3f}")

Learning Paths
--------------

.. raw:: html

   <style>
   .path-cards {
       display: flex;
       gap: 20px;
       margin: 30px 0;
       flex-wrap: wrap;
   }
   .path-card {
       flex: 1;
       min-width: 250px;
       border: 2px solid #e1e4e8;
       border-radius: 8px;
       padding: 20px;
       background: #f8f9fa;
   }
   .path-card h3 {
       color: #0969da;
       margin-top: 0;
   }
   </style>

   <div class="path-cards">
   
   <div class="path-card">
   <h3>🚀 I want to evaluate now</h3>
   <p><strong>Time: 5-30 minutes</strong></p>
   <p>Jump straight into running evaluations on your data.</p>
   <ul>
   <li><a href="quickstart.html">Quickstart Guide</a></li>
   <li><a href="guides/user_guide.html">User Guide</a></li>
   <li><a href="guides/configuration_reference.html">Configuration</a></li>
   </ul>
   </div>
   
   <div class="path-card">
   <h3>📚 I want to understand</h3>
   <p><strong>Time: 30-60 minutes</strong></p>
   <p>Learn the theory and methodology behind CJE.</p>
   <ul>
   <li><a href="theory/mathematical_foundations.html">Mathematical Foundations</a></li>
   <li><a href="api/estimators.html">Estimator Details</a></li>
   <li><a href="guides/weight_processing.html">Technical Deep-Dives</a></li>
   </ul>
   </div>
   
   <div class="path-card">
   <h3>🔧 I want to extend</h3>
   <p><strong>Time: 1-2 hours</strong></p>
   <p>Build custom components and integrations.</p>
   <ul>
   <li><a href="api/index.html">API Reference</a></li>
   <li><a href="guides/custom_components.html">Custom Components</a></li>
   <li><a href="developer/teacher_forcing.html">Developer Guides</a></li>
   </ul>
   </div>
   
   </div>

Key Concepts
------------

**What CJE Does:**

1. **Reuses existing data** - No need to generate new responses from target policies
2. **Corrects for bias** - Causal inference handles distribution shift between policies  
3. **Provides uncertainty** - Confidence intervals and variance estimates included
4. **Supports multiple estimators** - Choose based on your bias-variance tradeoff needs

**Core Pipeline:**

.. code-block:: text

   Historical Data → Importance Weights → Judge Scores → Causal Estimation → Policy Rankings
         ↓                    ↓                ↓                ↓                    ↓
   Your logged data    π_target/π_logging   AI evaluation   DR-CPO/MRDR/IPS    Best policy

Common Use Cases
----------------

**📝 Prompt Engineering**
   Test different prompts without deploying them
   → See :doc:`guides/user_guide`

**🤖 Model Comparison**  
   Evaluate GPT-4 vs GPT-3.5 vs Claude using historical GPT-3.5 data
   → See :doc:`tutorials/pairwise_evaluation`

**🎯 Parameter Tuning**
   Find optimal temperature/top-p settings
   → See :doc:`guides/configuration_reference`

**🏆 ChatBot Arena Analysis**
   Large-scale evaluation with human preferences
   → See :doc:`guides/arena_analysis`

Estimator Selection
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Estimator
     - Use When
     - Pros
     - Cons
   * - **IPS**
     - Quick tests, large data
     - Fastest, simple
     - High variance
   * - **SNIPS**
     - Medium data, better than IPS
     - Lower variance than IPS
     - Small bias
   * - **DR-CPO** ⭐
     - **Most use cases**
     - Robust, low variance
     - Needs target samples
   * - **MRDR**
     - Small data, max precision
     - Lowest variance
     - Slow, complex

**Recommendation:** Start with DR-CPO for most applications.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   guides/user_guide

.. toctree::
   :maxdepth: 2
   :caption: Guides
   
   guides/configuration_reference
   guides/arena_analysis
   guides/oracle_evaluation
   guides/troubleshooting
   guides/custom_components
   guides/teacher_forcing

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/estimators
   api/index
   theory/mathematical_foundations

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/pairwise_evaluation

Getting Help
------------

**🐛 Issues?** Check :doc:`guides/troubleshooting` first

**💬 Questions?** Open an issue on `GitHub <https://github.com/fondutech/causal-judge-evaluation>`_

**📖 Paper:** See our research paper for theoretical foundations

Citation
--------

.. code-block:: bibtex

   @article{landesberg2025cje,
     title={Causal Judge Evaluation: Fast, Accurate LLM Evaluation},
     author={Landesberg et al.},
     year={2025}
   }