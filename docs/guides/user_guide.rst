User Guide
==========

*Practical guide for real-world CJE usage*

Overview
--------

This guide covers practical CJE usage for production scenarios. For getting started, see the :doc:`../quickstart` first.

The CJE Pipeline
----------------

CJE converts logs into unbiased policy estimates through 5 steps:

.. code-block:: text

   Dataset → Log → Judge → Calibrate → Estimate → result.json

**Single command**: ``cje run --cfg-path cje.conf --cfg-name experiment``

This automatically runs all necessary steps based on existing files.

📋 Quick Reference
------------------

For the absolute fastest path see :doc:`../quickstart`. For a full list of Typer commands jump to :doc:`../api/index`. The remainder of this guide focuses on day-to-day usage details, estimator choices, and troubleshooting.

Essential Commands
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run full pipeline
   cje run --cfg-path cje.conf --cfg-name experiment
   
   # Validate configuration file
   cje validate config --cfg-path cje.conf --cfg-name experiment
   
   # Validate data file
   cje validate data my_data.jsonl
   
   # Quick data health check
   cje validate quick my_data.jsonl

The ``cje validate config`` command validates your YAML configuration files for:

- Required fields and proper structure
- Valid parameter values and types  
- Consistency between configuration sections
- Compatibility with available estimators and providers

.. code-block:: bash

   # Validate with detailed output
   cje validate config --cfg-path configs --cfg-name my_experiment --verbose
   
   # Basic validation
   cje validate config --cfg-name experiment

Estimator Quick Guide
~~~~~~~~~~~~~~~~~~~~~

.. note::
   **📖 Complete Estimator Reference**: See :doc:`../api/estimators` for detailed estimator documentation.

.. list-table:: Estimator Selection Guide
   :header-rows: 1
   :widths: 15 20 25 40

   * - Estimator
     - Best For
     - Required Parameters
     - Notes
   * - **IPS**
     - Quick baselines
     - ``sampler``, ``clip``
     - Fastest, simple
   * - **SNIPS**
     - Better than IPS
     - ``sampler``, ``clip``
     - Good robustness
   * - **DRCPO**
     - Most use cases
     - ``sampler``, ``k``, ``clip``
     - Best balance of speed/accuracy
   * - **MRDR**
     - Maximum robustness
     - ``sampler``, ``k``, ``clip``
     - Best for benchmarking

.. tip::
   **💡 Recommendation**: Start with DRCPO for most applications.

Basic Estimator Usage
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Estimator configuration
   estimator:
     name: "DRCPO"           # Doubly-robust (recommended)
     k: 5                    # Cross-validation folds
     n_jobs: -1              # Use all CPU cores for parallel cross-fitting

**Common Patterns:**

.. code-block:: yaml

   # Quick baseline (IPS)
   estimator:
     name: "IPS"

   # Robust estimation (DRCPO - recommended)
   estimator:
     name: "DRCPO"
     k: 5                    # Cross-validation folds

   # Maximum robustness (MRDR)
   estimator:
     name: "MRDR"
     k: 5                    # Cross-validation folds

🚨 Common Issues & Solutions
----------------------------

For detailed troubleshooting, see the :doc:`troubleshooting` guide. Quick tips:

- **Wide confidence intervals** → More data or use SNIPS
- **Estimators disagree** → Check calibration plots
- **Slow performance** → Try IPS or reduce mc_samples
- **Config errors** → Run ``cje validate`` first

💡 Pro Tips
-----------

- **Start small**: 10-100 samples for initial testing
- **Validate first**: Use ``cje validate`` to catch configuration errors
- **Set seeds**: Use ``estimator.seed=42`` for reproducible results
- **Monitor clipping**: Keep clipped weight mass < 2%
- **Compare estimators**: Always validate with multiple methods

🔄 Typical Workflow
-------------------

1. **Setup** → Install CJE, prepare data, create config
2. **Validate** → ``cje validate`` to check configuration
3. **Test** → Small test run, check intermediate files  
4. **Scale** → Full dataset, compare estimators
5. **Deploy** → Use CI bounds for deployment decisions

Extended Conversation Support
-----------------------------

CJE automatically handles multi-turn conversations in multiple formats:

**Standard Format:**

.. code-block:: text

   Human: What is machine learning?
   AI: Machine learning enables computers to learn from data.
   Human: Can you give me an example?

**Token Format:**

.. code-block:: text

   <|user|>What is deep learning?<|assistant|>Deep learning uses neural networks.<|user|>How many layers?

**JSON Format:**

.. code-block:: json

   [
     {"role": "user", "content": "Hello"}, 
     {"role": "assistant", "content": "Hi!"}, 
     {"role": "user", "content": "Help me"}
   ]

CJE preserves conversation context while applying your system prompts and message templates.

Core Workflows
--------------

1. System Prompt Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Most common use case** - testing different communication styles by varying the system prompt while keeping other parameters constant.

.. seealso::
   For a complete example of system prompt comparison, see the "Common Use Case: System Prompt Comparison" section in :doc:`configuration_reference`.

2. Model Comparison
~~~~~~~~~~~~~~~~~~~

Compare different models or versions to evaluate upgrades or alternatives.

.. seealso::
   See the "Model Comparison" section in :doc:`configuration_reference` for configuration examples.

3. Parameter Tuning
~~~~~~~~~~~~~~~~~~~

Test different hyperparameters like temperature, top_p, or max_tokens to optimize generation quality.

.. seealso::
   See the "Parameter Tuning" section in :doc:`configuration_reference` for examples of testing different parameter values.

4. Multi-Provider Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare models from different providers (OpenAI vs Anthropic vs Google, etc.) for the same use case.

.. seealso::
   See the "Multi-Provider Comparison" section in :doc:`configuration_reference` for examples comparing different providers.

Data Requirements
-----------------

Option 1: Use Existing Logs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you let the pipeline generate (or back-fill) logs it automatically includes two diagnostic fields which are optional but often useful:

* ``token_logps`` – list of per-token log-probabilities for the generated response (same order as the tokens).
* ``action`` – string identifier of the model / checkpoint that produced the response. This becomes a categorical feature in ``RichFeaturizer``.

``logp`` remains the **sum** of ``token_logps``; the estimators use that value for propensity weighting.

.. note::
   **Teacher Forcing Considerations**: Log probabilities must be computed consistently between logging and target policies. Due to API limitations, only certain providers (Fireworks, Together) support true teacher forcing. See :doc:`../developer/teacher_forcing` for critical implementation details.

If missing ``logp``, backfill it:

.. code-block:: bash

   cje backfill backfill-logp \
     your_logs.jsonl \
     logs_with_logp.jsonl \
     --model-name gpt-4o-mini

Option 2: Use CSV/Spreadsheet Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perfect for data science workflows and spreadsheet-based experiments:

**Basic CSV format:**

.. code-block:: csv

   context,response,y_true,logp
   "What is machine learning?","ML is a subset of AI...",1,-12.3
   "Explain neural networks","Neural networks are...",0,-8.7
   "Define artificial intelligence","AI enables machines...",1,-10.2

**Minimal CSV (context only):**

.. code-block:: csv

   context
   "What is machine learning?"
   "Explain neural networks"
   "Define artificial intelligence"

**Configuration:**

.. code-block:: yaml

   # Dataset configuration
   dataset:
     name: "./data/my_data.csv"  # Supports .csv and .tsv files
     # split is ignored for file-based datasets

**Pandas/Jupyter Integration:**

.. code-block:: python

   import pandas as pd
   from cje.data import CSVDataset

   # From existing DataFrame
   df = pd.DataFrame({
       'context': ['What is AI?', 'Explain ML'],
       'response': ['AI is...', 'ML is...'],
       'y_true': [0.9, 0.8],
       'experiment_id': ['exp_1', 'exp_1']  # Extra columns go to meta
   })

   # Load into CJE
   dataset = CSVDataset.from_dataframe(df, name="my_experiment")

   # Use in configuration
   from cje.config import simple_config
   config = simple_config(dataset_name="./my_data.csv")

**CSV Features:**

- **Required**: ``context`` column only
- **Optional**: ``uid``, ``response``, ``y_true``, ``logp``
- **Extra columns**: Automatically stored in ``meta`` field
- **Missing values**: NaN automatically converted to None
- **TSV support**: Auto-detected by ``.tsv`` extension

Option 3: Generate Fresh Logs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cje log run \
     --dataset ./data/my_data.jsonl \
     --model gpt-4o-mini \
     --out logs.jsonl

Option 4: External Data Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Advanced**: For users with pre-computed target policy data.

**Current format** (map-based for multiple policies):

.. code-block:: json

   {
     "context": "What is ML?",
     "response": "ML is...",
     "logp": -12.3,
     "y_true": 0.85,
     "logp_target_all": {
       "low": -10.8,
       "medium": -11.2,
       "high": -11.5
     },
     "target_samples": {
       "low": ["ML teaches computers...", "ML is pattern recognition..."],
       "medium": ["ML algorithms learn..."],
       "high": ["ML enables autonomous learning...", "ML creates intelligent systems..."]
     }
   }

.. note::
   Pre-computed data is automatically detected when ``logp_target_all`` contains a dict mapping policy names to log probabilities (preferred), or a list in the same order as configured policies.

Option 5: Use Pairwise Comparison Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   See the dedicated :doc:`../tutorials/pairwise_evaluation` for an end-to-end tutorial.

**Recommended**: When you have human preference data (A/B comparisons).

CJE can use pairwise comparison data (like LMSYS Chatbot Arena) where humans choose between responses. The Bradley-Terry model converts these preferences to scalar utilities.

**Built-in Chatbot Arena dataset:**

.. code-block:: python

   from cje.data import ChatbotArenaDataset

   # Load with model-level utilities
   dataset = ChatbotArenaDataset.download(
       split="train",
       model_aware=True,  # Get one utility per model
       regularization=0.01
   )

   # See model rankings
   rankings = dataset.get_model_rankings()
   for model, utility in list(rankings.items())[:5]:
       print(f"{model}: {utility:.3f}")

**Use in configuration:**

.. code-block:: yaml

   # Built-in pairwise dataset
   dataset:
     name: "ChatbotArena"
     split: "train"

   # Or use generic pairwise adapter
   dataset:
     name: "PairwiseComparison"
     # Will look for your own pairwise data

**Custom pairwise data:**

.. code-block:: python

   from cje.data.pairwise import BradleyTerryModel

   # Your comparison data: (winner_id, loser_id, weight)
   comparisons = [
       ("response_1", "response_2", 1.0),  # 1 beat 2
       ("response_3", "response_1", 1.0),  # 3 beat 1  
       ("response_2", "response_3", 0.5),  # Tie
       ("response_3", "response_2", 0.5),  # Tie (both directions)
   ]

   # Fit Bradley-Terry model
   bt_model = BradleyTerryModel(regularization=0.01)
   bt_model.fit(comparisons)

   # Get utilities
   utility_1 = bt_model.get_utility("response_1")  # ~0.33
   utility_2 = bt_model.get_utility("response_2")  # ~0.33
   utility_3 = bt_model.get_utility("response_3")  # ~0.66

**Key advantages:**

- **Human-grounded**: Based on actual human preferences
- **Handles intransitivity**: Robust to cycles via regularization
- **No absolute labels needed**: Only requires "A > B" judgments
- **Global consistency**: Bradley-Terry ensures coherent utilities

**Bradley-Terry model:**

.. math::

   P(A \text{ beats } B) = \frac{\exp(u_A)}{\exp(u_A) + \exp(u_B)}

Where :math:`u_A, u_B` are latent utilities, normalized to [0,1].

Estimator Selection
-------------------

.. note::
   **📖 Complete Reference**: See :doc:`../api/estimators` for detailed estimator specifications and parameters.

**Multi-Policy Support**: All estimators automatically support multiple target policies using CJE's unified architecture. When you specify multiple ``target_policies`` in your configuration, results are returned as dictionaries where each key corresponds to one policy.

**Parallel Cross-Fitting**: DRCPO and MRDR support parallel cross-validation for faster computation on multi-core systems. **Parallelization is enabled by default** (``n_jobs=-1``).

**Configuration**:

.. code-block:: yaml

   # Estimator configuration with parallelization
   estimator:
     name: "DRCPO"           # Doubly-robust (recommended)
     k: 5                    # Cross-validation folds
     n_jobs: -1              # Use all CPU cores for parallel cross-fitting

**Performance Tips**:

- Parallelization is enabled by default (``n_jobs=-1``) for optimal performance
- Use ``n_jobs=1`` to disable parallelization if needed (e.g., debugging)
- For small datasets (< 100 samples), you may want ``n_jobs=1`` to avoid overhead
- Parallel cross-fitting is most beneficial with ``k >= 5`` folds and complex outcome models

Judge Configuration
-------------------

CJE uses AI judges to evaluate response quality. Judges are critical for translating raw model responses into calibrated quality scores.

Quick Judge Setup
~~~~~~~~~~~~~~~~~

**Default (recommended for most cases):**

.. code-block:: yaml

   # Judge configuration
   judge:
     provider: "openai"
     model_name: "gpt-4o-mini"
     template: "quick_judge"
     temperature: 0.0        # Deterministic for consistency

**High-quality evaluation:**

.. code-block:: yaml

   # Judge configuration for higher quality
   judge:
     provider: "openai"
     model_name: "gpt-4o"
     template: "detailed_judge"
     temperature: 0.0        # Deterministic for consistency

.. seealso::
   For comprehensive judge configuration including custom templates, see :doc:`custom_components`. 