Arena Analysis Guide
====================

The Arena Analysis Runner provides a one-command solution for executing the complete Token-Level Counterfactual Judgment Estimation (CJE) pipeline on Chatbot Arena data.

Overview
--------

This pipeline implements the "Sequence-Propensity CJE Experiment on Chatbot-Arena Prompts" methodology, which:

1. **Logs responses** with token-level propensities from a base model
2. **Calibrates** a cheap LLM judge to a sparse oracle LLM  
3. **Estimates** counterfactual utility of alternative models using Doubly Robust CPO (DR-CPO)
4. **Validates** results against oracle truth with statistical confidence intervals

Quick Start
-----------

Prerequisites
~~~~~~~~~~~~~

1. **API Key**: Set your OpenAI API key in the environment:

   .. code-block:: bash

      export OPENAI_API_KEY="sk-your-key-here"
   
   Or for Anthropic models:

   .. code-block:: bash

      export ANTHROPIC_API_KEY="your-anthropic-key"

2. **Dependencies**: Ensure the CJE package is installed:

   .. code-block:: bash

      pip install -e .

Basic Usage
~~~~~~~~~~~

Run the analysis with default settings:

.. code-block:: bash

   python scripts/run_arena_analysis.py

This will:

- Process 1000 Arena prompts  
- Use 25% for oracle labeling
- Evaluate 5 target models (GPT-3.5, GPT-4, Claude-2, Claude-3-Opus, GPT-4o)
- Create timestamped results in ``outputs/arena_runs/run_<timestamp>/``

Configuration Options
---------------------

Core Parameters
~~~~~~~~~~~~~~~

.. code-block:: bash

   python scripts/run_arena_analysis.py \
       --max-samples 500 \
       --oracle-fraction 0.3 \
       --base-model "gpt-3.5-turbo" \
       --oracle-model "gpt-4o" \
       --proxy-model "gpt-3.5-turbo"

.. list-table:: Core Parameters
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``--max-samples``
     - 1000
     - Number of Arena prompts to process
   * - ``--oracle-fraction``
     - 0.25
     - Fraction of samples to label with oracle (0.0-1.0)
   * - ``--base-model``
     - ``gpt-3.5-turbo``
     - Model for stochastic log generation
   * - ``--oracle-model``
     - ``gpt-4o``
     - High-quality model for ground truth labels
   * - ``--proxy-model``
     - ``gpt-3.5-turbo``
     - Cheap model for full-coverage judging

Sampling Parameters
~~~~~~~~~~~~~~~~~~~

Control the stochasticity of the base model responses:

.. code-block:: bash

   python scripts/run_arena_analysis.py \
       --temperature 0.7 \
       --top-p 0.9

.. list-table:: Sampling Parameters
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``--temperature``
     - 0.7
     - Sampling temperature (0.0-2.0)
   * - ``--top-p``
     - 0.9
     - Nucleus sampling probability mass (0.0-1.0)

Target Models
~~~~~~~~~~~~~

Specify which models to evaluate:

.. code-block:: bash

   python scripts/run_arena_analysis.py \
       --target-models gpt-4 claude-3-opus gpt-4o-mini

**Default target models:**

- ``gpt-3.5-turbo``
- ``gpt-4`` 
- ``claude-2``
- ``claude-3-opus``
- ``gpt-4o``

Reproducibility
~~~~~~~~~~~~~~~

Set a random seed for reproducible results:

.. code-block:: bash

   python scripts/run_arena_analysis.py --seed 42

Example Workflows
-----------------

Quick Test Run
~~~~~~~~~~~~~~

For development and testing:

.. code-block:: bash

   python scripts/run_arena_analysis.py \
       --max-samples 100 \
       --oracle-fraction 0.5 \
       --target-models gpt-3.5-turbo gpt-4

High-Quality Analysis
~~~~~~~~~~~~~~~~~~~~~

For production analysis with maximum statistical power:

.. code-block:: bash

   python scripts/run_arena_analysis.py \
       --max-samples 5000 \
       --oracle-fraction 0.2 \
       --oracle-model "gpt-4o" \
       --temperature 0.4 \
       --top-p 0.9

Budget-Conscious Run
~~~~~~~~~~~~~~~~~~~~

Minimize API costs while maintaining quality:

.. code-block:: bash

   python scripts/run_arena_analysis.py \
       --max-samples 1000 \
       --oracle-fraction 0.15 \
       --base-model "gpt-3.5-turbo" \
       --oracle-model "gpt-4" \
       --proxy-model "gpt-3.5-turbo"

Understanding the Output
------------------------

Directory Structure
~~~~~~~~~~~~~~~~~~~

Each run creates a timestamped directory:

.. code-block:: text

   outputs/arena_runs/run_2024-01-15_14-30-22/
   ├── stochastic_log.jsonl              # Raw interaction logs
   ├── token_level_cje_results.json      # Final results & statistics  
   ├── calibration_curve.png             # Judge calibration plot
   ├── comparison_heatmap.png             # Policy comparison p-values
   └── .cache/                           # Cached intermediate results
       ├── interaction_logs.pkl
       ├── oracle_labels_gpt-4o.pkl
       └── proxy_scores_gpt-3.5-turbo.pkl

Plus a ZIP file: ``run_2024-01-15_14-30-22.zip`` containing all artifacts.

Key Output Files
~~~~~~~~~~~~~~~~

``token_level_cje_results.json``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Contains the main results:

.. code-block:: json

   {
     "results": {
       "gpt-4": {
         "v_hat": 0.742,           // Point estimate of expected utility
         "se": 0.023,              // Standard error
         "ci_low": 0.697,          // 95% confidence interval lower bound
         "ci_high": 0.787,         // 95% confidence interval upper bound
         "oracle_truth": 0.758,    // True oracle utility (if validation ran)
         "absolute_error": 0.016   // |v_hat - oracle_truth|
       }
     },
     "p_values_corrected": [...],    // Holm-corrected pairwise p-values
     "guard_rail_violations": [...]  // Quality control warnings
   }

``stochastic_log.jsonl``
^^^^^^^^^^^^^^^^^^^^^^^^

Raw interaction data (one JSON object per line):

.. code-block:: json

   {
     "uid": "sample_0",
     "prompt": "What is machine learning?",
     "answer": "Machine learning is...",
     "token_logprobs": [-0.1, -0.3, -0.2, ...],
     "action": "gpt-3.5-turbo",
     "sequence_logp": -45.7,
     "pi0": 2.34e-20,
     "oracle_score": 8.5,
     "judge_raw": 0.82,
     "calibrated_reward": 8.3
   }

Console Output
~~~~~~~~~~~~~~

The script provides real-time progress updates:

.. code-block:: text

   ──────────────── Loading Chatbot Arena prompts ────────────────
   Loaded 1000 prompts.

   Step 1: Creating stochastic interaction logs
   ✓ Created 1000 interaction logs

   Step 2: Obtaining oracle utility labels  
   ✓ Obtained 250 oracle labels

   Step 3: Scoring with proxy judge
   ✓ Scored all 1000 rows with proxy judge

   Step 4: Calibrating proxy judge
   Spearman correlation (proxy vs oracle): 0.834
   ✓ Calibration complete for all 1000 samples

   ...

   ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
   ┃ Policy        ┃ Estimate ┃ SE    ┃ 95% CI         ┃  
   ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩
   │ gpt-4         │ 0.742    │ 0.023 │ [0.697, 0.787] │
   │ claude-3-opus │ 0.738    │ 0.025 │ [0.689, 0.787] │
   │ gpt-4o        │ 0.756    │ 0.022 │ [0.713, 0.799] │
   └───────────────┴──────────┴───────┴────────────────┘

Quality Control
---------------

Automatic Weight Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Arena analysis automatically includes comprehensive weight diagnostics to catch issues early:

.. code-block:: text

   📊 Importance Weight Summary
   
   | Policy | ESS | Mean Weight | Status | Issues |
   |--------|-----|-------------|--------|--------|
   | gpt-4 | 85.2% | 1.245 | ✅ GOOD | None |
   | claude-3-opus | 72.1% | 0.987 | ✅ GOOD | None |
   | gpt-4o | 45.3% | 2.341 | 🟡 WARNING | Moderate ESS |

The diagnostics check for:

- **Effective Sample Size (ESS)**: Percentage of samples contributing meaningfully
- **Extreme weights**: Weights > 1000 or < 0.001 indicating distribution mismatch
- **Identical policy consistency**: Weights should be ≈1.0 for policies identical to logging policy
- **Zero weights**: Complete probability mass mismatches

Interactive weight analysis is available via:

.. code-block:: python

   from examples.arena_interactive import ArenaAnalyzer
   
   analyzer = ArenaAnalyzer()
   analyzer.plot_weight_diagnostics()  # Visual diagnostic dashboard
   analyzer.diagnose_weights()         # Detailed diagnostic objects

Guard Rail Checks
~~~~~~~~~~~~~~~~~~

The pipeline automatically detects potential issues:

- **Insufficient oracle samples**: Too few samples for reliable calibration
- **Low proxy-oracle correlation**: Judge calibration may be unreliable  
- **Poor CI coverage**: Statistical intervals may be miscalibrated
- **Extreme importance weights**: Target policies very different from base model

Recommendations
~~~~~~~~~~~~~~~

- **Oracle fraction**: Use 20-30% for reliable calibration
- **Sample size**: Minimum 500 samples; 2000+ for high precision
- **Model selection**: Choose base model representative of your target policies
- **Temperature**: Lower values (0.3-0.7) provide more stable importance weights

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

API Key Not Found
^^^^^^^^^^^^^^^^^

.. code-block:: text

   No provider API key found! Please set OPENAI_API_KEY or ANTHROPIC_API_KEY

**Solution**: Export your API key before running.

Rate Limiting
^^^^^^^^^^^^^

.. code-block:: text

   RateLimitError: You exceeded your current quota

**Solution**: Reduce ``--max-samples`` or use cheaper models for base/proxy.

Low Correlation Warning
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Guard-rail violations detected:
     - Low proxy-oracle correlation: 0.423

**Solution**: Try different proxy model or increase oracle fraction.

Import Errors
^^^^^^^^^^^^^

.. code-block:: text

   ModuleNotFoundError: No module named 'cje'

**Solution**: Install package with ``pip install -e .``

Performance Tips
~~~~~~~~~~~~~~~~

1. **Caching**: The script caches intermediate results. Remove ``.cache/`` to force regeneration.

2. **Memory**: Large sample sizes may require significant RAM. Monitor with:

   .. code-block:: bash

      htop  # or Activity Monitor on macOS

3. **API Costs**: Estimate costs before large runs:

   - Oracle calls: ``max_samples * oracle_fraction * $0.01``
   - Proxy calls: ``max_samples * $0.0005``  
   - Target evaluation: ``len(target_models) * oracle_samples * $0.01``

Analyzing Results
-----------------

After running the arena analysis, use the provided example script to explore your results:

.. code-block:: bash

   # Analyze latest results with detailed statistics and plots
   python examples/analyze_arena_results.py

   # Analyze a specific run
   python examples/analyze_arena_results.py --run-dir outputs/arena_runs/run_2024-01-15_14-30-22

The analysis script provides:

- **Policy Summary**: Best model identification, confidence intervals, oracle validation
- **Judge Calibration**: Correlation analysis between proxy and oracle judges
- **Model Comparisons**: Pairwise utility differences and statistical significance  
- **Visualization**: Confidence interval plots
- **Export**: CSV data for further analysis

Advanced Usage
--------------

Custom Model Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For models not in the default list, you may need to modify the script's model configurations in ``experiments/arena_token_cje.py``.

Batch Processing
~~~~~~~~~~~~~~~~

Run multiple configurations:

.. code-block:: bash

   #!/bin/bash
   for temp in 0.3 0.5 0.7; do
       python scripts/run_arena_analysis.py \
           --temperature $temp \
           --max-samples 1000 \
           --seed 42
   done

Integration with Notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~

Load results for further analysis:

.. code-block:: python

   import json
   from pathlib import Path

   # Load latest results
   results_dir = Path("outputs/arena_runs")
   latest_run = max(results_dir.iterdir(), key=lambda p: p.stat().st_mtime)
   with open(latest_run / "token_level_cje_results.json") as f:
       results = json.load(f)

   print(f"Best model: {max(results['results'].keys(), key=lambda k: results['results'][k]['v_hat'])}")

Extending the Pipeline
~~~~~~~~~~~~~~~~~~~~~~

The modular design allows easy extension:

1. **New Models**: Add to ``target_models`` list
2. **Custom Features**: Modify ``_extract_features()`` in ``ArenaTokenCJE``  
3. **Alternative Estimators**: Replace ``_compute_dr_cpo_estimate()``
4. **Custom Judges**: Modify ``JudgeFactory`` templates

Best Practices
--------------

1. **Start Small**: Use ``--max-samples 100`` for initial testing
2. **Version Control**: Save configuration with results for reproducibility
3. **Monitor Quality**: Check guard rail violations in output
4. **Validate Results**: Use oracle validation to verify estimates
5. **Document Experiments**: Keep notes on model choices and rationale

Support
-------

For questions or issues:

1. Check the logs in the output directory
2. Review guard rail violations  
3. Consult the main CJE documentation
4. Open an issue with configuration details and error messages

Full CLI Reference
------------------

.. code-block:: bash

   python scripts/run_arena_analysis.py [OPTIONS]

   Options:
     --max-samples INTEGER       Number of Arena prompts [default: 1000]
     --oracle-fraction FLOAT     Oracle labeling fraction [default: 0.25]  
     --base-model TEXT           Base model for logging [default: gpt-3.5-turbo]
     --temperature FLOAT         Sampling temperature [default: 0.7]
     --top-p FLOAT              Nucleus sampling mass [default: 0.9]
     --oracle-model TEXT         Oracle model [default: gpt-4o]
     --proxy-model TEXT          Proxy judge model [default: gpt-3.5-turbo]
     --target-models TEXT...     Models to evaluate [default: multiple]
     --seed INTEGER              Random seed [default: 42]
     --help                      Show this message and exit 