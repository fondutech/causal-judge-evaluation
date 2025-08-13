CLI Reference
=============

The CJE command-line interface provides convenient access to common analysis tasks.

Installation
------------

After installing CJE, the CLI is available via the ``cje`` module::

    python -m cje --help

Commands
--------

analyze
~~~~~~~

Run CJE analysis on a dataset with various estimators.

**Basic Usage**::

    python -m cje analyze data.jsonl

**With Options**::

    python -m cje analyze data.jsonl \
        --estimator dr-cpo \
        --output results.json \
        --fresh-draws-dir ./responses

**Arguments:**

* ``dataset`` - Path to JSONL dataset file (required)

**Options:**

* ``--estimator`` - Estimation method (default: calibrated-ips)
  
  * ``calibrated-ips`` - Variance-controlled IPS (recommended)
  * ``raw-ips`` - Standard importance sampling
  * ``dr-cpo`` - Doubly robust with isotonic outcome model
  * ``mrdr`` - Multiple robust DR
  * ``tmle`` - Targeted maximum likelihood

* ``--output, -o`` - Path to save results JSON
* ``--fresh-draws-dir`` - Directory with fresh draw responses (for DR estimators)
* ``--estimator-config`` - JSON configuration for estimator
* ``--judge-field`` - Metadata field containing judge scores (default: judge_score)
* ``--oracle-field`` - Metadata field containing oracle labels (default: oracle_label)
* ``--verbose, -v`` - Enable verbose output
* ``--quiet, -q`` - Suppress non-essential output

**Examples**::

    # Basic IPS analysis
    python -m cje analyze data.jsonl

    # DR analysis with fresh draws
    python -m cje analyze data.jsonl \
        --estimator dr-cpo \
        --fresh-draws-dir ./responses \
        --output dr_results.json

    # Custom estimator configuration
    python -m cje analyze data.jsonl \
        --estimator mrdr \
        --estimator-config '{"n_folds": 10, "omega_mode": "w2"}'

    # Verbose output for debugging
    python -m cje analyze data.jsonl \
        --verbose

validate
~~~~~~~~

Validate dataset format and check for common issues.

**Usage**::

    python -m cje validate data.jsonl

**Options:**

* ``-v, --verbose`` - Show detailed validation results

**What it checks:**

* Required fields (prompt, response, log probabilities)
* Consistent target policies across samples
* Presence of judge scores and oracle labels
* Data completeness and validity

**Example Output**::

    $ python -m cje validate data.jsonl
    Validating data.jsonl...
    ✓ Loaded 1000 samples
    ✓ Target policies: gpt4, claude, llama
    ✓ Rewards: 1000/1000 samples
    ✓ Dataset is valid and ready for analysis

    $ python -m cje validate data.jsonl -v
    Validating data.jsonl...
    ✓ Loaded 1000 samples
    ✓ Target policies: gpt4, claude, llama
    
    Detailed Statistics:
    ----------------------------------------
    Judge scores: 1000 samples
      Range: [0.100, 1.000]
      Mean: 0.724
    Oracle labels: 500 samples
      Range: [0.000, 1.000]
      Mean: 0.681
    
    Valid samples per policy:
      gpt4: 1000/1000
      claude: 998/1000
      llama: 995/1000

Output Formats
--------------

The ``analyze`` command can export results in multiple formats.

JSON Output
~~~~~~~~~~~

Use ``--output results.json`` to save comprehensive results::

    {
      "timestamp": "2024-01-15T10:30:45",
      "method": "calibrated_ips",
      "estimates": [0.724, 0.812, 0.693],
      "standard_errors": [0.015, 0.018, 0.021],
      "confidence_intervals": {
        "alpha": 0.05,
        "lower": [0.695, 0.777, 0.652],
        "upper": [0.753, 0.847, 0.734]
      },
      "metadata": {
        "dataset_path": "data.jsonl",
        "estimator": "calibrated-ips",
        "target_policies": ["gpt4", "claude", "llama"]
      },
      "diagnostics": {
        "gpt4": {
          "weights": {
            "ess": 850.5,
            "cv": 0.42,
            "max": 5.2
          },
          "status": "green"
        }
      },
      "per_policy_results": {
        "gpt4": {
          "estimate": 0.724,
          "standard_error": 0.015,
          "ci_lower": 0.695,
          "ci_upper": 0.753,
          "n_samples": 1000
        }
      }
    }

CSV Export (via Python)
~~~~~~~~~~~~~~~~~~~~~~~~

For tabular analysis, export to CSV using the Python API::

    from cje import analyze_dataset, export_results_csv

    results = analyze_dataset("data.jsonl")
    export_results_csv(results, "results.csv")

The CSV format includes::

    policy,estimate,standard_error,ci_lower,ci_upper,n_samples,method
    gpt4,0.724,0.015,0.695,0.753,1000,calibrated_ips
    claude,0.812,0.018,0.777,0.847,998,calibrated_ips
    llama,0.693,0.021,0.652,0.734,995,calibrated_ips

Best Practices
--------------

1. **Start with validation**::

    python -m cje validate data.jsonl -v

2. **Use calibrated-ips for initial analysis**::

    python -m cje analyze data.jsonl --output initial_results.json

3. **Try DR estimators if you have fresh draws**::

    python -m cje analyze data.jsonl \
        --estimator dr-cpo \
        --fresh-draws-dir ./responses \
        --output dr_results.json

4. **Export results for downstream analysis**::

    python -m cje analyze data.jsonl --output results.json
    # Then load in Python/R/Excel for further analysis

Environment Variables
---------------------

The CLI respects these environment variables:

* ``OPENAI_API_KEY`` - For judge evaluation (if using OpenAI)
* ``FIREWORKS_API_KEY`` - For log probability computation
* ``LOG_LEVEL`` - Set logging level (DEBUG, INFO, WARNING, ERROR)

Example::

    LOG_LEVEL=DEBUG python -m cje analyze data.jsonl --verbose

Troubleshooting
---------------

**"No oracle labels found"**

The dataset is missing oracle labels needed for calibration. Either:

1. Add oracle labels to your dataset
2. Use pre-calibrated rewards
3. Use raw IPS without calibration (not recommended)

**"Fresh draws missing for N prompts"**

DR estimators require fresh draws for all prompts. Either:

1. Generate fresh draws for missing prompts
2. Use ``--estimator calibrated-ips`` instead
3. Let the system use synthetic fresh draws (less accurate)

**"Insufficient oracle samples"**

You need at least 10 oracle labels (50+ recommended). Either:

1. Label more samples with oracle
2. Use a simpler estimator
3. Reduce the number of CV folds

See Also
--------

* :doc:`/getting_started` - Tutorial and examples
* :doc:`/api/analysis` - Python API reference
* :doc:`/estimators` - Detailed estimator descriptions