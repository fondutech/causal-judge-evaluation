Pairwise Evaluation Tutorial
============================

This tutorial covers how to use **pairwise human preference data** with CJE. Instead of scalar labels (⭐ ratings, clicks, revenue) you may only have **comparisons** like "A > B". CJE converts these *ordinal* preferences into continuous utilities with the **Bradley-Terry** model and then runs the usual causal inference pipeline.

When to Use Pairwise Data
-------------------------

Use pairwise evaluation when:

• **Crowd-sourcing A/B judgements** is cheaper than collecting numeric scores
• **Users can reliably say which answer they prefer** even when they cannot provide exact ratings  
• **Leader-boards exist** such as **LMSYS Chatbot Arena** with huge pairwise datasets
• **You have comparison data** that says *which* of two items wins (or that they tie)

How It Works
------------

1. **Bradley-Terry fit** – CJE fits utilities :math:`u_i` such that

   .. math::
   
      \Pr\left(i \text{ beats } j \right)=\frac{e^{u_i}}{e^{u_i}+e^{u_j}}

2. **Utility scaling** – utilities are pushed through a sigmoid so they lie in :math:`[0,1]`; this makes them interchangeable with the usual ``y_true`` labels used elsewhere in CJE

3. **CJESample emission** – for *each* side of the comparison we emit one sample with the computed utility

After that the pipeline (judge, estimator, statistics) is identical to the scalar-label workflow.

Quick Start Example
-------------------

Let's walk through a complete example using the ChatBot Arena dataset:

.. code-block:: python

   from cje.data import ChatbotArenaDataset
   from cje.config.unified import simple_config

   # 1. Download dataset (≈50 MB cached in ~/.cache/cje/pairwise)
   dataset = ChatbotArenaDataset.download(split="train", model_aware=True)
   print("Samples:", len(dataset))          # 2 × conversations
   print("Top models", dataset.get_model_rankings()[:5])

   # 2. Run a temperature experiment on the Arena utilities
   config = simple_config(
       dataset_name="ChatbotArena",   # built-in alias
       logging_model="gpt-4o-mini",   # current production model
       logging_provider="openai",
       target_model="gpt-4o-mini",    # same model with different temp
       target_provider="openai",
       target_changes={"temperature": 0.3},  # proposed tweak
       judge_model="gpt-4o",
       judge_provider="openai",
       estimator_name="DRCPO"
   )
   results = config.run()

   # 3. Analyze results
   print(f"Value estimate: {results.v_hat[0]:.3f}")
   ci_low, ci_high = results.confidence_interval()
   print(f"95% CI: [{ci_low[0]:.3f}, {ci_high[0]:.3f}]")

Expected Output
~~~~~~~~~~~~~~~

.. code-block:: text

   Samples: 45732
   Top models [('gpt-4', 0.847), ('claude-3-opus', 0.823), ...]
   Value estimate: 0.156
   95% CI: [0.089, 0.223]

This means the temperature change would improve quality by about 15.6% with statistical significance.

Command-Line Alternative
~~~~~~~~~~~~~~~~~~~~~~~~

The same analysis can be run from the command line:

.. code-block:: bash

   cje run \
     dataset.name=ChatbotArena \
     logging_model=gpt-4o-mini \
     target_changes.temperature=0.3

Using Your Own Pairwise Data
----------------------------

To use your own pairwise comparison data:

1. **Store the data** in **Parquet** or **Arrow** (recommended) or any HuggingFace-compatible dataset format

2. **Ensure proper schema** with these required columns:
   
   - Two responses (``response_a``, ``response_b``)
   - Comparison identifier (``conversation_id``, ``id``, etc.)
   - Winner column with values ``"A"``, ``"B"`` or ``"tie"``

3. **Load with the generic adapter**:

.. code-block:: python

   from cje.data.pairwise import PairwiseComparisonDataset

   dataset = PairwiseComparisonDataset.download(
       dataset_name="your_org/your_pairwise_dataset",
       split="train",
       regularization=0.02,   # optional L2 strength
   )

Custom Schema Handling
~~~~~~~~~~~~~~~~~~~~~~

If your schema differs, subclass :class:`~cje.data.pairwise.PairwiseComparisonDataset` and override ``_extract_comparisons()``:

.. code-block:: python

   from cje.data.pairwise import PairwiseComparisonDataset

   class CustomPairwiseDataset(PairwiseComparisonDataset):
       def _extract_comparisons(self, dataset):
           """Extract comparisons from your custom format."""
           comparisons = []
           for row in dataset:
               # Your custom logic here
               winner = "A" if row["preference"] == "left" else "B"
               comparisons.append({
                   "item_a": row["left_response"],
                   "item_b": row["right_response"], 
                   "winner": winner,
                   "comparison_id": row["id"]
               })
           return comparisons

Fine-Tuning the Bradley-Terry Model
-----------------------------------

For advanced use cases, you can directly work with the Bradley-Terry model:

.. code-block:: python

   from cje.data.pairwise import BradleyTerryModel

   # Define comparisons: (winner_id, loser_id, weight)
   comparisons = [
       ("item_1", "item_2", 1.0),  # 1 beats 2
       ("item_2", "item_3", 0.5),  # tie
       ("item_3", "item_1", 1.0),  # 3 beats 1
   ]

   # Fit the model
   bt = BradleyTerryModel(regularization=0.005)
   bt.fit(comparisons)
   
   # Get utilities
   print(f"Item 1 utility: {bt.get_utility('item_1'):.3f}")
   print(f"Item 2 utility: {bt.get_utility('item_2'):.3f}")
   print(f"Item 3 utility: {bt.get_utility('item_3'):.3f}")

Important Notes
~~~~~~~~~~~~~~~

- **Regularization** stabilizes utilities when the comparison graph is sparse or cyclic
- **Ties are encoded** with weight ``0.5`` **in both directions** so the log-likelihood gradients are correct
- **Global utilities** - re-fit the Bradley-Terry model whenever you add new data

Pipeline Visualization
----------------------

Here's how pairwise data flows through the CJE pipeline:

.. code-block:: text

   Pairwise Data    Bradley-Terry    Utilities      CJESample       Estimator
   (A > B, C > A)  ──────────────→  u_A, u_B, u_C ──────────────→ (IPS/DRCPO/...)
        │                              │                │               │
        └── fit preferences             └── sigmoid      └── y_true      └── Confidence
            to utilities                    [0,1]           = u_i           Intervals

The beauty of this approach is that after Bradley-Terry fitting, everything else in CJE works exactly the same as with scalar labels.

Best Practices
--------------

.. admonition:: Recommended Guidelines
   :class: tip

   1. **≥ 5 comparisons per item** for reliable utilities
   2. **Add regularization** (``0.005 – 0.05``) to handle intransitivity
   3. **Include ties explicitly** - do *not* drop them as they carry information!
   4. **Re-fit regularly** - utilities are global and should be updated when adding data

Common Issues and Solutions
---------------------------

**Sparse Comparison Graph**
   *Problem*: Some items have very few comparisons
   
   *Solution*: Increase regularization parameter (try ``0.02`` to ``0.05``)

**Cyclic Preferences** 
   *Problem*: A beats B, B beats C, C beats A
   
   *Solution*: This is normal! Bradley-Terry handles intransitivity via regularization

**Utility Scaling Issues**
   *Problem*: Utilities don't seem meaningful
   
   *Solution*: Remember utilities are relative - only differences matter for CJE

**Poor Convergence**
   *Problem*: Bradley-Terry fitting fails
   
   *Solution*: Check data format, increase regularization, ensure enough comparisons

Advanced Example: Multi-Model Arena
-----------------------------------

Here's a more complex example comparing multiple models using Arena data:

.. code-block:: python

   from cje.config.unified import UnifiedConfig
   from cje.data import ChatbotArenaDataset

   # Load arena data
   dataset = ChatbotArenaDataset.download(split="train", model_aware=True)
   
   # Get top models for comparison
   rankings = dataset.get_model_rankings()
   top_models = [model for model, _ in rankings[:3]]
   
   # Create multi-policy comparison
   target_policies = [
       {"name": f"{model}_baseline", "model_name": model, "temperature": 0.7}
       for model in top_models
   ]
   
   # Build config manually for multi-policy
   config = UnifiedConfig(
       dataset={"name": "ChatbotArena"},
       logging_policy={
           "provider": "openai",
           "model_name": "gpt-3.5-turbo"
       },
       target_policies=target_policies,
       judge={
           "provider": "openai",
           "model_name": "gpt-4o"
       },
       estimator={"name": "DRCPO"}
   )
   
   results = config.run()
   
   # Compare all policies
   for i, policy in enumerate(results.policy_names):
       est = results.v_hat[i]
       se = results.se[i]
       print(f"{policy}: {est:.3f} ± {se:.3f}")

This approach lets you use real human preference data to compare different model configurations.

Limitations and Future Work
---------------------------

Current limitations:

• **Single latent utility** - assumes one utility per item, doesn't model context-dependent preferences
• **No listwise/ranking data** - currently only handles pairwise comparisons
• **Missing API integration** - drops API costs/log probabilities

Future enhancements:

• Support for ranking data and listwise comparisons
• Context-dependent preference modeling  
• Integration with ``backfill compute`` for API costs

Further Reading
---------------

**Academic References:**
   • Bradley & Terry (1952). *Rank Analysis of Incomplete Block Designs*
   • Tsukida & Gupta (2011). *A survey of Bradley-Terry model and its extensions*

**Related Documentation:**
   • :doc:`../theory/mathematical_foundations` - Mathematical foundations
   • :doc:`../api/estimators` - Estimator implementation details
   • :doc:`../guides/weight_processing` - Technical pipeline details
   • :doc:`arena_analysis` - Arena-specific analysis techniques

Next Steps
----------

Now that you understand pairwise evaluation:

1. **Try the examples** with your own preference data
2. **Explore arena analysis** for comprehensive model comparisons  
3. **Learn about custom components** for specialized use cases
4. **Check the theory section** for mathematical details 