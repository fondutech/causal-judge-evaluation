Understanding Oracle Types in CJE
=================================

CJE supports two distinct types of "oracle" for ground truth validation, which serve different purposes and should not be confused:

Automated Oracle (Model-Based)
------------------------------

**What it is**: Uses a stronger AI model (e.g., GPT-4o, Claude-3-Opus) to generate ground truth labels.

**When to use**:
- Quick validation during development
- When human labels are impractical or too expensive
- For rapid experimentation with different configurations

**Configuration example**:

.. code-block:: yaml

   # In config files like arena_test.yaml
   oracle:
     enabled: true
     provider: "openai"
     model_name: "gpt-4o"
     temperature: 0.0

**Cost**: ~$0.01-0.03 per sample (depending on model)

**Advantages**:
- Fast and automated
- Consistent labeling
- No manual coordination required
- Can be run on-demand

**Limitations**:
- Still an AI model, not true human judgment
- May have systematic biases
- More expensive than proxy judges

Human Oracle (Crowdsourced)
---------------------------

**What it is**: Collects ground truth labels from human annotators via crowdsourcing platforms.

**When to use**:
- Final validation for research papers
- When human judgment is the gold standard
- For high-stakes production deployments

**Configuration example**:

.. code-block:: yaml

   # In experiments like arena_10k_oracle
   oracle:
     enabled: false  # Human labels imported separately
     provider: "human"
     platform: "surge"  # Crowdsourcing platform
     cost_per_vote: 0.08
     votes_per_sample: 3

**Cost**: ~$0.08-0.30 per sample (depending on platform and voting)

**Advantages**:
- True human judgment
- No model biases
- Gold standard for validation
- Can capture nuanced preferences

**Limitations**:
- Slow (days to collect)
- Requires manual coordination
- More expensive overall
- Potential annotator inconsistency

Key Differences
---------------

.. list-table:: Oracle Type Comparison
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Automated Oracle
     - Human Oracle
   * - **Speed**
     - Minutes to hours
     - Days to weeks
   * - **Cost per label**
     - $0.01-0.03
     - $0.08-0.30
   * - **Integration**
     - Fully automated in CJE
     - Manual export/import process
   * - **Quality**
     - Strong model approximation
     - True human judgment
   * - **Use case**
     - Development & testing
     - Final validation & research

Implementation Notes
--------------------

**Automated Oracle**:
- Set ``oracle.enabled: true`` in config
- Runs automatically during CJE pipeline
- Results included in standard output

**Human Oracle**:
- Set ``oracle.enabled: false`` in config
- Use export scripts to prepare data
- Coordinate with crowdsourcing platform
- Import results back for analysis
- See ``experiments/arena_10k_oracle`` for complete example

Common Confusion Points
-----------------------

1. **Config naming**: Both use ``oracle:`` config section but are fundamentally different
2. **Enable flag**: Human oracle has ``enabled: false`` because labels are imported, not generated
3. **Directory names**: ``arena_10k_oracle`` refers to human oracle experiment, not automated
4. **Documentation**: Always specify which type of oracle you're discussing

Best Practices
--------------

1. **Start with automated oracle** during development for quick iteration
2. **Use human oracle** for final validation and paper results  
3. **Be explicit** in documentation about which oracle type you're using
4. **Budget accordingly**: Human oracle can be 10x more expensive but provides true ground truth
5. **Consider hybrid**: Use automated oracle for most samples, human oracle for validation subset