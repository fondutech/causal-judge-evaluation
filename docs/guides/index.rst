How-To Guides
=============

Practical guides for real-world CJE usage, organized by experience level and use case.

Getting Started Guides
----------------------

**New to CJE-Core?** Start here with the essentials:

.. toctree::
   :maxdepth: 2
   :caption: Essential Workflows

   user_guide

**Ready for specific use cases?** Choose your path:

.. toctree::
   :maxdepth: 2
   :caption: Core Use Cases

   arena_analysis
   configuration_reference

Advanced Guides
---------------

**Need specialized techniques?** Dive deeper:

.. toctree::
   :maxdepth: 2
   :caption: Specialized Topics

   weight_processing
   teacher_forcing
   oracle_analysis  
   trajectory_methods
   custom_components

📚 Learning Paths by Experience
-------------------------------

**🌟 Newcomer (0-1 weeks)**
   New to off-policy evaluation or CJE-Core
   
   1. :doc:`../quickstart` - 5-minute introduction
   2. :doc:`user_guide` - Essential workflows and troubleshooting  
   3. :doc:`configuration_reference` - Complete config documentation

**⚙️ Practitioner (1+ weeks)**
   Comfortable with basics, need specific solutions
   
   1. :doc:`arena_analysis` - Large-scale evaluation workflows
   2. :doc:`weight_processing` - Performance optimization and debugging
   3. :doc:`oracle_analysis` - High-precision validation techniques

**🔬 Advanced User (1+ months)**
   Need custom solutions or deep technical understanding
   
   1. :doc:`custom_components` - Build custom estimators and judges
   2. :doc:`trajectory_methods` - Multi-turn conversation analysis

🎯 Use Case Quick Navigation
---------------------------

**System Prompt Engineering**
   Testing different communication styles and formats
   → :doc:`user_guide` → "System Prompt Engineering"

**Model Comparison**
   Comparing different LLMs, versions, or configurations
   → :doc:`user_guide` → "Model Comparison"

**ChatBot Arena Analysis**  
   Large-scale human preference evaluation
   → :doc:`arena_analysis` → Complete workflow

**A/B Testing**
   Converting pairwise comparisons to utilities
   → :doc:`../tutorials/pairwise_evaluation` → Bradley-Terry modeling


**Performance Optimization**
   Debugging slow or unreliable evaluations
   → :doc:`weight_processing` → Advanced diagnostics

**Research & Experimentation**
   Custom methods and theoretical extensions
   → :doc:`custom_components` → Extension guide

🔧 Configuration Quick Reference
--------------------------------

**All configuration examples are now centralized** in :doc:`configuration_reference` to reduce duplication and improve maintainability.

**Quick Config Templates:**

.. code-block:: yaml

   # Minimal configuration
   dataset: {name: "./my_data.csv"}
   target_policies: [{name: "test", model_name: "gpt-4o-mini"}]
   
   # Production configuration  
   estimator: {name: "DRCPO", k: 5, clip: 20.0}
   judge: {provider: "openai", model_name: "gpt-4o-mini"}

🚨 Common Issues & Quick Fixes
------------------------------

**Wide confidence intervals**
   → More data, similar policies, try SNIPS → :doc:`user_guide` → "Troubleshooting"

**Estimators disagree**  
   → Check calibration, more labels → :doc:`weight_processing` → "Diagnostics"

**Slow performance**
   → Reduce samples, simpler models → :doc:`weight_processing` → "Optimization"

**Configuration errors**
   → Use ``cje validate`` → :doc:`configuration_reference` → "Validation"

**Weight processing issues**
   → Advanced diagnostics → :doc:`weight_processing` → "Pipeline Details"

🎓 Learning Resources
---------------------

**Hands-On Tutorials**
   Step-by-step examples with real data
   → :doc:`../tutorials/index`

**Mathematical Background**
   Theory and foundations
   → :doc:`../theory/index`

**API Documentation**
   Complete technical reference
   → :doc:`../api/index`

**Working Examples**
   Copy-paste code samples
   → ``examples/`` directory in the repository

Support & Community
-------------------

* **Search Documentation**: Use the search box (top-left) for specific topics
* **Configuration Issues**: See :doc:`configuration_reference` for complete syntax
* **Performance Problems**: Start with :doc:`weight_processing` diagnostics  
* **GitHub Issues**: Report bugs with configuration details and error messages
* **Examples**: All guides include working code examples you can modify 