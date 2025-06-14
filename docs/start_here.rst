Start Here: Choose Your Path
=============================

Welcome to CJE! Let's get you to the right place in 30 seconds.

.. raw:: html

   <style>
   .track-container {
       display: flex;
       gap: 20px;
       margin: 30px 0;
       flex-wrap: wrap;
   }
   .track-card {
       flex: 1;
       min-width: 250px;
       border: 2px solid #e1e4e8;
       border-radius: 8px;
       padding: 20px;
       background: #f8f9fa;
       transition: all 0.2s;
   }
   .track-card:hover {
       border-color: #0969da;
       background: #f3f4f6;
       transform: translateY(-2px);
   }
   .track-title {
       font-size: 1.3em;
       font-weight: bold;
       color: #0969da;
       margin-bottom: 10px;
   }
   .track-time {
       color: #57606a;
       font-size: 0.9em;
       margin-bottom: 15px;
   }
   .track-description {
       margin-bottom: 20px;
       line-height: 1.6;
   }
   .track-steps {
       border-top: 1px solid #d1d5da;
       padding-top: 15px;
       margin-top: 15px;
   }
   .track-button {
       display: inline-block;
       background: #0969da;
       color: white;
       padding: 8px 16px;
       border-radius: 6px;
       text-decoration: none;
       font-weight: bold;
       margin-top: 10px;
   }
   .track-button:hover {
       background: #0860ca;
       color: white;
       text-decoration: none;
   }
   </style>

   <div class="track-container">
   
   <div class="track-card">
   <div class="track-title">🏃 Track 1: Run CJE</div>
   <div class="track-time">Time: 5 minutes</div>
   <div class="track-description">
   <strong>I want to evaluate my LLM system right now</strong><br><br>
   Perfect for: Quick experiments, demos, or exploring what CJE can do
   </div>
   <div class="track-steps">
   <strong>You'll learn:</strong><br>
   ✓ Run your first evaluation<br>
   ✓ Understand the results<br>
   ✓ Try different configurations<br><br>
   <a href="quickstart.html" class="track-button">Start Running →</a>
   </div>
   </div>
   
   <div class="track-card">
   <div class="track-title">🔧 Track 2: Integrate CJE</div>
   <div class="track-time">Time: 30 minutes</div>
   <div class="track-description">
   <strong>I need to add CJE to my production system</strong><br><br>
   Perfect for: ML Engineers, DevOps, Platform teams
   </div>
   <div class="track-steps">
   <strong>You'll learn:</strong><br>
   ✓ Set up API providers<br>
   ✓ Configure for your data<br>
   ✓ Build evaluation pipelines<br><br>
   <a href="guides/user_guide.html" class="track-button">Start Integrating →</a>
   </div>
   </div>
   
   <div class="track-card">
   <div class="track-title">🎓 Track 3: Understand CJE</div>
   <div class="track-time">Time: 45 minutes</div>
   <div class="track-description">
   <strong>I want to understand the theory and methodology</strong><br><br>
   Perfect for: Researchers, Data Scientists, Academics
   </div>
   <div class="track-steps">
   <strong>You'll learn:</strong><br>
   ✓ Causal inference foundations<br>
   ✓ Algorithm deep-dives<br>
   ✓ Statistical guarantees<br><br>
   <a href="theory/mathematical_foundations.html" class="track-button">Start Learning →</a>
   </div>
   </div>
   
   </div>

Quick Decision Helper
---------------------

Still not sure? Answer this:

**What's your immediate goal?**

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - "I have data and want results NOW"
     - → :doc:`quickstart` (Track 1: Run)
   * - "I'm comparing prompt variations"
     - → :doc:`quickstart` (Track 1: Run)
   * - "I need to evaluate model upgrades"
     - → :doc:`guides/user_guide` (Track 2: Integrate)
   * - "I'm building an eval pipeline"
     - → :doc:`guides/user_guide` (Track 2: Integrate)
   * - "I want to understand the math"
     - → :doc:`theory/mathematical_foundations` (Track 3: Understand)
   * - "I'm implementing custom estimators"
     - → :doc:`guides/custom_components` (Track 3: Understand)

Common Starting Points
----------------------

**🔥 Most Popular Path** (80% of users)
   :doc:`quickstart` → :doc:`guides/configuration_reference` → :doc:`guides/user_guide`

**📊 Data Science Path**
   :doc:`quickstart` → :doc:`tutorials/pairwise_evaluation` → :doc:`guides/arena_analysis`

**🏗️ Engineering Path**
   :doc:`installation` → :doc:`api/index` → :doc:`guides/custom_components`

**📚 Research Path**
   :doc:`theory/mathematical_foundations` → :doc:`api/estimators` → :doc:`guides/weight_processing`

Prerequisites Check
-------------------

Before you start, make sure you have:

✅ **For Track 1 (Run)**: 
   - Python 3.9+ and git
   - An OpenAI or Anthropic API key
   - 5 minutes

✅ **For Track 2 (Integrate)**:
   - Everything from Track 1
   - Your logged interaction data (CSV/JSON)
   - Basic understanding of your evaluation metrics

✅ **For Track 3 (Understand)**:
   - Everything from Track 1
   - Familiarity with basic statistics
   - Interest in causal inference (we'll teach the rest!)

Need Help Choosing?
-------------------

.. tip::
   **When in doubt, start with Track 1** (:doc:`quickstart`). You can always dive deeper later. The quickstart gives you a working example in 5 minutes, which helps everything else make sense.

Ready? Pick your track above and let's go! 🚀