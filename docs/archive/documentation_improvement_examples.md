# Documentation Improvement Examples

## Example 1: Improved Homepage (index.rst)

### Current (Overwhelming)
```rst
CJE: Causal Judge Evaluation Toolkit
=====================================
[275 lines including theory, math, installation, etc.]
```

### Improved (Focused)
```rst
CJE: Test LLM Changes Without Deployment
========================================

**What if you could know if GPT-4 is worth the upgrade cost?**
CJE tells you - using your existing chat logs.

🚀 Try it in 2 minutes
----------------------
```python
# No installation needed - run in Colab!
!pip install cje
from cje import quick_evaluate

result = quick_evaluate(
    your_data="customer_chats.csv",
    old_model="gpt-3.5-turbo", 
    new_model="gpt-4",
    metric="customer_satisfaction"
)
print(f"Expected improvement: {result.improvement:.1%}")
print(f"Confidence: {result.confidence:.1%}")
```

[Try in Colab →] [See Full Example →] [Watch 2-min Video →]

Choose Your Starting Point
--------------------------
🏃 **Just need results?** → [2-Minute Quick Test]
🔧 **Building a system?** → [Integration Guide]  
🎓 **Want the science?** → [How It Works]

Real Teams Using CJE
--------------------
• **E-commerce:** "Tested 5 chatbot variants without annoying customers"
• **Support:** "Found 18% improvement from new prompts before deploying"
• **Research:** "Evaluated 50 model configurations in one afternoon"
```

## Example 2: True Quick Start Experience

### Current Quickstart Issues:
- Assumes installation done
- Requires config file creation
- No sample data
- Shows code without outputs

### Improved Quickstart:
```rst
2-Minute Quick Test
===================

Let's see CJE in action with real data - no setup needed!

Step 1: Open This Colab Notebook
---------------------------------
[Open in Colab - Click Here]

The notebook has:
✓ CJE pre-installed
✓ Sample customer service data  
✓ API keys for testing (rate-limited)

Step 2: Run One Cell
--------------------
```python
# This is already in the Colab - just press Run!
from cje import quick_evaluate

result = quick_evaluate(
    data="sample_support_chats.csv",  # 100 real examples
    old="Your current GPT-3.5 assistant",
    new="GPT-4 with improved prompt",
    judge="customer_satisfaction"
)
```

Step 3: See Your Results
------------------------
```
📊 Evaluation Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Old Policy Score: 72.3% satisfied
New Policy Score: 85.7% satisfied  
Improvement: +13.4% (95% CI: [11.2%, 15.6%])

💡 Recommendation: HIGH CONFIDENCE improvement
   Deploying the new policy would likely improve
   customer satisfaction by 11-16%.

📈 See detailed analysis → result.show_details()
💾 Download report → result.save_report()
```

What Just Happened?
-------------------
CJE used your historical chat logs to predict how GPT-4 would perform - without actually deploying it! [Learn more →]

Try Your Own Data
-----------------
```python
# Upload your CSV with columns: [user_message, bot_response]
from google.colab import files
uploaded = files.upload()

# Evaluate your scenarios
result = quick_evaluate(
    data=uploaded['your_file.csv'],
    old="gpt-3.5-turbo",
    new="gpt-4",
    # Automatic judge selection based on your data
)
```

Next: Real Installation
-----------------------
Happy with the results? Let's set up CJE locally:
→ [Installation Guide (10 min)]
→ [First Real Evaluation (15 min)]
→ [Production Setup (30 min)]
```

## Example 3: Better Navigation Structure

### Current Structure (Confusing):
```
index.rst → start_here.rst → quickstart.rst → user_guide.rst → ???
     ↑              ↓                               ↑
     └──────────────┴───────────────────────────────┘
```

### Improved Structure:
```
Home
├── 🚀 Quick Win (2 min Colab)
├── 📚 Learning Paths
│   ├── Path 1: First Evaluation (15 min)
│   │   ├── 1. Install CJE
│   │   ├── 2. Prepare Your Data  
│   │   ├── 3. Run Evaluation
│   │   └── 4. Interpret Results
│   ├── Path 2: Production Setup (45 min)
│   │   ├── 1. Architecture Overview
│   │   ├── 2. Configure Pipelines
│   │   ├── 3. Set Up Monitoring
│   │   └── 4. Deploy to Production
│   └── Path 3: Deep Dive (2 hours)
│       ├── 1. Causal Inference 101
│       ├── 2. CJE Algorithm Explained
│       ├── 3. Statistical Guarantees
│       └── 4. Research Applications
├── 🔧 Reference
│   ├── Configuration (single source)
│   ├── API Docs
│   ├── Troubleshooting
│   └── FAQ
└── 💡 Examples
    ├── Compare Models
    ├── Test Prompts
    ├── Analyze A/B Tests
    └── Custom Judges
```

## Example 4: Progressive Disclosure Pattern

### Current (Everything at Once):
```rst
Estimators
----------
CJE provides four estimators: IPS (unbiased, high variance), 
SNIPS (≈unbiased, lower variance), DR-CPO (single-rate efficient, 
double robust), and MRDR (semiparametric optimal, variance-minimizing).
The DR-CPO estimator implements Algorithm 1 from the paper...
```

### Improved (Progressive Disclosure):
```rst
Choosing an Estimator
--------------------

🎯 **Quick Recommendation:** Use `DRCPO` for most cases.

<details>
<summary>Want to choose based on your situation?</summary>

**Have lots of data? (10k+ examples)**
→ Use `IPS` - fastest and simplest

**Working with limited data? (<1k examples)**  
→ Use `MRDR` - best for small samples

**Need the best balance?**
→ Use `DRCPO` - recommended default

</details>

<details>
<summary>Curious about the technical details?</summary>

**IPS (Inverse Propensity Scoring)**
- ✅ Unbiased estimator
- ✅ Very fast computation  
- ❌ Can have high variance
- 📊 Best when: Large datasets, simple comparisons

**SNIPS (Self-Normalized IPS)**
- ✅ More stable than IPS
- ✅ Still fast
- ⚠️ Small bias for finite samples
- 📊 Best when: Medium datasets, robust baseline needed

[See full technical comparison →]
</details>
```

## Example 5: Better Error Messages & Troubleshooting

### Current:
```
ModuleNotFoundError: No module named 'cje'
```

### Improved:
```
❌ CJE Module Not Found

This usually means CJE isn't installed in your current environment.

Quick fixes:
1. If using pip:     pip install cje
2. If using poetry:  poetry add cje  
3. If developing:    pip install -e .

Still having issues? Check:
• Python version: python --version  (need 3.9+)
• Virtual env:    which python      (correct env?)
• Import test:    python -c "import cje; print(cje.__version__)"

📚 Full installation guide: https://cje.ai/install
💬 Get help: https://github.com/cje/issues
```

## Example 6: Interactive Config Builder

Instead of requiring users to write YAML:

```html
<!-- Embedded in documentation -->
<div id="cje-config-builder">
  <h3>Build Your Config</h3>
  
  <label>What's your current model?</label>
  <select id="current-model">
    <option>gpt-3.5-turbo</option>
    <option>gpt-4</option>
    <option>claude-2</option>
  </select>
  
  <label>What do you want to test?</label>
  <select id="test-scenario">
    <option>Different model</option>
    <option>New system prompt</option>
    <option>Temperature change</option>
  </select>
  
  <label>How much data do you have?</label>
  <select id="data-size">
    <option><100 examples</option>
    <option>100-1000 examples</option>
    <option>>1000 examples</option>
  </select>
  
  <button onclick="generateConfig()">Generate Config</button>
  
  <div id="config-output">
    <!-- Generated YAML appears here -->
  </div>
  
  <button onclick="downloadConfig()">Download</button>
  <button onclick="copyConfig()">Copy</button>
</div>
```

## Example 7: FAQ Section (Currently Missing)

```rst
Frequently Asked Questions
==========================

Getting Started
---------------

**Q: Do I need to retrain any models?**
No! CJE works with your existing models and data.

**Q: How much data do I need?**
Minimum ~100 examples, but 1000+ gives more reliable results.

**Q: Which providers are supported?**
OpenAI, Anthropic, Google, Fireworks, and any OpenAI-compatible API.

**Q: Can I use local models?**
Yes! Use HuggingFace transformers or any model with a scoring API.

Common Issues
-------------

**Q: Why are my confidence intervals so wide?**
This usually means:
1. Your policies are very different (check ESS diagnostic)
2. You need more data
3. Try SNIPS estimator for tighter bounds

**Q: Can I compare more than 2 policies?**
Yes! Add multiple entries to `target_policies` in your config.

**Q: How do I interpret negative improvements?**
A negative value means the new policy performs worse. The confidence
interval tells you how certain we are about this.

[See all FAQs →] [Ask a question →]
```

These examples demonstrate how to make the documentation more:
- **Accessible**: Start with outcomes, not theory
- **Practical**: Show real results immediately  
- **Progressive**: Reveal complexity gradually
- **Actionable**: Clear next steps at every point
- **Friendly**: Conversational tone, visual aids