# CJE Examples

**Practical code examples for real-world CJE usage** - Copy, modify, and run these examples to jumpstart your evaluation workflows.

## 📖 Documentation Integration

**For complete documentation** → [CJE Documentation](https://causal-judge-evaluation.readthedocs.io/)

**For configuration help** → [Configuration Reference](https://causal-judge-evaluation.readthedocs.io/en/latest/guides/configuration_reference.html)

## 🚀 Quick Start Examples

### Interactive Arena Analysis

**File**: `arena_interactive.py` (26KB, 761 lines)

**What it does**: Complete interactive tool for ChatBot Arena-style evaluation with visualization, statistical analysis, and export capabilities.

**Perfect for**: Data scientists, researchers doing arena analysis, anyone who wants a comprehensive evaluation tool

```bash
# Run with built-in sample data
python examples/arena_interactive.py

# Run with your own data
python examples/arena_interactive.py --data-path ./my_arena_data.jsonl
```

**Features**:
- 🎯 **Policy Comparison**: Confidence intervals, significance testing
- 📊 **Visualizations**: Automated plots saved as PNG files  
- 🔍 **Judge Analysis**: Correlation statistics between proxy and oracle judges
- 📋 **Export**: CSV summaries for further analysis
- ⚡ **Quick Test Mode**: Built-in sample data for immediate testing

### Python Interface Examples

**File**: `arena_analysis_python.py` (8KB, 224 lines)

**What it does**: Clean Python API examples showing how to integrate CJE into your existing workflows.

**Perfect for**: ML engineers, platform developers, anyone integrating CJE into larger systems

```python
from examples.arena_analysis_python import run_basic_evaluation

# Basic evaluation
results = run_basic_evaluation(
    data_path="./my_data.csv",
    target_model="gpt-4o",
    baseline_model="gpt-3.5-turbo"
)

print(f"Improvement: {results.improvement}")
print(f"Confidence: {results.confidence_interval}")
```

**See**: `README_python_interface.md` for detailed API examples

## 🔧 Configuration & Setup Examples

### Environment Variables (Recommended)

**What it does**: Simple API key setup using standard environment variables.

**Quick Setup**:

```bash
# Set your API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."
export GOOGLE_API_KEY="..."  # optional
export FIREWORKS_API_KEY="..."  # optional

# Run CJE
cje run --cfg-path configs --cfg-name arena_test
```

**Key Features**:
- 🚀 **Simple Setup**: Just set environment variables
- 🔐 **Secure**: No secrets in code or config files
- 🌍 **Standard Approach**: Works with all API providers
- 📋 **Flexible**: Add any API keys you need

## 📚 Learning Path by User Type

### 🌟 New Users (First Time)
1. **Start here**: `arena_interactive.py` with quick test mode
2. **Understand results**: Built-in explanations and visualizations
3. **Try your data**: Replace sample data with your CSV/JSONL files

### ⚙️ Practitioners (Integration Focus)  
1. **API patterns**: `arena_analysis_python.py` → Python integration
2. **Configuration**: Set environment variables for your API keys
3. **Production**: Scale with your preferred deployment approach

### 🔬 Researchers (Customization Focus)
1. **Complete tool**: `arena_interactive.py` → Full feature set
2. **Statistical methods**: Examine the analysis functions  
3. **Extend**: Use these as templates for custom analysis

## 🎯 Use Case Examples

### Arena Analysis Workflow

**Complete End-to-End Example**:

```bash
# 1. Generate logs (if needed)
cje log run --dataset ./data/queries.csv --model gpt-3.5-turbo --out logs.jsonl

# 2. Run arena analysis
python examples/arena_interactive.py --data-path logs.jsonl

# 3. Analyze results  
# → Automatic plots saved as PNG
# → CSV summary exported
# → Statistical significance testing
```

### Quick API Integration

**Copy-Paste Python Code**:

```python
# From arena_analysis_python.py
from cje.pipeline import run_pipeline

config = {
    "dataset": {"name": "./my_data.csv"},
    "target_policies": [{
        "name": "enhanced", 
        "model_name": "gpt-4o",
        "provider": "openai"
    }],
    "estimator": {"name": "DRCPO"}
}

results = run_pipeline(config)
print(f"Best policy: {results.best_policy()}")
```

### Production Configuration

**Environment-Based Configuration**:

```python
# From arena_analysis_python.py  
import os
from cje.config import create_config

# Use environment variables for API keys
config = {
    "dataset": {"name": "./my_data.csv"},
    "target_policies": [{
        "name": "enhanced", 
        "model_name": "gpt-4o",
        "provider": "openai"  # Uses OPENAI_API_KEY env var
    }],
    "estimator": {"name": "DRCPO"}
}

results = run_pipeline(config)
```

## 🔄 Running Examples

### Prerequisites

```bash
# Install CJE
pip install cje

# Install optional dependencies for examples
pip install pandas matplotlib seaborn scipy boto3
```

### Quick Tests

```bash
# Test arena analysis (uses built-in sample data)
python examples/arena_interactive.py

# Test Python API
python examples/arena_analysis_python.py
```

### With Your Data

```bash
# Arena analysis with your data
python examples/arena_interactive.py --data-path ./your_data.csv

# Skip visualizations (for headless environments)
python examples/arena_interactive.py --data-path ./your_data.csv --no-plots

# Specify output directory
python examples/arena_interactive.py --output-dir ./my_results
```

## 📊 Example Outputs

### What You'll Get

**Console Output**:
- Policy comparison tables with confidence intervals
- Statistical significance tests
- Judge calibration analysis
- Performance benchmarks

**File Outputs**:
- `policy_comparison.png` - Visualization of results
- `policy_summary.csv` - Exportable data for further analysis
- `detailed_results.json` - Complete results for programmatic access

**Example Results**:
```
Policy Comparison Results:
┌─────────────┬──────────────┬─────────────┬─────────────┐
│ Policy      │ Estimate     │ Std Error   │ 95% CI      │
├─────────────┼──────────────┼─────────────┼─────────────┤
│ enhanced    │ 0.742        │ 0.023       │ [0.69, 0.79]│
│ baseline    │ 0.684        │ 0.025       │ [0.63, 0.74]│
└─────────────┴──────────────┴─────────────┴─────────────┘

Improvement: +0.058 (p < 0.05) ✓ Significant
```

## 🛠️ Extending Examples

### Customization Patterns

**Add Your Own Analysis**:
```python
# Extend arena_interactive.py
class MyCustomAnalyzer(ArenaAnalyzer):
    def custom_analysis(self):
        # Your custom analysis code
        pass
```

**Integration Templates**:
- Jupyter notebook integration
- Flask/FastAPI web service  
- Automated reporting pipelines
- Research experiment frameworks

### Contributing Examples

**Want to add an example?**
1. Follow the existing code style and documentation patterns
2. Include both console output and file export options
3. Add comprehensive docstrings and error handling
4. Test with sample data and document prerequisites

## 🔗 Related Documentation

- **[Complete User Guide](https://causal-judge-evaluation.readthedocs.io/en/latest/guides/user_guide.html)** - Essential workflows and troubleshooting
- **[Arena Analysis Guide](https://causal-judge-evaluation.readthedocs.io/en/latest/guides/arena_analysis.html)** - Detailed arena methodology
- **[Configuration Reference](https://causal-judge-evaluation.readthedocs.io/en/latest/guides/configuration_reference.html)** - All configuration options
- **[API Documentation](https://causal-judge-evaluation.readthedocs.io/en/latest/api/index.html)** - Complete technical reference

---

**Quick Links**: [📖 Full Documentation](https://causal-judge-evaluation.readthedocs.io/) | [🚀 5-Minute Quickstart](https://causal-judge-evaluation.readthedocs.io/en/latest/quickstart.html) | [⚙️ Configuration Help](https://causal-judge-evaluation.readthedocs.io/en/latest/guides/configuration_reference.html) 