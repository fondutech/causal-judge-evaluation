#!/usr/bin/env python3
"""Check setup status for Arena 10K experiment with llama.cpp"""

import os
import sys
from pathlib import Path

print("🦙 Arena 10K Setup Status Check")
print("=" * 50)

# Check llama-cpp-python
try:
    import llama_cpp

    print("✅ llama-cpp-python is installed")
except ImportError:
    print("❌ llama-cpp-python NOT installed")
    print("   Run: pip install llama-cpp-python")

# Check model file
model_path = Path(
    "/Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/models/Llama-3.2-3B-Instruct-Q6_K.gguf"
)
if model_path.exists():
    size_gb = model_path.stat().st_size / (1024**3)
    print(f"✅ Model file exists ({size_gb:.1f} GB)")
else:
    print("❌ Model file NOT found")
    print(
        "   Check: /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/models/"
    )

# Check API keys
openai_key = os.environ.get("OPENAI_API_KEY")
if openai_key:
    print("✅ OPENAI_API_KEY is set (for judge/oracle)")
else:
    print("❌ OPENAI_API_KEY NOT set")
    print(
        "   Run: source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh"
    )

# Check data directories
if Path("phase1_dataset_preparation/data").exists():
    print("⚠️  Old data directory exists - will be overwritten")
else:
    print("✅ Clean start - no existing data")

print("\n" + "=" * 50)

# Overall status
if model_path.exists() and openai_key:
    print("✅ READY TO RUN!")
    print("\nNext step:")
    print("  cd phase1_dataset_preparation")
    print("  python run_phase1_pipeline.py 10")
else:
    print("❌ NOT READY - Please fix the issues above")
