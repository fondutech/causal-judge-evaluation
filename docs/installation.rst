Installation
============

CJE-Core requires Python 3.9 or later and can be installed via pip or from source.

Quick Installation
-----------------

.. code-block:: bash

   pip install cje-core

Development Installation
-----------------------

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/fondutech/CJE.git
   cd CJE
   poetry install

System Requirements
------------------

**Python Version**
   Python 3.9 or later

**Operating Systems**
   - Linux (tested on Ubuntu 20.04+)
   - macOS (tested on 10.15+) 
   - Windows (tested on Windows 10+)

**Memory Requirements**
   - Minimum: 4GB RAM
   - Recommended: 8GB+ RAM for large datasets

Dependencies
-----------

Core dependencies are automatically installed:

.. code-block:: text

   numpy>=1.26
   pandas>=2.2
   scikit-learn>=1.4
   torch>=2.2
   transformers>=4.52
   hydra-core>=1.3
   rich>=13.7

Optional Dependencies
--------------------

For enhanced functionality:

.. code-block:: bash

   # For API-based policy evaluation
   pip install openai anthropic google-generativeai
   
   # For advanced plotting
   pip install seaborn matplotlib
   
   # For development
   pip install black mypy pytest pre-commit

Verification
-----------

Verify your installation:

.. code-block:: python

   import cje
   print(f"CJE-Core version: {cje.__version__}")
   
   # Run basic test
   from cje.estimators import get_estimator
   estimator = get_estimator("IPS")
   print("✅ Installation successful!")

Common Issues
------------

**Import Errors**
   Make sure you're using the correct Python environment:
   
   .. code-block:: bash
   
      which python
      python -c "import sys; print(sys.path)"

**CUDA Issues**
   For GPU support, ensure PyTorch is installed with CUDA:
   
   .. code-block:: bash
   
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

**Memory Issues**
   For large datasets, consider:
   
   - Using smaller batch sizes
   - Enabling gradient checkpointing
   - Using mixed precision training 