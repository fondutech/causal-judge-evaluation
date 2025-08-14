# Arena 10K Experiment Script Audit

## 📊 Overview
Total scripts: 20 Python files
Total lines: ~5,200 lines of code

## 🎯 Core Pipeline Scripts

### Main Entry Points
1. **`generate_arena_data.py`** (357 lines) ✅
   - **Purpose**: Main pipeline orchestrator
   - **Status**: ACTIVE - Primary entry point
   - **Dependencies**: All pipeline_steps/*, experiment_config
   - **Usage**: `poetry run python generate_arena_data.py --n-samples 5000`

2. **`analyze_dataset.py`** (939 lines) ✅
   - **Purpose**: Run CJE analysis on generated data
   - **Status**: ACTIVE - Analysis entry point
   - **Dependencies**: reward_utils, validation, oracle_comparison
   - **Usage**: `poetry run python analyze_dataset.py --data data/cje_dataset.jsonl`

### Pipeline Steps (`pipeline_steps/`)
3. **`prepare_arena_data.py`** (213 lines) ✅
   - **Purpose**: Extract prompts from ChatBot Arena dataset
   - **Status**: ACTIVE - Step 1 of pipeline
   - **Called by**: generate_arena_data.py

4. **`generate_responses.py`** (532 lines) ✅
   - **Purpose**: Generate responses using different policies
   - **Status**: ACTIVE - Step 2 of pipeline
   - **Features**: Retry logic, resume capability

5. **`add_scores_with_resume.py`** (372 lines) ✅
   - **Purpose**: Add judge/oracle scores with resume capability
   - **Status**: ACTIVE - Steps 3-4 of pipeline
   - **Features**: Batch processing, progress tracking

6. **`compute_logprobs.py`** (323 lines) ✅
   - **Purpose**: Compute log probabilities for responses
   - **Status**: ACTIVE - Step 5 of pipeline

7. **`prepare_cje_data.py`** (235 lines) ✅
   - **Purpose**: Combine all data into final CJE dataset
   - **Status**: ACTIVE - Step 6 of pipeline

## 🔧 Configuration & Utilities

8. **`experiment_config.py`** (198 lines) ✅
   - **Purpose**: Centralized configuration
   - **Status**: ACTIVE - Core configuration
   - **Contains**: Models, batch sizes, policies, validation

9. **`evaluation_utils.py`** (253 lines) ✅
   - **Purpose**: Shared evaluation utilities
   - **Status**: ACTIVE - Shared by scoring scripts
   - **Features**: FireworksEvaluator, structured outputs

10. **`reward_utils.py`** (246 lines) ✅
    - **Purpose**: Reward configuration and application
    - **Status**: ACTIVE - Used by analyze_dataset.py
    - **Features**: Oracle coverage handling

11. **`validation.py`** (87 lines) ✅
    - **Purpose**: Validation utilities
    - **Status**: ACTIVE - Prevents analysis mistakes

12. **`oracle_comparison.py`** (190 lines) ✅
    - **Purpose**: Compare estimates to oracle ground truth
    - **Status**: ACTIVE - Used for evaluation

## 🧪 Test Scripts

13. **`test_full_pipeline.py`** (247 lines) ✅
    - **Purpose**: End-to-end pipeline testing
    - **Status**: ACTIVE - Integration tests
    - **Usage**: `poetry run python test_full_pipeline.py --n-samples 10`

14. **`test_resume_pipeline.py`** (195 lines) ✅
    - **Purpose**: Test resume functionality
    - **Status**: ACTIVE - Resume tests

15. **`test_reward_handling.py`** (208 lines) ✅
    - **Purpose**: Test reward handling logic
    - **Status**: ACTIVE - Unit tests

## 📈 Analysis & Experiments

16. **`analyze_oracle_coverage.py`** (363 lines) ⚠️
    - **Purpose**: Oracle coverage ablation studies
    - **Status**: EXPERIMENTAL - Research tool
    - **Note**: For studying impact of oracle coverage levels

17. **`oracle_diagnostics.py`** (331 lines) ⚠️
    - **Purpose**: Detailed oracle diagnostics
    - **Status**: SPECIALIZED - Advanced diagnostics
    - **Note**: R²(S→Y) analysis, outcome model fidelity

18. **`create_oracle_coverage_variants.py`** (261 lines) ⚠️
    - **Purpose**: Create dataset variants with different oracle coverage
    - **Status**: EXPERIMENTAL - Research tool
    - **Note**: For ablation studies

## 🔍 Helper Scripts

19. **`verify_setup.py`** (52 lines) ✅
    - **Purpose**: Pre-flight checks
    - **Status**: ACTIVE - Setup verification
    - **Usage**: `poetry run python verify_setup.py`

20. **`__init__.py`** (1 line) ✅
    - **Purpose**: Package marker
    - **Status**: ACTIVE

## 🚨 Findings & Recommendations

### ✅ Well-Organized Scripts
- Core pipeline is clean and modular
- Clear separation between pipeline steps
- Good test coverage
- Configuration properly centralized

### ⚠️ Experimental/Research Scripts
These are research tools, not part of the main pipeline:
- `analyze_oracle_coverage.py` - Keep for research
- `oracle_diagnostics.py` - Keep for advanced analysis
- `create_oracle_coverage_variants.py` - Keep for experiments

### 🎯 No Redundant Scripts Found
All scripts serve distinct purposes. The codebase is well-organized with:
- Clear pipeline flow (generate → analyze)
- Proper separation of concerns
- Good test coverage
- Research tools clearly separated

## 📝 Script Dependencies Graph

```
generate_arena_data.py
├── experiment_config.py
├── pipeline_steps/
│   ├── prepare_arena_data.py
│   ├── generate_responses.py
│   ├── add_scores_with_resume.py
│   │   └── evaluation_utils.py
│   ├── compute_logprobs.py
│   └── prepare_cje_data.py

analyze_dataset.py
├── experiment_config.py
├── reward_utils.py
├── validation.py
├── oracle_comparison.py
└── oracle_diagnostics.py (optional)
```

## 💡 Maintenance Notes

1. **Core Pipeline**: 7 scripts that must be maintained
2. **Analysis Tools**: 2 primary analysis scripts
3. **Tests**: 3 test scripts ensure reliability
4. **Research**: 3 experimental scripts for ablations
5. **Config/Utils**: 5 support scripts

Total: 20 well-organized scripts with clear purposes.