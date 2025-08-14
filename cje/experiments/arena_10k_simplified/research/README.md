# Research Scripts

These scripts are experimental tools for ablation studies and advanced diagnostics. They are **not** part of the core pipeline.

## Scripts

### `oracle_diagnostics.py`
- **Purpose**: Advanced diagnostics for outcome model fidelity
- **Features**: R²(S→Y) analysis, oracle comparison metrics
- **Usage**: Standalone analysis tool for research

### `create_oracle_coverage_variants.py`
- **Purpose**: Create dataset variants with different oracle coverage levels
- **Features**: Systematic oracle label masking for ablation studies
- **Usage**: Research experiments on oracle coverage impact

### `analyze_oracle_coverage.py`
- **Purpose**: Run systematic oracle coverage experiments
- **Features**: Batch analysis across coverage levels
- **Usage**: `python research/analyze_oracle_coverage.py --data data/cje_dataset.jsonl`

## Note
These tools are preserved for research purposes but are not required for standard CJE analysis. Use the main pipeline scripts for production work.