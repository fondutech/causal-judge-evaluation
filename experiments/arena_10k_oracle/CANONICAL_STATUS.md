# Canonical Arena 10K Setup - Status

## ✅ Everything is Canonical and Consistent

### Documentation
- **Single README.md**: Consolidated all information
- **Removed**: 5 redundant documentation files
- **Phase-specific docs**: Kept in subdirectories

### Code Consistency
- **4 Target Policies** everywhere (including pi_clone)
- **Script names** match actual files
- **Cost estimates** updated for 4 policies
- **API call counts** consistent across all files

### Fixed Issues
1. Updated all "3 policies" → "4 policies"
2. Fixed script references in `run_full_pipeline.sh`
3. Updated monitoring tools for correct counts
4. Consolidated redundant documentation

### Canonical Target Policies
```python
{
    "pi_clone": "Same as P0 (baseline)",
    "pi_cot": "Chain-of-thought prompting", 
    "pi_bigger_model": "Larger model (maverick)",
    "pi_bad": "Deliberately poor policy"
}
```

### Canonical Costs (Full Run)
- **API Calls**: ~140,000
- **Total Cost**: ~$60
- **Time**: 50-75 hours

### Canonical File Structure
```
arena_10k_oracle/
├── README.md                    # Single source of truth
├── data/                        # All data files
├── phase1_dataset_preparation/  # Data generation
│   ├── *.py scripts            # Canonical pipeline
│   ├── sample_run/             # 1% testing tools
│   └── README_PHASE1.md        # Phase-specific details
└── phase2_cje_ablations/       # Analysis experiments
    └── configs/                # All include pi_clone
```

## Ready to Run

No conflicting information remains. All references are consistent:
- 4 target policies (including pi_clone)
- Updated costs and counts
- Correct script names
- Single authoritative README

The experiment is ready for the 1% sample test.