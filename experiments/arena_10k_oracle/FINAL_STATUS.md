# Arena 10K Oracle Experiment - Final Status

## ✅ Completed Work

### 1. Critical Bug Fix in CJE
- **File**: `/cje/loggers/api_policy.py`
- **Issue**: API failures were returning `0.0` log probability (implies P=1.0)
- **Fix**: Now raises `RuntimeError` on API failures instead of silent corruption
- **Impact**: Prevents importance weight corruption in off-policy evaluation

### 2. Data Quality Improvements
- **Rescored 139 P0 samples** that had incorrect `-50.0` log probabilities
- Most (120) turned out to have legitimate `0.0` log prob (model is very confident)
- Only 1 sample failed rescoring due to empty response
- **Created**: `p0_with_target_logps_fully_fixed.jsonl` with corrected values

### 3. Re-ran CJE Ablations
With the fixed data (1,824 samples):
- `pi_bigger_model` now has highest IPS estimate (as expected)
- `pi_bad` still has highest SNIPS (0.910) but gap is much smaller
- ESS remains very low (<7%) for all policies - more samples needed

### 4. Aggressive Cleanup
- Removed 16 temporary Python scripts
- Removed 16 temporary data files
- Archived important analysis scripts and checkpoints to `archive_20250626/`
- Cleaned directory structure for clarity

## 📊 Current State

### Active Processes
- **Teacher Forcing**: ~37% complete (3,712/10,000 samples)
- Process PID: 22291 (check with `ps aux | grep 02c_compute`)

### Key Data Files
- `data/p0_with_target_logps_fully_fixed.jsonl` - Best available data (1,824 samples)
- `data/p0_with_target_logps.checkpoint.jsonl` - Active checkpoint (growing)
- `data/p0_scored_*.jsonl` - Judge scores for all samples

### Key Scripts
- `phase1_dataset_preparation/` - Complete pipeline scripts
- `phase2_cje_ablations/run_direct_ablations.py` - Main ablation runner
- `phase2_cje_ablations/visualize_results.py` - Result visualization

## 🔮 Next Steps

1. **Wait for Teacher Forcing Completion**
   - Currently at ~37%, estimate 4-6 more hours
   - Will produce complete `p0_with_target_logps.jsonl`

2. **Run Full Ablations**
   ```bash
   cd phase2_cje_ablations
   python run_direct_ablations.py \
     --input ../data/p0_with_target_logps.jsonl \
     --output-dir ../results/ablations_final
   ```

3. **Submit PR to CJE Repository**
   - Fix for `api_policy.py` error handling
   - Prevents silent log probability failures

## 📈 Key Findings

1. **Distribution Mismatch**: Extreme mismatch between P0 and target policies causes very low ESS
2. **API Reliability**: Silent API failures can corrupt importance weights
3. **SNIPS vs IPS**: SNIPS can give misleading rankings with low ESS
4. **Model Confidence**: Many simple responses get log prob = 0.0 (P = 1.0)

## 🗂️ Archive Contents
All temporary analysis scripts and intermediate checkpoints saved to `archive_20250626/`